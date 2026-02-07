#!/usr/bin/env python3
"""Detect formats, normalize to UTF-8 text, clean corpus, and export PTB POS templates."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import re
import sys
import time
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Iterator
from xml.etree import ElementTree as ET


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".log",
    ".cfg",
    ".ini",
    ".yaml",
    ".yml",
}
HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}
XML_EXTENSIONS = {".xml"}
CSV_EXTENSIONS = {".csv", ".tsv"}
JSON_EXTENSIONS = {".json", ".jsonl", ".ndjson"}
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


PTB_PUNCT_TAGS = {
    ",": ",",
    ".": ".",
    ":": ":",
    ";": ":",
    "?": ".",
    "!": ".",
    "(": "-LRB-",
    ")": "-RRB-",
    "[": "-LSB-",
    "]": "-RSB-",
    "{": "-LCB-",
    "}": "-RCB-",
    "``": "``",
    "''": "''",
    '"': '"',
    "'": "'",
    "-": ":",
    "--": ":",
    "...": ":",
    "#": "#",
    "$": "$",
}

UTF8_ENCODINGS = ["utf-8", "utf-8-sig"]
UTF16_ENCODINGS = ["utf-16", "utf-16-le", "utf-16-be"]
LEGACY_ENCODINGS = ["gb18030", "big5", "shift_jis", "cp1252"]


a = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
}
PRONOUNS = {
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
}
POSSESSIVE_PRONOUNS = {"mine", "yours", "hers", "ours", "theirs", "whose"}
CONJUNCTIONS = {"and", "or", "but", "nor", "yet", "so"}
PREPOSITIONS = {
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "with",
    "by",
    "of",
    "into",
    "over",
    "under",
    "across",
    "between",
    "through",
}
MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
BE_VERBS = {
    "am": "VBP",
    "is": "VBZ",
    "are": "VBP",
    "was": "VBD",
    "were": "VBD",
    "be": "VB",
    "been": "VBN",
    "being": "VBG",
}


@dataclass
class Record:
    source: str
    file_size_bytes: int
    detected_format: str
    source_encoding: str
    converted_to_utf8: bool
    output_file: str
    metadata_file: str
    chars_raw: int
    chars_clean: int
    tokens_clean: int


@dataclass
class AnnotationRecord:
    source_cleaned: str
    output_file: str
    tokens: int


@dataclass
class ErrorRecord:
    timestamp_epoch: float
    source: str
    detected_format: str
    error_code: str
    message: str
    hint: str


@dataclass
class ExtractionResult:
    text: str
    source_encoding: str
    converted_to_utf8: bool
    page_texts: list[str] | None = None
    doc_metadata: dict | None = None


class CorpusError(Exception):
    def __init__(self, code: str, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint


class ErrorLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.count = 0

    def log(self, source: Path, detected_format: str, code: str, message: str, hint: str = "") -> None:
        record = ErrorRecord(
            timestamp_epoch=time.time(),
            source=str(source.resolve()),
            detected_format=detected_format,
            error_code=code,
            message=message,
            hint=hint,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        self.count += 1
        if hint:
            print(f"[WARN] {source}: {message} | Hint: {hint}", file=sys.stderr)
        else:
            print(f"[WARN] {source}: {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cleaned UTF-8 corpus from mixed files and emit PTB POS template outputs."
    )
    parser.add_argument("inputs", nargs="+", help="Files or directories to ingest.")
    parser.add_argument("--output-dir", required=True, help="Directory where all outputs are written.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into input directories.")
    parser.add_argument("--skip-empty", action="store_true", help="Skip files that clean to empty text.")
    parser.add_argument(
        "--single-file-limit-mb",
        type=float,
        default=100.0,
        help="Ask for confirmation when a single file exceeds this size (MB). Default: 100.",
    )
    parser.add_argument(
        "--total-size-limit-gb",
        type=float,
        default=2.0,
        help="Ask for confirmation when total input size exceeds this limit (GB). Default: 2.",
    )
    parser.add_argument(
        "--total-file-limit",
        type=int,
        default=20000,
        help="Ask for confirmation when file count exceeds this threshold. Default: 20000.",
    )
    parser.add_argument(
        "--assume-yes",
        action="store_true",
        help="Skip interactive confirmations for large-file safety checks.",
    )
    return parser.parse_args()


def iter_files(inputs: list[str], recursive: bool) -> Iterator[Path]:
    for item in inputs:
        path = Path(item)
        if not path.exists():
            print(f"[WARN] Missing path: {path}", file=sys.stderr)
            continue
        if path.is_file():
            yield path
            continue

        iterator: Iterable[Path]
        iterator = path.rglob("*") if recursive else path.glob("*")
        for subpath in iterator:
            if subpath.is_file():
                yield subpath


def detect_format(path: Path, sample: bytes) -> str:
    ext = path.suffix.lower()
    if sample.startswith(b"%PDF-") or ext == ".pdf":
        return "pdf"
    if ext in HTML_EXTENSIONS:
        return "html"
    if ext in XML_EXTENSIONS:
        return "xml"
    if ext == ".docx":
        return "docx"
    if ext in JSON_EXTENSIONS:
        return "json"
    if ext in CSV_EXTENSIONS:
        return "csv" if ext == ".csv" else "tsv"
    if ext in TEXT_EXTENSIONS:
        return "text"

    sniff = sample.decode("ascii", errors="ignore").lower()
    if "<html" in sniff or "<!doctype html" in sniff:
        return "html"
    if sniff.lstrip().startswith("<?xml"):
        return "xml"
    return "text"


def maybe_confirm(message: str, assume_yes: bool) -> None:
    print(f"[ATTENTION] {message}")
    if assume_yes:
        print("[INFO] Continue due to --assume-yes")
        return

    if not sys.stdin.isatty():
        raise CorpusError(
            "NEEDS_CONFIRMATION",
            "Input size exceeded safety threshold and interactive confirmation is required.",
            "Re-run with --assume-yes after reviewing time/cost estimates.",
        )

    answer = input("Continue? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        raise CorpusError("USER_ABORT", "User declined to continue.", "Reduce input size and retry.")


def human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MB"
    return f"{num_bytes / (1024**3):.2f} GB"


def preflight_checks(paths: list[Path], args: argparse.Namespace) -> int:
    sizes: list[tuple[Path, int]] = []
    for path in paths:
        try:
            sizes.append((path, path.stat().st_size))
        except OSError:
            sizes.append((path, 0))

    total_bytes = sum(size for _, size in sizes)
    single_limit = int(args.single_file_limit_mb * 1024 * 1024)
    total_limit = int(args.total_size_limit_gb * 1024 * 1024 * 1024)

    oversized = [(p, s) for p, s in sizes if s > single_limit]
    if oversized:
        preview = ", ".join(f"{p.name} ({human_size(s)})" for p, s in oversized[:5])
        maybe_confirm(
            "Detected file(s) exceeding single-file threshold "
            f"{args.single_file_limit_mb:.1f} MB: {preview}",
            args.assume_yes,
        )

    if total_bytes > total_limit or len(paths) > args.total_file_limit:
        pdf_count = sum(1 for p in paths if p.suffix.lower() == ".pdf")
        # Conservative estimate for local parsing throughput.
        eta_seconds = (total_bytes / (15 * 1024 * 1024)) + (pdf_count * 0.7) + (len(paths) * 0.02)
        eta_minutes = max(1, int(round(eta_seconds / 60)))
        est_tokens = int(total_bytes / 4)
        maybe_confirm(
            "Input scope is large. "
            f"Files: {len(paths)}, Size: {human_size(total_bytes)}, "
            f"Estimated local runtime: ~{eta_minutes} minutes. "
            "This script itself has no model API cost. "
            f"If you later send all text to an LLM, rough scale is ~{est_tokens / 1_000_000:.2f}M tokens.",
            args.assume_yes,
        )

    return total_bytes


def decode_to_text(raw: bytes, path: Path) -> tuple[str, str, bool]:
    for encoding in UTF8_ENCODINGS:
        try:
            decoded = raw.decode(encoding, errors="strict")
            normalized_encoding = "utf-8" if encoding in {"utf-8", "utf-8-sig"} else encoding
            converted = normalized_encoding != "utf-8"
            return decoded, normalized_encoding, converted
        except UnicodeDecodeError:
            continue

    looks_utf16 = False
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        looks_utf16 = True
    else:
        sample = raw[:4096]
        if sample:
            looks_utf16 = (sample.count(0) / len(sample)) > 0.2

    if looks_utf16:
        for encoding in UTF16_ENCODINGS:
            try:
                decoded = raw.decode(encoding, errors="strict")
                return decoded, encoding, True
            except UnicodeDecodeError:
                continue

    for encoding in LEGACY_ENCODINGS:
        try:
            decoded = raw.decode(encoding, errors="strict")
            return decoded, encoding, True
        except UnicodeDecodeError:
            continue

    raise CorpusError(
        "ENCODING_ERROR",
        f"Failed to decode file into UTF-8 text: {path}",
        "Convert the file to UTF-8 manually and retry.",
    )


def extract_text(path: Path, fmt: str) -> ExtractionResult:
    if fmt == "pdf":
        return extract_pdf(path)
    if fmt == "html":
        return extract_html(path)
    if fmt == "xml":
        return extract_xml(path)
    if fmt == "docx":
        return extract_docx(path)
    if fmt in {"csv", "tsv"}:
        return extract_delimited(path, fmt)
    if fmt == "json":
        return extract_json(path)
    return extract_text_file(path)


def extract_pdf(path: Path) -> ExtractionResult:
    text_parts: list[str] = []
    page_texts: list[str] = []
    doc_metadata: dict = {}
    errors: list[str] = []

    try:
        from pypdf import PdfReader  # type: ignore

        with contextlib.redirect_stderr(io.StringIO()):
            reader = PdfReader(str(path))
            raw_meta = getattr(reader, "metadata", None)
            if raw_meta:
                for key, value in raw_meta.items():
                    clean_key = str(key).lstrip("/")
                    if value is None:
                        continue
                    doc_metadata[clean_key] = str(value)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                page_texts.append(page_text)
                text_parts.append(page_text)
        text = "\n\n".join(text_parts)
        return ExtractionResult(
            text=text,
            source_encoding="utf-8",
            converted_to_utf8=False,
            page_texts=page_texts,
            doc_metadata=doc_metadata or None,
        )
    except Exception as exc:
        errors.append(f"pypdf: {exc}")

    try:
        import pdfplumber  # type: ignore

        with contextlib.redirect_stderr(io.StringIO()):
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    page_texts.append(page_text)
                    text_parts.append(page_text)
        text = "\n\n".join(text_parts)
        return ExtractionResult(
            text=text,
            source_encoding="utf-8",
            converted_to_utf8=False,
            page_texts=page_texts,
            doc_metadata=doc_metadata or None,
        )
    except Exception as exc:
        errors.append(f"pdfplumber: {exc}")

    raise CorpusError(
        "PARSE_ERROR",
        f"Cannot parse PDF: {path}",
        "Install pypdf/pdfplumber, or verify the PDF is not corrupted.",
    ) from Exception(" | ".join(errors))


def extract_html(path: Path) -> ExtractionResult:
    raw = path.read_bytes()
    decoded, source_encoding, converted = decode_to_text(raw, path)
    parser = _HTMLTextExtractor()
    parser.feed(decoded)
    return ExtractionResult(text=unescape(parser.get_text()), source_encoding=source_encoding, converted_to_utf8=converted)


def extract_xml(path: Path) -> ExtractionResult:
    raw = path.read_bytes()
    decoded, source_encoding, converted = decode_to_text(raw, path)
    try:
        root = ET.fromstring(decoded)
    except ET.ParseError as exc:
        raise CorpusError(
            "PARSE_ERROR",
            f"Cannot parse XML: {path}",
            "Check whether the XML file is broken or truncated.",
        ) from exc

    text_parts = [node.text for node in root.iter() if node.text and node.text.strip()]
    return ExtractionResult(text="\n".join(text_parts), source_encoding=source_encoding, converted_to_utf8=converted)


def extract_docx(path: Path) -> ExtractionResult:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            xml_data = archive.read("word/document.xml")
    except zipfile.BadZipFile as exc:
        raise CorpusError(
            "PARSE_ERROR",
            f"DOCX container is corrupted: {path}",
            "Re-export the document and retry.",
        ) from exc
    except KeyError as exc:
        raise CorpusError(
            "PARSE_ERROR",
            f"DOCX missing word/document.xml: {path}",
            "The file may not be a valid DOCX document.",
        ) from exc

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as exc:
        raise CorpusError(
            "PARSE_ERROR",
            f"Cannot parse DOCX XML content: {path}",
            "The document appears damaged; re-export it.",
        ) from exc

    paragraphs: list[str] = []
    current: list[str] = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "t" and element.text:
            current.append(element.text)
        if tag == "p":
            if current:
                paragraphs.append("".join(current))
            current = []
    if current:
        paragraphs.append("".join(current))

    return ExtractionResult(text="\n\n".join(paragraphs), source_encoding="utf-8", converted_to_utf8=False)


def extract_delimited(path: Path, fmt: str) -> ExtractionResult:
    delimiter = "," if fmt == "csv" else "\t"
    raw = path.read_bytes()
    decoded, source_encoding, converted = decode_to_text(raw, path)

    rows: list[str] = []
    handle = io.StringIO(decoded)
    reader = csv.reader(handle, delimiter=delimiter)
    for row in reader:
        if row:
            rows.append(" ".join(cell.strip() for cell in row if cell.strip()))

    return ExtractionResult(text="\n".join(rows), source_encoding=source_encoding, converted_to_utf8=converted)


def extract_json(path: Path) -> ExtractionResult:
    raw = path.read_bytes()
    decoded, source_encoding, converted = decode_to_text(raw, path)
    texts: list[str] = []

    def visit(value) -> None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                texts.append(stripped)
            return
        if isinstance(value, dict):
            for subvalue in value.values():
                visit(subvalue)
            return
        if isinstance(value, list):
            for subvalue in value:
                visit(subvalue)

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        for line_no, line in enumerate(decoded.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                visit(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CorpusError(
                    "PARSE_ERROR",
                    f"Invalid JSONL at line {line_no}: {path}",
                    "Fix JSONL formatting issues and retry.",
                ) from exc
    else:
        try:
            visit(json.loads(decoded))
        except json.JSONDecodeError as exc:
            raise CorpusError(
                "PARSE_ERROR",
                f"Invalid JSON structure: {path}",
                "Fix JSON syntax and retry.",
            ) from exc

    return ExtractionResult(text="\n".join(texts), source_encoding=source_encoding, converted_to_utf8=converted)


def extract_text_file(path: Path) -> ExtractionResult:
    raw = path.read_bytes()
    decoded, source_encoding, converted = decode_to_text(raw, path)
    return ExtractionResult(text=decoded, source_encoding=source_encoding, converted_to_utf8=converted)


def isoformat_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
    except OSError:
        ts = time.time()
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def looks_like_sentence(line: str) -> bool:
    words = re.findall(r"[A-Za-z]+", line)
    if len(words) < 6:
        return False
    return bool(re.search(r"[.!?]$", line) or len(line) > 80)


def looks_like_toc_line(line: str) -> bool:
    lower = line.lower().strip()
    if not lower:
        return False
    if "table of contents" in lower or lower == "contents":
        return True
    if re.search(r"\.{2,}\s*\d{1,4}$", line):
        return True
    if re.search(r"\s\d{1,4}$", line) and len(re.findall(r"[A-Za-z]+", line)) >= 2:
        return True
    return False


def is_all_caps_header(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 12:
        return False
    if not re.search(r"[A-Za-z]", stripped):
        return False
    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    return upper_ratio > 0.9 and len(letters) >= 8


def is_footnote_like(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^\[\d+\]\s+", stripped):
        return True
    if re.match(r"^\d+[A-Za-z]", stripped):
        return True
    if re.match(r"^\d+\s+", stripped):
        lowered = stripped.lower()
        if "http" in lowered or "www." in lowered or "doi" in lowered:
            return True
        words = re.findall(r"[A-Za-z]+", stripped)
        if len(words) <= 12 and len(stripped) < 120:
            return True
    return False


def extract_pdf_body_and_metadata(
    path: Path,
    extraction: ExtractionResult,
    file_size: int,
) -> tuple[str, dict]:
    lines = [normalize_line(line) for line in extraction.text.splitlines()]
    lines = [line for line in lines if line]

    base_metadata = {
        "source_file_name": path.name,
        "source_path": str(path.resolve()),
        "detected_format": "pdf",
        "file_size_bytes": file_size,
        "last_modified_utc": isoformat_mtime(path),
    }
    if extraction.doc_metadata:
        base_metadata["pdf_embedded_metadata"] = extraction.doc_metadata
    if extraction.page_texts:
        base_metadata["pdf_page_count"] = len(extraction.page_texts)

    metadata_blocks: dict[str, str] = {}
    indices_to_remove: set[int] = set()
    line_counts = Counter(lines)

    # 1) Leading cover/title pages metadata block.
    leading_block: list[str] = []
    for idx, line in enumerate(lines[:80]):
        lower = line.lower()
        if looks_like_sentence(line) and not (
            "prepared by" in lower or "copyright" in lower or "implementation plan" in lower
        ):
            break
        leading_block.append(line)
        indices_to_remove.add(idx)
        if len(leading_block) >= 40:
            break

    if leading_block:
        metadata_blocks["leading_metadata_block"] = "\n".join(leading_block).strip()

    # Remove known boilerplate blank page marker.
    for idx, line in enumerate(lines):
        if line.lower().strip() == "this page is intentionally left blank.":
            indices_to_remove.add(idx)

    # 2) Copyright block.
    for idx, line in enumerate(lines):
        lower = line.lower()
        if "copyright information" in lower or re.search(r"\bcopyright\b", lower):
            block = [line]
            indices_to_remove.add(idx)
            for next_idx in range(idx + 1, min(len(lines), idx + 30)):
                candidate = lines[next_idx]
                lowered_candidate = candidate.lower().strip()
                if lowered_candidate.startswith("contents") or lowered_candidate.startswith("table of contents"):
                    break
                if lowered_candidate.startswith("preface"):
                    break
                if len(block) >= 6 and (is_all_caps_header(candidate) or looks_like_toc_line(candidate)):
                    break
                block.append(candidate)
                indices_to_remove.add(next_idx)
            metadata_blocks["copyright_block"] = "\n".join(block).strip()
            break

    # 3) Remove explicit TOC sections.
    in_toc = False
    for idx, line in enumerate(lines):
        lowered = line.lower().strip()
        if not in_toc and (
            lowered == "contents"
            or "table of contents" in lowered
            or ("contents" in lowered and len(re.findall(r"[A-Za-z]+", lowered)) <= 10)
        ):
            in_toc = True
            indices_to_remove.add(idx)
            continue
        if in_toc:
            if lowered.startswith("preface"):
                in_toc = False
                continue
            if looks_like_sentence(line):
                in_toc = False
                continue
            indices_to_remove.add(idx)

    # 4) Remove repeated all-caps running headers.
    for idx, line in enumerate(lines):
        if line_counts[line] >= 2 and is_all_caps_header(line):
            indices_to_remove.add(idx)

    # 5) Remove TOC-like line fragments.
    idx = 0
    while idx < len(lines):
        if looks_like_toc_line(lines[idx]):
            start = idx
            end = idx
            for next_idx in range(idx + 1, min(len(lines), idx + 250)):
                candidate = lines[next_idx]
                if not candidate:
                    end = next_idx
                    continue
                if looks_like_toc_line(candidate) or re.match(r"^[A-Za-z][A-Za-z\s&,-]{2,}$", candidate):
                    end = next_idx
                    continue
                if looks_like_sentence(candidate):
                    break
                end = next_idx
            for rm in range(start, end + 1):
                indices_to_remove.add(rm)
            idx = end + 1
            continue
        idx += 1

    # 6) Remove references tail.
    for idx, line in enumerate(lines):
        lowered = line.lower().strip()
        if lowered in {"references", "bibliography", "works cited"}:
            metadata_blocks["references_heading"] = line
            for rm in range(idx, len(lines)):
                indices_to_remove.add(rm)
            break

    # 7) Remove footnote-like lines.
    for idx, line in enumerate(lines):
        if is_footnote_like(line):
            indices_to_remove.add(idx)

    body_lines = [line for idx, line in enumerate(lines) if idx not in indices_to_remove]
    base_metadata["removed_metadata_blocks"] = metadata_blocks
    return "\n".join(body_lines).strip(), base_metadata


def extract_generic_metadata(path: Path, fmt: str, file_size: int) -> dict:
    return {
        "source_file_name": path.name,
        "source_path": str(path.resolve()),
        "detected_format": fmt,
        "file_size_bytes": file_size,
        "last_modified_utc": isoformat_mtime(path),
    }


def preprocess_text_and_metadata(
    path: Path,
    fmt: str,
    extraction: ExtractionResult,
    file_size: int,
) -> tuple[str, dict]:
    if fmt == "pdf":
        body_text, metadata = extract_pdf_body_and_metadata(path, extraction, file_size)
        return body_text, metadata
    return extraction.text, extract_generic_metadata(path, fmt, file_size)


class _HTMLTextExtractor(HTMLParser):
    """Extract visible text from HTML while skipping script/style blocks."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag in {"p", "div", "section", "article", "li", "br", "tr", "h1", "h2", "h3", "h4"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


def clean_text(text: str, aggressive_pdf_filter: bool = False) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = URL_PATTERN.sub(" ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines: list[str] = []
    for line in text.split("\n"):
        cleaned = line.strip()
        if not cleaned:
            lines.append("")
            continue

        lowered = cleaned.lower()
        if re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", lowered):
            continue
        if re.fullmatch(r"\d+", cleaned):
            continue
        if lowered == "this page is intentionally left blank.":
            continue
        if lowered in {"contents", "table of contents", "references", "bibliography"}:
            continue
        if lowered.startswith("implementation plan 2025-2026"):
            continue
        if "interagency arctic research policy committee" in lowered and "arctic research plan 2022-2026" in lowered:
            continue

        cleaned = URL_PATTERN.sub("", cleaned).strip()
        if not cleaned:
            continue

        # Remove isolated non-sentence fragments/noisy leftovers.
        alpha_words = re.findall(r"[A-Za-z]+", cleaned)
        has_sentence_punct = bool(re.search(r"[.!?]$", cleaned))
        if len(alpha_words) <= 1 and not has_sentence_punct:
            continue
        if len(alpha_words) <= 3 and len(cleaned) < 24 and not has_sentence_punct:
            continue
        if len(alpha_words) <= 7 and not has_sentence_punct:
            continue
        if is_all_caps_header(cleaned):
            continue
        if cleaned.lower().endswith("available at"):
            continue
        if re.match(r"^[^A-Za-z]*\d+\s+[A-Za-z]", cleaned):
            continue
        if not re.search(r"[A-Za-z]", cleaned):
            continue

        lines.append(cleaned)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if not aggressive_pdf_filter:
        return text

    # Extra denoising for PDF corpora: merge wrapped lines and keep narrative paragraphs.
    merged: list[str] = []
    buffer = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            continue
        if not buffer:
            buffer = stripped
            continue
        if re.search(r"[.!?]$", buffer):
            merged.append(buffer.strip())
            buffer = stripped
        else:
            buffer = f"{buffer} {stripped}"
    if buffer:
        merged.append(buffer.strip())

    filtered_paragraphs: list[str] = []
    for para in merged:
        words = re.findall(r"[A-Za-z]+", para)
        if len(words) < 8:
            continue
        if not re.search(r"[.!?]", para):
            continue
        lowered = para.lower()
        if "lead agency pending" in lowered:
            continue
        if re.search(r"\bobjective\s+\d+(\.\d+)?\b", lowered):
            continue
        if re.search(r"\b(doc-noaa|doi-usgs|nsf\s*\(lead\))\b", lowered):
            continue
        filtered_paragraphs.append(para)

    return "\n\n".join(filtered_paragraphs).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|\.{3}|--|[^\w\s]", text)


def guess_ptb_tag(token: str) -> str:
    if token in PTB_PUNCT_TAGS:
        return PTB_PUNCT_TAGS[token]

    lower = token.lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return "CD"
    if lower in a:
        return "DT"
    if lower in PRONOUNS:
        return "PRP"
    if lower in POSSESSIVE_PRONOUNS or lower.endswith("'s"):
        return "PRP$"
    if lower in CONJUNCTIONS:
        return "CC"
    if lower in PREPOSITIONS:
        return "IN"
    if lower == "to":
        return "TO"
    if lower in MODALS:
        return "MD"
    if lower in BE_VERBS:
        return BE_VERBS[lower]
    if lower.endswith("ing"):
        return "VBG"
    if lower.endswith("ed"):
        return "VBD"
    if lower.endswith("ly"):
        return "RB"
    if lower.endswith(("ous", "ful", "able", "ible", "ive", "ic", "al")):
        return "JJ"
    if token[0].isupper() and token[1:].islower():
        return "NNP"
    if lower.endswith("s") and len(lower) > 3:
        return "NNS"
    return "NN"


def build_pos_template(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        tokens = tokenize(stripped)
        tagged = [f"{tok}_{guess_ptb_tag(tok)}" for tok in tokens]
        lines.append(" ".join(tagged))
    return "\n".join(lines).strip()


def count_tokens_and_types(text: str) -> tuple[int, int]:
    tokens = tokenize(text)
    normalized = [token.lower() for token in tokens]
    return len(tokens), len(set(normalized))


def sanitize_name(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "document"


def ensure_unique(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique output name for {path}")


def process_file(
    path: Path,
    output_text_dir: Path,
    metadata_dir: Path,
    skip_empty: bool,
    logger: ErrorLogger,
) -> Record | None:
    file_size = 0
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            sample = handle.read(4096)
    except OSError as exc:
        logger.log(path, "unknown", "FILE_IO_ERROR", f"Cannot read file: {exc}", "Check file permissions.")
        return None

    fmt = detect_format(path, sample)

    try:
        extraction = extract_text(path, fmt)
    except CorpusError as exc:
        logger.log(path, fmt, exc.code, str(exc), exc.hint)
        return None
    except Exception as exc:
        logger.log(
            path,
            fmt,
            "UNEXPECTED_ERROR",
            f"Unexpected extraction failure: {exc}",
            "Inspect file integrity or parser dependencies.",
        )
        return None

    raw_text, metadata_payload = preprocess_text_and_metadata(path, fmt, extraction, file_size)

    metadata_path = ensure_unique(metadata_dir / f"{sanitize_name(path)}_metadata.json")
    metadata_payload["metadata_saved_at_utc"] = datetime.now(tz=timezone.utc).isoformat()
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not raw_text.strip():
        logger.log(
            path,
            fmt,
            "TEXT_UNREADABLE",
            "No readable text content extracted.",
            "For scanned PDFs or images, run OCR first.",
        )
        return None

    cleaned = clean_text(raw_text, aggressive_pdf_filter=(fmt == "pdf"))
    if not cleaned:
        if skip_empty:
            logger.log(
                path,
                fmt,
                "EMPTY_AFTER_CLEAN",
                "Text became empty after cleaning.",
                "Adjust cleaning rules if you expect content from this file.",
            )
            return None

        logger.log(
            path,
            fmt,
            "EMPTY_AFTER_CLEAN",
            "Text became empty after cleaning; file skipped.",
            "Adjust cleaning rules and retry if needed.",
        )
        return None

    output_path = ensure_unique(output_text_dir / f"{sanitize_name(path)}.txt")
    output_path.write_text(cleaned + "\n", encoding="utf-8")

    token_count, _ = count_tokens_and_types(cleaned)
    return Record(
        source=str(path.resolve()),
        file_size_bytes=file_size,
        detected_format=fmt,
        source_encoding=extraction.source_encoding,
        converted_to_utf8=extraction.converted_to_utf8,
        output_file=str(output_path.resolve()),
        metadata_file=str(metadata_path.resolve()),
        chars_raw=len(raw_text),
        chars_clean=len(cleaned),
        tokens_clean=token_count,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_files(paths: list[Path], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as out:
        for index, path in enumerate(paths):
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            if index > 0:
                out.write("\n\n")
            out.write(text)
            out.write("\n")


def main() -> int:
    args = parse_args()

    files = list(iter_files(args.inputs, recursive=args.recursive))
    if not files:
        print("[WARN] No input files found.", file=sys.stderr)
        return 1

    try:
        total_input_bytes = preflight_checks(files, args)
    except CorpusError as exc:
        print(f"[WARN] {exc}", file=sys.stderr)
        if exc.hint:
            print(f"[HINT] {exc.hint}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    clean_dir = output_dir / "cleaned_corpus"
    clean_txt_dir = clean_dir / "cleaned_txt"
    pos_dir = output_dir / "pos_annotated_corpus"
    pos_txt_dir = pos_dir / "annotated_txt"
    metadata_dir = output_dir / "metadata"
    metadata_files_dir = metadata_dir / "per_file"
    logs_dir = output_dir / "logs"

    clean_txt_dir.mkdir(parents=True, exist_ok=True)
    pos_txt_dir.mkdir(parents=True, exist_ok=True)
    metadata_files_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    error_log_path = logs_dir / "errors.jsonl"
    logger = ErrorLogger(error_log_path)

    records: list[Record] = []
    for file_path in files:
        record = process_file(file_path, clean_txt_dir, metadata_files_dir, args.skip_empty, logger)
        if record is None:
            continue
        records.append(record)
        print(f"[OK] {file_path} -> {record.output_file}")

    clean_manifest_path = clean_dir / "manifest.jsonl"
    clean_corpus_path = clean_dir / "corpus.txt"
    metadata_manifest_path = metadata_dir / "metadata_manifest.jsonl"
    write_jsonl(clean_manifest_path, [asdict(r) for r in records])
    write_jsonl(
        metadata_manifest_path,
        [
            {
                "source": r.source,
                "detected_format": r.detected_format,
                "metadata_file": r.metadata_file,
            }
            for r in records
        ],
    )
    merge_files([Path(r.output_file) for r in records], clean_corpus_path)

    annotation_records: list[AnnotationRecord] = []
    pos_paths: list[Path] = []
    for record in records:
        source_cleaned = Path(record.output_file)
        cleaned_text = source_cleaned.read_text(encoding="utf-8")
        pos_text = build_pos_template(cleaned_text)
        pos_path = ensure_unique(pos_txt_dir / f"{source_cleaned.stem}_pos.txt")
        pos_path.write_text(pos_text + ("\n" if pos_text else ""), encoding="utf-8")
        tokens, _ = count_tokens_and_types(cleaned_text)
        annotation_records.append(
            AnnotationRecord(
                source_cleaned=str(source_cleaned.resolve()),
                output_file=str(pos_path.resolve()),
                tokens=tokens,
            )
        )
        pos_paths.append(pos_path)

    pos_manifest_path = pos_dir / "manifest_pos.jsonl"
    pos_corpus_path = pos_dir / "corpus_pos_template.txt"
    write_jsonl(pos_manifest_path, [asdict(r) for r in annotation_records])
    merge_files(pos_paths, pos_corpus_path)

    clean_corpus_text = clean_corpus_path.read_text(encoding="utf-8") if clean_corpus_path.exists() else ""
    pos_corpus_text = pos_corpus_path.read_text(encoding="utf-8") if pos_corpus_path.exists() else ""

    clean_tokens, clean_types = count_tokens_and_types(clean_corpus_text)
    pos_tokens, pos_types = count_tokens_and_types(pos_corpus_text)

    format_dist = Counter(r.detected_format for r in records)
    converted_count = sum(1 for r in records if r.converted_to_utf8)
    report = {
        "input_files_total": len(files),
        "processed_files": len(records),
        "skipped_files": len(files) - len(records),
        "errors_logged": logger.count,
        "total_input_size_bytes": total_input_bytes,
        "total_input_size_human": human_size(total_input_bytes),
        "converted_to_utf8_files": converted_count,
        "format_distribution": dict(sorted(format_dist.items())),
        "cleaned_corpus": {
            "path": str(clean_corpus_path.resolve()),
            "token_count": clean_tokens,
            "type_count": clean_types,
            "char_count": len(clean_corpus_text),
        },
        "pos_annotated_corpus": {
            "path": str(pos_corpus_path.resolve()),
            "token_count": pos_tokens,
            "type_count": pos_types,
            "char_count": len(pos_corpus_text),
        },
        "logs": {
            "error_log": str(error_log_path.resolve()),
        },
        "metadata": {
            "metadata_manifest": str(metadata_manifest_path.resolve()),
            "metadata_files_dir": str(metadata_files_dir.resolve()),
        },
    }
    report_path = output_dir / "corpus_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[DONE] Processed files: {len(records)}")
    print(f"[DONE] Clean corpus dir: {clean_dir}")
    print(f"[DONE] POS corpus dir: {pos_dir}")
    print(f"[DONE] Metadata dir: {metadata_dir}")
    print(f"[DONE] Error log: {error_log_path}")
    print(f"[DONE] Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

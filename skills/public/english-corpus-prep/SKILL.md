---
name: english-corpus-prep
description: Build corpus-ready English TXT data from mixed file formats. Use when Codex needs to ingest raw text from PDF, TXT/Markdown, HTML/XML, DOCX, JSON/JSONL, CSV/TSV, or unknown text-like files; detect input formats at the start; clean and normalize extracted text; and produce presentable, analysis-ready corpus outputs.
---

# English Corpus Prep

Prepare standardized UTF-8 TXT corpus output with deterministic format detection, extraction, cleaning, error logging, and Penn Treebank POS template export. Use the bundled script first; patch it only if a new format or project-specific rule is required.

## Workflow

1. Gather all input files or directories.
2. Run preflight checks for oversized files/data volumes; request confirmation when thresholds are exceeded.
3. Detect format before extraction for each file.
4. Extract raw text with the format-specific handler.
5. Decode non-UTF-8 text to UTF-8 when possible; log and skip on conversion failure.
6. Extract and save per-file metadata separately; remove metadata blocks from corpus text.
7. For PDF, keep narrative body text only (remove cover/title metadata, copyright blocks, TOC, references, and footnote-like noise).
8. Clean and normalize text using the default profile.
9. Log parse/encoding/readability failures and skip bad files.
10. Export presentable outputs:
   - per-file cleaned TXT files
   - combined clean corpus text
   - per-file PTB POS template files
   - combined POS template corpus
   - metadata files/manifests, error logs, and corpus report stats
11. Spot-check 3-5 outputs and adjust cleaning rules if needed.

## Run The Pipeline

Use `scripts/build_corpus.py`:

```bash
python3 scripts/build_corpus.py <input-path> [<input-path> ...] --output-dir <output-dir> [--recursive] [--skip-empty] [--assume-yes]
```

Examples:

```bash
python3 scripts/build_corpus.py ./raw --output-dir ./corpus_out --recursive
python3 scripts/build_corpus.py ./raw/a.pdf ./raw/b.html --output-dir ./corpus_out
python3 scripts/build_corpus.py ./raw --output-dir ./corpus_out --recursive --assume-yes
```

## Format Detection And Extraction Behavior

Detection priority:

1. File signature and extension (`.pdf`, `.html`, `.xml`, `.docx`, `.json`, `.jsonl`, `.csv`, `.tsv`, text extensions)
2. Lightweight content sniffing (`<html`, `<!doctype html`, `<?xml`)
3. Fallback to text decode

Built-in extractors:

- PDF: `pypdf` first, `pdfplumber` fallback (if installed)
- HTML/XML: tag-aware extraction and entity unescaping
- DOCX: direct XML extraction from `word/document.xml`
- JSON/JSONL: recursive extraction of string fields
- CSV/TSV: cell text flattening row by row
- Text-like unknown files: best-effort decoding

Failure handling:

- Log parse failures, encoding failures, and unreadable-text failures to `logs/errors.jsonl`
- Skip failed files and continue processing remaining inputs
- Save per-file metadata to `metadata/per_file/*_metadata.json` and remove metadata blocks from cleaned corpus text

## Output Contract

Given `--output-dir out`, write:

- `out/cleaned_corpus/cleaned_txt/*.txt`: one cleaned UTF-8 TXT per source
- `out/cleaned_corpus/corpus.txt`: merged cleaned corpus
- `out/cleaned_corpus/manifest.jsonl`: file-level cleaned manifest
- `out/metadata/per_file/*_metadata.json`: per-file metadata (including PDF copyright/cover metadata blocks)
- `out/metadata/metadata_manifest.jsonl`: metadata manifest
- `out/pos_annotated_corpus/annotated_txt/*_pos.txt`: PTB-style POS template per file (`token_TAG`, punctuation included)
- `out/pos_annotated_corpus/corpus_pos_template.txt`: merged POS template corpus
- `out/pos_annotated_corpus/manifest_pos.jsonl`: annotation manifest
- `out/logs/errors.jsonl`: error log for skipped files
- `out/corpus_report.json`: corpus stats (`token_count`, `type_count`, format distribution, processed/skipped counts)

## Cleaning Profile

Apply the default profile described in `references/cleaning-profile.md`:

- Normalize Unicode (`NFKC`) and newlines
- Remove soft hyphens and de-hyphenate line wraps
- Collapse noisy spacing
- Remove common page-number/header noise patterns
- Preserve paragraph-level readability in final TXT

UTF-8 policy:

- Encode all outputs as UTF-8
- Attempt conversion for non-UTF-8 inputs
- Log and skip inputs that cannot be decoded to UTF-8

PDF body policy:

- Remove metadata blocks from corpus text and store them separately
- Drop TOC/references/footnote-like fragments before corpus cleaning
- Perform POS template generation only after cleaning is complete

## Quality Checks

After each run:

1. Confirm every line in `cleaned_corpus/manifest.jsonl` has non-empty `source` and `detected_format`.
2. Check `logs/errors.jsonl` and resolve recurring causes (corruption, encoding, OCR gaps).
3. Open a few `cleaned_txt` and `annotated_txt` files from different formats.
4. Verify both corpora (`corpus.txt`, `corpus_pos_template.txt`) have readable output and proper punctuation tagging.
5. Inspect `corpus_report.json` for token/type counts and format distribution.
6. If cleaning is too aggressive or too weak, patch `clean_text()` in `scripts/build_corpus.py` and re-run.

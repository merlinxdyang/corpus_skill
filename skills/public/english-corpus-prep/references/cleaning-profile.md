# Cleaning Profile

Use this profile as the default for English corpus preparation.

## Goals

- Keep semantic text content.
- Remove extraction artifacts and layout noise.
- Preserve enough paragraph structure for downstream NLP.

## Rules

1. Normalize text to Unicode `NFKC`.
2. Convert all newline styles to `\n`.
3. Remove soft hyphen (`\u00ad`) characters.
4. Join hyphenated line breaks where a word was split by layout.
5. Collapse repeated spaces and tabs into one space.
6. Limit blank-line runs to two newlines.
7. Remove page-number lines such as:
   - `Page 4`
   - `Page 4 of 12`
   - lines containing only digits
8. Trim leading/trailing whitespace.
9. Remove URLs and obvious non-English/non-ASCII symbols.
10. Remove isolated non-sentence fragments (especially OCR leftovers and TOC-like debris).

## Optional Project-Specific Adjustments

Adjust `clean_text()` in `scripts/build_corpus.py` when a corpus requires:

- preserving table-like spacing
- preserving line-level poetry/drama structure
- keeping legal headers and section numbering
- removing specific recurring boilerplate blocks

Always keep changes deterministic and document any added regex in comments.

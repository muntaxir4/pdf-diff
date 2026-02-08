#!/usr/bin/env python3
"""pdfdiff: Compare two PDFs and output JSON diff summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import pdfplumber
from diff_pdf import render_github_diff_pdf


@dataclass
class Block:
    block_type: str  # paragraph | table | image
    content: Any
    norm: str
    page_index: int
    block_index: int
    bbox: Optional[Tuple[float, float, float, float]] = None


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_text(value: str) -> str:
    """Clean text from PDF-specific characters and artifacts."""
    # \u00ad is soft hyphen, \u200b is zero-width space, \u200d is zero-width joiner
    return (
        value.replace("\u00ad", "").replace("\u200b", "").replace("\u200d", "").strip()
    )


def is_bullet_start(text: str) -> bool:
    """Check if text starts with a bullet point character."""
    # Comprehensive list of bullet characters including unicode ranges
    bullet_chars = (
        "•",
        "●",
        "◦",
        "▪",
        "▫",
        "■",
        "□",
        "∙",
        "◦",
        "‣",
        "",
        "·",
        "§",
        "*",
        "-",
        "–",
        "—",
        "»",
        "›",
        "■",
    )
    trimmed = clean_text(text)
    if not trimmed:
        return False
    # Also check for common bullet patterns like "1.", "a)", etc. if needed,
    # but for now let's stick to symbols.
    return trimmed.startswith(bullet_chars) or (
        len(trimmed) > 1 and trimmed[0] in bullet_chars
    )


def _split_paragraphs_from_lines(lines: List[Dict[str, Any]]) -> List[str]:
    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda l: (l.get("top", 0), l.get("x0", 0)))
    gaps: List[float] = []
    x0_values: List[float] = []
    for prev, curr in zip(sorted_lines, sorted_lines[1:]):
        prev_bottom = prev.get("bottom")
        curr_top = curr.get("top")
        if isinstance(prev_bottom, (int, float)) and isinstance(curr_top, (int, float)):
            gaps.append(curr_top - prev_bottom)
        curr_x0 = curr.get("x0")
        if isinstance(curr_x0, (int, float)):
            x0_values.append(float(curr_x0))

    gap_threshold = 12.0
    median_gap = 0.0
    if gaps:
        median_gap = sorted(gaps)[len(gaps) // 2]
        gap_threshold = max(12.0, median_gap * 1.5)

    def is_sentence_end(value: str) -> bool:
        trimmed = clean_text(value)
        if not trimmed:
            return False
        if trimmed.endswith(("-", "‐", "‑", "–", "—")):
            return False
        return trimmed.endswith((".", "!", "?", ".”", '!"', '?"', '".'))

    def starts_new_sentence(value: str) -> bool:
        trimmed = clean_text(value)
        if not trimmed:
            return False
        first = trimmed[0]
        return first.isupper() or first.isdigit()

    def is_section_header(value: str) -> bool:
        trimmed = clean_text(value)
        if not trimmed:
            return False
        if len(trimmed) > 40:
            return False
        letters = [c for c in trimmed if c.isalpha()]
        if not letters:
            return False
        return all(c.isupper() for c in letters)

    min_x0 = min(x0_values) if x0_values else 0.0
    indent_tolerance = 8.0
    column_break_threshold = 24.0

    def join_buffer(buf: List[str]) -> str:
        if not buf:
            return ""
        text = buf[0]
        for line in buf[1:]:
            if text.endswith(("-", "‐", "‑")):
                # Likely hyphenated word at line break
                text = text[:-1] + line
            else:
                text += " " + line
        return text

    paragraphs: List[str] = []
    buffer: List[str] = []
    prev_bottom: Optional[float] = None
    prev_text: str = ""
    prev_x0: Optional[float] = None
    for line in sorted_lines:
        text = line.get("text") or ""
        cleaned = clean_text(text)
        if not cleaned:
            continue
        top = line.get("top")
        x0 = line.get("x0")
        line_x0 = float(x0) if isinstance(x0, (int, float)) else None
        gap = None
        if prev_bottom is not None and isinstance(top, (int, float)):
            gap = top - prev_bottom

        # Trigger break on bullet or section header
        if buffer and (
            is_bullet_start(cleaned)
            or (
                len(cleaned) < 40
                and cleaned.isupper()
                and any(c.isalpha() for c in cleaned)
            )
        ):
            paragraphs.append(join_buffer(buffer))
            buffer = []

        if (
            gap is not None
            and buffer
            and (
                gap > gap_threshold
                or (
                    gap >= max(2.0, median_gap * 0.8)
                    and is_sentence_end(prev_text)
                    and starts_new_sentence(cleaned)
                )
                or (
                    is_sentence_end(prev_text)
                    and starts_new_sentence(cleaned)
                    and line_x0 is not None
                    and abs(line_x0 - min_x0) <= indent_tolerance
                )
                or (
                    line_x0 is not None
                    and prev_x0 is not None
                    and abs(line_x0 - prev_x0) >= column_break_threshold
                )
            )
        ):
            paragraphs.append(join_buffer(buffer))
            buffer = []
        buffer.append(cleaned)
        prev_bottom = (
            line.get("bottom")
            if isinstance(line.get("bottom"), (int, float))
            else prev_bottom
        )
        prev_x0 = line_x0 if line_x0 is not None else prev_x0
        prev_text = cleaned

    if buffer:
        paragraphs.append(join_buffer(buffer))

    return paragraphs


def extract_blocks(pdf_path: str) -> List[Block]:
    blocks: List[Block] = []
    # Use tighter tolerance (defaults: x=3, y=3) to avoid merging words
    # x_tolerance=2 helps with resumes that often have tight kerning or missing spaces
    extraction_settings = {"x_tolerance": 2, "y_tolerance": 3}

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            text_lines = page.extract_text_lines(**extraction_settings) or []
            paragraphs = _split_paragraphs_from_lines(text_lines)
            if not paragraphs:
                text = page.extract_text(**extraction_settings) or ""
                paragraphs = (
                    [
                        " ".join(
                            line.strip() for line in text.splitlines() if line.strip()
                        )
                    ]
                    if text.strip()
                    else []
                )
            for i, para in enumerate(paragraphs):
                blocks.append(
                    Block(
                        block_type="paragraph",
                        content=para,
                        norm=normalize_text(para),
                        page_index=page_index,
                        block_index=i,
                    )
                )

            tables = page.extract_tables() or []
            for i, table in enumerate(tables):
                table_text = "\n".join(
                    ["\t".join([(cell or "").strip() for cell in row]) for row in table]
                )
                blocks.append(
                    Block(
                        block_type="table",
                        content=table,
                        norm=normalize_text(table_text),
                        page_index=page_index,
                        block_index=i,
                    )
                )

            images = page.images or []
            for i, img in enumerate(images):
                # Try to get the image bytes if available via stream; otherwise hash metadata.
                img_hash = None
                try:
                    if "object_id" in img:
                        pdf_obj = getattr(page, "pdf", None)
                        streams = getattr(pdf_obj, "streams", None)
                        if streams and img["object_id"] in streams:
                            raw = streams[img["object_id"]].get_data()
                            img_hash = hash_bytes(raw)
                except Exception:
                    img_hash = None

                if not img_hash:
                    img_hash = hash_bytes(
                        json.dumps(img, sort_keys=True, default=str).encode("utf-8")
                    )

                x0 = img.get("x0")
                top = img.get("top")
                x1 = img.get("x1")
                bottom = img.get("bottom")
                bbox: Optional[Tuple[float, float, float, float]] = None
                if all(v is not None for v in (x0, top, x1, bottom)):
                    bbox = (
                        float(cast(float, x0)),
                        float(cast(float, top)),
                        float(cast(float, x1)),
                        float(cast(float, bottom)),
                    )

                blocks.append(
                    Block(
                        block_type="image",
                        content=img_hash,
                        norm=img_hash,
                        page_index=page_index,
                        block_index=i,
                        bbox=bbox,
                    )
                )

    return blocks


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def word_diff(old_text: str, new_text: str) -> List[Dict[str, str]]:
    """Generate a word-level diff between two strings."""
    # Split while preserving some punctuation if possible, but split() is simple
    old_tokens = old_text.split()
    new_tokens = new_text.split()

    # autojunk=False helps with small texts where common words might be ignored
    sm = SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
    diffs: List[Dict[str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for token in old_tokens[i1:i2]:
                diffs.append({"type": "equal", "value": token})
        elif tag == "delete":
            for token in old_tokens[i1:i2]:
                diffs.append({"type": "delete", "value": token})
        elif tag == "insert":
            for token in new_tokens[j1:j2]:
                diffs.append({"type": "insert", "value": token})
        elif tag == "replace":
            for token in old_tokens[i1:i2]:
                diffs.append({"type": "delete", "value": token})
            for token in new_tokens[j1:j2]:
                diffs.append({"type": "insert", "value": token})
    return diffs


def build_diff(
    old_blocks: List[Block],
    new_blocks: List[Block],
    threshold: float,
) -> List[Dict[str, Any]]:
    """Build a comprehensive diff between two lists of blocks."""
    # Use SequenceMatcher on block norms to find the best global alignment
    old_norms = [b.norm for b in old_blocks]
    new_norms = [b.norm for b in new_blocks]

    # We use a custom isjunk to ignore empty/very short blocks if needed
    sm = SequenceMatcher(None, old_norms, new_norms, autojunk=False)

    results: List[Dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # Blocks are identical (according to norm)
            for i, j in zip(range(i1, i2), range(j1, j2)):
                old = old_blocks[i]
                new = new_blocks[j]
                # Even if norm is equal, check for actual content diff (e.g. case)
                diff = word_diff(old.content, new.content)
                change = "unchanged"
                if any(t["type"] != "equal" for t in diff):
                    change = "modified"

                results.append(
                    {
                        "block_type": old.block_type,
                        "change": change,
                        "old_index": i,
                        "new_index": j,
                        "old_page": old.page_index,
                        "new_page": new.page_index,
                        "word_diff": diff,
                    }
                )
        elif tag == "delete":
            for i in range(i1, i2):
                old = old_blocks[i]
                results.append(
                    {
                        "block_type": old.block_type,
                        "change": "deleted",
                        "old_index": i,
                        "new_index": None,
                        "old_page": old.page_index,
                        "new_page": None,
                        "word_diff": word_diff(old.content, ""),
                    }
                )
        elif tag == "insert":
            for j in range(j1, j2):
                new = new_blocks[j]
                results.append(
                    {
                        "block_type": new.block_type,
                        "change": "added",
                        "old_index": None,
                        "new_index": j,
                        "old_page": None,
                        "new_page": new.page_index,
                        "word_diff": word_diff("", new.content),
                    }
                )
        elif tag == "replace":
            # Heuristic for matching blocks in a 'replace' range
            sub_old_idx = list(range(i1, i2))
            sub_new_idx = list(range(j1, j2))

            # Try to match blocks based on similarity threshold
            matched = []
            used_new = set()
            for oi in sub_old_idx:
                best_ji = None
                best_score = 0.0
                for ni in sub_new_idx:
                    if ni in used_new:
                        continue
                    if old_blocks[oi].block_type != new_blocks[ni].block_type:
                        continue

                    score = similarity(old_blocks[oi].norm, new_blocks[ni].norm)
                    if score > best_score:
                        best_score = score
                        best_ji = ni

                if best_ji is not None and best_score >= threshold:
                    matched.append((oi, best_ji))
                    used_new.add(best_ji)

            # Now we have some matched pairs. The rest are deleted/added.
            matched_old = {m[0] for m in matched}
            matched_new = {m[1] for m in matched}

            # To maintain some order, we can output deletions, then matched/modified, then insertions
            # or try to interleave. Let's just follow the order of the old document for deletions/matches
            # and then the new document for remaining insertions.

            for oi in sub_old_idx:
                if oi in matched_old:
                    # Find which new index it matched
                    ni = next(m[1] for m in matched if m[0] == oi)
                    old = old_blocks[oi]
                    new = new_blocks[ni]
                    diff = word_diff(old.content, new.content)
                    results.append(
                        {
                            "block_type": old.block_type,
                            "change": "modified",
                            "old_index": oi,
                            "new_index": ni,
                            "old_page": old.page_index,
                            "new_page": new.page_index,
                            "word_diff": diff,
                        }
                    )
                else:
                    old = old_blocks[oi]
                    results.append(
                        {
                            "block_type": old.block_type,
                            "change": "deleted",
                            "old_index": oi,
                            "new_index": None,
                            "old_page": old.page_index,
                            "new_page": None,
                            "word_diff": word_diff(old.content, ""),
                        }
                    )

            for ni in sub_new_idx:
                if ni not in matched_new:
                    new = new_blocks[ni]
                    results.append(
                        {
                            "block_type": new.block_type,
                            "change": "added",
                            "old_index": None,
                            "new_index": ni,
                            "old_page": None,
                            "new_page": new.page_index,
                            "word_diff": word_diff("", new.content),
                        }
                    )

    return results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two PDFs and output JSON diff."
    )
    parser.add_argument("old_pdf", help="Path to the original PDF")
    parser.add_argument("new_pdf", help="Path to the updated PDF")
    parser.add_argument("--out", "-o", required=True, help="Output JSON path")
    parser.add_argument("--pdf", help="Output diff PDF path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for block matching (0-1)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not os.path.isfile(args.old_pdf):
        print(f"Error: not found: {args.old_pdf}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.new_pdf):
        print(f"Error: not found: {args.new_pdf}", file=sys.stderr)
        return 2

    try:
        old_blocks = extract_blocks(args.old_pdf)
        new_blocks = extract_blocks(args.new_pdf)
    except Exception as exc:
        print(f"Error: failed to parse PDFs: {exc}", file=sys.stderr)
        return 3

    diff = build_diff(old_blocks, new_blocks, args.threshold)

    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(diff, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"Error: failed to write output: {exc}", file=sys.stderr)
        return 4

    if args.pdf:
        try:
            render_github_diff_pdf(diff, args.pdf)
        except Exception as exc:
            print(f"Error: failed to write diff PDF: {exc}", file=sys.stderr)
            return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

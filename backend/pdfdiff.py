#!/usr/bin/env python3
"""pdfdiff: Compare two PDFs and output JSON diff summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import base64
import io
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import pdfplumber
from diff_pdf_report import render_github_diff_pdf


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
        "○",
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


def _split_paragraphs_from_lines(
    lines: List[Dict[str, Any]], page_width: float = 600.0
) -> List[Dict[str, Any]]:
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
        sorted_gaps = sorted(gaps)
        median_gap = sorted_gaps[(len(sorted_gaps) - 1) // 2]
        gap_threshold = max(6.0, median_gap * 1.25)

    def join_buffer(buf: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not buf:
            return {"text": "", "bbox": None}

        # Merge text
        text_parts = []
        for i, line in enumerate(buf):
            t = line.get("text", "").strip()
            t = clean_text(t)
            if not t:
                continue
            if i > 0:
                prev_raw = buf[i - 1].get("text", "").strip()
                prev_clean = clean_text(prev_raw)
                if prev_clean.endswith(("-", "‐", "‑")):
                    if text_parts:
                        text_parts[-1] = text_parts[-1][:-1] + t
                    else:
                        text_parts.append(t)
                else:
                    text_parts.append(t)
            else:
                text_parts.append(t)

        merged_text = " ".join(text_parts)

        # Merge bbox
        x0s = [float(l["x0"]) for l in buf if l.get("x0") is not None]
        tops = [float(l["top"]) for l in buf if l.get("top") is not None]
        x1s = [float(l["x1"]) for l in buf if l.get("x1") is not None]
        bottoms = [float(l["bottom"]) for l in buf if l.get("bottom") is not None]

        bbox = None
        if x0s and tops and x1s and bottoms:
            bbox = (min(x0s), min(tops), max(x1s), max(bottoms))

        return {"text": merged_text, "bbox": bbox}

    paragraphs: List[Dict[str, Any]] = []
    buffer: List[Dict[str, Any]] = []

    prev_bottom: Optional[float] = None
    prev_x0: Optional[float] = None
    prev_x1: Optional[float] = None

    column_break_threshold = 24.0
    short_line_threshold = 60.0  # pixels to edge

    # Estimate content right margin from statistics if possible
    max_content_x1 = 0.0
    for l in sorted_lines:
        lx1 = l.get("x1")
        if isinstance(lx1, (int, float)) and lx1 > max_content_x1:
            max_content_x1 = float(lx1)

    reference_right = max_content_x1 if max_content_x1 > 0 else page_width

    for line in sorted_lines:
        text = line.get("text", "")
        cleaned = clean_text(text)
        if not cleaned:
            continue

        top = line.get("top")
        x0 = line.get("x0")
        x1 = line.get("x1")
        line_x0 = float(x0) if isinstance(x0, (int, float)) else None
        line_x1 = float(x1) if isinstance(x1, (int, float)) else None

        gap = None
        if prev_bottom is not None and isinstance(top, (int, float)):
            gap = top - prev_bottom

        should_split = False

        if buffer and is_bullet_start(cleaned):
            should_split = True

        if not should_split and gap is not None and gap > gap_threshold:
            should_split = True

        if not should_split and line_x0 is not None and prev_x0 is not None:
            if abs(line_x0 - prev_x0) >= column_break_threshold:
                should_split = True

        # Short line heuristic
        if not should_split and prev_x1 is not None and gap is not None:
            dist_to_right = reference_right - prev_x1
            if dist_to_right > short_line_threshold and gap > 0:
                should_split = True

        if buffer and should_split:
            paragraphs.append(join_buffer(buffer))
            buffer = []

        buffer.append(line)

        prev_bottom = (
            line.get("bottom")
            if isinstance(line.get("bottom"), (int, float))
            else prev_bottom
        )
        prev_x0 = line_x0 if line_x0 is not None else prev_x0
        prev_x1 = line_x1 if line_x1 is not None else prev_x1

    if buffer:
        paragraphs.append(join_buffer(buffer))

    return paragraphs


@dataclass
class PageInfo:
    width: float
    height: float
    index: int


@dataclass
class ExtractionResult:
    blocks: List[Block]
    pages: List[PageInfo]


def extract_blocks(pdf_path: str) -> ExtractionResult:
    blocks: List[Block] = []
    pages: List[PageInfo] = []

    # Use tighter tolerance (defaults: x=3, y=3) to avoid merging words
    # x_tolerance=2 helps with resumes that often have tight kerning or missing spaces
    extraction_settings = {"x_tolerance": 2, "y_tolerance": 3}

    # We can also attempt to extract physical layout first to guide paragraph merging
    # But pdfplumber text_lines is usually good enough if we handle gaps right.

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            pages.append(
                PageInfo(
                    width=float(page.width), height=float(page.height), index=page_index
                )
            )
            text_lines = page.extract_text_lines(**extraction_settings) or []
            paragraphs_data = _split_paragraphs_from_lines(
                text_lines, page_width=float(page.width)
            )

            if not paragraphs_data:
                # Fallback for empty line extraction
                text = page.extract_text(**extraction_settings) or ""
                if text.strip():
                    # No bbox for fallback
                    blocks.append(
                        Block(
                            block_type="paragraph",
                            content=text.strip(),
                            norm=normalize_text(text.strip()),
                            page_index=page_index,
                            block_index=0,
                            bbox=None,
                        )
                    )

            for i, p_data in enumerate(paragraphs_data):
                content = p_data["text"]
                bbox = p_data["bbox"]
                blocks.append(
                    Block(
                        block_type="paragraph",
                        content=content,
                        norm=normalize_text(content),
                        page_index=page_index,
                        block_index=i,
                        bbox=bbox,
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
                # Try to get the image bytes for hashing and display
                img_hash = None
                data_uri = None

                x0 = img.get("x0")
                top = img.get("top")
                x1 = img.get("x1")
                bottom = img.get("bottom")

                # Create a valid bbox tuple
                bbox: Optional[Tuple[float, float, float, float]] = None
                if all(v is not None for v in (x0, top, x1, bottom)):
                    bbox = (
                        float(x0),
                        float(top),
                        float(x1),
                        float(bottom),
                    )

                if bbox:
                    # Generate visual representation for the frontend
                    # We crop the page to the image area and convert to base64
                    try:
                        # crop() expects (x0, top, x1, bottom)
                        cropped = page.crop(bbox)
                        # to_image() returns a PageImage, .original is the PIL Image
                        pil_img = cropped.to_image(resolution=72).original

                        buffered = io.BytesIO()
                        pil_img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        data_uri = f"data:image/png;base64,{img_str}"

                        # Use the pixel data hash for diffing if we have it
                        img_hash = hash_bytes(buffered.getvalue())
                    except Exception as e:
                        # Fallback if cropping fails
                        # print(f"Image extraction warning: {e}", file=sys.stderr)
                        pass

                # Fallback hash calculation if extraction failed or no bbox
                if not img_hash:
                    try:
                        # Try object ID method
                        if "object_id" in img:
                            pdf_obj = getattr(page, "pdf", None)
                            streams = getattr(pdf_obj, "streams", None)
                            if streams and img["object_id"] in streams:
                                raw = streams[img["object_id"]].get_data()
                                img_hash = hash_bytes(raw)
                    except Exception:
                        pass

                if not img_hash:
                    # Last resort: metadata hash
                    img_hash = hash_bytes(
                        json.dumps(img, sort_keys=True, default=str).encode("utf-8")
                    )

                blocks.append(
                    Block(
                        block_type="image",
                        content=(
                            data_uri if data_uri else "[Image]"
                        ),  # Send base64 to frontend
                        norm=img_hash,  # Diff on hash
                        page_index=page_index,
                        block_index=i,
                        bbox=bbox,
                    )
                )

    return ExtractionResult(blocks=blocks, pages=pages)


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def word_diff(old_text: Any, new_text: Any) -> List[Dict[str, str]]:
    """Generate a word-level diff between two strings."""
    # Handle non-string content (e.g., images or tables) by treating as raw string or skipping
    if not isinstance(old_text, str):
        old_text = str(old_text) if old_text is not None else ""
    if not isinstance(new_text, str):
        new_text = str(new_text) if new_text is not None else ""

    # If it's a data URI (image), we don't word diff it.
    if old_text.startswith("data:image") or new_text.startswith("data:image"):
        if old_text == new_text:
            return [{"type": "equal", "value": "[Image]"}]
        else:
            if old_text and not new_text:
                return [{"type": "delete", "value": "[Image]"}]
            elif not old_text and new_text:
                return [{"type": "insert", "value": "[Image]"}]
            else:
                return [
                    {"type": "delete", "value": "[Image]"},
                    {"type": "insert", "value": "[Image]"},
                ]

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
                        "old_bbox": old.bbox,
                        "new_bbox": new.bbox,
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
                        "old_bbox": old.bbox,
                        "new_bbox": None,
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
                        "old_bbox": None,
                        "new_bbox": new.bbox,
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
                            "old_bbox": old.bbox,
                            "new_bbox": new.bbox,
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
                            "old_bbox": old.bbox,
                            "new_bbox": None,
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
                            "old_bbox": None,
                            "new_bbox": new.bbox,
                        }
                    )

    # Post-processing: Detect Moves
    # Identify items that were marked 'deleted' and 'added' but have identical/similar content
    deleted_indices = [i for i, r in enumerate(results) if r["change"] == "deleted"]
    added_indices = [i for i, r in enumerate(results) if r["change"] == "added"]

    # Simple exact match heuristic (can be expanded to high similarity)
    # We loop through deleted items and try to find a match in added items
    used_added_indices = set()

    for d_idx in deleted_indices:
        d_item = results[d_idx]
        d_content_norm = old_blocks[d_item["old_index"]].norm

        # Try to find a match in added items
        match_found_idx = None
        for a_idx in added_indices:
            if a_idx in used_added_indices:
                continue

            a_item = results[a_idx]
            if d_item["block_type"] != a_item["block_type"]:
                continue

            a_content_norm = new_blocks[a_item["new_index"]].norm

            # Use exact match for now to be safe, or high threshold
            if d_content_norm == a_content_norm and len(d_content_norm) > 10:
                match_found_idx = a_idx
                break

        if match_found_idx is not None:
            # Link them as moved
            a_item = results[match_found_idx]
            used_added_indices.add(match_found_idx)

            # Update the 'deleted' entry to be 'moved_source' (or just moved)
            # Update the 'added' entry to be 'moved_target'
            # Or better: merge them logic wise for the viewer?
            # For the viewer to show them on both sides, we need to keep both entries
            # but change their status so UI can color them differently.

            results[d_idx]["change"] = "moved"
            results[d_idx]["new_index"] = a_item["new_index"]
            results[d_idx]["new_page"] = a_item["new_page"]
            results[d_idx]["new_bbox"] = a_item["new_bbox"]
            results[d_idx]["word_diff"] = word_diff(
                old_blocks[d_item["old_index"]].content,
                new_blocks[a_item["new_index"]].content,
            )

            results[match_found_idx]["change"] = "moved"
            results[match_found_idx]["old_index"] = d_item["old_index"]
            results[match_found_idx]["old_page"] = d_item["old_page"]
            results[match_found_idx]["old_bbox"] = d_item["old_bbox"]
            results[match_found_idx]["word_diff"] = results[d_idx]["word_diff"]

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
        old_res = extract_blocks(args.old_pdf)
        new_res = extract_blocks(args.new_pdf)
        old_blocks = old_res.blocks
        new_blocks = new_res.blocks
    except Exception as exc:
        print(f"Error: failed to parse PDFs: {exc}", file=sys.stderr)
        return 3

    diff = build_diff(old_blocks, new_blocks, args.threshold)

    # Structure output to include page info
    output = {
        "diff": diff,
        "old_pages": [
            {"width": p.width, "height": p.height, "index": p.index}
            for p in old_res.pages
        ],
        "new_pages": [
            {"width": p.width, "height": p.height, "index": p.index}
            for p in new_res.pages
        ],
    }

    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
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

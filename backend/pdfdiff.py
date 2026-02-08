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


def extract_blocks(pdf_path: str) -> List[Block]:
    blocks: List[Block] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                paragraphs = [p for p in (seg.strip() for seg in text.split("\n")) if p]
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
                        json.dumps(img, sort_keys=True).encode("utf-8")
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


def match_blocks(
    old_blocks: List[Block], new_blocks: List[Block], threshold: float
) -> Tuple[Dict[int, int], List[int], List[int]]:
    matched_old_to_new: Dict[int, int] = {}
    used_new: set[int] = set()

    for i, old in enumerate(old_blocks):
        best_j = None
        best_score = 0.0
        for j, new in enumerate(new_blocks):
            if j in used_new or old.block_type != new.block_type:
                continue
            score = similarity(old.norm, new.norm)
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is not None and best_score >= threshold:
            matched_old_to_new[i] = best_j
            used_new.add(best_j)

    unmatched_old = [i for i in range(len(old_blocks)) if i not in matched_old_to_new]
    unmatched_new = [j for j in range(len(new_blocks)) if j not in used_new]
    return matched_old_to_new, unmatched_old, unmatched_new


def word_diff(old_text: str, new_text: str) -> List[Dict[str, str]]:
    old_tokens = old_text.split()
    new_tokens = new_text.split()
    sm = SequenceMatcher(None, old_tokens, new_tokens)
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
    matched, unmatched_old, unmatched_new = match_blocks(
        old_blocks, new_blocks, threshold
    )

    results: List[Dict[str, Any]] = []

    for old_i, new_i in matched.items():
        old = old_blocks[old_i]
        new = new_blocks[new_i]
        change = "unchanged"
        if old_i != new_i:
            change = "moved"
        if old.block_type == "paragraph":
            diff = word_diff(old.content, new.content)
        else:
            diff = []

        results.append(
            {
                "block_type": old.block_type,
                "change": change,
                "old_index": old_i,
                "new_index": new_i,
                "old_page": old.page_index,
                "new_page": new.page_index,
                "word_diff": diff,
            }
        )

    for old_i in unmatched_old:
        old = old_blocks[old_i]
        results.append(
            {
                "block_type": old.block_type,
                "change": "deleted",
                "old_index": old_i,
                "new_index": None,
                "old_page": old.page_index,
                "new_page": None,
                "word_diff": [],
            }
        )

    for new_i in unmatched_new:
        new = new_blocks[new_i]
        results.append(
            {
                "block_type": new.block_type,
                "change": "added",
                "old_index": None,
                "new_index": new_i,
                "old_page": None,
                "new_page": new.page_index,
                "word_diff": [],
            }
        )

    return sorted(
        results, key=lambda x: (x["block_type"], x["change"], x.get("old_index") or -1)
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two PDFs and output JSON diff."
    )
    parser.add_argument("old_pdf", help="Path to the original PDF")
    parser.add_argument("new_pdf", help="Path to the updated PDF")
    parser.add_argument("--out", "-o", required=True, help="Output JSON path")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

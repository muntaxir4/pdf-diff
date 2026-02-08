"""Render a GitHub-style diff PDF using reportlab."""

from __future__ import annotations

from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

Token = Dict[str, str]
DiffItem = Dict[str, Any]


def _draw_text_line(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    font_name: str,
    font_size: int,
    color=colors.black,
) -> float:
    c.setFont(font_name, font_size)
    c.setFillColor(color)
    c.drawString(x, y, text)
    return y


def _draw_wrapped_tokens(
    c: canvas.Canvas,
    tokens: List[Token],
    x: float,
    y: float,
    max_width: float,
    line_height: float,
    font_name: str,
    font_size: int,
) -> float:
    space_w = c.stringWidth(" ", font_name, font_size)
    cursor_x = x
    cursor_y = y

    def new_line() -> None:
        nonlocal cursor_x, cursor_y
        cursor_x = x
        cursor_y -= line_height

    for token in tokens:
        value = token["value"]
        token_type = token["type"]
        if token_type == "insert":
            color = colors.green
        elif token_type == "delete":
            color = colors.red
        else:
            color = colors.black

        text_w = c.stringWidth(value, font_name, font_size)
        if cursor_x + text_w > x + max_width:
            new_line()
        c.setFont(font_name, font_size)
        c.setFillColor(color)
        c.drawString(cursor_x, cursor_y, value)
        cursor_x += text_w + space_w

    return cursor_y


def render_github_diff_pdf(diff: List[DiffItem], out_path: str) -> None:
    c = canvas.Canvas(out_path, pagesize=letter)
    width, height = letter
    margin = 50
    y = height - margin
    line_height = 14
    max_width = width - (2 * margin)

    def page_break() -> None:
        nonlocal y
        c.showPage()
        y = height - margin

    def ensure_space(lines: int = 1) -> None:
        nonlocal y
        if y - (lines * line_height) < margin:
            page_break()

    changed_items = [i for i in diff if i.get("change") != "unchanged"]
    total_inserted = 0
    total_deleted = 0
    for item in diff:
        if item.get("block_type") == "paragraph":
            for token in item.get("word_diff", []):
                if token.get("type") == "insert":
                    total_inserted += 1
                elif token.get("type") == "delete":
                    total_deleted += 1

    _draw_text_line(c, "PDF Diff Report", margin, y, "Helvetica-Bold", 14)
    y -= line_height
    _draw_text_line(
        c,
        f"Changes: {len(changed_items)} / Total blocks: {len(diff)}",
        margin,
        y,
        "Helvetica",
        10,
    )
    y -= line_height
    _draw_text_line(
        c,
        f"Words inserted: {total_inserted} | Words deleted: {total_deleted}",
        margin,
        y,
        "Helvetica",
        10,
    )
    y -= line_height * 2

    for index, item in enumerate(diff, start=1):
        ensure_space(3)
        header = (
            f"{index}. {item['block_type']} | {item['change']} | "
            f"old_index={item.get('old_index')} new_index={item.get('new_index')}"
        )
        _draw_text_line(c, header, margin, y, "Helvetica-Bold", 10)
        y -= line_height

        if item["block_type"] == "paragraph" and item.get("word_diff"):
            inserted = sum(1 for t in item["word_diff"] if t.get("type") == "insert")
            deleted = sum(1 for t in item["word_diff"] if t.get("type") == "delete")
            _draw_text_line(
                c,
                f"Words: +{inserted} / -{deleted}",
                margin,
                y,
                "Helvetica",
                9,
            )
            y -= line_height
            ensure_space(2)
            y = _draw_wrapped_tokens(
                c,
                item["word_diff"],
                margin,
                y,
                max_width,
                line_height,
                "Helvetica",
                10,
            )
            y -= line_height
        else:
            _draw_text_line(c, "(no word diff)", margin, y, "Helvetica", 9)
            y -= line_height

        y -= line_height / 2

    c.save()

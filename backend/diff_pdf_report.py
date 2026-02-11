"""Render a GitHub-style diff PDF using reportlab."""

from __future__ import annotations

from typing import Any, Dict, List
import base64
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

Token = Dict[str, str]
DiffItem = Dict[str, Any]

def render_github_diff_pdf(diff_data: Dict[str, Any], out_path: str) -> None:
    """
    Render a PDF report closely resembling the UI visualization.
    We restart the canvas for each page in the new/old sets and draw the boxes.
    Requires the full diff_data object containing 'diff', 'old_pages', and 'new_pages'.
    """
    if not isinstance(diff_data, dict) or "diff" not in diff_data:
        # Fallback if just list passed (legacy support)
        # We process it as empty pages ref if not provided
        diff_list = diff_data if isinstance(diff_data, list) else []
        if isinstance(diff_data, dict): return
        
        # If we really just got a list, we can't do the visual diff properly without page sizes.
        # But let's assume the caller will be fixed.
        return

    diff_list = diff_data["diff"]
    c = canvas.Canvas(out_path) # Page size will be set per page

    # Helper to draw a set of pages (like 'Original Version' or 'New Version')
    def draw_pages_side(pages_info, type_key, label):
        if not pages_info:
            return

        for page in pages_info:
            p_width = page["width"]
            p_height = page["height"]
            idx = page["index"]
            
            c.setPageSize((p_width, p_height))
            
            # White background
            c.setFillColor(colors.white)
            c.rect(0, 0, p_width, p_height, fill=1)
            
            # Draw Title at top
            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(20, p_height - 30, f"{label} - Page {idx + 1}")

            # Filter blocks for this page
            page_blocks = [
                b for b in diff_list 
                if (b["old_page"] == idx if type_key == "old" else b["new_page"] == idx)
            ]

            for item in page_blocks:
                change = item.get("change")
                bg_color = None
                border_color = None
                
                if change == "deleted":
                    bg_color = colors.Color(1, 0, 0, alpha=0.1)
                    border_color = colors.Color(1, 0, 0, alpha=0.5)
                elif change == "added":
                    bg_color = colors.Color(0, 1, 0, alpha=0.1)
                    border_color = colors.Color(0, 1, 0, alpha=0.5)
                elif change == "modified":
                    bg_color = colors.Color(1, 0.65, 0, alpha=0.1)
                    border_color = colors.Color(1, 0.65, 0, alpha=0.5)
                elif change == "moved":
                    bg_color = colors.Color(0, 0, 1, alpha=0.1)
                    border_color = colors.Color(0, 0, 1, alpha=0.5)
                
                bbox = item.get("old_bbox") if type_key == "old" else item.get("new_bbox")
                if not bbox:
                    continue
                
                # pdfplumber bbox: (x0, top, x1, bottom) from top-left
                # ReportLab: (x, y) from bottom-left
                x0, top, x1, bottom = bbox
                rect_x = x0
                rect_y = p_height - bottom
                rect_w = x1 - x0
                rect_h = bottom - top
                
                # Draw Box
                if bg_color:
                    c.setFillColor(bg_color)
                    c.setStrokeColor(border_color)
                    c.rect(rect_x, rect_y, rect_w, rect_h, fill=1, stroke=1)
                
                # Draw Text or Image Placeholder
                c.setStrokeColor(colors.black)
                c.setFillColor(colors.black)
                
                word_diff = item.get("word_diff")
                
                if item.get("block_type") == "image":
                    found_img = False
                    if word_diff:
                        val = ""
                        for t in word_diff:
                            if isinstance(t, dict):
                                v = t.get("value", "")
                                if v.startswith("data:image"):
                                    val = v
                                    break
                                    
                        if val.startswith("data:image"):
                             try:
                                 header, encoded = val.split(",", 1)
                                 data = base64.b64decode(encoded)
                                 img_stream = BytesIO(data)
                                 img_reader = ImageReader(img_stream)
                                 c.drawImage(img_reader, rect_x, rect_y, width=rect_w, height=rect_h, preserveAspectRatio=True, mask='auto')
                                 found_img = True
                             except Exception:
                                 pass
                    
                    if not found_img:
                        c.setFont("Helvetica-Oblique", 8)
                        c.drawString(rect_x + 2, rect_y + rect_h/2, "[Image]")

                elif word_diff:
                    text_y = rect_y + rect_h - 10 
                    text_x = rect_x + 2
                    c.setFont("Helvetica", 10)
                    
                    for token in word_diff:
                        if not isinstance(token, dict): continue
                        
                        val = token.get("value", "")
                        t_type = token.get("type")
                        
                        # In Old Version, don't show inserted text. In New Version, don't show deleted text.
                        if type_key == "old" and t_type == "insert": continue
                        if type_key == "new" and t_type == "delete": continue
                        
                        t_color = colors.black
                        if type_key == "old" and t_type == "delete": t_color = colors.red
                        if type_key == "new" and t_type == "insert": t_color = colors.green
                        
                        c.setFillColor(t_color)
                        c.drawString(text_x, text_y, val)
                        text_x += c.stringWidth(val, "Helvetica", 10) + 2
                        
                        if text_x > rect_x + rect_w:
                             text_x = rect_x + 2
                             text_y -= 12 # simple line wrapping
                             if text_y < rect_y: break
            
            c.showPage()

    # Draw Old Version Pages
    draw_pages_side(diff_data.get("old_pages", []), "old", "Original Version")
    
    # Draw New Version Pages
    draw_pages_side(diff_data.get("new_pages", []), "new", "New Version")

    c.save()

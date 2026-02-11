from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import tempfile
import sys
from typing import List, Dict, Any

# Add parent directory to sys.path to import pdfdiff
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pdfdiff
from diff_pdf_report import render_github_diff_pdf

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/diff")
async def calculate_diff(
    old_pdf: UploadFile = File(...), new_pdf: UploadFile = File(...)
):
    # Create temporary files
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as old_tmp, tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as new_tmp:

        try:
            shutil.copyfileobj(old_pdf.file, old_tmp)
            shutil.copyfileobj(new_pdf.file, new_tmp)

            old_tmp_path = old_tmp.name
            new_tmp_path = new_tmp.name

            # Close files so pdfplumber can open them
            old_tmp.close()
            new_tmp.close()

            # Extract blocks using existing logic
            old_res = pdfdiff.extract_blocks(old_tmp_path)
            new_res = pdfdiff.extract_blocks(new_tmp_path)

            # Build diff
            diff_results = pdfdiff.build_diff(
                old_res.blocks, new_res.blocks, threshold=0.7
            )

            return {
                "diff": diff_results,
                "old_pages": [
                    {"width": p.width, "height": p.height, "index": p.index}
                    for p in old_res.pages
                ],
                "new_pages": [
                    {"width": p.width, "height": p.height, "index": p.index}
                    for p in new_res.pages
                ],
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up temporary files
            if os.path.exists(old_tmp_path):
                os.remove(old_tmp_path)
            if os.path.exists(new_tmp_path):
                os.remove(new_tmp_path)


@app.post("/diff-report")
async def generate_diff_report(
    old_pdf: UploadFile = File(...), new_pdf: UploadFile = File(...)
):
    # Create temporary files
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as old_tmp, tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as new_tmp:
        out_path = tempfile.mktemp(suffix=".pdf")

        try:
            shutil.copyfileobj(old_pdf.file, old_tmp)
            shutil.copyfileobj(new_pdf.file, new_tmp)

            old_tmp_path = old_tmp.name
            new_tmp_path = new_tmp.name

            old_tmp.close()
            new_tmp.close()

            old_res = pdfdiff.extract_blocks(old_tmp_path)
            new_res = pdfdiff.extract_blocks(new_tmp_path)

            diff_results = pdfdiff.build_diff(
                old_res.blocks, new_res.blocks, threshold=0.7
            )

            diff_data = {
                "diff": diff_results,
                "old_pages": [
                    {"width": p.width, "height": p.height, "index": p.index}
                    for p in old_res.pages
                ],
                "new_pages": [
                    {"width": p.width, "height": p.height, "index": p.index}
                    for p in new_res.pages
                ],
            }

            # Generate the PDF report
            render_github_diff_pdf(diff_data, out_path)

            return FileResponse(
                out_path, filename="diff_report.pdf", media_type="application/pdf"
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if os.path.exists(old_tmp_path):
                os.remove(old_tmp_path)
            if os.path.exists(new_tmp_path):
                os.remove(new_tmp_path)
                # Note: we don't delete out_path immediately because FileResponse needs it.
                # In a real app, use a BackgroundTask to cleanup.


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)

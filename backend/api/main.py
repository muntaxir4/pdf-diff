from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
import sys
from typing import List, Dict, Any

# Add parent directory to sys.path to import pdfdiff
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pdfdiff

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
            old_blocks = pdfdiff.extract_blocks(old_tmp_path)
            new_blocks = pdfdiff.extract_blocks(new_tmp_path)

            # Build diff
            diff_results = pdfdiff.build_diff(old_blocks, new_blocks, threshold=0.7)

            return diff_results

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up temporary files
            if os.path.exists(old_tmp_path):
                os.remove(old_tmp_path)
            if os.path.exists(new_tmp_path):
                os.remove(new_tmp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

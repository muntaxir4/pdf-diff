import { useState, useEffect, useRef } from "react";
import PdfUploader from "./components/PdfUploader";
import DiffResult from "./components/DiffResult";
import "./App.css";

function App() {
  const [diff, setDiff] = useState(null);
  const oldFileRef = useRef(null);
  const newFileRef = useRef(null);

  const handleDiffResult = (result, files) => {
    setDiff(result);
    // Store files for report generation
    if (files) {
      oldFileRef.current = files.oldFile;
      newFileRef.current = files.newFile;
    }
  };

  useEffect(() => {
    const handleDownload = async () => {
      if (!oldFileRef.current || !newFileRef.current) {
        alert("Files not available for report generation.");
        return;
      }

      const formData = new FormData();
      formData.append("old_pdf", oldFileRef.current);
      formData.append("new_pdf", newFileRef.current);

      try {
        const response = await fetch("http://localhost:8080/diff-report", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error("Report generation failed");
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "diff_report.pdf";
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
      } catch (error) {
        console.error("Download error:", error);
        alert("Failed to download report.");
      }
    };

    window.addEventListener("downloadReport", handleDownload);
    return () => window.removeEventListener("downloadReport", handleDownload);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="mx-auto">
        <header className="mb-12 text-center">
          <h1 className="text-4xl font-extrabold text-gray-900 mb-2">
            PDF Diff Tool
          </h1>
          <p className="text-lg text-gray-600">
            Upload two versions of a PDF to see the changes.
          </p>
        </header>

        <main>
          <PdfUploader onDiffResult={handleDiffResult} />
          {diff && <DiffResult diff={diff} />}
        </main>
      </div>
    </div>
  );
}

export default App;

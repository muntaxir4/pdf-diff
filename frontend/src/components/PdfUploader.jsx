import React, { useState } from "react";

const PdfUploader = ({ onDiffResult }) => {
  const [oldFile, setOldFile] = useState(null);
  const [newFile, setNewFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!oldFile || !newFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("old_pdf", oldFile);
    formData.append("new_pdf", newFile);

    try {
      const response = await fetch("http://localhost:8000/diff", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to calculate diff");
      }

      const data = await response.json();
      onDiffResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md mb-8">
      <h2 className="text-xl font-bold mb-4">Upload PDFs to Compare</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Original PDF (Old)
            </label>
            <input
              type="file"
              accept=".pdf"
              onChange={(e) => setOldFile(e.target.files[0])}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Updated PDF (New)
            </label>
            <input
              type="file"
              accept=".pdf"
              onChange={(e) => setNewFile(e.target.files[0])}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
          </div>
        </div>
        <button
          type="submit"
          disabled={loading || !oldFile || !newFile}
          className={`w-full py-2 px-4 rounded-md text-white font-bold ${
            loading || !oldFile || !newFile
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {loading ? "Comparing..." : "Compare PDFs"}
        </button>
      </form>
      {error && <p className="mt-4 text-red-600 text-sm">{error}</p>}
    </div>
  );
};

export default PdfUploader;

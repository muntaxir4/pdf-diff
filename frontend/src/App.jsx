import { useState } from "react";
import PdfUploader from "./components/PdfUploader";
import DiffResult from "./components/DiffResult";
import "./App.css";

function App() {
  const [diff, setDiff] = useState(null);

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <header className="mb-12 text-center">
          <h1 className="text-4xl font-extrabold text-gray-900 mb-2">
            PDF Diff Tool
          </h1>
          <p className="text-lg text-gray-600">
            Upload two versions of a PDF to see the changes.
          </p>
        </header>

        <main>
          <PdfUploader onDiffResult={setDiff} />
          {diff && <DiffResult diff={diff} />}
        </main>
      </div>
    </div>
  );
}

export default App;

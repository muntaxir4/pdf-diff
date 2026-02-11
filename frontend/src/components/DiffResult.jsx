import React, { useMemo } from "react";

const PageRenderer = ({ page, blocks, type }) => {
  // Constants for display
  const PAGE_WIDTH_PX = 500; // Fixed display width for layout calculations
  const scale = PAGE_WIDTH_PX / page.width;
  const heightPx = page.height * scale;

  return (
    <div
      className="mb-8 relative shadow-lg bg-white border border-gray-200 mx-auto"
      style={{ width: PAGE_WIDTH_PX, height: heightPx }}
    >
      {/* Page Number Label */}
      <div className="absolute -top-6 left-0 text-xs text-gray-500 font-bold uppercase tracking-wider">
        Page {page.index + 1}
      </div>

      {blocks.map((item, idx) => {
        const bbox = type === "old" ? item.old_bbox : item.new_bbox;
        if (!bbox) return null;

        const [x0, top, x1, bottom] = bbox;
        const left = x0 * scale;
        const t = top * scale;
        const w = (x1 - x0) * scale;
        const h = (bottom - top) * scale;

        let bgColor = "transparent";
        let borderColor = "transparent";

        if (item.change === "deleted") {
          bgColor = "rgba(255, 0, 0, 0.2)"; // Red for deleted
          borderColor = "rgba(255, 0, 0, 0.5)";
        } else if (item.change === "added") {
          bgColor = "rgba(0, 255, 0, 0.2)"; // Green for added
          borderColor = "rgba(0, 255, 0, 0.5)";
        } else if (item.change === "modified") {
          bgColor = "rgba(255, 165, 0, 0.2)"; // Orange for modified
          borderColor = "rgba(255, 165, 0, 0.5)";
        } else if (item.change === "moved") {
          bgColor = "rgba(0, 0, 255, 0.2)"; // Blue for moved
          borderColor = "rgba(0, 0, 255, 0.5)";
        }

        // Reconstruct text content for display
        const textContent = item.word_diff
          ? item.word_diff
              .filter((w) =>
                type === "old" ? w.type !== "insert" : w.type !== "delete",
              )
              .map((w) => w.value)
              .join(" ")
          : "";

        const isImage =
          item.block_type === "image" && textContent.startsWith("data:image");

        // Render individual words inside the block to simulate proper highlighting
        const renderWords = () => {
          if (isImage) {
            return (
              <img
                src={textContent}
                alt="Extracted"
                className="w-full h-full object-contain"
              />
            );
          }
          if (item.block_type === "image") return <span>[Image]</span>;
          if (!item.word_diff) return textContent;

          return item.word_diff
            .filter((w) =>
              type === "old" ? w.type !== "insert" : w.type !== "delete",
            )
            .map((w, i) => {
              let wordBg = "transparent";
              if (
                type === "old" &&
                w.type === "delete" &&
                item.change === "modified"
              ) {
                wordBg = "rgba(255, 0, 0, 0.4)";
              } else if (
                type === "new" &&
                w.type === "insert" &&
                item.change === "modified"
              ) {
                wordBg = "rgba(0, 255, 0, 0.4)";
              }

              return (
                <span key={i} style={{ backgroundColor: wordBg }}>
                  {w.value}{" "}
                </span>
              );
            });
        };

        return (
          <div
            key={idx}
            className="absolute leading-none hover:z-50 hover:outline hover:outline-blue-500 select-none cursor-pointer flex items-start flex-wrap content-start"
            title={isImage ? "Image" : textContent}
            style={{
              left: left,
              top: t,
              width: w,
              height: h,
              backgroundColor: bgColor,
              border: `1px solid ${borderColor}`,
              color: "rgba(0,0,0,0.8)",
              fontSize: "9px",
              lineHeight: 1,
              whiteSpace: "pre-wrap",
              wordBreak: "break-all",
              overflow: "hidden",
            }}
          >
            {renderWords()}
          </div>
        );
      })}
    </div>
  );
};

const DiffResult = ({ diff: diffData }) => {
  if (!diffData || !diffData.diff) return null;

  const { diff, old_pages, new_pages } = diffData;

  // Need to persist the file objects to re-send them for report generation
  // Since we don't have access to the original file objects securely here unless passed down
  // simple way: Assume the parent component holds them or we create a new specialized component.
  // For now, let's just make the UI logic.
  // To actually download, the parent App.jsx needs to pass a "handleDownloadReport" function
  // or we need to access the file inputs again.
  //
  // Let's assume passed as props or context.

  // Filter blocks for a specific page efficiently
  const getBlocksForPage = (pageIndex, type) => {
    return diff.filter((item) =>
      type === "old"
        ? item.old_page === pageIndex
        : item.new_page === pageIndex,
    );
  };

  return (
    <div className="w-full mt-8 font-sans">
      <div className="max-w-[1400px] mx-auto min-w-[1000px]">
        <div className="flex justify-between items-center bg-white p-4 shadow-sm rounded-lg mb-6 border border-gray-200">
          <div className="flex items-center gap-4">
            <h2 className="text-lg font-bold text-gray-800">
              Visual PDF Comparison
            </h2>
            <button
              className="px-3 py-1 bg-gray-800 text-white text-xs rounded hover:bg-gray-700 transition"
              id="download-report-btn"
              onClick={() => {
                // We need to trigger the download from the parent or use a global event
                // dispatch event for App to handle
                const event = new CustomEvent("downloadReport");
                window.dispatchEvent(event);
              }}
            >
              Download Report
            </button>
          </div>
          <div className="flex gap-4 text-sm">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 bg-red-200 border border-red-400 block rounded-sm"></span>
              <span className="text-gray-600">Deleted (Old)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 bg-green-200 border border-green-400 block rounded-sm"></span>
              <span className="text-gray-600">Added (New)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 bg-orange-200 border border-orange-400 block rounded-sm"></span>
              <span className="text-gray-600">Modified</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 bg-blue-200 border border-blue-400 block rounded-sm"></span>
              <span className="text-gray-600">Moved</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-8 px-4">
          {/* Old Version Column */}
          <div className="bg-gray-100/50 p-6 rounded-xl border border-gray-200">
            <h3 className="text-center font-bold text-gray-700 mb-6 bg-white py-2 rounded shadow-sm border border-gray-100">
              Original Version
            </h3>
            <div className="flex flex-col items-center justify-center">
              {old_pages.map((page) => (
                <PageRenderer
                  key={`old-${page.index}`}
                  page={page}
                  blocks={getBlocksForPage(page.index, "old")}
                  type="old"
                />
              ))}
            </div>
          </div>

          {/* New Version Column */}
          <div className="bg-gray-100/50 p-6 rounded-xl border border-gray-200">
            <h3 className="text-center font-bold text-gray-700 mb-6 bg-white py-2 rounded shadow-sm border border-gray-100">
              New Version
            </h3>
            <div className="flex flex-col items-center justify-center">
              {new_pages.map((page) => (
                <PageRenderer
                  key={`new-${page.index}`}
                  page={page}
                  blocks={getBlocksForPage(page.index, "new")}
                  type="new"
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DiffResult;

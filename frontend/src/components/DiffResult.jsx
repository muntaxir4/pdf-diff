import React, { useState } from "react";

const DiffResult = ({ diff }) => {
  const [viewMode, setViewMode] = useState("unified"); // 'unified' | 'split'

  if (!diff || diff.length === 0) return null;

  // Calculate stats
  const stats = {
    total: diff.length,
    modified: diff.filter((i) => i.change === "modified").length,
    added: diff.filter((i) => i.change === "added").length,
    deleted: diff.filter((i) => i.change === "deleted").length,
    wordsInserted: diff.reduce(
      (acc, item) =>
        acc +
        (item.word_diff
          ? item.word_diff.filter((t) => t.type === "insert").length
          : 0),
      0,
    ),
    wordsDeleted: diff.reduce(
      (acc, item) =>
        acc +
        (item.word_diff
          ? item.word_diff.filter((t) => t.type === "delete").length
          : 0),
      0,
    ),
  };

  const reconstructText = (wordDiff, version) => {
    if (!wordDiff) return "";
    return wordDiff
      .filter((t) =>
        version === "old" ? t.type !== "insert" : t.type !== "delete",
      )
      .map((t) => t.value)
      .join(" ");
  };

  return (
    <div className="w-full max-w-6xl mx-auto mt-8 bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden font-sans">
      {/* Header / Toolbar */}
      <div className="bg-gray-50 border-b border-gray-200 px-4 py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold text-gray-700">Diff Summary</h2>
          <div className="text-xs text-gray-500 mt-1 flex gap-3">
            <span className="flex items-center text-green-600 font-medium">
              <span className="text-lg mr-1">+</span>
              {stats.wordsInserted} words
            </span>
            <span className="flex items-center text-red-600 font-medium">
              <span className="text-lg mr-1">-</span>
              {stats.wordsDeleted} words
            </span>
            <span className="flex items-center text-gray-600">
              {stats.total} blocks changed
            </span>
          </div>
        </div>

        <div className="flex bg-white rounded-md shadow-sm border border-gray-300 overflow-hidden shrink-0">
          <button
            onClick={() => setViewMode("unified")}
            className={`px-4 py-1.5 text-xs font-medium cursor-pointer transition-colors ${
              viewMode === "unified"
                ? "bg-blue-50 text-blue-700 border-r border-gray-200"
                : "text-gray-600 hover:bg-gray-50 border-r border-gray-200"
            }`}
          >
            Unified
          </button>
          <button
            onClick={() => setViewMode("split")}
            className={`px-4 py-1.5 text-xs font-medium cursor-pointer transition-colors ${
              viewMode === "split"
                ? "bg-blue-50 text-blue-700"
                : "text-gray-600 hover:bg-gray-50"
            }`}
          >
            Split
          </button>
        </div>
      </div>

      {/* Changes List */}
      <div className="divide-y divide-gray-200">
        {diff.map((item, idx) => {
          const prevItem = idx > 0 ? diff[idx - 1] : null;
          // Show page header if page changed or first item
          const currentOldPage = item.old_page;
          const currentNewPage = item.new_page;
          const prevOldPage = prevItem ? prevItem.old_page : -1;
          const prevNewPage = prevItem ? prevItem.new_page : -1;

          const showHeader =
            idx === 0 ||
            (currentOldPage !== null && currentOldPage !== prevOldPage) ||
            (currentNewPage !== null && currentNewPage !== prevNewPage);

          return (
            <div key={idx} className="bg-white">
              {showHeader && (
                <div className="bg-[#f6f8fa] px-4 py-2 border-y border-gray-200 text-xs font-mono text-gray-500 flex items-center mt-[-1px]">
                  <span>
                    {item.old_page !== null
                      ? `Page ${item.old_page + 1}`
                      : "..."}
                    {" → "}
                    {item.new_page !== null
                      ? `Page ${item.new_page + 1}`
                      : "..."}
                  </span>
                </div>
              )}

              <div className="text-sm">
                {/* Unified View */}
                {viewMode === "unified" && (
                  <div
                    className={`
                    ${
                      item.change === "added"
                        ? "bg-[#e6ffec]"
                        : item.change === "deleted"
                          ? "bg-[#ffebe9]"
                          : "bg-white"
                    } 
                    hover:bg-opacity-80 transition-colors
                  `}
                  >
                    <div className="flex">
                      {/* Line Numbers */}
                      <div className="flex w-16 shrink-0 border-r border-gray-100 bg-white/50 select-none">
                        <div className="w-8 text-right pr-2 py-2 text-xs text-gray-400 font-mono">
                          {item.old_index !== null ? item.old_index + 1 : ""}
                        </div>
                        <div className="w-8 text-right pr-2 py-2 text-xs text-gray-400 font-mono">
                          {item.new_index !== null ? item.new_index + 1 : ""}
                        </div>
                      </div>

                      {/* Content */}
                      <div className="grow p-2 break-words whitespace-pre-wrap leading-relaxed font-mono text-[13px]">
                        {item.block_type === "image" ? (
                          <span className="text-gray-500 italic">
                            [Image Comparison - Hash Diff (Not rendered)]
                          </span>
                        ) : item.word_diff ? (
                          item.word_diff.map((token, tIdx) => (
                            <span
                              key={tIdx}
                              className={`${
                                token.type === "insert"
                                  ? "bg-[#abf2bc] text-black decoration-0 mx-0.5 px-0.5 rounded-[2px]"
                                  : token.type === "delete"
                                    ? "bg-[#ffc0c0] text-black decoration-0 mx-0.5 px-0.5 rounded-[2px]"
                                    : ""
                              }`}
                            >
                              {token.value}{" "}
                            </span>
                          ))
                        ) : (
                          // Fallback
                          <span>
                            {item.change === "added"
                              ? reconstructText(item.word_diff, "new")
                              : reconstructText(item.word_diff, "old")}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Split View */}
                {viewMode === "split" && (
                  <div className="grid grid-cols-2 divide-x divide-gray-200">
                    {/* Left Side (Old) */}
                    <div
                      className={`${
                        item.change === "deleted" || item.change === "modified"
                          ? "bg-[#ffebe9]"
                          : "bg-white"
                      } flex`}
                    >
                      <div className="w-8 shrink-0 border-r border-gray-100/50 bg-gray-50/50 text-right pr-2 py-2 text-xs text-gray-400 font-mono select-none">
                        {item.old_index !== null ? item.old_index + 1 : ""}
                      </div>
                      <div className="grow p-2 break-words whitespace-pre-wrap font-mono text-[13px] text-gray-800">
                        {item.change === "added" ? (
                          <div className="bg-gray-50/50 h-full w-full"></div>
                        ) : (
                          reconstructText(item.word_diff, "old")
                        )}
                      </div>
                    </div>

                    {/* Right Side (New) */}
                    <div
                      className={`${
                        item.change === "added" || item.change === "modified"
                          ? "bg-[#e6ffec]"
                          : "bg-white"
                      } flex`}
                    >
                      <div className="w-8 shrink-0 border-r border-gray-100/50 bg-gray-50/50 text-right pr-2 py-2 text-xs text-gray-400 font-mono select-none">
                        {item.new_index !== null ? item.new_index + 1 : ""}
                      </div>
                      <div className="grow p-2 break-words whitespace-pre-wrap font-mono text-[13px] text-gray-800">
                        {item.change === "deleted" ? (
                          <div className="bg-gray-50/50 h-full w-full"></div>
                        ) : (
                          reconstructText(item.word_diff, "new")
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default DiffResult;

"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { listAnalyses } from "@/lib/api/analyses";

export default function HistoryPage() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");

  useEffect(() => {
    async function loadHistory() {
      try {
        setLoading(true);
        const data = await listAnalyses({ skip: 0, limit: 100 });
        const items = (data.items || []).map((item) => {
          // Generic crop label — capitalize whatever Model 2 returned
          const cropLabel = item.crop_check?.label
            ? item.crop_check.label.charAt(0).toUpperCase() + item.crop_check.label.slice(1)
            : "—";

          // Disease-supported unsupported crops show crop name, no disease result
          const isDiseaseUnsupported =
            item.prediction?.disease_supported === false;

          const resultTitle = isDiseaseUnsupported
            ? `${cropLabel} — disease classification not supported`
            : item.prediction?.display_name ||
              (item.status === "healthy"
                ? cropLabel
                  ? `${cropLabel} — Healthy`
                  : "Healthy result"
                : item.status === "not_leaf"
                ? "No leaf detected"
                : item.status === "unsupported_crop"
                ? `${cropLabel || "Crop"} detected`
                : "Analysis failed");

          // Backend returns 0–1; multiply by 100 for display. Guard null.
          const scorePct =
            item.prediction?.score != null
              ? Math.round(item.prediction.score * 100)
              : null;

          return {
            id: item.analysis_id,
            imageName: item.image?.filename || "specimen.jpg",
            crop: cropLabel,
            cropId: item.crop_check?.label || "other",
            result: resultTitle,
            resultStatus: item.status,
            isDiseaseUnsupported,
            confidenceScore: scorePct,
            qualityStatus:
              item.quality?.status === "passed"
                ? "Quality: Passed"
                : "Needs Attention",
            date: new Date(item.created_at).toLocaleDateString([], {
              month: "short",
              day: "numeric",
              year: "numeric",
            }),
            imageUrl: "/images/results/sample-healthy-leaf.jpg",
          };
        });
        setRecords(items);
      } catch (err) {
        console.warn("Failed to load analysis history:", err);
      } finally {
        setLoading(false);
      }
    }

    loadHistory();
  }, []);

  const filteredRecords = records.filter((record) => {
    if (filter === "healthy" && record.resultStatus !== "healthy") return false;
    if (filter === "disease" && record.resultStatus !== "disease_detected") return false;
    if (filter === "unsupported" && !record.isDiseaseUnsupported) return false;

    if (search.trim()) {
      const q = search.toLowerCase();
      const matchName = record.imageName.toLowerCase().includes(q);
      const matchResult = record.result.toLowerCase().includes(q);
      const matchCrop = record.crop.toLowerCase().includes(q);
      return matchName || matchResult || matchCrop;
    }

    return true;
  });

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="font-serif text-3xl font-bold text-stone-900">Analysis history</h1>
        <p className="mt-1 text-sm text-stone-600">
          Review previous leaf image analyses, quality scores, and classification outcomes.
        </p>
      </div>

      {/* Controls Bar */}
      <div className="bg-white border border-stone-200 rounded-lg p-4 space-y-4 sm:space-y-0 sm:flex sm:items-center sm:justify-between sm:gap-4">
        {/* Filters */}
        <div className="flex flex-wrap items-center gap-1.5">
          {[
            { id: "all", label: "All" },
            { id: "healthy", label: "Healthy" },
            { id: "disease", label: "Potential disease" },
            { id: "unsupported", label: "Disease unsupported" },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setFilter(item.id)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                filter === item.id
                  ? "bg-emerald-800 text-white"
                  : "bg-stone-100 text-stone-700 hover:bg-stone-200"
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>

        {/* Search Input */}
        <div className="relative min-w-[200px] sm:w-64">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search history..."
            className="w-full text-xs bg-stone-50 border border-stone-300 rounded-md px-3 py-2 text-stone-900 placeholder-stone-400 focus:outline-none focus:ring-2 focus:ring-emerald-700"
          />
        </div>
      </div>

      {/* History List */}
      <div className="bg-white border border-stone-200 rounded-lg overflow-hidden shadow-xs">
        {loading ? (
          <div className="p-12 text-center text-stone-500 text-xs flex flex-col items-center justify-center space-y-2">
            <div className="w-5 h-5 border-2 border-emerald-800 border-t-transparent rounded-full animate-spin" />
            <span>Loading history from backend...</span>
          </div>
        ) : filteredRecords.length === 0 ? (
          <div className="p-8 text-center text-stone-500 text-sm">
            No analysis records match your selected filter or search query.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-stone-50 border-b border-stone-200 text-stone-600 font-semibold">
                  <th className="py-3 px-4">Specimen</th>
                  <th className="py-3 px-4">Crop</th>
                  <th className="py-3 px-4">Result</th>
                  <th className="py-3 px-4">Model Confidence</th>
                  <th className="py-3 px-4">Date</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-200">
                {filteredRecords.map((rec) => (
                  <tr key={rec.id} className="hover:bg-stone-50/80 transition-colors">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded bg-stone-200 overflow-hidden relative shrink-0 border border-stone-300">
                          <Image
                            src={rec.imageUrl}
                            alt={rec.imageName}
                            fill
                            className="object-cover"
                          />
                        </div>
                        <div>
                          <p className="font-medium text-stone-900 truncate max-w-[160px]">
                            {rec.imageName}
                          </p>
                          <p className="text-[11px] text-stone-500">{rec.qualityStatus}</p>
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-4 font-medium text-stone-800">{rec.crop}</td>
                    <td className="py-3 px-4">
                      <span
                        className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium border ${
                          rec.resultStatus === "healthy"
                            ? "bg-emerald-50 text-emerald-800 border-emerald-200"
                            : rec.resultStatus === "disease_detected"
                            ? "bg-amber-50 text-amber-900 border-amber-200"
                            : "bg-stone-100 text-stone-700 border-stone-200"
                        }`}
                      >
                        {rec.result}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-stone-700">
                      {rec.confidenceScore !== null ? (
                        <span className="font-mono">{rec.confidenceScore}%</span>
                      ) : (
                        <span className="text-stone-400">—</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-stone-500 whitespace-nowrap">{rec.date}</td>
                    <td className="py-3 px-4 text-right">
                      <Link
                        href={`/app/results/${rec.id}`}
                        className="inline-flex items-center text-xs font-medium text-emerald-800 hover:text-emerald-950 underline underline-offset-2"
                      >
                        View result
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

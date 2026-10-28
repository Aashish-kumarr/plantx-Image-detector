"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import ResultWorkspace from "@/components/results/ResultWorkspace";
import { getAnalysis } from "@/lib/api/analyses";

/**
 * ResultPageClient Component
 *
 * Resolves route `id` and `?state=` query parameters.
 * If `id` is a real analysis UUID, fetches the real record from GET /api/v1/analyses/{id}.
 * If `id` or `?state=` matches a canonical demo fixture, renders the demo fixture for visual QA.
 */
export default function ResultPageClient({ id, initialDemoState }) {
  const searchParams = useSearchParams();
  const queryState = searchParams.get("state") || initialDemoState;

  const validDemoStates = [
    "healthy",
    "disease_detected",
    "not_leaf",
    "unsupported_crop",
    "analysis_failed",
    "demo",
  ];

  const isDemo = validDemoStates.includes(id) || Boolean(queryState);

  const [realResult, setRealResult] = useState(null);
  const [loading, setLoading] = useState(!isDemo);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (isDemo || !id) return;

    let isMounted = true;
    async function loadRealAnalysis() {
      try {
        setLoading(true);
        setError(null);
        const data = await getAnalysis(id);

        if (!isMounted) return;

        // Map backend AnalysisResponse to ResultWorkspace structure
        const formatted = {
          id: data.analysis_id,
          status: data.status,
          badgeText:
            data.status === "healthy"
              ? "No supported disease pattern identified"
              : data.status === "disease_detected"
              ? "Analysis complete"
              : data.status === "not_leaf"
              ? "Leaf not detected"
              : data.status === "unsupported_crop"
              ? "Crop not supported"
              : "Analysis failed",
          title:
            data.status === "healthy"
              ? (data.prediction?.crop ? (data.prediction.crop.charAt(0).toUpperCase() + data.prediction.crop.slice(1) + " Healthy") : "Healthy result")
              : data.prediction?.display_name ||
              (data.status === "not_leaf"
              ? "No leaf pattern detected"
              : data.status === "unsupported_crop"
              ? "Plant leaf detected"
              : "Analysis failed"),
          subheading:
            data.status === "healthy"
              ? "Among the conditions supported by the current model, this image most closely matched the healthy leaf class."
              : data.status === "disease_detected"
              ? `Observed visual patterns in leaf specimen closely align with symptomatic features of ${data.prediction?.display_name || "disease"}.`
              : data.status === "not_leaf"
              ? "No plant leaf was detected in this image. Further classification was halted to avoid false diagnostic readings."
              : data.status === "unsupported_crop"
              ? "Crop detected successfully. Disease classification is not currently supported for this crop."
              : "PlantDx encountered an unexpected problem while processing the analysis.",
          leafCheck: data.leaf_check ? { ...data.leaf_check, score: data.leaf_check.score != null ? data.leaf_check.score * 100 : null } : null,
          cropCheck: data.crop_check ? { ...data.crop_check, score: data.crop_check.score != null ? data.crop_check.score * 100 : null } : null,
          prediction: data.prediction ? { ...data.prediction, score: data.prediction.score != null ? data.prediction.score * 100 : null } : null,
          confidenceNote:
            "Model confidence reflects activation scores among supported classes, not Bayesean probabilities or laboratory certainty.",
          image: {
            name: data.image?.filename || "uploaded-leaf.jpg",
            previewUrl: `/images/results/sample-healthy-leaf.jpg`, // Preview fallback
            typeFormatted: (data.image?.mime_type || "image/jpeg").split("/")[1]?.toUpperCase() || "JPG",
            sizeFormatted: `${((data.image?.size_bytes || 0) / (1024 * 1024)).toFixed(1)} MB`,
            analyzedAt: new Date(data.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
            imageTag: `Crop: ${data.crop_check?.label || "Unknown"}`,
          },
          qualityCheckText: data.quality?.status === "passed" ? "Passed (4/4)" : "Needs Attention",
          explanation: {
            title: "Why this result?",
            text: `PlantDx completed hierarchical pipeline analysis for ${data.image?.filename || "this specimen"}.`,
          },
        };

        setRealResult(formatted);
      } catch (err) {
        if (!isMounted) return;
        console.error("Failed to load real analysis:", err);
        setError(err.message || "Could not retrieve analysis result.");
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    loadRealAnalysis();
    return () => {
      isMounted = false;
    };
  }, [id, isDemo]);

  if (!isDemo) {
    if (loading) {
      return (
        <div className="min-h-[400px] flex flex-col items-center justify-center space-y-3">
          <div className="w-6 h-6 border-2 border-emerald-800 border-t-transparent rounded-full animate-spin" />
          <p className="text-xs text-stone-500 font-medium">Loading analysis result...</p>
        </div>
      );
    }

    if (error || !realResult) {
      return (
        <div className="p-6 bg-red-50 border border-red-200 rounded-lg space-y-2 text-red-800 max-w-2xl mx-auto my-8">
          <h2 className="font-semibold text-sm">Analysis Not Found</h2>
          <p className="text-xs">{error || "The requested analysis record was not found or has been deleted."}</p>
        </div>
      );
    }

    return <ResultWorkspace resultData={realResult} />;
  }

  let demoState = "healthy";
  if (queryState && validDemoStates.includes(queryState)) {
    demoState = queryState;
  } else if (id && validDemoStates.includes(id)) {
    demoState = id;
  }

  return <ResultWorkspace demoState={demoState} />;
}

"use client";

import { useState, useEffect, useCallback, useTransition } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Icon from "@/components/ui/Icon";
import AnalyzeProgress from "./AnalyzeProgress";
import PhotoGuidance from "./PhotoGuidance";
import UploadState from "./UploadState";
import ImageSelectedState from "./ImageSelectedState";
import QualityPassedState from "./QualityPassedState";
import QualityNeedsImprovementState from "./QualityNeedsImprovementState";
import AnalyzingState from "./AnalyzingState";
import { mockAnalyzeData } from "@/lib/mock/analyze";
import { uploadImage, checkQuality } from "@/lib/api/images";
import { createAnalysis } from "@/lib/api/analyses";

export default function AnalyzeWorkspace() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [, startTransition] = useTransition();

  // Workflow states: upload | selected | checking_quality | quality_passed | quality_needs_improvement | analyzing
  const [currentState, setCurrentState] = useState("upload");
  const [selectedImage, setSelectedImage] = useState(null);
  const [activeObjectUrl, setActiveObjectUrl] = useState(null);

  // Real API integration state
  const [uploadedImageId, setUploadedImageId] = useState(null);
  const [qualityResult, setQualityResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [loading, setLoading] = useState(false);

  // Handle demo query parameter for visual QA
  useEffect(() => {
    const demoParam = searchParams.get("demo");

    startTransition(() => {
      if (demoParam === "upload") {
        setCurrentState("upload");
        setSelectedImage(null);
      } else if (demoParam === "selected") {
        setCurrentState("selected");
        setSelectedImage(mockAnalyzeData.defaultSampleImage);
      } else if (demoParam === "quality-passed") {
        setCurrentState("quality_passed");
        setSelectedImage(mockAnalyzeData.defaultSampleImage);
      } else if (demoParam === "quality-needs-improvement") {
        setCurrentState("quality_needs_improvement");
        setSelectedImage(mockAnalyzeData.defaultSampleImage);
      } else if (demoParam === "analyzing") {
        setCurrentState("analyzing");
        setSelectedImage(mockAnalyzeData.defaultSampleImage);
      }
    });
  }, [searchParams]);

  // Clean up ObjectURL when component unmounts
  useEffect(() => {
    return () => {
      if (activeObjectUrl) {
        URL.revokeObjectURL(activeObjectUrl);
      }
    };
  }, [activeObjectUrl]);

  // Helper to format bytes
  const formatBytes = (bytes) => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  // Helper to get extension
  const getFormat = (filename, mimeType) => {
    if (mimeType === "image/jpeg") return "JPG";
    if (mimeType === "image/png") return "PNG";
    if (mimeType === "image/webp") return "WEBP";
    const ext = filename?.split(".").pop()?.toUpperCase();
    return ext || "JPG";
  };

  // User file selection handler
  const handleFileSelected = useCallback((file) => {
    setErrorMsg(null);
    if (activeObjectUrl) {
      URL.revokeObjectURL(activeObjectUrl);
    }

    const objectUrl = URL.createObjectURL(file);
    setActiveObjectUrl(objectUrl);

    const imageInfo = {
      name: file.name,
      sizeFormatted: formatBytes(file.size),
      typeFormatted: getFormat(file.name, file.type),
      sizeBytes: file.size,
      previewUrl: objectUrl,
      file,
    };

    setSelectedImage(imageInfo);
    setUploadedImageId(null);
    setQualityResult(null);
    setCurrentState("selected");
  }, [activeObjectUrl]);

  // Select sample image handler
  const handleSelectSample = useCallback(async () => {
    setErrorMsg(null);
    if (activeObjectUrl) {
      URL.revokeObjectURL(activeObjectUrl);
      setActiveObjectUrl(null);
    }

    try {
      // Fetch the sample image and turn into a real File object for uploading
      const res = await fetch(mockAnalyzeData.defaultSampleImage.previewUrl);
      const blob = await res.blob();
      const sampleFile = new File([blob], "sample-leaf.jpg", { type: "image/jpeg" });
      handleFileSelected(sampleFile);
    } catch (err) {
      // Fallback
      setSelectedImage(mockAnalyzeData.defaultSampleImage);
      setCurrentState("selected");
    }
  }, [activeObjectUrl, handleFileSelected]);

  // Remove current image
  const handleRemoveImage = useCallback(() => {
    if (activeObjectUrl) {
      URL.revokeObjectURL(activeObjectUrl);
      setActiveObjectUrl(null);
    }
    setSelectedImage(null);
    setUploadedImageId(null);
    setQualityResult(null);
    setErrorMsg(null);
    setCurrentState("upload");
  }, [activeObjectUrl]);

  // Trigger quality check (POST /api/v1/images -> POST /api/v1/images/{id}/quality)
  const handleCheckQuality = useCallback(async () => {
    setErrorMsg(null);
    if (!selectedImage?.file) {
      // If demo mode or no File object, use fixture transition
      setCurrentState("quality_passed");
      return;
    }

    setLoading(true);
    try {
      // Step 1: Upload image to backend
      let imgId = uploadedImageId;
      if (!imgId) {
        const uploadRes = await uploadImage(selectedImage.file);
        imgId = uploadRes.image_id;
        setUploadedImageId(imgId);
      }

      // Step 2: Run product quality check
      const qualityRes = await checkQuality(imgId);
      setQualityResult(qualityRes);

      if (qualityRes.status === "passed") {
        setCurrentState("quality_passed");
      } else {
        setCurrentState("quality_needs_improvement");
      }
    } catch (err) {
      console.error("Quality check error:", err);
      setErrorMsg(err.message || "Failed to process image quality. Please try again.");
      setCurrentState("selected");
    } finally {
      setLoading(false);
    }
  }, [selectedImage, uploadedImageId]);

  // Start analysis (POST /api/v1/analyses -> redirect /app/results/{analysis_id})
  const handleStartAnalysis = useCallback(async () => {
    setErrorMsg(null);
    if (!uploadedImageId) {
      // Demo fallback if no uploaded image ID
      setCurrentState("analyzing");
      return;
    }

    setCurrentState("analyzing");
    try {
      const analysisRes = await createAnalysis(uploadedImageId);
      // Redirect to real result page
      if (analysisRes && analysisRes.analysis_id) {
        router.push(`/app/results/${analysisRes.analysis_id}`);
      }
    } catch (err) {
      console.error("Analysis error:", err);
      setErrorMsg(err.message || "Failed to run analysis. Please try again.");
      setCurrentState("quality_passed");
    }
  }, [uploadedImageId, router]);

  // Cancel analysis
  const handleCancelAnalysis = useCallback(() => {
    setCurrentState("quality_passed");
  }, []);

  return (
    <div className="flex flex-col gap-6 w-full">
      {/* Header & Workflow Progress Container */}
      <section className="flex flex-col gap-4 pb-4 border-b border-surface-variant">
        {/* Dynamic Context Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div className="flex flex-col gap-1.5">
            {currentState === "upload" || currentState === "selected" ? (
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-sm bg-surface-container-high text-on-surface-variant font-mono text-[11px] font-medium border border-surface-variant/40">
                  <span className="w-1.5 h-1.5 rounded-full bg-surface-tint" />
                  Supported: Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, Tomato
                </span>
              </div>
            ) : currentState === "quality_passed" ? (
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-sm bg-primary/10 text-primary font-mono text-[11px] font-semibold border border-primary/20">
                  <span className="w-1.5 h-1.5 rounded-full bg-surface-tint" />
                  Ready for analysis
                </span>
              </div>
            ) : currentState === "quality_needs_improvement" ? (
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-sm bg-[#F5EEDC] text-[#6E4E10] border border-[#D8C7A0] font-mono text-[11px] font-semibold">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#8C6418]" />
                  Needs improvement
                </span>
              </div>
            ) : (
              // analyzing
              <div className="flex items-center gap-2 font-mono text-[11px] text-outline uppercase tracking-wider">
                <span className="w-1.5 h-1.5 rounded-full bg-surface-tint" />
                <span>PlantDx Analysis</span>
              </div>
            )}

            <h1 className="text-2xl sm:text-3xl font-semibold text-primary font-display tracking-tight">
              {currentState === "upload" || currentState === "selected"
                ? "Analyze a leaf"
                : currentState === "quality_passed" || currentState === "quality_needs_improvement"
                ? "Image quality check"
                : "Analyzing your leaf"}
            </h1>

            <p className="text-xs sm:text-sm text-on-surface-variant">
              {currentState === "upload"
                ? "Upload a clear photo of a crop leaf to begin quality check and disease analysis."
                : currentState === "selected"
                ? "Review your image before continuing."
                : currentState === "quality_passed"
                ? "Your photo is suitable for analysis."
                : currentState === "quality_needs_improvement"
                ? "This photo may not provide enough detail for a reliable analysis."
                : "PlantDx is inspecting leaf patterns against supported disease models."}
            </p>
          </div>
        </div>

        {/* 4-Step Progress Indicator */}
        <AnalyzeProgress currentState={currentState} />
      </section>

      {/* Global Error Banner */}
      {errorMsg && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3 text-xs text-red-800">
          <Icon name="error" className="w-4 h-4 text-red-600 mt-0.5 shrink-0" />
          <div className="flex flex-col gap-0.5">
            <span className="font-semibold">Request Failed</span>
            <span>{errorMsg}</span>
          </div>
        </div>
      )}

      {/* Main Workspace Layout */}
      {currentState === "analyzing" ? (
        <AnalyzingState
          image={selectedImage || mockAnalyzeData.defaultSampleImage}
          stages={mockAnalyzeData.analyzingStages}
          onCancel={handleCancelAnalysis}
        />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Primary Column (Left 7 cols) */}
          <div className="lg:col-span-7 flex flex-col gap-4">
            {currentState === "upload" && (
              <UploadState
                onFileSelected={handleFileSelected}
                onSelectSample={handleSelectSample}
              />
            )}

            {currentState === "selected" && (
              <ImageSelectedState
                image={selectedImage || mockAnalyzeData.defaultSampleImage}
                loading={loading}
                onReplaceImage={handleFileSelected}
                onRemoveImage={handleRemoveImage}
                onCheckQuality={handleCheckQuality}
              />
            )}

            {currentState === "quality_passed" && (
              <QualityPassedState
                image={selectedImage || mockAnalyzeData.defaultSampleImage}
                qualityData={qualityResult || mockAnalyzeData.qualityPassed}
                onReplaceImage={() => setCurrentState("selected")}
                onStartAnalysis={handleStartAnalysis}
              />
            )}

            {currentState === "quality_needs_improvement" && (
              <QualityNeedsImprovementState
                image={selectedImage || mockAnalyzeData.defaultSampleImage}
                qualityData={qualityResult || mockAnalyzeData.qualityNeedsImprovement}
                onReplaceImage={handleFileSelected}
              />
            )}
          </div>

          {/* Secondary Column: Field Guidance / Why Quality Matters (Right 5 cols) */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            {currentState === "quality_passed" ? (
              <aside className="flex flex-col gap-4" aria-label="Quality protocol notes">
                {/* Informational Panel: Why Quality Matters */}
                <div className="bg-surface border border-surface-variant rounded-xl p-4 sm:p-5 flex flex-col gap-3 shadow-sm">
                  <div className="flex items-center justify-between border-b border-surface-variant pb-2.5">
                    <h3 className="text-sm sm:text-base font-semibold text-primary font-display">
                      Why image quality matters
                    </h3>
                    <span className="font-mono text-[10px] sm:text-[11px] uppercase text-outline">
                      Overview
                    </span>
                  </div>
                  <p className="text-xs sm:text-sm text-on-surface-variant leading-relaxed">
                    Clear, well-lit images help preserve leaf details used during analysis.
                  </p>
                  <div className="flex flex-col gap-1 pt-2 border-t border-surface-variant">
                    <span className="font-mono text-[10px] sm:text-[11px] uppercase tracking-wider text-outline font-semibold">
                      Next: Leaf analysis
                    </span>
                    <p className="text-xs text-on-surface-variant leading-relaxed">
                      PlantDx evaluates visible characteristics across leaf surface, texture, and coloration.
                    </p>
                  </div>
                </div>

                {/* Image Privacy Note */}
                <div className="bg-surface-container-low border border-surface-variant rounded-lg p-3.5 flex items-start gap-2.5">
                  <Icon name="lock" className="w-4 h-4 text-outline mt-0.5 shrink-0" />
                  <div className="flex flex-col">
                    <span className="text-xs sm:text-sm font-semibold text-on-surface">
                      About your image
                    </span>
                    <span className="text-[11px] sm:text-xs text-outline leading-relaxed mt-0.5">
                      Your photograph is processed through the PlantDx quality gate and hierarchical diagnostic pipeline.
                    </span>
                  </div>
                </div>
              </aside>
            ) : (
              <PhotoGuidance showPrivacy={false} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

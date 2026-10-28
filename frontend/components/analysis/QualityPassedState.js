/* eslint-disable @next/next/no-img-element */
'use client';

import Icon from '@/components/ui/Icon';

/**
 * Quality Passed State View
 *
 * Displays:
 * 1. Image review with optional user crop context
 * 2. 4 verified quality checks (Resolution, Sharpness, Lighting, Contrast)
 * 3. Leaf visibility protocol note
 * 4. Primary action: "Analyze leaf"
 * 5. Secondary action: "Use another image"
 */
export default function QualityPassedState({
  image,
  qualityData,
  onReplaceImage,
  onStartAnalysis,
}) {
  return (
    <div className="flex flex-col gap-4">
      {/* Main Review & Quality Assessment Card */}
      <div className="bg-surface border border-surface-variant rounded-xl overflow-hidden shadow-sm flex flex-col">
        {/* File Info Bar */}
        <div className="px-4 py-2.5 bg-surface-container-low border-b border-surface-variant flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <Icon name="image" className="w-4 h-4 text-outline" />
            <span className="text-xs sm:text-sm font-semibold text-on-surface truncate max-w-[180px] sm:max-w-xs">
              {image?.name || 'tomato-leaf.jpg'}
            </span>
            <span className="text-[10px] sm:text-[11px] font-mono uppercase text-outline">
              {image?.typeFormatted || 'JPG'} • {image?.sizeFormatted || '2.4 MB'}
            </span>
          </div>

          <button
            type="button"
            onClick={onReplaceImage}
            className="inline-flex items-center gap-1 text-on-surface-variant hover:text-on-surface text-xs font-medium py-1 px-1.5 rounded transition-colors cursor-pointer"
          >
            <Icon name="sync" className="w-3.5 h-3.5" />
            <span>Replace image</span>
          </button>
        </div>

        {/* Botanical Image Preview */}
        <div className="relative bg-surface-container w-full h-64 sm:h-72 overflow-hidden flex items-center justify-center border-b border-surface-variant">
          <img
            src={image?.previewUrl}
            alt={image?.name || 'Selected leaf preview'}
            className="w-full h-full object-cover"
          />
        </div>

        {/* Quality Assessment Panel */}
        <div className="p-4 sm:p-6 flex flex-col gap-4 bg-surface">
          <div className="flex items-center justify-between border-b border-surface-variant pb-3">
            <div className="flex items-center gap-2">
              <h3 className="text-sm sm:text-base font-semibold text-primary font-display">
                Image quality
              </h3>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-sm bg-surface-container-high text-primary font-mono text-[10px] sm:text-[11px] font-semibold">
                <Icon name="verified" className="w-3.5 h-3.5 text-surface-tint" />
                {qualityData?.summaryBadge || '4 checks passed'}
              </span>
            </div>
          </div>

          {/* 4 Verification Checks Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 sm:gap-3">
            {Array.isArray(qualityData?.checks) ? qualityData.checks.map((check) => (
              <div
                className="p-3 rounded-lg border border-surface-variant bg-surface-container-low flex flex-col gap-1"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs sm:text-sm font-semibold text-on-surface">
                  </span>
                  <span className="inline-flex items-center gap-1 font-mono text-[10px] sm:text-[11px] text-surface-tint font-semibold">
                    <Icon name="check_circle" className="w-3.5 h-3.5" />
                  </span>
                </div>
                <p className="text-[11px] sm:text-xs text-on-surface-variant leading-relaxed">
                </p>
              </div>
            )) : qualityData?.checks ? Object.entries(qualityData.checks).map(([key, value]) => (
              <div
                key={key}
                className="p-3 rounded-lg border border-surface-variant bg-surface-container-low flex flex-col gap-1"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs sm:text-sm font-semibold text-on-surface capitalize">
                    {key}
                  </span>
                  <span className="inline-flex items-center gap-1 font-mono text-[10px] sm:text-[11px] text-surface-tint font-semibold">
                    <Icon name="check_circle" className="w-3.5 h-3.5" />
                    {value.status.toUpperCase()}
                  </span>
                </div>
                <p className="text-[11px] sm:text-xs text-on-surface-variant leading-relaxed">
                  {key === 'resolution' && 'Image dimensions meet minimum requirements.'}
                  {key === 'sharpness' && 'Image is in focus and textures are clear.'}
                  {key === 'lighting' && 'Illumination is balanced without harsh glare.'}
                  {key === 'contrast' && 'Colors and features are distinctly separated.'}
                </p>
              </div>
            )) : null}

          </div>

          {/* Leaf Visibility Notice */}
          <div className="pt-2 border-t border-surface-variant">
            <div className="p-3 rounded bg-surface-container-high flex items-start gap-2.5">
              <Icon name="visibility" className="w-4 h-4 text-outline mt-0.5 shrink-0" />
              <div className="flex flex-col">
                <span className="text-xs font-semibold text-on-surface">
                  Leaf visibility
                </span>
                <span className="text-[11px] text-outline leading-relaxed mt-0.5">
                  Visually review that the leaf is clearly shown. Automated checks evaluate clarity and balance, not botanical specimen placement.
                </span>
              </div>
            </div>
          </div>

          {/* Primary Action Callout */}
          <div className="flex flex-col gap-2 pt-3 border-t border-surface-variant">
            <button
              type="button"
              onClick={onStartAnalysis}
              className="w-full min-h-[44px] px-6 py-2.5 bg-primary hover:bg-primary-container active:bg-primary-container text-on-primary rounded font-medium text-sm flex items-center justify-center gap-2 transition-colors shadow-sm cursor-pointer"
            >
              <Icon name="biotech" className="w-4 h-4" />
              <span>Analyze leaf</span>
            </button>

            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pt-1 text-center sm:text-left">
              <p className="text-[11px] text-outline">
                PlantDx will analyze the leaf image for signs associated with supported plant conditions.
              </p>
              <button
                type="button"
                onClick={onReplaceImage}
                className="text-outline hover:text-on-surface text-xs font-medium transition-colors shrink-0 cursor-pointer self-center sm:self-auto"
              >
                Use another image
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

'use client';

import Icon from '@/components/ui/Icon';

/**
 * UnsupportedCropResult Component
 *
 * Early-stop result component when Model 1 passes (leaf) but Model 2 identifies crop as `other`.
 *
 * IMPORTANT CONTRACT:
 * Disease models did NOT run.
 * Does NOT guess or display an unsupported species name.
 * Does NOT display disease prediction or confidence scores.
 */
export default function UnsupportedCropResult({ result, renderImage, renderActions, renderLimitations }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      {/* Primary Column (Left 7 cols) */}
      <div className="lg:col-span-7 flex flex-col gap-6">
        {/* Image Assessment Card */}
        <div className="bg-surface border border-surface-variant rounded-xl overflow-hidden shadow-sm flex flex-col p-5 sm:p-6 gap-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <span className="font-mono text-[10px] sm:text-[11px] uppercase text-outline font-semibold">
              Image details
            </span>
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-sm bg-[#F5EEDC] text-[#6E4E10] border border-[#D8C7A0] font-mono text-[11px] font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-[#8C6418]" />
              Crop not supported
            </span>
          </div>

          <div className="flex flex-col gap-1.5 border-b border-surface-variant pb-4">
            <h2 className="text-xl sm:text-2xl font-semibold text-primary font-display">
              {result?.title || 'Plant leaf detected'}
            </h2>
            <p className="text-xs sm:text-sm text-on-surface-variant leading-relaxed">
              {result?.subheading || 'Crop detected successfully. Disease classification is not currently supported for this crop.'}
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-surface-container-low border border-surface-variant flex flex-col gap-1.5">
            <span className="text-xs font-semibold text-on-surface">
              Why didn&apos;t disease analysis run?
            </span>
            <p className="text-xs text-on-surface-variant leading-relaxed">
              {result?.reasonText || 'PlantDx detected your leaf and identified the crop, but disease analysis models are only available for specific crops. Disease classification for this crop is not yet included in the current model set.'}
            </p>
          </div>

          {/* Currently Supported Crops Box */}
          <div className="p-3.5 rounded-lg border border-surface-variant bg-surface flex flex-col gap-3">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-1.5 text-on-surface text-xs font-semibold">
                <Icon name="verified" className="w-4 h-4 text-primary" />
                <span>Crops with disease analysis</span>
              </div>
              <span className="font-mono text-[10px] uppercase text-outline">
                9 crop models
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {['Apple', 'Cherry', 'Corn', 'Grape', 'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato'].map((crop) => (
                <div key={crop} className="p-2 rounded bg-surface-container-low border border-surface-variant flex items-center gap-2">
                  <Icon name="eco" className="w-3.5 h-3.5 text-surface-tint shrink-0" />
                  <span className="text-xs font-medium text-on-surface">{crop}</span>
                </div>
              ))}
            </div>

            <p className="text-[11px] sm:text-xs text-outline">
              PlantDx currently provides disease analysis for the 9 crops listed above.
            </p>
          </div>
        </div>

        {/* Uploaded Leaf Image Card */}
        {renderImage}

        {/* Guidance section for real-image submissions */}
        <div className="bg-surface border border-surface-variant rounded-xl p-5 sm:p-6 flex flex-col gap-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-surface-variant pb-3">
            <h3 className="text-sm sm:text-base font-semibold text-primary font-display">
              Getting better results
            </h3>
            <span className="font-mono text-[10px] sm:text-[11px] uppercase text-outline">
              Guidance
            </span>
          </div>

          <p className="text-xs sm:text-sm text-on-surface-variant leading-relaxed">
            Follow these recommendations to help PlantDx match your leaf to the correct crop model:
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {result?.guidanceTips?.map((tip, idx) => (
              <div
                key={idx}
                className="p-3 rounded bg-surface-container-low border border-surface-variant flex items-start gap-2.5"
              >
                <Icon name={tip.icon || 'center_focus_strong'} className="w-4 h-4 text-surface-tint mt-0.5 shrink-0" />
                <div className="flex flex-col gap-0.5">
                  <span className="text-xs font-semibold text-on-surface">
                    {tip.title}
                  </span>
                  <span className="text-[11px] text-outline leading-relaxed">
                    {tip.desc}
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div className="p-3 rounded bg-surface-container-high text-on-surface-variant text-xs flex items-center gap-2">
            <Icon name="rotate_right" className="w-4 h-4 text-outline shrink-0" />
            <span>
              <strong className="font-semibold text-on-surface">Try another angle:</strong> Photographing from a different angle can reveal distinguishing leaf characteristics.
            </span>
          </div>
        </div>
      </div>

      {/* Secondary Column (Right 5 cols) */}
      <div className="lg:col-span-5 flex flex-col gap-4">
        {/* Actions Module */}
        {renderActions}

        {/* Limitations Module */}
        {renderLimitations}

        {/* Early-Stop Analysis Details Card */}
        <div className="bg-surface border border-surface-variant rounded-xl p-4 sm:p-5 flex flex-col gap-3 shadow-sm">
          <div className="flex items-center justify-between border-b border-surface-variant pb-2.5">
            <h3 className="text-sm sm:text-base font-semibold text-primary font-display">
              Analysis details
            </h3>
            <span className="font-mono text-[10px] sm:text-[11px] uppercase text-outline">
              Record
            </span>
          </div>

          <div className="flex flex-col gap-2 text-xs sm:text-sm">
            <div className="flex items-center justify-between py-1 border-b border-surface-variant">
              <span className="text-outline">Analyzed</span>
              <span className="text-on-surface font-medium">{result?.image?.analyzedAt}</span>
            </div>
            <div className="flex items-center justify-between py-1 border-b border-surface-variant">
              <span className="text-outline">Image file</span>
              <span className="text-on-surface font-medium truncate max-w-[150px]">{result?.image?.name}</span>
            </div>
            <div className="flex items-center justify-between py-1 border-b border-surface-variant">
              <span className="text-outline">Quality check</span>
              <span className="inline-flex items-center gap-1 text-surface-tint font-medium">
                <Icon name="check_circle" className="w-3.5 h-3.5 text-surface-tint" />
                {result?.qualityCheckText || 'Passed (4/4)'}
              </span>
            </div>
            <div className="flex items-center justify-between py-1 border-b border-surface-variant">
              <span className="text-outline">Leaf check</span>
              <span className="inline-flex items-center gap-1 text-surface-tint font-medium">
                <Icon name="check_circle" className="w-3.5 h-3.5 text-surface-tint" />
                Matched
              </span>
            </div>
            <div className="flex items-center justify-between py-1 border-b border-surface-variant">
              <span className="text-outline">Crop check</span>
              <span className="inline-flex items-center gap-1 text-[#8E4E08] font-semibold">
                <Icon name="warning" className="w-3.5 h-3.5 text-[#8E4E08]" />
                Not supported
              </span>
            </div>
            <div className="flex items-center justify-between py-1">
              <span className="text-outline">Analysis pipeline</span>
              <span className="text-on-surface-variant font-medium">Unsupported crop</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

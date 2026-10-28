function normalizeChecks(checks) {
  if (Array.isArray(checks)) return checks;
  if (!checks) return [];
  const mapDesc = {
    resolution: "Image dimensions meet minimum requirements.",
    sharpness: "Image is in focus and textures are clear.",
    lighting: "Illumination is balanced without harsh glare.",
    contrast: "Colors and features are distinctly separated."
  };
  return Object.entries(checks).map(([key, value]) => ({
    id: key,
    label: key.charAt(0).toUpperCase() + key.slice(1),
    statusText: value.status === 'good' ? 'GOOD' : 'POOR',
    description: mapDesc[key] || "Verified check."
  }));
}

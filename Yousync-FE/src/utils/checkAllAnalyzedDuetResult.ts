export function checkAllAnalyzedDuetResult(
    captions: { script: string }[],
    latestResultByScript: Record<string, any>,
    normalizeScript: (script: string) => string
  ): boolean {
    const analyzedCount = captions.filter(caption => {
      const scriptKey = normalizeScript(caption.script);
      return !!latestResultByScript[scriptKey];
    }).length;
    return analyzedCount === captions.length;
  }
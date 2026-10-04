import { describe, expect, it } from 'vitest';
import { analyzeWaveform } from './ecgAnalysis';

describe('analyzeWaveform', () => {
  it('returns a warning-level suggestion for PVC morphology', () => {
    const result = analyzeWaveform('pvc', 82);

    expect(result.severity).toBe('warning');
    expect(result.confidence).toBeGreaterThan(80);
    expect(result.cues.some((cue) => cue.includes('幅広'))).toBe(true);
  });

  it('returns a normal status for sinus rhythm', () => {
    const result = analyzeWaveform('normal_sinus', 72);

    expect(result.severity).toBe('normal');
    expect(result.cues.some((cue) => cue.includes('P波'))).toBe(true);
  });
});

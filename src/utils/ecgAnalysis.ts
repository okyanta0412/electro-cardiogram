export type ECGWaveformType = 'pvc' | 'af' | 'st_change' | 'normal_sinus' | 'noise' | 'other';

export type ECGSeverity = 'critical' | 'warning' | 'normal' | 'artifact';

export type ECGAnalysisResult = {
  label: string;
  severity: ECGSeverity;
  confidence: number;
  cues: string[];
  summary: string;
  recommendation: string;
};

export function analyzeWaveform(
  waveformType: ECGWaveformType,
  heartRate: number,
): ECGAnalysisResult {
  switch (waveformType) {
    case 'normal_sinus':
      return {
        label: '正常洞調律',
        severity: 'normal',
        confidence: 94,
        cues: [
          'P波先行あり',
          'RR間隔が規則的',
          heartRate > 0 ? `推定HR ${heartRate}bpm` : '心拍数未計測',
        ],
        summary: '規則正しい洞調律で、基準内のリズムと判断されます。',
        recommendation: '経過観察を継続し、通常のモニタリングを行ってください。',
      };
    case 'pvc':
      return {
        label: '心室性期外収縮（PVC）',
        severity: 'warning',
        confidence: 88,
        cues: [
          '幅広・異型QRSが出現',
          '早期に発生する不整像',
          heartRate > 0 ? `推定HR ${heartRate}bpm` : '心拍数未計測',
        ],
        summary: 'PVCの典型的なパターンが見られ、単発または連発の可能性があります。',
        recommendation: '看護師リーダーへ共有し、発生頻度と患者状態を再確認してください。',
      };
    case 'af':
      return {
        label: '心房細動（AF）',
        severity: 'warning',
        confidence: 90,
        cues: [
          'RR間隔が不規則',
          'P波が消失または不明瞭',
          heartRate > 0 ? `推定HR ${heartRate}bpm` : '心拍数未計測',
        ],
        summary: '不整脈と考えられるため、継続観察と医療者への共有が重要です。',
        recommendation: '当直医または担当医への報告を検討し、変化の有無を確認してください。',
      };
    case 'st_change':
      return {
        label: 'ST変化の可能性',
        severity: 'critical',
        confidence: 82,
        cues: [
          'ST部分の変化が疑われる',
          '虚血または心筋障害を示唆',
          heartRate > 0 ? `推定HR ${heartRate}bpm` : '心拍数未計測',
        ],
        summary: 'ST変化は重篤な虚血や梗塞のサインとして扱う必要があります。',
        recommendation: '直ちに医師へ連絡し、臨床所見と併せて最終判断してください。',
      };
    case 'noise':
      return {
        label: '体動ノイズ / アーティファクト',
        severity: 'artifact',
        confidence: 92,
        cues: [
          '基線が大きく揺れている',
          '電極の浮きや体動の影響が推測される',
          '特徴的なP波やQRSを判定しにくい',
        ],
        summary: '解析対象の心拍情報を妨げるノイズが強く、再測定が必要です。',
        recommendation: '電極の再装着と読み取り条件の見直しを行ってください。',
      };
    default:
      return {
        label: 'その他・要精査',
        severity: 'warning',
        confidence: 66,
        cues: [
          '分類が明確ではない',
          '追加の画像確認または再測定が必要',
        ],
        summary: '波形パターンが確定していないため、再確認が必要です。',
        recommendation: '他のリードや臨床情報と照合して再評価してください。',
      };
  }
}

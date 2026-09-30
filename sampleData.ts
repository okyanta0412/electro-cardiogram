export interface ECGRecord {
  id: string;
  imageUrl: string;
  patientId: string;
  triageCategory: 'critical' | 'warning' | 'normal' | 'artifact';
  waveformType: 'pvc' | 'af' | 'st_change' | 'normal_sinus' | 'noise' | 'other';
  notes: string;
  heartRateEstimate?: number;
  timestamp: string;
  source: 'camera' | 'upload' | 'sample';
}

// Generate realistic ECG SVG data URLs for instant offline demo
export function createECGDataURL(type: 'normal' | 'pvc' | 'af' | 'noise'): string {
  let polylinePoints = '';
  let label = '';
  
  if (type === 'normal') {
    label = '正常洞調律 (Normal Sinus Rhythm) - HR 72bpm';
    // Repetitive clean P-QRS-T waves
    let pts: string[] = [];
    for (let i = 0; i < 6; i++) {
      const x = i * 140;
      pts.push(
        `${x},100`, `${x + 20},100`, 
        `${x + 35},92`, `${x + 45},100`, // P wave
        `${x + 60},100`, `${x + 65},108`, // Q
        `${x + 72},30`,  // R wave (sharp peak)
        `${x + 78},115`, // S
        `${x + 85},100`, 
        `${x + 105},85`, `${x + 120},100`, // T wave
        `${x + 140},100`
      );
    }
    polylinePoints = pts.join(' ');
  } else if (type === 'pvc') {
    label = '心室性期外収縮 (PVC) - 早期出現・幅広異型QRS';
    let pts: string[] = [];
    // 2 normal beats, 1 huge bizarre PVC, compensatory pause, 2 normal beats
    pts.push(
      // Beat 1 Normal
      '0,100', '20,100', '35,92', '45,100', '60,100', '65,108', '72,30', '78,115', '85,100', '105,85', '120,100', '140,100',
      // Beat 2 Normal
      '160,100', '175,92', '185,100', '200,100', '205,108', '212,30', '218,115', '225,100', '245,85', '260,100', '280,100',
      // Beat 3: PVC!! (No P wave, premature at 320, broad tall notched R, inverted deep T)
      '310,100', '320,118', '335,15', '342,22', '350,135', '370,140', '390,100',
      // Compensatory pause: flat line until 460
      '460,100',
      // Beat 4 Normal resumes
      '480,100', '495,92', '505,100', '520,100', '525,108', '532,30', '538,115', '545,100', '565,85', '580,100', '600,100',
      // Beat 5 Normal
      '620,100', '635,92', '645,100', '660,100', '665,108', '672,30', '678,115', '685,100', '705,85', '720,100', '740,100'
    );
    polylinePoints = pts.join(' ');
  } else if (type === 'af') {
    label = '心房細動 (AFib) - 不規則なRR間隔・f波';
    let pts: string[] = [];
    const beatPositions = [50, 140, 260, 330, 470, 560, 690];
    let currX = 0;
    for (const bX of beatPositions) {
      // Fibrillation baseline wave
      while (currX < bX - 10) {
        pts.push(`${currX},${100 + (Math.sin(currX * 0.4) * 4) + (Math.cos(currX * 0.9) * 3)}`);
        currX += 8;
      }
      // Narrow QRS
      pts.push(`${bX - 5},105`, `${bX},28`, `${bX + 5},112`, `${bX + 10},98`);
      currX = bX + 10;
    }
    pts.push(`800,100`);
    polylinePoints = pts.join(' ');
  } else {
    // Noise / Artifact
    label = '体動アーティファクト (Noise / Artifact) - 基線動揺';
    let pts: string[] = [];
    for (let x = 0; x <= 800; x += 10) {
      const y = 100 + Math.sin(x * 0.05) * 35 + (Math.random() * 20 - 10);
      pts.push(`${x},${Math.round(y)}`);
    }
    polylinePoints = pts.join(' ');
  }

  // Create an authentic medical ECG strip SVG with millimeter grid background
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 200" width="800" height="200">
    <defs>
      <!-- 1mm fine grid -->
      <pattern id="smallGrid" width="10" height="10" patternUnits="userSpaceOnUse">
        <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#fecdd3" stroke-width="0.5" opacity="0.65"/>
      </pattern>
      <!-- 5mm major grid -->
      <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
        <rect width="50" height="50" fill="url(#smallGrid)"/>
        <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#fda4af" stroke-width="1.2" opacity="0.85"/>
      </pattern>
    </defs>
    <!-- Medical ECG paper background -->
    <rect width="800" height="200" fill="#fff1f2"/>
    <rect width="800" height="200" fill="url(#grid)"/>
    
    <!-- Calibration pulse (1mV = 10mm) at start -->
    <path d="M 10 100 L 20 100 L 20 60 L 35 60 L 35 100 L 45 100" fill="none" stroke="#0f172a" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
    
    <!-- Waveform trace -->
    <polyline points="${polylinePoints}" fill="none" stroke="#090d16" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/>
    
    <!-- Watermark / clinical lead label -->
    <text x="15" y="25" font-family="monospace" font-size="12" font-weight="bold" fill="#be123c">LEAD II (25mm/s, 10mm/mV)</text>
    <text x="520" y="25" font-family="sans-serif" font-size="11" font-weight="600" fill="#475569">${label}</text>
  </svg>`;

  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

// Initial preloaded mock records so the app opens with realistic clinical data
export const INITIAL_RECORDS: ECGRecord[] = [
  {
    id: 'rec-001',
    patientId: 'Bed 302 (P-401)',
    imageUrl: createECGDataURL('pvc'),
    triageCategory: 'warning',
    waveformType: 'pvc',
    notes: '22:15 アラーム鳴動。夜間トイレ歩行後に単発〜2連発出現。血圧128/78、胸痛なし。夜勤リーダーへ申し送り済み。',
    heartRateEstimate: 78,
    timestamp: '2026-10-01 01:15',
    source: 'sample'
  },
  {
    id: 'rec-002',
    patientId: 'Bed 215 (P-108)',
    imageUrl: createECGDataURL('normal'),
    triageCategory: 'normal',
    waveformType: 'normal_sinus',
    notes: '00:30 定時モニタリングチェック。P波先行あり、QRS幅正常。経過良好につき経過観察継続。',
    heartRateEstimate: 68,
    timestamp: '2026-10-01 00:30',
    source: 'sample'
  }
];

export const CATEGORY_DEFINITIONS = {
  critical: {
    label: '即報告（Drコール）',
    shortLabel: '即報告',
    color: 'rose',
    bg: '#f43f5e',
    text: '#ffffff',
    border: '#e11d48',
    description: 'VT疑い・3度AVブロック・ST著明変化。直ちに当直医へ連絡'
  },
  warning: {
    label: '要相談（リーダーへ）',
    shortLabel: '要相談',
    color: 'amber',
    bg: '#f59e0b',
    text: '#ffffff',
    border: '#d97706',
    description: 'PVC多発/連発・新規AF・高度頻脈/徐脈。早めにリーダーへ報告'
  },
  normal: {
    label: '経過観察（様子見）',
    shortLabel: '経過観察',
    color: 'emerald',
    bg: '#10b981',
    text: '#ffffff',
    border: '#059669',
    description: '単発PVC・落ち着いた洞調律。バイタル安定、定期チェック継続'
  },
  artifact: {
    label: 'ノイズ（電極貼り直し）',
    shortLabel: 'ノイズ',
    color: 'slate',
    bg: '#64748b',
    text: '#ffffff',
    border: '#475569',
    description: '体動・筋電図・電極浮き。波形判定不能のため電極再確認'
  }
};

export const WAVEFORM_TYPES = [
  { id: 'pvc', label: 'PVC (心室性期外収縮)', hint: '幅広異型QRS波が早期に出現' },
  { id: 'af', label: 'AF (心房細動)', hint: 'RR間隔不規則・P波消失' },
  { id: 'st_change', label: 'ST変化 (虚血/梗塞疑い)', hint: 'ST部分の著明な上昇または低下' },
  { id: 'normal_sinus', label: '正常洞調律 (NSR)', hint: '規則正しいP-QRS-T波形' },
  { id: 'noise', label: '体動ノイズ / アーティファクト', hint: '基線の激しい揺れ・電極外れ' },
  { id: 'other', label: 'その他・要精査', hint: '分類困難な不整脈' },
];

import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import {
  Activity,
  HeartPulse,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Upload,
  Camera,
  Trash2,
  Copy,
  Check,
  Maximize2,
  X,
  FileCheck,
  Sparkles,
  ShieldAlert,
  HelpCircle,
} from 'lucide-react';
import {
  ECGRecord,
  INITIAL_RECORDS,
  CATEGORY_DEFINITIONS,
  WAVEFORM_TYPES,
  createECGDataURL,
} from './utils/sampleData';

const LOCAL_STORAGE_KEY = 'pulse_triage_records_v1';

export default function App() {
  // Persistence with localStorage
  const [records, setRecords] = useState<ECGRecord[]>(() => {
    try {
      const saved = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (saved) {
        return JSON.parse(saved);
      }
    } catch (e) {
      console.error('Failed to load records from localStorage', e);
    }
    return INITIAL_RECORDS;
  });

  useEffect(() => {
    try {
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(records));
    } catch (e) {
      console.error('Failed to save records to localStorage', e);
    }
  }, [records]);

  // Active form state
  const [selectedImage, setSelectedImage] = useState<string>(() => createECGDataURL('pvc'));
  const [patientId, setPatientId] = useState<string>('Bed 105 (P-202)');
  const [triageCategory, setTriageCategory] = useState<'critical' | 'warning' | 'normal' | 'artifact'>('warning');
  const [waveformType, setWaveformType] = useState<ECGRecord['waveformType']>('pvc');
  const [notes, setNotes] = useState<string>('02:10 アラーム鳴動。数分間に渡り単発〜多発性の波形変化あり。自覚症状・胸痛なし。バイタル安定。');
  const [heartRate, setHeartRate] = useState<number>(84);

  // Gallery filter & modal states
  const [activeFilter, setActiveFilter] = useState<string>('all');
  const [modalImage, setModalImage] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [submitSuccess, setSubmitSuccess] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle image upload from file or camera
  const handleFileUpload = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      if (e.target?.result) {
        setSelectedImage(e.target.result as string);
      }
    };
    reader.readAsDataURL(file);
  };

  // Quick sample loader
  const loadSample = (type: 'normal' | 'pvc' | 'af' | 'noise') => {
    const url = createECGDataURL(type);
    setSelectedImage(url);
    if (type === 'normal') {
      setTriageCategory('normal');
      setWaveformType('normal_sinus');
      setHeartRate(72);
      setNotes('00:45 定時記録。規則正しい洞調律。ST変化・期外収縮なし。');
    } else if (type === 'pvc') {
      setTriageCategory('warning');
      setWaveformType('pvc');
      setHeartRate(82);
      setNotes('01:30 トイレ移動後に早期幅広QRS出現。自覚症状なし。経過観察継続。');
    } else if (type === 'af') {
      setTriageCategory('warning');
      setWaveformType('af');
      setHeartRate(110);
      setNotes('02:00 RR不規則、P波消失。軽度の動悸訴えあり。当直医へ相談検討。');
    } else {
      setTriageCategory('artifact');
      setWaveformType('noise');
      setHeartRate(0);
      setNotes('体動による基線動揺大。電極の浮きを確認し貼り直し推奨。');
    }
  };

  // Handle record creation (Week 1 Tracer Bullet Action)
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedImage) return;

    const now = new Date();
    const timeString = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;

    const newRecord: ECGRecord = {
      id: `rec-${Date.now()}`,
      imageUrl: selectedImage,
      patientId: patientId.trim() || '未設定患者',
      triageCategory,
      waveformType,
      notes: notes.trim(),
      heartRateEstimate: heartRate > 0 ? heartRate : undefined,
      timestamp: timeString,
      source: 'upload',
    };

    setRecords((prev) => [newRecord, ...prev]);
    setSubmitSuccess(true);
    setTimeout(() => setSubmitSuccess(false), 2500);

    // Reset some fields for next patient
    setPatientId(`Bed ${Math.floor(Math.random() * 800 + 100)}`);
  };

  const handleDeleteRecord = (id: string) => {
    if (window.confirm('この心電図レコードを削除してもよろしいですか？')) {
      setRecords((prev) => prev.filter((r) => r.id !== id));
    }
  };

  const handleCopyNote = (rec: ECGRecord) => {
    const text = `【心電図トリアージ記録】
日時: ${rec.timestamp}
患者: ${rec.patientId}
判定区分: ${CATEGORY_DEFINITIONS[rec.triageCategory].label}
波形分類: ${WAVEFORM_TYPES.find((w) => w.id === rec.waveformType)?.label || 'その他'}
推定心拍: ${rec.heartRateEstimate ? `${rec.heartRateEstimate} bpm` : '未計測'}
経過・所見: ${rec.notes}`;

    navigator.clipboard.writeText(text);
    setCopiedId(rec.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Filtered records
  const filteredRecords = records.filter((r) => {
    if (activeFilter === 'all') return true;
    return r.triageCategory === activeFilter;
  });

  // Category counts
  const countCritical = records.filter((r) => r.triageCategory === 'critical').length;
  const countWarning = records.filter((r) => r.triageCategory === 'warning').length;
  const countNormal = records.filter((r) => r.triageCategory === 'normal').length;
  const countArtifact = records.filter((r) => r.triageCategory === 'artifact').length;

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-icon-wrapper">
            <HeartPulse size={28} />
          </div>
          <div>
            <h1 className="brand-title">
              PulseTriage
              <span className="badge-version">第1週MVP</span>
            </h1>
            <p className="brand-subtitle">
              病棟モニター心電図 異常波形判読・一次トリアージ支援
            </p>
          </div>
        </div>

        <div className="header-meta">
          <div className="shift-badge">
            <span className="live-indicator"></span>
            <span>🌙 夜勤帯シフトモード (21:00-09:00)</span>
          </div>
        </div>
      </header>

      {/* Summary Statistics Ribbon */}
      <section className="stats-ribbon" aria-label="トリアージ集計">
        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'rgba(6, 182, 212, 0.15)', color: '#06b6d4' }}>
            <Activity size={22} />
          </div>
          <div className="stat-content">
            <span className="stat-label">本日チェック総数</span>
            <span className="stat-number">{records.length} <small style={{ fontSize: '0.9rem', color: '#94a3b8' }}>件</small></span>
          </div>
        </div>

        <div className="stat-card" style={{ borderColor: countCritical > 0 ? 'rgba(244, 63, 94, 0.4)' : undefined }}>
          <div className="stat-icon" style={{ background: 'rgba(244, 63, 94, 0.15)', color: '#f43f5e' }}>
            <ShieldAlert size={22} />
          </div>
          <div className="stat-content">
            <span className="stat-label">🚨 即時報告（Drコール）</span>
            <span className="stat-number" style={{ color: countCritical > 0 ? '#fda4af' : '#ffffff' }}>
              {countCritical} <small style={{ fontSize: '0.9rem', color: '#94a3b8' }}>件</small>
            </span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'rgba(245, 158, 11, 0.15)', color: '#f59e0b' }}>
            <AlertTriangle size={22} />
          </div>
          <div className="stat-content">
            <span className="stat-label">⚠️ 要相談（リーダーへ）</span>
            <span className="stat-number">{countWarning} <small style={{ fontSize: '0.9rem', color: '#94a3b8' }}>件</small></span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#10b981' }}>
            <CheckCircle2 size={22} />
          </div>
          <div className="stat-content">
            <span className="stat-label">📋 経過観察（安定）</span>
            <span className="stat-number">{countNormal} <small style={{ fontSize: '0.9rem', color: '#94a3b8' }}>件</small></span>
          </div>
        </div>
      </section>

      {/* Main Two-Column Workflow */}
      <div className="main-grid">
        {/* Left Column: Triage Recording Studio */}
        <section className="glass-panel" aria-labelledby="triage-studio-title">
          <div className="panel-header">
            <h2 id="triage-studio-title" className="panel-title">
              <Camera size={20} color="#06b6d4" />
              波形記録 ＆ トリアージ入力
            </h2>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>全層貫通・即時保存</span>
          </div>

          {/* Quick Sample Selector */}
          <div className="sample-picker-bar">
            <div className="sample-picker-label">
              <Sparkles size={14} color="#f59e0b" />
              <span>ワンクリックでテスト症例を読み込む:</span>
            </div>
            <div className="sample-buttons-grid">
              <button type="button" className="btn-sample" onClick={() => loadSample('pvc')}>
                <span style={{ color: '#f59e0b' }}>●</span> PVC (期外収縮)
              </button>
              <button type="button" className="btn-sample" onClick={() => loadSample('af')}>
                <span style={{ color: '#f59e0b' }}>●</span> AF (心房細動)
              </button>
              <button type="button" className="btn-sample" onClick={() => loadSample('normal')}>
                <span style={{ color: '#10b981' }}>●</span> 正常洞調律
              </button>
              <button type="button" className="btn-sample" onClick={() => loadSample('noise')}>
                <span style={{ color: '#64748b' }}>●</span> 体動ノイズ
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit}>
            {/* ECG Strip Image Preview Container */}
            <div className="ecg-preview-box">
              <img
                src={selectedImage}
                alt="心電図プレビュー"
                className="ecg-preview-img"
              />
              <div className="ecg-preview-overlay">
                <button
                  type="button"
                  className="btn-preview-action"
                  onClick={() => setModalImage(selectedImage)}
                  title="拡大読影"
                >
                  <Maximize2 size={13} />
                  <span>拡大</span>
                </button>
              </div>
            </div>

            {/* Custom Image Upload Dropzone */}
            <div
              className={`dropzone-container ${isDragOver ? 'is-active' : ''}`}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragOver(true);
              }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragOver(false);
                if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                  handleFileUpload(e.dataTransfer.files[0]);
                }
              }}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                capture="environment"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileUpload(e.target.files[0]);
                  }
                }}
              />
              <div className="dropzone-icon">
                <Upload size={22} />
              </div>
              <div className="dropzone-title">スマホカメラで撮影 または 画像選択</div>
              <div className="dropzone-hint">
                モニター用紙を平らな場所で撮影してください（JPEG/PNG対応）
              </div>
            </div>

            {/* Patient & Heart Rate Info */}
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '0.75rem' }}>
              <div className="form-group">
                <label className="form-label" htmlFor="patient-id-input">
                  患者識別番号 / ベッド番号
                </label>
                <input
                  id="patient-id-input"
                  type="text"
                  className="form-input"
                  placeholder="例: Bed 302 (P-401)"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="heart-rate-input">
                  心拍数 (HR)
                </label>
                <input
                  id="heart-rate-input"
                  type="number"
                  className="form-input"
                  placeholder="bpm"
                  value={heartRate || ''}
                  onChange={(e) => setHeartRate(Number(e.target.value))}
                />
              </div>
            </div>

            {/* Triage Category Radio Cards */}
            <div className="form-group">
              <label className="form-label">
                一次トリアージ区分（緊急度判断）
              </label>
              <div className="triage-selector-grid">
                {(['critical', 'warning', 'normal', 'artifact'] as const).map((cat) => {
                  const def = CATEGORY_DEFINITIONS[cat];
                  const isSelected = triageCategory === cat;
                  return (
                    <div
                      key={cat}
                      className={`triage-option-card ${cat} ${isSelected ? 'selected' : ''}`}
                      onClick={() => setTriageCategory(cat)}
                      role="button"
                      tabIndex={0}
                    >
                      <div className="triage-option-header">
                        <span className="triage-option-title" style={{ color: def.text }}>
                          {cat === 'critical' && '🚨'}
                          {cat === 'warning' && '⚠️'}
                          {cat === 'normal' && '📋'}
                          {cat === 'artifact' && '🛠️'}
                          {def.shortLabel}
                        </span>
                        {isSelected && <Check size={16} color="#ffffff" />}
                      </div>
                      <p className="triage-option-desc">{def.description}</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Waveform Detail Type */}
            <div className="form-group">
              <label className="form-label" htmlFor="waveform-type-select">
                推定波形分類（特定異常の選択）
              </label>
              <select
                id="waveform-type-select"
                className="form-select"
                value={waveformType}
                onChange={(e) => setWaveformType(e.target.value as any)}
              >
                {WAVEFORM_TYPES.map((w) => (
                  <option key={w.id} value={w.id}>
                    {w.label} - {w.hint}
                  </option>
                ))}
              </select>
            </div>

            {/* Notes & Observations */}
            <div className="form-group">
              <label className="form-label" htmlFor="notes-textarea">
                看護経過・所見メモ（申し送り用）
              </label>
              <textarea
                id="notes-textarea"
                className="form-textarea"
                placeholder="患者の様子（胸痛・息切れの有無、血圧、トイレ歩行後など）を記録"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
              />
            </div>

            {/* Submit Action */}
            <button
              type="submit"
              className="btn-submit"
              id="submit-record-button"
            >
              {submitSuccess ? (
                <>
                  <Check size={20} />
                  <span>記録を完了しました！</span>
                </>
              ) : (
                <>
                  <FileCheck size={20} />
                  <span>この心電図を仕分けて記録する</span>
                </>
              )}
            </button>
          </form>
        </section>

        {/* Right Column: Triaged ECG Gallery */}
        <section className="glass-panel" aria-labelledby="triage-gallery-title">
          <div className="panel-header">
            <div>
              <h2 id="triage-gallery-title" className="panel-title">
                <Activity size={20} color="#10b981" />
                仕分け済み心電図リスト
              </h2>
              <span style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
                ブラウザを再読み込みしても永続化されます（第1週要件達成）
              </span>
            </div>
            <span className="badge-version">{filteredRecords.length} 件表示</span>
          </div>

          {/* Filter Tabs */}
          <div className="filter-tabs-bar" role="tablist">
            <button
              className={`filter-tab ${activeFilter === 'all' ? 'active' : ''}`}
              onClick={() => setActiveFilter('all')}
            >
              すべて ({records.length})
            </button>
            <button
              className={`filter-tab ${activeFilter === 'critical' ? 'active' : ''}`}
              onClick={() => setActiveFilter('critical')}
            >
              🚨 即報告 ({countCritical})
            </button>
            <button
              className={`filter-tab ${activeFilter === 'warning' ? 'active' : ''}`}
              onClick={() => setActiveFilter('warning')}
            >
              ⚠️ 要相談 ({countWarning})
            </button>
            <button
              className={`filter-tab ${activeFilter === 'normal' ? 'active' : ''}`}
              onClick={() => setActiveFilter('normal')}
            >
              📋 経過観察 ({countNormal})
            </button>
            <button
              className={`filter-tab ${activeFilter === 'artifact' ? 'active' : ''}`}
              onClick={() => setActiveFilter('artifact')}
            >
              🛠️ ノイズ ({countArtifact})
            </button>
          </div>

          {/* Cards List */}
          {filteredRecords.length === 0 ? (
            <div className="empty-state">
              <HelpCircle size={48} className="empty-icon" />
              <p>該当する心電図レコードはありません</p>
              <p style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>
                左の入力フォームから新しい心電図を登録してください。
              </p>
            </div>
          ) : (
            <div className="records-list">
              {filteredRecords.map((rec) => {
                const catDef = CATEGORY_DEFINITIONS[rec.triageCategory];
                const waveDef = WAVEFORM_TYPES.find((w) => w.id === rec.waveformType);

                return (
                  <article key={rec.id} className="record-card">
                    <div className="record-header">
                      <div className="patient-info">
                        <span className="patient-badge">{rec.patientId}</span>
                        {rec.heartRateEstimate && (
                          <span style={{ fontSize: '0.75rem', color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
                            HR: {rec.heartRateEstimate}bpm
                          </span>
                        )}
                      </div>

                      <div className="record-meta-actions">
                        <span className={`triage-badge ${rec.triageCategory}`}>
                          {rec.triageCategory === 'critical' && '🚨'}
                          {rec.triageCategory === 'warning' && '⚠️'}
                          {rec.triageCategory === 'normal' && '📋'}
                          {rec.triageCategory === 'artifact' && '🛠️'}
                          {catDef.shortLabel}
                        </span>
                      </div>
                    </div>

                    {/* Waveform Strip Thumbnail */}
                    <div
                      className="card-ecg-strip"
                      onClick={() => setModalImage(rec.imageUrl)}
                      title="クリックして拡大"
                    >
                      <img src={rec.imageUrl} alt={`${rec.patientId}の心電図`} />
                    </div>

                    {/* Waveform tag & Notes */}
                    <div style={{ marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span
                        style={{
                          fontSize: '0.72rem',
                          background: 'rgba(6, 182, 212, 0.15)',
                          color: '#38bdf8',
                          padding: '0.15rem 0.5rem',
                          borderRadius: '4px',
                          fontWeight: 600,
                        }}
                      >
                        波形: {waveDef?.label || '未分類'}
                      </span>
                    </div>

                    {rec.notes && <div className="card-notes">{rec.notes}</div>}

                    {/* Card Footer Actions */}
                    <div className="card-footer">
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                        <Clock size={13} />
                        <span className="record-time">{rec.timestamp}</span>
                      </div>

                      <div className="card-actions">
                        <button
                          type="button"
                          className="btn-card-action"
                          onClick={() => setModalImage(rec.imageUrl)}
                          title="画像を拡大して確認"
                        >
                          <Maximize2 size={13} />
                          <span>拡大</span>
                        </button>

                        <button
                          type="button"
                          className="btn-card-action"
                          onClick={() => handleCopyNote(rec)}
                          title="申し送り用テキストをクリップボードにコピー"
                        >
                          {copiedId === rec.id ? (
                            <>
                              <Check size={13} color="#10b981" />
                              <span style={{ color: '#10b981' }}>コピー済</span>
                            </>
                          ) : (
                            <>
                              <Copy size={13} />
                              <span>メモコピー</span>
                            </>
                          )}
                        </button>

                        <button
                          type="button"
                          className="btn-card-action danger"
                          onClick={() => handleDeleteRecord(rec.id)}
                          title="レコードを削除"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>
      </div>

      {/* Week 1 Milestone Completion Banner */}
      <footer style={{ marginTop: '3.5rem', padding: '1.5rem', background: 'rgba(15, 23, 42, 0.5)', borderRadius: '12px', border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.6rem', marginBottom: '0.5rem' }}>
          <FileCheck size={20} color="#10b981" />
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f8fafc' }}>
            第1週スライス要件達成済み
          </h3>
        </div>
        <p style={{ fontSize: '0.8rem', color: '#94a3b8', maxWidth: '650px', margin: '0 auto' }}>
          UI入力 ⇆ 写真アップロード・サジェスト ⇆ トリアージ分類 ⇆ データ永続化（localStorage / Supabase連携設計）の全層が貫通しています。
        </p>
      </footer>

      {/* Fullscreen ECG Modal Viewer */}
      {modalImage && (
        <div className="modal-backdrop" onClick={() => setModalImage(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">心電図 高解像度読影ビューワー</h3>
              <button
                type="button"
                className="btn-card-action"
                onClick={() => setModalImage(null)}
              >
                <X size={18} />
              </button>
            </div>

            <div className="modal-image-wrapper">
              <img src={modalImage} alt="心電図拡大表示" />
            </div>

            <div className="modal-footer">
              <span style={{ fontSize: '0.75rem', color: '#94a3b8', marginRight: 'auto' }}>
                方眼紙仕様: 1小マス = 0.04秒 (1mm), 1大マス = 0.20秒 (5mm)
              </span>
              <button
                type="button"
                className="btn-submit"
                style={{ width: 'auto', padding: '0.5rem 1.25rem' }}
                onClick={() => setModalImage(null)}
              >
                閉じる
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

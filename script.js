const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png"]);
const HEIC_TYPES = new Set(["image/heic", "image/heif", "image/heic-sequence", "image/heif-sequence"]);
const EMPTY_PREVIEW_TEXT = "選択したモニター写真のプレビューがここに表示されます";
const CLASSIFY_URL = "/api/classify";
const LABEL_NAMES = {
  NORMAL: "正常・対象所見なし",
  AF: "心房細動",
  PVC: "心室期外収縮",
  PAC: "心房性期外収縮候補",
  REVIEW: "判定保留（AF/PVC要確認）",
  ventricular_tachycardia: "心室頻拍",
  third_degree_av_block: "3度房室ブロック候補",
  nonspecific_st_elevation: "ST上昇候補",
  second_degree_av_block: "2度房室ブロック候補",
  nonspecific_st_depression: "ST低下候補",
  ventricular_escape_beat: "心室補充収縮",
  high_grade_av_block: "高度房室ブロック",
  pacing_failure: "ペーシング不全",
  torsades_de_pointes: "TdP（トルサード・ド・ポワンツ）",
  ventricular_fibrillation: "心室細動",
  asystole: "心静止",
};
const LABEL_CODES = {
  NORMAL: "NORMAL",
  AF: "AF",
  PVC: "PVC",
  REVIEW: "REVIEW",
  ventricular_tachycardia: "VT",
  third_degree_av_block: "3AVB",
  nonspecific_st_elevation: "STE",
  second_degree_av_block: "2AVB",
  nonspecific_st_depression: "STD",
  ventricular_escape_beat: "ESCAPE",
  high_grade_av_block: "HG-AVB",
  pacing_failure: "PACE-FAIL",
  torsades_de_pointes: "TdP",
  ventricular_fibrillation: "VF",
  asystole: "ASYSTOLE",
};
const PROTOTYPE_LABEL_NAMES = {
  ventricular_tachycardia: "心室頻拍",
  third_degree_av_block: "3度房室ブロック",
  st_elevation: "ST上昇",
  st_depression: "ST低下",
  ventricular_escape_beat: "心室補充収縮",
  high_grade_av_block: "高度房室ブロック",
  pacing_failure: "ペーシング不全",
  torsades_de_pointes: "TdP（トルサード・ド・ポワンツ）",
  ventricular_fibrillation: "心室細動",
  asystole: "心静止",
};
const PROTOTYPE_LABEL_CODES = {
  ventricular_tachycardia: "VT",
  third_degree_av_block: "3AVB",
  st_elevation: "STE",
  st_depression: "STD",
  ventricular_escape_beat: "ESCAPE",
  high_grade_av_block: "HG-AVB",
  pacing_failure: "PACE-FAIL",
  torsades_de_pointes: "TdP",
  ventricular_fibrillation: "VF",
  asystole: "ASYSTOLE",
};

const fileInput = document.querySelector("#ecg-photo");
const inputMessage = document.querySelector("#input-message");
const previewImage = document.querySelector("#preview-image");
const previewEmpty = document.querySelector("#preview-empty");
const previewMeta = document.querySelector("#preview-meta");
const previewFilename = document.querySelector("#preview-filename");
const removeImageButton = document.querySelector("#remove-image");
const analyzeButton = document.querySelector("#analyze-button");
const analysisMessage = document.querySelector("#analysis-message");
const resultState = document.querySelector("#result-state");
const resultPlaceholder = document.querySelector(".result-placeholder");
const resultTitle = document.querySelector("#result-title");
const resultDescription = document.querySelector("#result-description");

let selectedFile = null;
let previewUrl = null;
let validationVersion = 0;

function setMessage(element, message, state) {
  element.textContent = message;
  element.dataset.state = state;
}

function releasePreview() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  previewImage.removeAttribute("src");
  previewImage.hidden = true;
  previewEmpty.hidden = false;
  previewEmpty.textContent = EMPTY_PREVIEW_TEXT;
  previewMeta.hidden = true;
  previewFilename.textContent = "";
  selectedFile = null;
  analyzeButton.disabled = true;
}

function resetResult() {
  resultPlaceholder.classList.remove("is-classified");
  resultState.textContent = "未解析";
  resultState.dataset.state = "idle";
  resultTitle.textContent = "解析結果はまだありません";
  resultDescription.textContent = "複数の異常所見を検出し、優先候補を主結果に表示します。";
}

function readAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(reader.result), { once: true });
    reader.addEventListener("error", () => reject(new Error("画像を読み込めませんでした。")), { once: true });
    reader.readAsDataURL(file);
  });
}

function showInvalidFile(message) {
  setMessage(inputMessage, message, "error");
  setMessage(analysisMessage, "有効な画像を選択してください。", "error");
}

async function handleFileSelection() {
  const file = fileInput.files?.[0];
  if (!file) return;

  const currentVersion = ++validationVersion;
  const isHeic = HEIC_TYPES.has(file.type) || /\.hei[cf]$/i.test(file.name);
  releasePreview();
  setMessage(inputMessage, "画像を確認しています…", "checking");
  setMessage(analysisMessage, "", "idle");
  resetResult();

  if (!ALLOWED_TYPES.has(file.type) && !isHeic) {
    showInvalidFile("JPEG、PNG、HEIC 形式の画像を選択してください。");
    return;
  }

  if (file.size === 0) {
    showInvalidFile("選択したファイルに画像データがありません。");
    return;
  }

  if (file.size > MAX_FILE_SIZE) {
    showInvalidFile("画像サイズが10MBを超えています。別の画像を選択してください。");
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  const image = new Image();
  image.src = objectUrl;

  try {
    await image.decode();
  } catch {
    if (isHeic) {
      if (currentVersion !== validationVersion) {
        URL.revokeObjectURL(objectUrl);
        return;
      }
      previewUrl = objectUrl;
      selectedFile = file;
      previewEmpty.textContent = "HEIC画像を選択しました。プレビューできない場合も、そのまま分類できます。";
      previewEmpty.hidden = false;
      previewMeta.hidden = false;
      previewFilename.textContent = file.name;
      analyzeButton.disabled = false;
      setMessage(inputMessage, "HEIC画像を読み込みました。内容を確認してください。", "success");
      setMessage(analysisMessage, "HEIC画像をサーバー側で変換して分類します。", "ready");
      return;
    }
    URL.revokeObjectURL(objectUrl);
    if (currentVersion === validationVersion) {
      showInvalidFile("画像を読み込めませんでした。ファイルを確認してください。");
    }
    return;
  }

  if (currentVersion !== validationVersion) {
    URL.revokeObjectURL(objectUrl);
    return;
  }

  previewUrl = objectUrl;
  selectedFile = file;
  previewImage.src = objectUrl;
  previewImage.hidden = false;
  previewEmpty.hidden = true;
  previewFilename.textContent = `${file.name} / ${image.naturalWidth} × ${image.naturalHeight}px`;
  previewMeta.hidden = false;
  analyzeButton.disabled = false;
  setMessage(inputMessage, "画像を読み込みました。内容を確認してください。", "success");
  setMessage(analysisMessage, "分類の準備ができました。出力は研究用試作です。", "ready");
}

fileInput.addEventListener("change", handleFileSelection);

removeImageButton.addEventListener("click", () => {
  validationVersion += 1;
  releasePreview();
  fileInput.value = "";
  setMessage(inputMessage, "画像を選択してください。", "idle");
  setMessage(analysisMessage, "分類APIに接続して画像を分類してください。", "idle");
  resetResult();
});

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile) return;

  const currentVersion = validationVersion;
  analyzeButton.disabled = true;
  resultState.textContent = "分類中";
  resultState.dataset.state = "working";
  resultTitle.textContent = "画像を分類しています";
  resultDescription.textContent = "Lead IIの波形抽出と試作モデルの推論を実行中です。";
  setMessage(analysisMessage, "分類APIへ画像を送信しています…", "checking");

  try {
    const image = await readAsDataUrl(selectedFile);
    let response;
    try {
      response = await fetch(CLASSIFY_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image }),
      });
    } catch {
      throw new Error("分類APIに接続できません。通信状態を確認してください。");
    }
    const payload = await response.json();
    if (!response.ok) {
      const detail = payload.error || "画像を分類できませんでした。";
      if (response.status === 413) throw new Error(`画像サイズの上限を超えています。${detail}`);
      if (response.status === 422 && detail.startsWith("Could not isolate a high-contrast waveform")) {
        throw new Error("背景から波形を分離できません。波形を1本に切り取り、反射や方眼の少ない写真をお試しください。");
      }
      if (response.status === 422) throw new Error(`この画像を処理できません。${detail}`);
      throw new Error(detail);
    }
    if (currentVersion !== validationVersion) return;

    resultPlaceholder.classList.add("is-classified");
    resultState.textContent = payload.photo_prototype_prediction?.label
      ? "試作判定"
      : payload.priority_finding ? "優先所見候補" : "試作判定";
    resultState.dataset.state = "demo";
    const resultNames = payload.classification_mode === "labeled_photo_prototype" ? PROTOTYPE_LABEL_NAMES : LABEL_NAMES;
    const resultCodes = payload.classification_mode === "labeled_photo_prototype" ? PROTOTYPE_LABEL_CODES : LABEL_CODES;
    resultTitle.textContent = `${resultNames[payload.label] || "判定不能"}（${resultCodes[payload.label] || payload.label || "UNREADABLE"}）`;
    resultDescription.textContent = payload.photo_prototype_prediction?.label
      ? ""
      : payload.priority_finding
      ? `基本リズム分類: ${LABEL_NAMES[payload.base_label] || payload.base_label}。${payload.warning || ""}`
      : payload.review_reason || payload.warning || "研究用試作の出力です。医療上の判断に使用しないでください。";
    setMessage(analysisMessage, "分類処理が完了しました。結果は未検証の試作出力です。", "success");
  } catch (error) {
    if (currentVersion !== validationVersion) return;
    resultPlaceholder.classList.add("is-classified");
    resultState.textContent = "判定できません";
    resultState.dataset.state = "unavailable";
    resultTitle.textContent = "分類を完了できませんでした";
    resultDescription.textContent = error instanceof Error ? error.message : "分類APIに接続できません。";
    setMessage(analysisMessage, error instanceof Error ? error.message : "分類APIに接続できません。", "error");
  } finally {
    if (currentVersion === validationVersion && selectedFile) analyzeButton.disabled = false;
  }
});

window.addEventListener("beforeunload", releasePreview);

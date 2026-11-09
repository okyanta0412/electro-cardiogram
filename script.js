const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png"]);
const CLASSIFY_URL = "/api/classify";
const LABEL_NAMES = {
  NORMAL: "正常・対象所見なし",
  AF: "心房細動",
  PVC: "心室期外収縮",
  PAC: "心房性期外収縮候補",
  REVIEW: "判定保留（AF/PVC要確認）",
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
  previewMeta.hidden = true;
  previewFilename.textContent = "";
  selectedFile = null;
  analyzeButton.disabled = true;
}

function resetResult() {
  resultState.textContent = "未解析";
  resultState.dataset.state = "idle";
  resultTitle.textContent = "解析結果はまだありません";
  resultDescription.textContent = "Lead II波形を3分類モデルとPAC候補検出器で処理します。";
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
  releasePreview();
  setMessage(inputMessage, "画像を確認しています…", "checking");
  setMessage(analysisMessage, "", "idle");
  resetResult();

  if (!ALLOWED_TYPES.has(file.type)) {
    showInvalidFile("JPEG または PNG 形式の画像を選択してください。");
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

    resultState.textContent = "試作判定";
    resultState.dataset.state = "demo";
    resultTitle.textContent = `${LABEL_NAMES[payload.label] || "判定不能"}（${payload.label || "UNREADABLE"}）`;
    const notes = [];
    if (payload.review_reason) notes.push(payload.review_reason);
    if (payload.pac_candidate) notes.push("PAC候補スコアはしきい値以上です。");
    else notes.push("PAC候補スコアはしきい値未満です（PACを否定するものではありません）。");
    notes.push(payload.warning || "研究用試作の出力です。医療上の判断に使用しないでください。");
    resultDescription.textContent = notes.join(" ");
    setMessage(analysisMessage, "分類処理が完了しました。結果は未検証の試作出力です。", "success");
  } catch (error) {
    if (currentVersion !== validationVersion) return;
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

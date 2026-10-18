const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png"]);

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
  resultState.textContent = "未解析";
  resultState.dataset.state = "idle";
  resultTitle.textContent = "解析結果はまだありません";
  resultDescription.textContent = "画像解析モデルを接続すると、判定された波形分類と判定根拠を表示します。";

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
  setMessage(analysisMessage, "写真の選択と確認が完了しました。", "ready");
}

fileInput.addEventListener("change", handleFileSelection);

removeImageButton.addEventListener("click", () => {
  validationVersion += 1;
  releasePreview();
  fileInput.value = "";
  setMessage(inputMessage, "画像を選択してください。", "idle");
  setMessage(analysisMessage, "画像を選択すると、解析の準備状態を確認できます。", "idle");
  resultState.textContent = "未解析";
  resultState.dataset.state = "idle";
  resultTitle.textContent = "解析結果はまだありません";
  resultDescription.textContent = "画像解析モデルを接続すると、判定された波形分類と判定根拠を表示します。";
});

analyzeButton.addEventListener("click", () => {
  if (!selectedFile) return;

  resultState.textContent = "未接続";
  resultState.dataset.state = "unavailable";
  resultTitle.textContent = "AI解析はまだ利用できません";
  resultDescription.textContent = "画像は確認できました。分類結果を出すには、後で解析 API と学習済みモデルを接続してください。";
  setMessage(analysisMessage, "画像の準備は完了していますが、AI解析エンジンは未接続です。", "error");
});

window.addEventListener("beforeunload", releasePreview);

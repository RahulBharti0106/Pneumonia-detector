// ── State ─────────────────────────────────────────────────────────────────────
let uploadedImage = null;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const fileInput    = document.getElementById('file-upload');
const dropZone     = document.getElementById('file-drag');
const uploadCaption = document.getElementById('upload-caption');
const imagePreview = document.getElementById('image-preview');
const imageDisplay = document.getElementById('image-display');
const predResult   = document.getElementById('pred-result');
const loader       = document.getElementById('loader');
const submitBtn    = document.getElementById('submit-btn');
const clearBtn     = document.getElementById('clear-btn');

// ── File selection ────────────────────────────────────────────────────────────
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) loadImage(file);
});

// ── Drag and drop ─────────────────────────────────────────────────────────────
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) loadImage(file);
});

// ── Load image preview ────────────────────────────────────────────────────────
function loadImage(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadedImage = e.target.result;

    // Show preview inside drop zone
    imagePreview.src = uploadedImage;
    imagePreview.classList.remove('hidden');
    uploadCaption.classList.add('hidden');

    // Clear any previous result
    resetResult();
    submitBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

// ── Submit ────────────────────────────────────────────────────────────────────
function submitImage() {
  if (!uploadedImage) return;

  setLoading(true);

  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(uploadedImage)
  })
  .then(res => res.json())
  .then(data => {
    setLoading(false);

    if (data.error) {
      showResult('Error: ' + data.error, 'result-pneumonia');
      return;
    }

    // Show the image in the result area
    imageDisplay.src = uploadedImage;
    imageDisplay.style.display = 'block';

    // Display result with colour coding
    const isPneumonia = data.result === 'PNEUMONIA';
    const cls         = isPneumonia ? 'result-pneumonia' : 'result-normal';
    const label       = isPneumonia ? '⚠ PNEUMONIA DETECTED' : '✓ NORMAL';
    const confidence  = data.confidence || '';

    showResult(`${label}\nConfidence: ${confidence}`, cls);
  })
  .catch(err => {
    setLoading(false);
    showResult('Server error — please try again.', 'result-pneumonia');
    console.error(err);
  });
}

// ── Clear ─────────────────────────────────────────────────────────────────────
function clearImage() {
  uploadedImage = null;
  fileInput.value = '';

  imagePreview.src = '';
  imagePreview.classList.add('hidden');
  uploadCaption.classList.remove('hidden');

  imageDisplay.src = '';
  imageDisplay.style.display = 'none';

  submitBtn.disabled = true;
  resetResult();
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function showResult(text, cls) {
  predResult.textContent = text;
  predResult.className   = cls;
  predResult.style.display = 'block';
}

function resetResult() {
  predResult.textContent   = '';
  predResult.className     = '';
  predResult.style.display = 'none';
}

function setLoading(state) {
  loader.style.display     = state ? 'block' : 'none';
  submitBtn.disabled       = state;
}
// OpenCV + WebGazer placeholder script
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext && canvas.getContext('2d');
const statusEl = document.getElementById('status');

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

async function initWebGazer() {
  try {
    // Try ESM import; fallback to global window.webgazer if available
    let wg;
    try {
      wg = await import('webgazer');
      wg = wg.default || wg;
    } catch (e) {
      wg = window.webgazer;
    }
    if (!wg) throw new Error('webgazer not found');

    // Begin webgazer; usage may vary depending on package build
    if (typeof wg.begin === 'function') {
      await wg.begin();
      statusEl.textContent = 'WebGazer started';
      if (typeof wg.showPredictionPoints === 'function') wg.showPredictionPoints(true);
    } else if (wg && wg.setGazeListener) {
      wg.setGazeListener((data, elapsed) => { /* handle gaze data */ });
      wg.begin();
      statusEl.textContent = 'WebGazer started';
    } else {
      statusEl.textContent = 'WebGazer loaded (unknown API)';
    }
  } catch (err) {
    console.warn('webgazer init failed', err);
    statusEl.textContent = 'WebGazer init failed';
  }
}

startCamera().then(() => initWebGazer()).catch(err => { console.error(err); statusEl.textContent = 'Camera error'; });

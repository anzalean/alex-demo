import { FaceMesh } from '@mediapipe/face_mesh';

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const headEl = document.getElementById('head-metrics');
const leftEyeEl = document.getElementById('left-eye');
const rightEyeEl = document.getElementById('right-eye');

// Load OpenCV via the package's `loadOpenCV` helper and wait for initialization.
async function loadOpenCV(timeout = 20000) {
  const start = Date.now();
  try {
    const mod = await import('@opencvjs/web');
    if (mod && typeof mod.loadOpenCV === 'function') {
      const cvModule = await mod.loadOpenCV();
      // The package returns the Module-like object; ensure it's available as window.cv
      if (cvModule) {
        window.cv = cvModule;
        if (typeof window.cv.Mat !== 'undefined') return window.cv;
      }
    }
  } catch (e) {
    // continue to polling fallback below
    console.warn('dynamic import/loadOpenCV failed:', e);
  }

  // fallback: poll for window.cv (in case user loaded OpenCV via <script> tag)
  return new Promise((resolve, reject) => {
    function check() {
      if (window.cv && typeof window.cv.Mat !== 'undefined') return resolve(window.cv);
      if (Date.now() - start > timeout) return reject(new Error('OpenCV not found or timed out'));
      setTimeout(check, 100);
    }
    check();
  });
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

function drawLandmarks(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'rgba(0,200,150,0.9)';
  for (const lm of landmarks) {
    const x = lm.x * canvas.width;
    const y = lm.y * canvas.height;
    ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2); ctx.fill();
  }
}

function avgPoints(landmarks, idxs) {
  const p = idxs.reduce((acc, i) => ({ x: acc.x + landmarks[i].x, y: acc.y + landmarks[i].y }), { x: 0, y: 0 });
  return { x: p.x / idxs.length, y: p.y / idxs.length };
}

function rotationMatrixToEuler(r) {
  // r: Float64Array length 9 (row-major)
  const r00 = r[0], r01 = r[1], r02 = r[2];
  const r10 = r[3], r11 = r[4], r12 = r[5];
  const r20 = r[6], r21 = r[7], r22 = r[8];
  const sy = Math.sqrt(r00 * r00 + r10 * r10);
  const singular = sy < 1e-6;
  let x, y, z;
  if (!singular) {
    x = Math.atan2(r21, r22);
    y = Math.atan2(-r20, sy);
    z = Math.atan2(r10, r00);
  } else {
    x = Math.atan2(-r12, r11);
    y = Math.atan2(-r20, sy);
    z = 0;
  }
  // degrees
  return { pitch: x * 180 / Math.PI, yaw: y * 180 / Math.PI, roll: z * 180 / Math.PI };
}

function computePoseAndEyes(cv, landmarks) {
  // FaceMesh indices (typical): nose tip 1, chin 152, left eye corner 33, right eye corner 263, left mouth 61, right mouth 291
  const ids = [1, 152, 33, 263, 61, 291];
  const imagePts = [];
  for (const i of ids) {
    imagePts.push(landmarks[i].x * canvas.width);
    imagePts.push(landmarks[i].y * canvas.height);
  }

  const objectPts = [
    0.0, 0.0, 0.0,        // nose tip
    0.0, -330.0, -65.0,   // chin
    -225.0, 170.0, -135.0, // left eye corner
    225.0, 170.0, -135.0,  // right eye corner
    -150.0, -150.0, -125.0, // left mouth
    150.0, -150.0, -125.0   // right mouth
  ];

  const objMat = cv.matFromArray(6, 3, cv.CV_64F, objectPts);
  const imgMat = cv.matFromArray(6, 2, cv.CV_64F, imagePts);
  const focal = canvas.width; // approximation
  const cx = canvas.width / 2.0;
  const cy = canvas.height / 2.0;
  const camMat = cv.matFromArray(3, 3, cv.CV_64F, [focal, 0, cx, 0, focal, cy, 0, 0, 1]);
  const distCoeffs = new cv.Mat.zeros(4, 1, cv.CV_64F);

  const rvec = new cv.Mat();
  const tvec = new cv.Mat();
  const success = cv.solvePnP(objMat, imgMat, camMat, distCoeffs, rvec, tvec, false, cv.SOLVEPNP_ITERATIVE);

  let angles = null;
  if (success) {
    const rotMat = new cv.Mat();
    cv.Rodrigues(rvec, rotMat);
    // copy to Float64Array
    const arr = new Float64Array(rotMat.data64F);
    angles = rotationMatrixToEuler(arr);
    rotMat.delete();
  }

  // Eye centers: prefer iris landmarks (468-472 left, 473-477 right) when refineLandmarks=true
  const leftIrisIdxs = [468, 469, 470, 471, 472];
  const rightIrisIdxs = [473, 474, 475, 476, 477];
  let leftCenter, rightCenter;
  try {
    leftCenter = avgPoints(landmarks, leftIrisIdxs);
    rightCenter = avgPoints(landmarks, rightIrisIdxs);
  } catch (e) {
    // fallback to eye corners
    leftCenter = avgPoints(landmarks, [33, 133]);
    rightCenter = avgPoints(landmarks, [362, 263]);
  }

  const leftEyeCorner = avgPoints(landmarks, [33, 133]);
  const rightEyeCorner = avgPoints(landmarks, [362, 263]);

  const leftGaze = { x: (leftCenter.x - leftEyeCorner.x).toFixed(4), y: (leftCenter.y - leftEyeCorner.y).toFixed(4) };
  const rightGaze = { x: (rightCenter.x - rightEyeCorner.x).toFixed(4), y: (rightCenter.y - rightEyeCorner.y).toFixed(4) };

  // cleanup
  objMat.delete(); imgMat.delete(); camMat.delete(); distCoeffs.delete(); rvec.delete(); tvec.delete();

  return { angles, leftGaze, rightGaze };
}

function onResults(results) {
  if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) return;
  const landmarks = results.multiFaceLandmarks[0];
  drawLandmarks(landmarks);

  if (!window.cv || typeof window.cv.Mat === 'undefined') {
    headEl.textContent = 'OpenCV loading...';
    return;
  }

  try {
    const { angles, leftGaze, rightGaze } = computePoseAndEyes(window.cv, landmarks);
    if (angles) headEl.textContent = `pitch:${angles.pitch.toFixed(1)}°, yaw:${angles.yaw.toFixed(1)}°, roll:${angles.roll.toFixed(1)}°`;
    leftEyeEl.textContent = `${leftGaze.x}, ${leftGaze.y}`;
    rightEyeEl.textContent = `${rightGaze.x}, ${rightGaze.y}`;
  } catch (err) {
    console.error('compute error', err);
  }
}

// Initialize MediaPipe FaceMesh
const faceMesh = new FaceMesh({ locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}` });
faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
faceMesh.onResults(onResults);

async function init() {
  try {
    const cv = await loadOpenCV();
    console.log('OpenCV ready', !!cv && typeof cv.Mat !== 'undefined');
  } catch (err) {
    console.warn('OpenCV init warning:', err);
  }
  await startCamera();

  // Use a simple requestAnimationFrame loop to feed frames to MediaPipe.
  // If you want to use `@mediapipe/camera_utils`, install it and replace this loop.
  async function loop() { await faceMesh.send({ image: video }); requestAnimationFrame(loop); }
  loop();
}

init().catch(console.error);


// OpenCV + MediaPipe demo script
// Load MediaPipe FaceMesh dynamically (supports dev bundler import, with CDN fallback for static hosts)

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const headEl = document.getElementById('head-metrics');
const leftEyeEl = document.getElementById('left-eye');
const rightEyeEl = document.getElementById('right-eye');
const presenceEl = document.getElementById('presence');
const leftEyeStateEl = document.getElementById('left-eye-state');
const rightEyeStateEl = document.getElementById('right-eye-state');
const attentionEl = document.getElementById('attention');
const loaderEl = document.getElementById('loader');

function showLoader(text = 'Loading...') {
  if (!loaderEl) return;
  loaderEl.innerHTML = `<div class="box">${text}</div>`;
  loaderEl.classList.remove('hidden');
}

function hideLoader() {
  if (!loaderEl) return;
  loaderEl.classList.add('hidden');
}

// Hysteresis / smoothing counters
let consecutiveFacing = 0;
let consecutiveAway = 0;
let consecutiveEyeOpen = 0;
let consecutiveEyeClosed = 0;
const AWAY_THRESHOLD = 1; // frames to consider 'away' (1 = immediate)
const FACE_THRESHOLD = 3; // frames required to re-consider 'facing'
const EYE_OPEN_THRESHOLD = 2; // frames required to consider eyes open
const EYE_CLOSED_THRESHOLD = 2; // frames for closed

// Load OpenCV via the package's `loadOpenCV` helper and wait for initialization.
async function loadOpenCV(timeout = 20000) {
  const start = Date.now();
  try {
    const mod = await import('@opencvjs/web');
    if (mod && typeof mod.loadOpenCV === 'function') {
      const cvModule = await mod.loadOpenCV();
      if (cvModule) {
        window.cv = cvModule;
        if (typeof window.cv.Mat !== 'undefined') return window.cv;
      }
    }
  } catch (e) {
    console.warn('dynamic import/loadOpenCV failed:', e);
  }

  // Fallback: insert OpenCV CDN script (docs.opencv.org provides a UMD build exposing `cv`)
  if (!document.querySelector('script[data-opencv-cdn]')) {
    const s = document.createElement('script');
    s.setAttribute('data-opencv-cdn', '1');
    s.src = 'https://docs.opencv.org/4.7.0/opencv.js';
    s.async = true;
    s.onload = () => console.log('OpenCV CDN script loaded');
    s.onerror = () => console.warn('Failed to load OpenCV from CDN');
    document.head.appendChild(s);
  }

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

// EAR (eye aspect ratio) helper for blink detection
function distance(a, b) {
  const dx = (a.x - b.x) * canvas.width;
  const dy = (a.y - b.y) * canvas.height;
  return Math.hypot(dx, dy);
}

function computeEAR(landmarks, indices) {
  // indices: [p1, p2, p3, p4, p5, p6] following common EAR formula
  const p1 = landmarks[indices[0]];
  const p2 = landmarks[indices[1]];
  const p3 = landmarks[indices[2]];
  const p4 = landmarks[indices[3]];
  const p5 = landmarks[indices[4]];
  const p6 = landmarks[indices[5]];
  const A = distance(p2, p6);
  const B = distance(p3, p5);
  const C = distance(p1, p4);
  if (C === 0) return 0;
  return (A + B) / (2.0 * C);
}

// smoothing buffers
const LEFT_EAR_BUFFER = [];
const RIGHT_EAR_BUFFER = [];
const EAR_BUFFER_SIZE = 5;


function rotationMatrixToEuler(r) {
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
  return { pitch: x * 180 / Math.PI, yaw: y * 180 / Math.PI, roll: z * 180 / Math.PI };
}

function normalizeAngleDeg(a) {
  // Map angles near +-180 to a small equivalent angle (handles gimbal flips)
  if (a <= -90) return a + 180;
  if (a >= 90) return a - 180;
  return a;
}

function computePoseAndEyes(cv, landmarks) {
  const ids = [1, 152, 33, 263, 61, 291];
  const imagePts = [];
  for (const i of ids) {
    imagePts.push(landmarks[i].x * canvas.width);
    imagePts.push(landmarks[i].y * canvas.height);
  }

  const objectPts = [
    0.0, 0.0, 0.0,
    0.0, -330.0, -65.0,
    -225.0, 170.0, -135.0,
    225.0, 170.0, -135.0,
    -150.0, -150.0, -125.0,
    150.0, -150.0, -125.0
  ];

  const objMat = cv.matFromArray(6, 3, cv.CV_64F, objectPts);
  const imgMat = cv.matFromArray(6, 2, cv.CV_64F, imagePts);
  const focal = canvas.width;
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
    const arr = new Float64Array(rotMat.data64F);
    angles = rotationMatrixToEuler(arr);
    // normalize extreme angles (fix 180deg flips)
    angles.norm = {
      pitch: normalizeAngleDeg(angles.pitch),
      yaw: normalizeAngleDeg(angles.yaw),
      roll: normalizeAngleDeg(angles.roll)
    };
    rotMat.delete();
  }

  const leftIrisIdxs = [468, 469, 470, 471, 472];
  const rightIrisIdxs = [473, 474, 475, 476, 477];
  let leftCenter, rightCenter;
  try {
    leftCenter = avgPoints(landmarks, leftIrisIdxs);
    rightCenter = avgPoints(landmarks, rightIrisIdxs);
  } catch (e) {
    leftCenter = avgPoints(landmarks, [33, 133]);
    rightCenter = avgPoints(landmarks, [362, 263]);
  }

  const leftEyeCorner = avgPoints(landmarks, [33, 133]);
  const rightEyeCorner = avgPoints(landmarks, [362, 263]);

  const leftGaze = { x: (leftCenter.x - leftEyeCorner.x).toFixed(4), y: (leftCenter.y - leftEyeCorner.y).toFixed(4) };
  const rightGaze = { x: (rightCenter.x - rightEyeCorner.x).toFixed(4), y: (rightCenter.y - rightEyeCorner.y).toFixed(4) };

  // Compute EAR for left and right eyes using MediaPipe landmark indices
  const leftEAR = computeEAR(landmarks, [33, 160, 158, 133, 153, 144]);
  const rightEAR = computeEAR(landmarks, [263, 387, 385, 362, 380, 373]);

  // update buffers
  LEFT_EAR_BUFFER.push(leftEAR);
  RIGHT_EAR_BUFFER.push(rightEAR);
  if (LEFT_EAR_BUFFER.length > EAR_BUFFER_SIZE) LEFT_EAR_BUFFER.shift();
  if (RIGHT_EAR_BUFFER.length > EAR_BUFFER_SIZE) RIGHT_EAR_BUFFER.shift();

  const avgLeftEAR = LEFT_EAR_BUFFER.length ? LEFT_EAR_BUFFER.reduce((a,b)=>a+b,0)/LEFT_EAR_BUFFER.length : 0;
  const avgRightEAR = RIGHT_EAR_BUFFER.length ? RIGHT_EAR_BUFFER.reduce((a,b)=>a+b,0)/RIGHT_EAR_BUFFER.length : 0;

  // eye open threshold (tuneable)
  const EAR_THRESH = 0.22;
  const leftOpen = avgLeftEAR > EAR_THRESH;
  const rightOpen = avgRightEAR > EAR_THRESH;

  // Presence: compute bounding box area of landmarks to detect when face removed
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y > maxY) maxY = lm.y;
  }
  const bboxW = (maxX - minX) * canvas.width;
  const bboxH = (maxY - minY) * canvas.height;
  const bboxAreaRel = (bboxW * bboxH) / (canvas.width * canvas.height);
  const PRESENCE_AREA_THRESH = 0.003; // tuneable
  const present = bboxAreaRel > PRESENCE_AREA_THRESH;

  // Attention logic: attentive only when present, facing forward, and at least one eye open.
  const YAW_THRESH = 30; // degrees (relaxed)
  const PITCH_THRESH = 30; // degrees (relaxed)
  const a = angles && angles.norm ? angles.norm : angles;
  const facingForward = a ? (Math.abs(a.yaw) < YAW_THRESH && Math.abs(a.pitch) < PITCH_THRESH) : false;
  const eyesClosed = !(leftOpen || rightOpen);
  const attentive = present && facingForward && !eyesClosed;

  objMat.delete(); imgMat.delete(); camMat.delete(); distCoeffs.delete(); rvec.delete(); tvec.delete();

  return { angles, leftGaze, rightGaze, avgLeftEAR, avgRightEAR, attentive, facingForward, eyesClosed, present };
}

// Promise resolver for first processed frame (used to hide loader)
let firstFrameResolve = null;
const firstFramePromise = new Promise((resolve) => { firstFrameResolve = resolve; });

function onResults(results) {
  // resolve first-frame promise if present
  if (typeof firstFrameResolve === 'function') {
    firstFrameResolve();
    firstFrameResolve = null;
  }
  if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
    // No face detected — clear metrics
    presenceEl.textContent = 'Absent';
    headEl.textContent = '—';
    leftEyeEl.textContent = '—';
    rightEyeEl.textContent = '—';
    leftEyeStateEl.textContent = '—';
    rightEyeStateEl.textContent = '—';
    attentionEl.textContent = '—';
    // clear canvas
    ctx.clearRect(0,0,canvas.width,canvas.height);
    return;
  }

  const landmarks = results.multiFaceLandmarks[0];
  drawLandmarks(landmarks);

  if (!window.cv || typeof window.cv.Mat === 'undefined') {
    headEl.textContent = 'OpenCV loading...';
    presenceEl.textContent = 'Present (no OpenCV)';
    return;
  }

  try {
    const { angles, leftGaze, rightGaze, avgLeftEAR, avgRightEAR, attentive, facingForward, eyesClosed, present } = computePoseAndEyes(window.cv, landmarks);
    presenceEl.textContent = 'Present';
    if (angles) headEl.textContent = `pitch:${angles.pitch.toFixed(1)}° (${angles.norm ? angles.norm.pitch.toFixed(1) : 'na'}), yaw:${angles.yaw.toFixed(1)}° (${angles.norm ? angles.norm.yaw.toFixed(1) : 'na'})`;
    leftEyeEl.textContent = `${leftGaze.x}, ${leftGaze.y}`;
    rightEyeEl.textContent = `${rightGaze.x}, ${rightGaze.y}`;
    // update hysteresis counters
    if (facingForward) { consecutiveFacing++; consecutiveAway = 0; } else { consecutiveAway++; consecutiveFacing = 0; }
    if (!eyesClosed) { consecutiveEyeOpen++; consecutiveEyeClosed = 0; } else { consecutiveEyeClosed++; consecutiveEyeOpen = 0; }

    // Show eye state only after buffer has accumulated
    leftEyeStateEl.textContent = LEFT_EAR_BUFFER.length >= 3 ? (avgLeftEAR > 0.22 ? 'Open' : 'Closed') : '—';
    rightEyeStateEl.textContent = RIGHT_EAR_BUFFER.length >= 3 ? (avgRightEAR > 0.22 ? 'Open' : 'Closed') : '—';

    // Determine final attentive state with hysteresis:
    // - If user turns head away for AWAY_THRESHOLD frames => immediately not attentive
    // - To become attentive again require FACE_THRESHOLD consecutive facing frames and eyes open for EYE_OPEN_THRESHOLD frames
    let attentiveFinal = false;
    if (!present) {
      attentiveFinal = false;
    } else if (consecutiveAway >= AWAY_THRESHOLD) {
      attentiveFinal = false;
    } else {
      attentiveFinal = (consecutiveFacing >= FACE_THRESHOLD) && (consecutiveEyeOpen >= EYE_OPEN_THRESHOLD);
    }

    attentionEl.textContent = attentiveFinal ? 'Attentive' : 'Not attentive';
  } catch (err) {
    console.error('compute error', err);
  }
}

let faceMesh = null;

async function ensureFaceMeshModule() {
  // Try bundler-resolved package first; if that fails (browser can't resolve bare specifier), fall back to CDN
  try {
    return await import('@mediapipe/face_mesh');
  } catch (err) {
    // Browser / static host won't resolve bare specifiers — import from jsDelivr instead
    const cdn = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js';
    try {
      return await import(/* @vite-ignore */ cdn);
    } catch (e) {
      // As a last resort, inject the script tag and rely on the global
      if (!document.querySelector('script[data-mediapipe-face_mesh]')) {
        const s = document.createElement('script');
        s.setAttribute('data-mediapipe-face_mesh', '1');
        s.src = cdn;
        s.async = true;
        document.head.appendChild(s);
      }
      // Return an object-like placeholder; the constructor may be on window later
      return {};
    }
  }
}

async function createFaceMesh() {
  if (faceMesh) return faceMesh;
  const mod = await ensureFaceMeshModule();
  let FaceMeshCtor = mod.FaceMesh || mod.default?.FaceMesh || mod.default || mod;
  if (typeof FaceMeshCtor !== 'function') {
    // try global value (common for UMD builds served via CDN)
    FaceMeshCtor = window.FaceMesh || window.MPFaceMesh || FaceMeshCtor;
  }
  if (typeof FaceMeshCtor !== 'function') {
    throw new Error('FaceMesh constructor not found after dynamic import and CDN fallback');
  }
  faceMesh = new FaceMeshCtor({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}` });
  faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
  faceMesh.onResults(onResults);
  return faceMesh;
}

async function init() {
  try {
    const cv = await loadOpenCV();
    console.log('OpenCV ready', !!cv && typeof cv.Mat !== 'undefined');
  } catch (err) {
    console.warn('OpenCV init warning:', err);
  }
  // show loader and progress
  showLoader('Loading OpenCV and MediaPipe...');
  await startCamera();
  showLoader('Starting video processing...');

  // ensure FaceMesh module is ready and bound
  try {
    await createFaceMesh();
  } catch (err) {
    console.warn('FaceMesh init warning:', err);
  }

  // feed frames and wait for first processed result or timeout
  async function loop() { if (faceMesh) await faceMesh.send({ image: video }); requestAnimationFrame(loop); }
  loop();

  // wait for first frame processed (or 5s timeout)
  await Promise.race([firstFramePromise, new Promise(r => setTimeout(r, 5000))]);
  hideLoader();
}

init().catch(console.error);

// OpenCV + TensorFlow demo script
// This file mirrors the MediaPipe demo behavior but uses TensorFlow's
// face-landmarks-detection model to get 3D landmarks directly.

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

// (debug panel removed)

function showLoader(text = 'Loading...') {
  if (!loaderEl) return;
  loaderEl.innerHTML = `<div class="box">${text}</div>`;
  loaderEl.classList.remove('hidden');
}

// processing canvas to stabilize input passed to the detector
const procCanvas = document.createElement('canvas');
const procCtx = procCanvas.getContext && procCanvas.getContext('2d');
procCanvas.style.display = 'none';
document.body && document.body.appendChild(procCanvas);

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

// Auto-calibration settings and globals (will be tuned at runtime)
const CALIBRATE_FRAMES = 80; // number of frames to gather baseline
let calibCount = 0;
let calibSumLeftEAR = 0;
let calibSumRightEAR = 0;
let calibSumYaw = 0;
let calibSumPitch = 0;
let calibSumYawSq = 0;
let calibSumPitchSq = 0;
let calibrated = false;

// Toggle auto-calibration: set to false to keep manual thresholds in effect
let AUTO_CALIBRATE = false;

// runtime thresholds (adjusted after calibration)
let EAR_THRESH = 0.22;
let YAW_THRESH = 70;
let PITCH_THRESH = 70;

// Performance tuning
const FRAME_SKIP = 2; // process model every Nth frame (1 = every frame)
let frameCounter = 0;
const MAX_PROC_DIM = 1024; // downscale longest side for model input

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
    // suppressed logging
  }

  // Fallback: insert OpenCV CDN script (docs.opencv.org provides a UMD build exposing `cv`)
  if (!document.querySelector('script[data-opencv-cdn]')) {
    const s = document.createElement('script');
    s.setAttribute('data-opencv-cdn', '1');
    s.src = 'https://docs.opencv.org/4.7.0/opencv.js';
    s.async = true;
    s.onload = () => {};
    s.onerror = () => {};
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
    const x = lm.x;
    const y = lm.y;
    ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2); ctx.fill();
  }
}

function avgPoints(landmarks, idxs) {
  const p = idxs.reduce((acc, i) => ({ x: acc.x + (landmarks[i].x || 0), y: acc.y + (landmarks[i].y || 0), z: acc.z + (landmarks[i].z || 0) }), { x: 0, y: 0, z: 0 });
  return { x: p.x / idxs.length, y: p.y / idxs.length, z: p.z / idxs.length };
}

// EAR (eye aspect ratio) helper for blink detection
function distance(a, b) {
  const dx = (a.x - b.x);
  const dy = (a.y - b.y);
  return Math.hypot(dx, dy);
}

function computeEAR(landmarks, indices) {
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
  if (a <= -90) return a + 180;
  if (a >= 90) return a - 180;
  return a;
}

function computePoseAndEyes(landmarks) {
  // Use model-provided 3D landmarks (pixels) to compute head orientation without
  // relying on hand-crafted 3D object points. We'll compute approximate pitch/yaw/roll
  // from 3D vectors between key landmarks.

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

  const leftGaze = { x: (leftCenter.x - leftEyeCorner.x).toFixed(2), y: (leftCenter.y - leftEyeCorner.y).toFixed(2) };
  const rightGaze = { x: (rightCenter.x - rightEyeCorner.x).toFixed(2), y: (rightCenter.y - rightEyeCorner.y).toFixed(2) };

  const leftEAR = computeEAR(landmarks, [33, 160, 158, 133, 153, 144]);
  const rightEAR = computeEAR(landmarks, [263, 387, 385, 362, 380, 373]);

  LEFT_EAR_BUFFER.push(leftEAR);
  RIGHT_EAR_BUFFER.push(rightEAR);
  if (LEFT_EAR_BUFFER.length > EAR_BUFFER_SIZE) LEFT_EAR_BUFFER.shift();
  if (RIGHT_EAR_BUFFER.length > EAR_BUFFER_SIZE) RIGHT_EAR_BUFFER.shift();

  const avgLeftEAR = LEFT_EAR_BUFFER.length ? LEFT_EAR_BUFFER.reduce((a,b)=>a+b,0)/LEFT_EAR_BUFFER.length : 0;
  const avgRightEAR = RIGHT_EAR_BUFFER.length ? RIGHT_EAR_BUFFER.reduce((a,b)=>a+b,0)/RIGHT_EAR_BUFFER.length : 0;

  const leftOpen = avgLeftEAR > EAR_THRESH;
  const rightOpen = avgRightEAR > EAR_THRESH;

  // Presence: compute bounding box area of landmarks to detect when face removed
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y > maxY) maxY = lm.y;
  }
  const bboxW = (maxX - minX);
  const bboxH = (maxY - minY);
  const bboxAreaRel = (bboxW * bboxH) / (canvas.width * canvas.height);
  const PRESENCE_AREA_THRESH = 0.003; // tuneable
  const present = bboxAreaRel > PRESENCE_AREA_THRESH;

  // Compute 3D orientation from landmarks: use vector from mid-eyes to nose tip.
  const noseIdx = 1; // mediapipe nose tip
  const nose = landmarks[noseIdx];
  const midEye = { x: (leftCenter.x + rightCenter.x) / 2, y: (leftCenter.y + rightCenter.y) / 2, z: (leftCenter.z + rightCenter.z) / 2 };
  const vec = { x: nose.x - midEye.x, y: nose.y - midEye.y, z: (nose.z || 0) - (midEye.z || 0) };

  // yaw: left/right rotation, pitch: up/down, roll: tilt
  const yaw = Math.atan2(vec.x, vec.z) * 180 / Math.PI;
  const pitch = Math.atan2(-vec.y, vec.z) * 180 / Math.PI;
  const eyeVec = { x: rightCenter.x - leftCenter.x, y: rightCenter.y - leftCenter.y };
  const roll = Math.atan2(eyeVec.y, eyeVec.x) * 180 / Math.PI;

  const angles = { pitch, yaw, roll, norm: { pitch: normalizeAngleDeg(pitch), yaw: normalizeAngleDeg(yaw), roll: normalizeAngleDeg(roll) } };

  const a = angles && angles.norm ? angles.norm : angles;
  const angleOk = a ? (Math.abs(a.yaw) < YAW_THRESH && Math.abs(a.pitch) < PITCH_THRESH) : false;

  // Pixel-based checks: nose center and gaze should also be near-center to be considered facing
  let pixelOk = true;
  try {
    const bboxCenterX = (minX + maxX) / 2.0;
    const bboxWpx = Math.max(1, (maxX - minX));
    const noseDx = nose.x - bboxCenterX;
    const NOSE_CENTER_REL = 0.12; // 12% of bbox width
    if (Math.abs(noseDx) > NOSE_CENTER_REL * bboxWpx) pixelOk = false;
    const gazeThresh = 0.12 * bboxWpx;
    if (Math.abs(leftGaze.x) > gazeThresh || Math.abs(rightGaze.x) > gazeThresh) pixelOk = false;
  } catch (e) {
    pixelOk = true;
  }

  // Require both angle and pixel checks when angles exist; otherwise use pixel check alone
  let facingForward = (angles ? (angleOk && pixelOk) : pixelOk);
  const eyesClosed = !(leftOpen || rightOpen);

  // Fallback: if solvePnP failed (no `angles`), estimate facing from pixel offsets
  if (!angles) {
    try {
      const bboxWpx = Math.max(1, bboxW);
      const bboxHpx = Math.max(1, bboxH);
      const nosePx = { x: nose.x, y: nose.y };
      const midEyePx = { x: midEye.x, y: midEye.y };
      const dx = nosePx.x - midEyePx.x;
      const dy = nosePx.y - midEyePx.y;
      // tighten fallback thresholds so small head rotations count as away
      const X_REL = 0.08 * bboxWpx;
      const Y_REL = 0.08 * bboxHpx;
      if (Math.abs(dx) < X_REL && Math.abs(dy) < Y_REL) facingForward = true;
    } catch (e) {
      // ignore fallback errors
    }
  }

  const attentive = present && facingForward && !eyesClosed;

  return { angles, leftGaze, rightGaze, avgLeftEAR, avgRightEAR, attentive, facingForward, eyesClosed, present };
}

// Promise resolver for first processed frame (used to hide loader)
let firstFrameResolve = null;
const firstFramePromise = new Promise((resolve) => { firstFrameResolve = resolve; });

function onPrediction(landmarks) {
  if (typeof firstFrameResolve === 'function') {
    firstFrameResolve();
    firstFrameResolve = null;
  }
  if (!landmarks || landmarks.length === 0) {
    presenceEl.textContent = 'Absent';
    headEl.textContent = '—';
    leftEyeEl.textContent = '—';
    rightEyeEl.textContent = '—';
    leftEyeStateEl.textContent = '—';
    rightEyeStateEl.textContent = '—';
    attentionEl.textContent = '—';
    ctx.clearRect(0,0,canvas.width,canvas.height);
    return;
  }

  drawLandmarks(landmarks);

  // drawn landmarks, computing pose

  if (!window.cv || typeof window.cv.Mat === 'undefined') {
    headEl.textContent = 'OpenCV loading...';
    presenceEl.textContent = 'Present (no OpenCV)';
    return;
  }

  try {
    const { angles, leftGaze, rightGaze, avgLeftEAR, avgRightEAR, attentive, facingForward, eyesClosed, present } = computePoseAndEyes(landmarks);
    // Auto-calibration: gather baseline values for EAR and angles while user looks straight
    if (!calibrated && present && typeof avgLeftEAR === 'number' && typeof avgRightEAR === 'number') {
      calibCount++;
      calibSumLeftEAR += avgLeftEAR;
      calibSumRightEAR += avgRightEAR;
      if (angles && angles.yaw != null && angles.pitch != null) {
        const yawVal = angles.yaw;
        const pitchVal = angles.pitch;
        calibSumYaw += yawVal;
        calibSumPitch += pitchVal;
        calibSumYawSq += yawVal * yawVal;
        calibSumPitchSq += pitchVal * pitchVal;
      }

      // update loader message while calibrating
      if (loaderEl) loaderEl.innerHTML = `<div class="box">Calibrating (${calibCount}/${CALIBRATE_FRAMES})...</div>`;

      if (calibCount >= CALIBRATE_FRAMES) {
        const meanLeft = calibSumLeftEAR / calibCount;
        const meanRight = calibSumRightEAR / calibCount;
        const meanEAR = (meanLeft + meanRight) / 2;
        const meanYaw = calibSumYaw / calibCount;
        const meanPitch = calibSumPitch / calibCount;
        const varYaw = Math.max(0, (calibSumYawSq / calibCount) - (meanYaw * meanYaw));
        const varPitch = Math.max(0, (calibSumPitchSq / calibCount) - (meanPitch * meanPitch));
        const stdYaw = Math.sqrt(varYaw);
        const stdPitch = Math.sqrt(varPitch);

        // Set thresholds conservatively based on observed baseline and noise
        EAR_THRESH = Math.min(0.28, Math.max(0.12, meanEAR * 0.68));
        YAW_THRESH = Math.max(15, Math.abs(meanYaw) + Math.max(12, stdYaw * 3));
        PITCH_THRESH = Math.max(15, Math.abs(meanPitch) + Math.max(12, stdPitch * 3));
        calibrated = true;
        if (loaderEl) loaderEl.classList.add('hidden');
      }
    }
    presenceEl.textContent = 'Present';
    if (angles) headEl.textContent = `pitch:${angles.pitch.toFixed(1)}° (${angles.norm ? angles.norm.pitch.toFixed(1) : 'na'}), yaw:${angles.yaw.toFixed(1)}° (${angles.norm ? angles.norm.yaw.toFixed(1) : 'na'})`;
    leftEyeEl.textContent = `${leftGaze.x}, ${leftGaze.y}`;
    rightEyeEl.textContent = `${rightGaze.x}, ${rightGaze.y}`;
    if (facingForward) { consecutiveFacing++; consecutiveAway = 0; } else { consecutiveAway++; consecutiveFacing = 0; }
    if (!eyesClosed) { consecutiveEyeOpen++; consecutiveEyeClosed = 0; } else { consecutiveEyeClosed++; consecutiveEyeOpen = 0; }

    leftEyeStateEl.textContent = LEFT_EAR_BUFFER.length >= 3 ? (avgLeftEAR > EAR_THRESH ? 'Open' : 'Closed') : '—';
    rightEyeStateEl.textContent = RIGHT_EAR_BUFFER.length >= 3 ? (avgRightEAR > EAR_THRESH ? 'Open' : 'Closed') : '—';

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
    // suppressed compute error logging
  }
}

// Initialize TensorFlow face-landmarks model and run loop
async function init() {
  try {
    await loadOpenCV();
    } catch (err) {
    // suppressed OpenCV init warning
  }
  showLoader('Loading OpenCV and TensorFlow...');
  await startCamera();
  showLoader('Starting detection...');

  // Load TF and model
    let model = null;
    let modelApi = null;
    let noFaceFrames = 0;
    let lastLandmarks = null;
    let lastLandmarksAge = 0;
    const LANDMARKS_CACHE_MAX_AGE = 1; // reuse last landmarks for up to N frames when detector misses
  try {
    const tf = await import('@tensorflow/tfjs');
    // register and load the WebGL backend package, then set backend on tf
    await import('@tensorflow/tfjs-backend-webgl');
    await tf.setBackend('webgl');
    await tf.ready();
    const faceLandmarksModule = await import('@tensorflow-models/face-landmarks-detection');
    const faceLandmarks = faceLandmarksModule.default || faceLandmarksModule;
    // Try legacy `load` API first, then fallback to the newer `createDetector` API.
      // let modelApi = null; // Removed inner redeclaration
    if (typeof faceLandmarks.load === 'function') {
      const pkg = (faceLandmarks.SupportedPackages && faceLandmarks.SupportedPackages.tfjs)
        ? faceLandmarks.SupportedPackages.tfjs
        : 'tfjs';
      model = await faceLandmarks.load(pkg, { maxFaces: 1, refineLandmarks: true });
      modelApi = 'legacy';
    } else if (typeof faceLandmarks.createDetector === 'function') {
      // new API: createDetector(SupportedModels.MediaPipeFaceMesh, config)
      const modelName = (faceLandmarks.SupportedModels && faceLandmarks.SupportedModels.MediaPipeFaceMesh)
        ? faceLandmarks.SupportedModels.MediaPipeFaceMesh
        : Object.values(faceLandmarks.SupportedModels || {})[0];
      const detectorConfig = { runtime: 'tfjs', refineLandmarks: true, maxFaces: 1 };
      model = await faceLandmarks.createDetector(modelName, detectorConfig);
      modelApi = 'detector';
    } else {
      throw new Error('face-landmarks-detection API not supported');
    }
    // model loaded
  } catch (e) {
    // suppressed model load error
  }

  async function loop() {
    if (model) {
      try {
        // frame processing
        // video readiness diagnostics
        // draw current video frame into processing canvas (stabilize input)
        if (procCanvas && procCtx && video.videoWidth && video.videoHeight) {
          // downscale the processing canvas to reduce model input size while preserving aspect
          const vw = video.videoWidth;
          const vh = video.videoHeight;
          const longest = Math.max(vw, vh);
          const scale = longest > MAX_PROC_DIM ? (MAX_PROC_DIM / longest) : 1;
          const pw = Math.max(1, Math.round(vw * scale));
          const ph = Math.max(1, Math.round(vh * scale));
          if (procCanvas.width !== pw || procCanvas.height !== ph) {
            procCanvas.width = pw;
            procCanvas.height = ph;
          }
          try { procCtx.drawImage(video, 0, 0, procCanvas.width, procCanvas.height); } catch (e) { /* ignore draw errors */ }
        }
        frameCounter++;
        let preds = null;
        // skip heavy model inference on some frames to reduce load
        const shouldRunModel = (frameCounter % FRAME_SKIP) === 0;
        // pass processing canvas to estimator for consistent input
        const inputForModel = (procCanvas && procCanvas.width) ? procCanvas : video;
        if (shouldRunModel) {
          if (modelApi === 'legacy' && typeof model.estimateFaces === 'function') {
            preds = await model.estimateFaces({ input: inputForModel, returnTensors: false, predictIrises: true });
          } else if (modelApi === 'detector' && typeof model.estimateFaces === 'function') {
            preds = await model.estimateFaces(inputForModel);
          }
        }

        // prediction result
        if (preds && preds.length > 0) {
          // reset noFaceFrames when we get a detection
          noFaceFrames = 0;
          // log brief summary of first face
          const f0 = preds[0];
          const kpCount = (f0.scaledMesh && f0.scaledMesh.length) || (f0.mesh && f0.mesh.length) || (f0.keypoints && f0.keypoints.length) || 0;
          // face summary
          const face = preds[0];
          let scaled = null;
          if (face.scaledMesh) scaled = face.scaledMesh;
          else if (face.mesh) scaled = face.mesh;
          else if (face.keypoints && face.keypoints.length) scaled = face.keypoints.map(k => [k.x, k.y, k.z || 0]);
          else if (face.keypoints3D) scaled = face.keypoints3D;

          if (scaled && scaled.length) {
            // normalize coords: model may return pixels or normalized [0..1]
            const landmarks = scaled.map(p => {
              let x = p[0]; let y = p[1]; let z = p[2] || 0;
              // model may return normalized coords or pixel coords relative to procCanvas
              if (x <= 1) x = x * procCanvas.width;
              if (y <= 1) y = y * procCanvas.height;
              // rescale landmarks from procCanvas space to overlay canvas space
              const sx = canvas.width / procCanvas.width;
              const sy = canvas.height / procCanvas.height;
              return { x: x * sx, y: y * sy, z: z * ((sx + sy) / 2) };
            });
            // cache landmarks to reduce flicker when detector briefly misses
            lastLandmarks = landmarks;
            lastLandmarksAge = 0;

            // deliver landmarks to UI/metrics
            onPrediction(landmarks);
          }
        } else {
          // no detection this frame
          noFaceFrames++;
          // reuse last landmarks for a few frames to reduce flicker
          if (lastLandmarks && lastLandmarksAge < LANDMARKS_CACHE_MAX_AGE) {
            lastLandmarksAge++;
            onPrediction(lastLandmarks);
          } else {
            // try ImageBitmap fallback occasionally when detector misses repeatedly
            if (noFaceFrames > 10 && procCanvas && procCanvas.width) {
              try {
                const bmp = await createImageBitmap(procCanvas);
                let fbPreds = null;
                if (modelApi === 'legacy' && typeof model.estimateFaces === 'function') {
                  fbPreds = await model.estimateFaces({ input: bmp, returnTensors: false, predictIrises: true });
                } else if (modelApi === 'detector' && typeof model.estimateFaces === 'function') {
                  fbPreds = await model.estimateFaces(bmp);
                }
                if (fbPreds && fbPreds.length > 0) {
                  noFaceFrames = 0;
                  const face = fbPreds[0];
                  let scaled = face.scaledMesh || face.mesh || (face.keypoints && face.keypoints.map(k => [k.x, k.y, k.z || 0])) || face.keypoints3D;
                  if (scaled && scaled.length) {
                    const landmarks = scaled.map(p => {
                      let x = p[0]; let y = p[1]; let z = p[2] || 0;
                      if (x <= 1) x = x * canvas.width;
                      if (y <= 1) y = y * canvas.height;
                      return { x, y, z };
                    });
                    lastLandmarks = landmarks; lastLandmarksAge = 0;
                    onPrediction(landmarks);
                  }
                } else {
                  onPrediction(null);
                }
              } catch (e) {
                onPrediction(null);
              }
            } else {
              onPrediction(null);
            }
          }
        }
      } catch (err) {
        // suppressed frame-processing error
      }

      // schedule next frame
      requestAnimationFrame(loop);
    } else {
      // model not loaded yet — try again later
      requestAnimationFrame(loop);
    }
  }

  // start loop
  requestAnimationFrame(loop);

  // hide loader after first detected frame (or after a short timeout)
  Promise.race([firstFramePromise, new Promise((r) => setTimeout(r, 5000))]).then(() => hideLoader());

}

// kick off initialization
init().catch(() => { /* suppressed init errors */ });

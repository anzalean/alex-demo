// OpenCV + TensorFlow placeholder script
import('@tensorflow/tfjs').then(tf => {
  console.log('tfjs loaded', !!tf && tf.version);
}).catch(err => console.warn('tfjs load failed', err));

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
  statusEl.textContent = 'Camera started';
}

startCamera().catch(err => { console.error(err); statusEl.textContent = 'Camera error'; });

// TODO: integrate TensorFlow model inference on video frames, using OpenCV for preprocessing if needed.

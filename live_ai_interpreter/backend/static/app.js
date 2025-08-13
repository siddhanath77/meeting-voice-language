const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const muteBtn = document.getElementById('muteBtn');
const clearBtn = document.getElementById('clearBtn');
const saveBtn = document.getElementById('saveBtn');
const asrPartial = document.getElementById('asrPartial');
const asrFinal = document.getElementById('asrFinal');
const mtPartial = document.getElementById('mtPartial');
const mtFinal = document.getElementById('mtFinal');
const player = document.getElementById('player');
const sampleRateSel = document.getElementById('sampleRate');
const targetLangSel = document.getElementById('targetLang');
const roomIdInput = document.getElementById('roomId');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const latencyEl = document.getElementById('latency');
let historyFetched = false;

let mediaStream = null;
let audioCtx = null;
let processor = null;
let source = null;
let ws = null;
let recording = false;
let lastPingTs = null;
let muted = false;

const WS_URL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/audio';

function int16BufferFromFloat32(input) {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    let s = Math.max(-1, Math.min(1, input[i]));
    if (muted) s = 0;
    output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return output.buffer;
}

function playMp3Chunk(bytes) {
  const blob = new Blob([bytes], { type: 'audio/mpeg' });
  const url = URL.createObjectURL(blob);
  player.src = url;
}

function handleWsMessage(ev) {
  if (typeof ev.data === 'string') {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'pong' && msg.ts_ms) {
        const now = Date.now();
        const rtt = Math.max(0, now - msg.ts_ms);
        latencyEl.textContent = rtt.toString();
        return;
      }
      if (msg.type === 'warning') {
        statusText.textContent = msg.message || 'Warning';
        return;
      }
      if (msg.type === 'history' && Array.isArray(msg.items)) {
        if (msg.items.length > 0) {
          const last = msg.items[msg.items.length - 1];
          asrFinal.textContent = last.text || '';
          mtFinal.textContent = last.translation || '';
        }
        return;
      }
      if (msg.type === 'partial') {
        asrPartial.textContent = msg.text || '';
        mtPartial.textContent = msg.translation || '';
      } else if (msg.type === 'final') {
        asrFinal.textContent = msg.text || '';
        mtFinal.textContent = msg.translation || '';
      }
    } catch (_) {}
  } else if (ev.data instanceof Blob) {
    // MP3 chunk
    ev.data.arrayBuffer().then(playMp3Chunk);
  }
}

async function start() {
  if (recording) return;
  recording = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;

  const sampleRate = parseInt(sampleRateSel.value, 10);
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate, channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }, video: false });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
  source = audioCtx.createMediaStreamSource(mediaStream);
  // smaller buffer => lower latency
  processor = audioCtx.createScriptProcessor(1024, 1, 1);
  source.connect(processor);
  processor.connect(audioCtx.destination);

  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = handleWsMessage;
  ws.onerror = () => { statusText.textContent = 'Error'; };
  ws.onopen = () => {
    const roomId = (roomIdInput.value || 'default').trim();
    ws.send(JSON.stringify({ type: 'config', sampleRate, frameSize: 320, targetLang: targetLangSel.value, roomId }));
    statusDot.classList.remove('offline');
    statusDot.classList.add('online');
    statusText.textContent = 'Connected';
    // start heartbeat
    const hb = setInterval(() => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify({ type: 'heartbeat', ts_ms: Date.now() }));
    }, 1000);
    ws._hb = hb;

    if (!historyFetched) {
      ws.send(JSON.stringify({ type: 'history_request' }));
      historyFetched = true;
    }
  };

  processor.onaudioprocess = (e) => {
    if (!recording || ws.readyState !== WebSocket.OPEN) return;
    const inBuf = e.inputBuffer.getChannelData(0);
    // update simple level meter
    const rms = Math.sqrt(inBuf.reduce((a, b) => a + b * b, 0) / inBuf.length);
    const pct = Math.min(100, Math.max(0, Math.round(rms * 120)));
    const meter = document.getElementById('meterFill');
    if (meter) meter.style.width = `${pct}%`;
    const ab = int16BufferFromFloat32(inBuf);
    ws.send(ab);
  };

  ws.onclose = () => {
    if (ws && ws._hb) clearInterval(ws._hb);
    statusDot.classList.remove('online');
    statusDot.classList.add('offline');
    statusText.textContent = 'Disconnected';
  };
}

function stop() {
  recording = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;

  if (processor) processor.disconnect();
  if (source) source.disconnect();
  if (audioCtx) audioCtx.close();
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
}

muteBtn.addEventListener('click', () => {
  muted = !muted;
  muteBtn.textContent = muted ? 'Unmute' : 'Mute';
});

clearBtn.addEventListener('click', () => {
  asrPartial.textContent = '';
  asrFinal.textContent = '';
  mtPartial.textContent = '';
  mtFinal.textContent = '';
});

saveBtn.addEventListener('click', () => {
  const text = `ASR (partial): ${asrPartial.textContent}\nASR (final): ${asrFinal.textContent}\nMT (partial): ${mtPartial.textContent}\nMT (final): ${mtFinal.textContent}\n`;
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `transcript_${Date.now()}.txt`;
  a.click();
  URL.revokeObjectURL(url);
});

startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', stop);



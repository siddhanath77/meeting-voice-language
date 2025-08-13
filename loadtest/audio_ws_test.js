import ws from 'k6/ws';
import { check, sleep } from 'k6';

export const options = {
  vus: 50,
  duration: '60s',
};

const HEADER_SIZE = 12; // uint32 + float64

function makeFrame(seq) {
  const ts = Date.now();
  const header = new ArrayBuffer(HEADER_SIZE);
  const view = new DataView(header);
  view.setUint32(0, seq, true);
  // float64 little-endian
  // JS lacks setFloat64 in some runtimes used by k6, so write via two Uint32
  const buf = new ArrayBuffer(8);
  const dv = new DataView(buf);
  dv.setFloat64(0, ts, true);
  view.setUint32(4, dv.getUint32(0, true), true);
  view.setUint32(8, dv.getUint32(4, true), true);

  // 10ms 48kHz mono s16le -> 480 samples -> 960 bytes
  const pcm = new Int16Array(480);
  // simple sine for test
  for (let i = 0; i < pcm.length; i++) {
    pcm[i] = Math.floor(1000 * Math.sin((2 * Math.PI * i) / pcm.length));
  }

  const payload = new Uint8Array(HEADER_SIZE + pcm.byteLength);
  payload.set(new Uint8Array(header), 0);
  payload.set(new Uint8Array(pcm.buffer), HEADER_SIZE);
  return payload.buffer;
}

export default function () {
  const url = __ENV.WS_URL || 'ws://localhost:5000/ws-audio';
  const params = { tags: { my_tag: 'websocket' } };

  ws.connect(url, params, function (socket) {
    socket.on('open', function () {
      socket.send(JSON.stringify({ type: 'config', sampleRate: 48000, channelCount: 1, frameSize: 480, format: 's16le' }));
      let seq = 0;
      const interval = setInterval(() => {
        socket.sendBinary(makeFrame(seq++));
      }, 10);

      socket.setInterval(function () {
        // keep alive
      }, 1000);

      socket.on('message', function () {});

      socket.on('close', function () {
        clearInterval(interval);
      });
    });

    socket.on('error', function () {});

    sleep(10);
  });
}



/**
 * Audio Recording and Waveform Helpers for SETH Web TUI.
 */

export const WAVEFORM_FRAMES = [
  '▁▂▃▅▇▅▃▂▁▂▃▅',
  '▂▃▅▇▅▃▂▁▂▃▅▇',
  '▃▅▇▅▃▂▁▂▃▅▇▅',
  '▅▇▅▃▂▁▂▃▅▇▅▃',
  '▇▅▃▂▁▂▃▅▇▅▃▂'
];

export class AudioRecorderManager {
  constructor() {
    this.mediaStream = null;
    this.mediaRecorder = null;
    this.recordedChunks = [];
    this.isRecording = false;
    this.seconds = 0;
    this.timer = null;
    this.waveIdx = 0;
    this.waveTimer = null;
  }

  static isSupported() {
    return Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
  }

  static getSupportedMimeType() {
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4'
    ];
    return candidates.find(t => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || '';
  }

  async start(onTick) {
    if (this.isRecording) return;
    this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = AudioRecorderManager.getSupportedMimeType();
    this.mediaRecorder = mimeType
      ? new MediaRecorder(this.mediaStream, { mimeType })
      : new MediaRecorder(this.mediaStream);

    this.recordedChunks = [];
    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        this.recordedChunks.push(e.data);
      }
    };

    this.mediaRecorder.start();
    this.isRecording = true;
    this.seconds = 0;
    this.waveIdx = 0;

    this.timer = setInterval(() => {
      this.seconds++;
      if (onTick) onTick({ seconds: this.seconds, waveFrame: WAVEFORM_FRAMES[this.waveIdx] });
    }, 1000);

    this.waveTimer = setInterval(() => {
      this.waveIdx = (this.waveIdx + 1) % WAVEFORM_FRAMES.length;
      if (onTick) onTick({ seconds: this.seconds, waveFrame: WAVEFORM_FRAMES[this.waveIdx] });
    }, 200);
  }

  async stop() {
    if (!this.isRecording) return null;

    clearInterval(this.timer);
    clearInterval(this.waveTimer);
    this.isRecording = false;

    const mins = Math.floor(this.seconds / 60);
    const secs = this.seconds % 60;
    const duration = `${mins}:${secs.toString().padStart(2, '0')}`;

    const blob = await new Promise((resolve) => {
      if (!this.mediaRecorder) {
        resolve(null);
        return;
      }
      this.mediaRecorder.onstop = () => {
        const mime = this.mediaRecorder.mimeType || 'audio/webm';
        resolve(new Blob(this.recordedChunks, { type: mime }));
      };
      this.mediaRecorder.stop();
    });

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    return { blob, duration };
  }
}

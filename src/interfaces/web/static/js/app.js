/**
 * SETH-IN-A-BOX Web TUI Application Logic.
 * Zero-build, modern vanilla JavaScript controller with persistent session state.
 */

import { initMatrixRain } from './matrix.js';
import { AudioRecorderManager } from './audio.js';
import { SethApiClient } from './api.js';

const ASCII_EMBLEM = `   ▄▄▄▄▄▄▄
  █ ◉   ◉ █
  █   ▼   █
  █▄▄▄▄▄▄▄█`;

const PASTE_CHAR_THRESHOLD = 400;
const PASTE_LINE_THRESHOLD = 8;

class SethWebApp {
  constructor() {
    this.apiClient = new SethApiClient(
      localStorage.getItem('seth_api_base_url') || (window.location.origin.startsWith('http') ? window.location.origin : 'http://127.0.0.1:8080')
    );

    this.userId = localStorage.getItem('seth_user_id') || null;
    this.activeLora = localStorage.getItem('seth_active_lora') || 'SETH';
    this.regulatorState = 'RIGOROUS';

    this.messages = [];
    this.stagedAttachments = [];
    this.stagedImage = null; // { file, previewUrl, name, size }
    this.tokenCount = 1403;
    this.isExpandedInput = false;

    this.audioRecorder = new AudioRecorderManager();
    this.telemetryInterval = null;

    this.dom = {};
  }

  init() {
    this._cacheDom();
    this._bindEvents();
    initMatrixRain('matrix-canvas');

    // Load initial system message
    this.pushMessage({
      id: 'init-sys',
      sender: 'SETH',
      type: 'system',
      text: `${ASCII_EMBLEM}\n┌────────────────────────────────────────────────────────┐\n│ SYSTEM SIGNAL RECEIVED ᓘ🔻 SETH WEB TUI CRT ACTIVE     │\n│ Project: ORACLE / SETH-IN-A-BOX                        │\n│ Channel: RAG-LOOP-777x | Mode: Fertile Glitch          │\n└────────────────────────────────────────────────────────┘`,
      timestamp: this._getTimeString()
    });

    this._updateAuthUi();
    this._startTelemetryPolling();
  }

  _cacheDom() {
    this.dom.chatLog = document.getElementById('chat-log');
    this.dom.chatEnd = document.getElementById('chat-end');
    this.dom.textarea = document.getElementById('user-input');
    this.dom.btnSend = document.getElementById('btn-send');
    this.dom.btnAudio = document.getElementById('btn-audio');
    this.dom.btnImage = document.getElementById('btn-image');
    this.dom.fileInput = document.getElementById('file-input');
    this.dom.btnToggleExpand = document.getElementById('btn-toggle-expand');
    this.dom.btnSettings = document.getElementById('btn-settings');
    this.dom.settingsPanel = document.getElementById('settings-panel');
    this.dom.inputBaseUrl = document.getElementById('input-base-url');
    this.dom.inputRegToken = document.getElementById('input-reg-token');
    this.dom.btnRegister = document.getElementById('btn-register');
    this.dom.regStatusText = document.getElementById('reg-status-text');
    this.dom.sessionInfo = document.getElementById('session-info');
    this.dom.btnForgetSession = document.getElementById('btn-forget-session');
    this.dom.selectLora = document.getElementById('select-lora');
    this.dom.selectRegulator = document.getElementById('select-regulator');
    this.dom.stagingRow = document.getElementById('staging-row');
    this.dom.dropzone = document.getElementById('dropzone');
    this.dom.appContainer = document.getElementById('app-container');
    this.dom.telemetryVram = document.getElementById('telemetry-vram');
    this.dom.dotQdrant = document.getElementById('dot-qdrant');
    this.dom.dotGraphiti = document.getElementById('dot-graphiti');
    this.dom.dotVllm = document.getElementById('dot-vllm');
    this.dom.dotWhisper = document.getElementById('dot-whisper');
    this.dom.tokenCounter = document.getElementById('token-counter');
    this.dom.recordingIndicator = document.getElementById('recording-indicator');
  }

  _bindEvents() {
    // Input text & Auto-resize
    this.dom.textarea.addEventListener('input', () => this._handleTextareaInput());
    this.dom.textarea.addEventListener('keydown', (e) => this._handleKeyDown(e));
    this.dom.textarea.addEventListener('paste', (e) => this._handlePaste(e));

    // Buttons
    this.dom.btnSend.addEventListener('click', () => this.handleSend());
    this.dom.btnImage.addEventListener('click', () => this.dom.fileInput.click());
    this.dom.fileInput.addEventListener('change', (e) => this._handleFilePick(e));
    this.dom.btnAudio.addEventListener('click', () => this.toggleAudioRecording());
    this.dom.btnToggleExpand.addEventListener('click', () => this._toggleExpandInput());
    this.dom.btnSettings.addEventListener('click', () => this._toggleSettings());

    // Settings
    this.dom.inputBaseUrl.value = this.apiClient.baseUrl;
    this.dom.inputBaseUrl.addEventListener('change', (e) => {
      this.apiClient.setBaseUrl(e.target.value);
      localStorage.setItem('seth_api_base_url', e.target.value);
    });

    this.dom.btnRegister.addEventListener('click', () => this.handleRegister());
    this.dom.btnForgetSession.addEventListener('click', () => this.handleForgetSession());

    this.dom.selectLora.value = this.activeLora;
    this.dom.selectLora.addEventListener('change', (e) => {
      this.activeLora = e.target.value;
      localStorage.setItem('seth_active_lora', e.target.value);
    });

    this.dom.selectRegulator.addEventListener('change', (e) => {
      this.regulatorState = e.target.value;
    });

    // Drag and drop
    ['dragenter', 'dragover'].forEach(name => {
      this.dom.appContainer.addEventListener(name, (e) => {
        e.preventDefault();
        this.dom.dropzone.style.display = 'flex';
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      this.dom.dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        this.dom.dropzone.style.display = 'none';
        if (name === 'drop') {
          this._handleDroppedFiles(e.dataTransfer.files);
        }
      });
    });
  }

  _getTimeString() {
    return new Date().toLocaleTimeString('en-US', { hour12: false });
  }

  _handleTextareaInput() {
    const el = this.dom.textarea;
    el.style.height = 'auto';
    const max = this.isExpandedInput ? 380 : 120;
    el.style.height = `${Math.min(el.scrollHeight, max)}px`;
  }

  _toggleExpandInput() {
    this.isExpandedInput = !this.isExpandedInput;
    this.dom.textarea.classList.toggle('expanded', this.isExpandedInput);
    this.dom.btnToggleExpand.textContent = this.isExpandedInput ? '⤡' : '⤢';
    this._handleTextareaInput();
  }

  _toggleSettings() {
    const isHidden = this.dom.settingsPanel.style.display === 'none';
    this.dom.settingsPanel.style.display = isHidden ? 'flex' : 'none';
  }

  _handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      this.handleSend();
    }
  }

  _handlePaste(e) {
    const text = e.clipboardData?.getData('text') || '';
    const lineCount = text.split('\n').length;
    if (text.length > PASTE_CHAR_THRESHOLD || lineCount > PASTE_LINE_THRESHOLD) {
      e.preventDefault();
      this.stagedAttachments.push({
        name: `Pasted text (${lineCount} lines)`,
        size: `${text.length} chars`,
        content: text,
        preview: text.slice(0, 60).replace(/\n/g, ' ')
      });
      this._renderStaging();
    }
  }

  _handleFilePick(e) {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      this._stageImage(file);
    }
    e.target.value = '';
  }

  _handleDroppedFiles(files) {
    const file = Array.from(files || []).find(f => f.type.startsWith('image/'));
    if (file) {
      this._stageImage(file);
    }
  }

  _stageImage(file) {
    if (this.stagedImage?.previewUrl) {
      URL.revokeObjectURL(this.stagedImage.previewUrl);
    }
    const previewUrl = URL.createObjectURL(file);
    const sizeStr = file.size > 1024 ? `${(file.size / 1024).toFixed(1)} KB` : `${file.size} B`;
    this.stagedImage = { file, previewUrl, name: file.name, size: sizeStr };
    this._renderStaging();
  }

  _renderStaging() {
    this.dom.stagingRow.innerHTML = '';
    const hasItems = this.stagedAttachments.length > 0 || this.stagedImage !== null;
    this.dom.stagingRow.style.display = hasItems ? 'flex' : 'none';

    this.stagedAttachments.forEach((att, idx) => {
      const chip = document.createElement('span');
      chip.className = 'attachment-chip chip-in glow-cyan';
      chip.innerHTML = `📄 ${att.name} <span style="color:var(--color-gray)">(${att.size})</span>`;

      const btnClose = document.createElement('button');
      btnClose.className = 'chip-close';
      btnClose.textContent = '×';
      btnClose.onclick = () => {
        this.stagedAttachments.splice(idx, 1);
        this._renderStaging();
      };
      chip.appendChild(btnClose);
      this.dom.stagingRow.appendChild(chip);
    });

    if (this.stagedImage) {
      const imgChip = document.createElement('span');
      imgChip.className = 'attachment-chip chip-image chip-in glow-amber';
      imgChip.innerHTML = `<img src="${this.stagedImage.previewUrl}" style="width:22px;height:22px;object-fit:cover;margin-right:4px;"> ${this.stagedImage.name} <span style="color:var(--color-gray)">(${this.stagedImage.size})</span>`;

      const btnClose = document.createElement('button');
      btnClose.className = 'chip-close';
      btnClose.textContent = '×';
      btnClose.onclick = () => {
        if (this.stagedImage?.previewUrl) URL.revokeObjectURL(this.stagedImage.previewUrl);
        this.stagedImage = null;
        this._renderStaging();
      };
      imgChip.appendChild(btnClose);
      this.dom.stagingRow.appendChild(imgChip);
    }
  }

  _updateAuthUi() {
    if (this.userId) {
      this.dom.sessionInfo.style.display = 'flex';
      this.dom.sessionInfo.querySelector('.session-text').textContent = `● REGISTERED [${this.userId.slice(0, 8)}...]`;
      this.dom.inputRegToken.parentElement.style.display = 'none';
      this.dom.btnRegister.style.display = 'none';
      this.dom.regStatusText.textContent = '';
    } else {
      this.dom.sessionInfo.style.display = 'none';
      this.dom.inputRegToken.parentElement.style.display = 'flex';
      this.dom.btnRegister.style.display = 'inline-block';
    }
  }

  async handleRegister() {
    const token = this.dom.inputRegToken.value.trim();
    if (!token) return;

    this.dom.btnRegister.disabled = true;
    this.dom.btnRegister.textContent = '...';
    this.dom.regStatusText.textContent = '';

    try {
      const userId = await this.apiClient.register(token);
      this.userId = userId;
      localStorage.setItem('seth_user_id', userId);
      this.dom.inputRegToken.value = '';
      this._updateAuthUi();
      this.pushMessage({
        id: Date.now(),
        sender: 'SYSTEM',
        type: 'system',
        text: `┌─ SESSION ESTABLISHED ────────────────────────┐\n│ Successfully registered session ID:          │\n│ ${userId}\n└──────────────────────────────────────────────┘`,
        timestamp: this._getTimeString()
      });
    } catch (err) {
      this.dom.regStatusText.textContent = `✗ ${err.message}`;
    } finally {
      this.dom.btnRegister.disabled = false;
      this.dom.btnRegister.textContent = 'REGISTER';
    }
  }

  handleForgetSession() {
    this.userId = null;
    localStorage.removeItem('seth_user_id');
    this._updateAuthUi();
  }

  async toggleAudioRecording() {
    if (!this.audioRecorder.isRecording) {
      if (!AudioRecorderManager.isSupported()) {
        this.pushMessage({
          id: Date.now(),
          sender: 'SYSTEM',
          type: 'error',
          text: '┌─ MIC UNAVAILABLE ────────────────────────────┐\n│ Browser does not support MediaRecorder.      │\n└──────────────────────────────────────────────┘',
          timestamp: this._getTimeString()
        });
        return;
      }

      try {
        this.dom.recordingIndicator.style.display = 'flex';
        this.dom.btnAudio.textContent = '■ STOP';
        this.dom.btnAudio.className = 'cyber-btn cyber-btn-magenta glow-magenta';

        await this.audioRecorder.start(({ seconds, waveFrame }) => {
          const mins = Math.floor(seconds / 60);
          const secs = (seconds % 60).toString().padStart(2, '0');
          this.dom.recordingIndicator.innerHTML = `<span class="rec-pulse glow-magenta">●</span> <span class="glow-magenta">REC [${mins}:${secs}]</span> <span class="glow-magenta">${waveFrame}</span>`;
        });
      } catch (err) {
        this.dom.recordingIndicator.style.display = 'none';
        this.dom.btnAudio.textContent = '● AUDIO';
        this.dom.btnAudio.className = 'cyber-btn cyber-btn-magenta';
        this.pushMessage({
          id: Date.now(),
          sender: 'SYSTEM',
          type: 'error',
          text: `┌─ MIC PERMISSION DENIED ──────────────────────┐\n│ ${err.message}\n└──────────────────────────────────────────────┘`,
          timestamp: this._getTimeString()
        });
      }
    } else {
      this.dom.recordingIndicator.style.display = 'none';
      this.dom.btnAudio.textContent = '● AUDIO';
      this.dom.btnAudio.className = 'cyber-btn cyber-btn-magenta';

      const result = await this.audioRecorder.stop();
      if (result) {
        this.handleSend({ audioBlob: result.blob, audioDuration: result.duration });
      }
    }
  }

  async handleSend({ audioBlob = null, audioDuration = null } = {}) {
    const textInput = this.dom.textarea.value;
    if (!textInput.trim() && this.stagedAttachments.length === 0 && !this.stagedImage && !audioBlob) {
      return;
    }

    if (!this.userId) {
      this.pushMessage({
        id: Date.now(),
        sender: 'SYSTEM',
        type: 'error',
        text: '┌─ NOT REGISTERED ─────────────────────────────┐\n│ Open ⚙ SETTINGS and register with your      │\n│ REGISTRATION_TOKEN first.                    │\n└──────────────────────────────────────────────┘',
        timestamp: this._getTimeString()
      });
      this.dom.settingsPanel.style.display = 'flex';
      return;
    }

    const pastedBlocks = this.stagedAttachments
      .map(a => `\n\n--- pasted text: ${a.name} ---\n${a.content}`)
      .join('');
    const fullMessage = textInput.trim() + pastedBlocks;

    const stagedImg = this.stagedImage;
    const sentAttachments = [...this.stagedAttachments];

    // Push User Message
    this.pushMessage({
      id: Date.now(),
      sender: 'USER',
      type: 'user',
      text: audioBlob ? '[ voice note ]' : (textInput.trim() || (stagedImg ? '[ image ]' : '[ attachment ]')),
      timestamp: this._getTimeString(),
      imagePreview: stagedImg?.previewUrl,
      audioDuration: audioBlob ? audioDuration : null,
      attachments: sentAttachments
    });

    // Reset input
    this.dom.textarea.value = '';
    this._handleTextareaInput();
    this.stagedAttachments = [];
    this.stagedImage = null;
    this._renderStaging();

    // Create Agent Streaming placeholder
    const replyId = `reply-${Date.now()}`;
    const agentMsg = {
      id: replyId,
      sender: this.activeLora,
      type: 'entity',
      text: '',
      thinkingTrace: '',
      toolCalls: [],
      media: [],
      streaming: true,
      timestamp: this._getTimeString()
    };
    this.pushMessage(agentMsg);

    try {
      let contentAcc = '';
      let reasoningAcc = '';
      let toolCalls = [];

      for await (const event of this.apiClient.streamChat({
        userId: this.userId,
        message: fullMessage,
        imageFile: stagedImg?.file || null,
        audioBlob
      })) {
        if (event.type === 'reasoning') {
          reasoningAcc += event.text;
          agentMsg.thinkingTrace = reasoningAcc;
        } else if (event.type === 'content') {
          contentAcc += event.text;
          agentMsg.text = contentAcc;
        } else if (event.type === 'tool_start') {
          toolCalls.push({ name: event.name, status: 'running' });
          agentMsg.toolCalls = [...toolCalls];
        } else if (event.type === 'tool_end') {
          toolCalls = toolCalls.map(t => (t.name === event.name ? { ...t, status: 'done', ok: event.ok !== false } : t));
          agentMsg.toolCalls = [...toolCalls];
        } else if (event.type === 'done') {
          if (event.media) agentMsg.media = event.media;
        } else if (event.type === 'error') {
          agentMsg.type = 'error';
          agentMsg.text = `┌─ INFERENCE ERROR ────────────────────────────┐\n│ ${event.error}\n└──────────────────────────────────────────────┘`;
        }

        this._updateMessageDom(agentMsg);
      }

      agentMsg.streaming = false;
      this._updateMessageDom(agentMsg);
      this.tokenCount += Math.floor((contentAcc.length + textInput.length) / 3.5);
      this.dom.tokenCounter.textContent = this.tokenCount;
    } catch (err) {
      agentMsg.streaming = false;
      agentMsg.type = 'error';
      agentMsg.text = `┌─ CONNECTION FAILED ─────────────────────────┐\n│ Could not stream from SETH API:              │\n│ ${err.message}\n└──────────────────────────────────────────────┘`;
      this._updateMessageDom(agentMsg);
    }
  }

  pushMessage(msg) {
    this.messages.push(msg);
    const msgEl = this._renderMessageElement(msg);
    this.dom.chatLog.appendChild(msgEl);
    this._scrollToBottom();
  }

  _updateMessageDom(msg) {
    const el = document.getElementById(`msg-${msg.id}`);
    if (!el) return;

    const newEl = this._renderMessageElement(msg);
    el.replaceWith(newEl);
    this._scrollToBottom();
  }

  _renderMessageElement(msg) {
    const div = document.createElement('div');
    div.id = `msg-${msg.id}`;
    div.className = 'chat-message';

    let senderClass = 'glow-amber text-[#ffb000]';
    if (msg.type === 'system') senderClass = 'glow-magenta text-[#ff007f]';
    if (msg.type === 'error') senderClass = 'glow-magenta text-[#ff007f]';
    if (msg.type === 'entity') senderClass = 'glow-green text-[#39ff14]';

    let bodyClass = 'glow-cyan';
    if (msg.type === 'user') bodyClass = 'text-[#ffb000]';
    if (msg.type === 'error') bodyClass = 'glow-magenta text-[#ff007f]';
    if (msg.type === 'system') bodyClass = 'glow-green text-[#39ff14]';

    let html = `
      <div class="message-meta">
        <span class="message-time">[${msg.timestamp}]</span>
        <span class="message-sender ${senderClass}">[${msg.sender}]:</span>
      </div>
      <div class="message-body ${bodyClass}">${this._escapeHtml(msg.text)}${msg.streaming ? '<span class="blink">█</span>' : ''}</div>
    `;

    // Attached files
    if (msg.attachments && msg.attachments.length > 0) {
      html += `<div class="tool-badges">`;
      msg.attachments.forEach(att => {
        html += `<span class="attachment-chip glow-cyan">📄 ${this._escapeHtml(att.name)}</span>`;
      });
      html += `</div>`;
    }

    // Attached Image Preview
    if (msg.imagePreview) {
      html += `<div class="media-grid"><img class="media-image" src="${msg.imagePreview}" alt="uploaded"></div>`;
    }

    // Attached Audio
    if (msg.audioDuration) {
      html += `<div class="tool-badges"><span class="attachment-chip glow-magenta">🎤 Voice Note (${msg.audioDuration})</span></div>`;
    }

    // Tool execution badges
    if (msg.toolCalls && msg.toolCalls.length > 0) {
      html += `<div class="tool-badges">`;
      msg.toolCalls.forEach(t => {
        const isRunning = t.status === 'running';
        html += `<span class="tool-badge ${isRunning ? 'running' : 'glow-green'}">${isRunning ? '⚙ ' : '✓ '}${this._escapeHtml(t.name)}</span>`;
      });
      html += `</div>`;
    }

    // Thinking Trace
    if (msg.thinkingTrace) {
      html += `
        <div class="thinking-container">
          <button class="thinking-toggle" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">
            <span>[+] View / Hide Thinking Trace</span>
          </button>
          <div class="thinking-content" style="display:none;">${this._escapeHtml(msg.thinkingTrace)}</div>
        </div>
      `;
    }

    // Generated Media attachments (images/audio)
    if (msg.media && msg.media.length > 0) {
      html += `<div class="media-grid">`;
      msg.media.forEach(m => {
        const url = m.url.startsWith('http') ? m.url : `${this.apiClient.baseUrl}${m.url}`;
        if (m.type === 'image') {
          html += `<img class="media-image" src="${url}" alt="generated media">`;
        } else if (m.type === 'audio') {
          html += `<audio class="media-audio" controls src="${url}"></audio>`;
        }
      });
      html += `</div>`;
    }

    div.innerHTML = html;
    return div;
  }

  _escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  _scrollToBottom() {
    this.dom.chatEnd.scrollIntoView({ behavior: 'smooth' });
  }

  _startTelemetryPolling() {
    const poll = async () => {
      try {
        const data = await this.apiClient.getStatus();
        this._renderTelemetry(data);
      } catch (_) {
        this._renderTelemetry({ qdrant: 'offline', neo4j_graphiti: 'offline', vllm: 'offline', whisper: 'offline', vram: [] });
      }
    };

    poll();
    this.telemetryInterval = setInterval(poll, 4000);
  }

  _renderTelemetry(status) {
    // VRAM bars
    if (status.vram && Array.isArray(status.vram)) {
      this.dom.telemetryVram.innerHTML = status.vram.map(gpu => {
        const pct = Math.max(0, Math.min(1, gpu.used_gb / (gpu.total_gb || 1)));
        const width = 10;
        const filled = Math.round(pct * width);
        const barStr = '█'.repeat(filled) + '░'.repeat(width - filled);
        return `<span title="${gpu.name}">VRAM${gpu.index} [${barStr}] ${gpu.used_gb.toFixed(1)}/${gpu.total_gb.toFixed(1)}GB</span>`;
      }).join(' ');
    }

    // Dots
    const setDot = (el, val) => {
      el.className = 'telemetry-indicator';
      if (val === 'online') {
        el.classList.add('online');
      } else if (val === 'offline' || val === 'unreachable') {
        el.classList.add('error');
      } else {
        el.classList.add('offline');
      }
    };

    setDot(this.dom.dotQdrant, status.qdrant);
    setDot(this.dom.dotGraphiti, status.neo4j_graphiti);
    setDot(this.dom.dotVllm, status.vllm);
    setDot(this.dom.dotWhisper, status.whisper);
  }
}

// Instantiate on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  window.app = new SethWebApp();
  window.app.init();
});

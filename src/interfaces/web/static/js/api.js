/**
 * SETH-IN-A-BOX Client API for Web TUI.
 * Handles SSE streaming, registration, and status polling.
 */

export class SethApiClient {
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  setBaseUrl(url) {
    this.baseUrl = url.replace(/\/$/, '');
  }

  async getStatus() {
    const url = `${this.baseUrl}/api/status`;
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    return await resp.json();
  }

  async register(token) {
    const url = `${this.baseUrl}/api/register`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: token.trim() })
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.detail || `HTTP ${resp.status}`);
    }
    return data.user_id;
  }

  async *streamChat({ userId, message = '', imageFile = null, audioBlob = null }) {
    const url = `${this.baseUrl}/api/chat`;
    const form = new FormData();
    form.append('message', message);

    if (imageFile) {
      form.append('image', imageFile);
    }

    if (audioBlob) {
      const ext = (audioBlob.type.split('/')[1] || 'webm').split(';')[0];
      form.append('audio', audioBlob, `recording.${ext}`);
    }

    const resp = await fetch(url, {
      method: 'POST',
      headers: {
        'X-Seth-User': userId
      },
      body: form
    });

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${resp.status} — ${resp.statusText}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        const dataStr = trimmed.slice(5).trim();
        if (dataStr === '[DONE]') {
          yield { type: 'done' };
          continue;
        }

        try {
          const payload = JSON.parse(dataStr);

          // 1. Tool execution events
          if (payload.seth_event) {
            yield payload.seth_event;
            continue;
          }

          // 2. Choice delta tokens (content & reasoning)
          const choice = payload.choices?.[0] || {};
          const delta = choice.delta || {};

          if (delta.reasoning) {
            yield { type: 'reasoning', text: delta.reasoning };
          }
          if (delta.content) {
            yield { type: 'content', text: delta.content };
          }

          // 3. Finish / Meta / Media attachments
          if (choice.finish_reason === 'error') {
            yield { type: 'error', error: payload.seth_meta?.error || 'Inference error' };
          } else if (choice.finish_reason === 'stop' && payload.seth_meta) {
            yield {
              type: 'done',
              media: payload.seth_meta.media || [],
              tool_calls_used: payload.seth_meta.tool_calls_used || []
            };
          }
        } catch (_) {
          // ignore malformed SSE line
        }
      }
    }
  }
}

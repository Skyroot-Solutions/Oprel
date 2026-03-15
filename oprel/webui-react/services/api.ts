/**
 * API Service for Oprel SDK
 * Mirrors and enhances logic from legacy webui/js/api.js
 */

export interface Model {
  model_id: string;
  name: string;
  size_gb: number;
  quantization: string;
  backend: string;
  loaded: boolean;
  status: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  last_updated: string;
  message_count: number;
  model_id: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string | any[];
}

export interface Metrics {
  cpu_usage: number;
  ram_total_gb: number;
  ram_used_gb: number;
  gpu_name: string | null;
  gpu_usage: number | null;
  vram_total_mb: number | null;
  vram_used_mb: number | null;
  generation_speed: number;
}

export interface UserSettings {
  temperature: number;
  top_p: number;
  top_k: number;
  repeat_penalty: number;
  max_tokens: number;
  system_instruction: string | null;
}

export interface UserProfile {
  name: string;
  role: string;
  initials?: string;
}

const API_BASE = (typeof window !== 'undefined' && window.location.port === '3000') 
? 'http://localhost:11435' 
: ''; 

export const API = {
  async fetchModels(): Promise<Model[]> {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) throw new Error('Failed to fetch models');
    return res.json();
  },

  async loadModel(modelId: string, params: any = {}): Promise<any> {
    const res = await fetch(`${API_BASE}/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId, ...params }),
    });
    if (!res.ok) throw new Error('Failed to load model');
    return res.json();
  },

  async unloadModel(modelId: string): Promise<any> {
    const res = await fetch(`${API_BASE}/unload/${encodeURIComponent(modelId)}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to unload model');
    return res.json();
  },

  async pullModel(modelId: string): Promise<any> {
    const res = await fetch(`${API_BASE}/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId }),
    });
    if (!res.ok) throw new Error('Failed to download model');
    return res.json();
  },

  async fetchConversations(): Promise<Conversation[]> {
    const res = await fetch(`${API_BASE}/conversations`);
    if (!res.ok) throw new Error('Failed to fetch conversations');
    return res.json();
  },

  async getConversation(id: string): Promise<ChatMessage[]> {
    const res = await fetch(`${API_BASE}/conversations/${id}`);
    if (!res.ok) throw new Error('Failed to load conversation');
    return res.json();
  },

  async deleteConversation(id: string): Promise<any> {
    const res = await fetch(`${API_BASE}/conversations/${id}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete conversation');
    return res.json();
  },

  async renameConversation(id: string, title: string): Promise<any> {
    const res = await fetch(`${API_BASE}/conversations/${id}/title`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
    if (!res.ok) throw new Error('Failed to rename conversation');
    return res.json();
  },

  async getMetrics(): Promise<Metrics> {
    const res = await fetch(`${API_BASE}/system/metrics`);
    if (!res.ok) throw new Error('Failed to fetch metrics');
    return res.json();
  },

  async fetchRegistryModels(): Promise<any> {
    const res = await fetch(`${API_BASE}/registry/models`);
    if (!res.ok) throw new Error('Failed to fetch registry models');
    return res.json();
  },

  async fetchUser(): Promise<UserProfile> {
    const res = await fetch(`${API_BASE}/user`);
    if (!res.ok) throw new Error('Failed to fetch user');
    return res.json();
  },

  async saveUser(name: string, role: string): Promise<UserProfile> {
    const res = await fetch(`${API_BASE}/user`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, role }),
    });
    if (!res.ok) throw new Error('Failed to save user');
    return res.json();
  },

  async fetchSettings(): Promise<UserSettings> {
    const res = await fetch(`${API_BASE}/user/settings`);
    if (!res.ok) throw new Error('Failed to fetch settings');
    return res.json();
  },

  async saveSettings(settings: UserSettings): Promise<UserSettings> {
    const res = await fetch(`${API_BASE}/user/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
    if (!res.ok) throw new Error('Failed to save settings');
    return res.json();
  },

  /**
   * Helper for SSE streaming
   */
  async chatCompletionStream(
    payload: any,
    onToken: (token: string) => void,
    onConversationId?: (id: string) => void
  ): Promise<void> {
    const { thinking, maxTokens, topP, topK, repeatPenalty, ...rest } = payload;
    
    // Map camelCase to snake_case for the backend
    const bodyData = { 
      ...rest, 
      stream: true, 
      thinking: Boolean(thinking),
      max_tokens: maxTokens,
      top_p: topP,
      top_k: topK,
      repeat_penalty: repeatPenalty,
    };
    
    const res = await fetch(`${API_BASE}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(bodyData),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      let msg = 'Streaming request failed';
      if (typeof err.detail === 'string') msg = err.detail;
      else if (Array.isArray(err.detail)) msg = err.detail.map((d: any) => d.msg).join(', ');
      else if (err.error) msg = err.error.message || err.error;
      throw new Error(msg);
    }

    const convId = res.headers.get('X-Conversation-ID');
    if (convId && onConversationId) {
      onConversationId(convId);
    }

    if (!res.body) return;

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        this._processLine(line, onToken);
      }
    }

    if (buffer.trim()) {
      this._processLine(buffer, onToken);
    }
  },

  _processLine(line: string, onToken: (token: string) => void) {
    const trimmed = line.trim();
    if (!trimmed || trimmed === 'data: [DONE]') return;
    
    if (trimmed.startsWith('data: ')) {
      const dataStr = trimmed.slice(6);
      
      if (dataStr.startsWith('[ERROR]')) {
        const errorMsg = dataStr.replace('[ERROR]', '').trim();
        throw new Error(errorMsg || 'Server-side generation error');
      }
      
      try {
        const json = JSON.parse(dataStr);
        const content = json.choices[0]?.delta?.content || '';
        if (content) onToken(content);
      } catch (e) {
        if (dataStr !== '[DONE]') {
          console.error('Error parsing SSE data:', e, line);
        }
      }
    }
  },
};

const API = {
    async fetchModels() {
        const res = await fetch('/models');
        if (!res.ok) throw new Error('Failed to fetch models');
        return res.json();
    },

    async loadModel(modelId, params = {}) {
        const res = await fetch('/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId, ...params })
        });
        if (!res.ok) throw new Error('Failed to load model');
        return res.json();
    },

    async unloadModel(modelId) {
        const res = await fetch(`/unload/${modelId}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Failed to unload model');
        return res.json();
    },

    async pullModel(modelId) {
        const res = await fetch('/pull', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId })
        });
        if (!res.ok) throw new Error('Failed to download model');
        return res.json();
    },

    async chatCompletion(payload) {
        const res = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error('Chat completion failed');
        return res;
    },

    /**
     * Helper for SSE streaming
     * @param {Object} payload 
     * @param {Function} onToken - Callback for each new text token
     */
    async chatCompletionStream(payload, onToken) {
        const bodyData = { ...payload, stream: true };
        const res = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(bodyData)
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
            let msg = 'Streaming request failed';
            if (typeof err.detail === 'string') msg = err.detail;
            else if (Array.isArray(err.detail)) msg = err.detail.map(d => d.msg).join(', ');
            else if (err.error) msg = err.error.message || err.error;
            throw new Error(msg);
        }

        const convId = res.headers.get('X-Conversation-ID');
        if (convId && payload.onConversationId) {
            payload.onConversationId(convId);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep the last partial line in buffer

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed || trimmed === 'data: [DONE]') continue;
                
                if (trimmed.startsWith('data: ')) {
                    const dataStr = trimmed.slice(6);
                    
                    // Handle server-side [ERROR] within the stream
                    if (dataStr.startsWith('[ERROR]')) {
                        const errorMsg = dataStr.replace('[ERROR]', '').trim();
                        throw new Error(errorMsg || 'Server-side generation error');
                    }
                    
                    try {
                        const json = JSON.parse(dataStr);
                        const content = json.choices[0]?.delta?.content || '';
                        if (content) onToken(content);
                    } catch (e) {
                        // Only log if it's not a known signal
                        if (dataStr !== '[DONE]') {
                            console.error('Error parsing SSE data:', e, trimmed);
                        }
                    }
                }
            }
        }
    },

    async fetchConversations() {
        const res = await fetch('/conversations');
        if (!res.ok) throw new Error('Failed to fetch conversations');
        return res.json();
    },

    async getConversation(id) {
        const res = await fetch(`/conversations/${id}`);
        if (!res.ok) throw new Error('Failed to load conversation');
        return res.json();
    },

    async renameConversation(id, title) {
        const res = await fetch(`/conversations/${id}/title`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title })
        });
        if (!res.ok) throw new Error('Failed to rename conversation');
        return res.json();
    },

    async deleteConversation(id) {
        const res = await fetch(`/conversations/${id}`, {
            method: 'DELETE'
        });
        if (!res.ok) throw new Error('Failed to delete conversation');
        return res.json();
    },

    async getMetrics() {
        const res = await fetch('/system/metrics');
        if (!res.ok) throw new Error('Failed to fetch metrics');
        return res.json();
    },

    async fetchRegistryModels() {
        const res = await fetch('/registry/models');
        if (!res.ok) throw new Error('Failed to fetch registry models');
        return res.json();
    },

    async fetchUser() {
        const res = await fetch('/user');
        if (!res.ok) throw new Error('Failed to fetch user');
        return res.json();
    },

    async saveUser(name, role) {
        const res = await fetch('/user', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, role })
        });
        if (!res.ok) throw new Error('Failed to save user');
        return res.json();
    }
};

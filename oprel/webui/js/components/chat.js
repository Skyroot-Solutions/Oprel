/**
 * Oprel Chat Module — Production-Grade Chat Interface
 *
 * Architecture:
 *   ChatPage
 *    ├── Header (model name, status, switch)
 *    ├── ChatContainer
 *    │     ├── Message
 *    │     │     ├── Avatar
 *    │     │     └── Content
 *    │     │            ├── MarkdownBlock
 *    │     │            ├── CodeCanvas
 *    │     │            ├── Table
 *    │     │            └── Callout
 *    │     └── Message ...
 *    ├── Actions (copy, retry, like, dislike, TPS)
 *    └── InputBar
 *
 * Uses MessageRenderer for AST-based rendering.
 */

class Chat {
    static state = {
        modelId: null,
        isGenerating: false,
        conversationId: null,
        history: [],
        messageCounter: 0,
        selectedImage: null,
        selectedImageName: null
    };

    static elements = {
        messages: null,
        input: null,
        sendBtn: null,
        emptyState: null,
        modelBtn: null
    };

    // ─── System Prompt ─────────────────────────────────────────────
    static SYSTEM_PROMPT = [
        'You are a helpful, expert AI assistant. Always format your responses in clean, structured Markdown.',
        '',
        'FORMATTING RULES (follow strictly):',
        '- Use ## or ### headings to organize long responses into clear sections.',
        '- Write short, focused paragraphs. Never write walls of text.',
        '- Use **bold** for key terms and `backticks` for code references inline.',
        '- Use bullet points (- item) or numbered lists (1. step) for multiple items or steps.',
        '- Always wrap code in fenced code blocks with the language tag:',
        '  ```python',
        '  code here',
        '  ```',
        '- Use tables for comparisons or structured data:',
        '  | Column | Column |',
        '  |--------|--------|',
        '  | data   | data   |',
        '- Use > blockquotes for notes or important callouts.',
        '- Keep responses concise and well-organized.',
        '- Reply in the same language the user uses.',
    ].join('\n');

    // ─── Init ──────────────────────────────────────────────────────
    static init() {
        this.elements.messages = document.getElementById('chat-scroll');
        this.elements.input = document.getElementById('chat-textarea');
        this.elements.sendBtn = document.getElementById('chat-send-btn');
        this.elements.modelBtn = document.getElementById('chat-model-name');

        // Initialize the renderer
        MessageRenderer.init();

        // Input handlers
        if (this.elements.input) {
            this.elements.input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.send();
                }
            });
            this.elements.input.addEventListener('input', () => {
                this.elements.input.style.height = 'auto';
                this.elements.input.style.height = Math.min(this.elements.input.scrollHeight, 200) + 'px';
            });
        }

        // Image input listener
        const imgInput = document.getElementById('image-input');
        if (imgInput) {
            imgInput.addEventListener('change', (e) => Chat.handleImageSelect(e));
        }
    }

    // ─── Load Conversation ─────────────────────────────────────────
    static async loadConversation(id) {
        if (typeof App !== 'undefined') App.switchView('chat');
        try {
            const history = await API.getConversation(id);
            this.state.conversationId = id;
            this.state.history = history;
            this._renderAll(history);
            
            // Persist
            localStorage.setItem('oprel_last_conversation_id', id);
        } catch (e) {
            console.error('Failed to load conversation:', e);
            localStorage.removeItem('oprel_last_conversation_id');
        }
    }

    static _renderAll(history) {
        const container = this.elements.messages;
        if (!container) return;
        container.innerHTML = '';
        history.forEach(msg => this.appendMessage(msg.role, msg.content));
    }

    // ─── New Conversation ──────────────────────────────────────────
    static newConversation() {
        if (typeof App !== 'undefined') App.switchView('chat');
        this.state.messageCounter = 0;
        this.state.isGenerating = false;
        
        localStorage.removeItem('oprel_last_conversation_id');
        
        const container = this.elements.messages;
        if (container) {
            container.innerHTML = '';
        }
        
        // Hide sidebar on mobile if needed
        if (window.innerWidth < 1024) {
            const sidebar = document.getElementById('sidebar');
            if (sidebar && !sidebar.classList.contains('-translate-x-full')) {
                // If you have a Sidebar.toggle method, you'd call it here
            }
        }
        this.clearImage();
    }

    // ─── Image Handling ──────────────────────────────────────────
    static handleImageSelect(e) {
        const file = e.target.files[0];
        if (!file) {
            console.warn("No file selected");
            return;
        }

        console.log("Image selected:", file.name);
        Chat.state.selectedImageName = file.name;
        
        const reader = new FileReader();
        reader.onload = (event) => {
            const dataUrl = event.target.result;
            Chat.state.selectedImage = dataUrl;
            
            // UI Update
            const container = document.getElementById('image-preview-container');
            const preview = document.getElementById('image-preview');
            const nameEl = document.getElementById('image-name');
            
            if (container && preview && nameEl) {
                console.log("Updating preview UI...");
                preview.src = dataUrl;
                nameEl.textContent = file.name;
                
                // Force visibility
                container.classList.remove('hidden');
                container.style.display = 'flex';
            } else {
                console.error("Preview elements not found: ", {container, preview, nameEl});
            }
        };
        reader.onerror = (err) => console.error("FileReader error:", err);
        reader.readAsDataURL(file);
    }

    static clearImage() {
        Chat.state.selectedImage = null;
        Chat.state.selectedImageName = null;
        const container = document.getElementById('image-preview-container');
        const input = document.getElementById('image-input');
        if (container) {
            container.classList.add('hidden');
            container.style.display = 'none';
        }
        if (input) input.value = '';
    }

    // ─── Message ID Generator ──────────────────────────────────────
    static _nextId() {
        return `msg_${++this.state.messageCounter}_${Date.now()}`;
    }

    // ─── Append Message (Core Renderer) ────────────────────────────
    static appendMessage(role, content, metrics = null) {
        const container = this.elements.messages;
        if (!container) return null;

        if (role === 'system') return null;

        const msgId = this._nextId();
        const isUser = role === 'user';
        const isLoading = !content && !isUser;

        const userInitials = (typeof App !== 'undefined' && App.state.user) ? App.state.user.initials : "TR";

        const msgEl = document.createElement('div');
        msgEl.id = msgId;
        msgEl.className = `oprel-message ${isUser ? 'oprel-msg-user' : 'oprel-msg-assistant'} fade-in`;
        msgEl.dataset.role = role;

        let renderedContent = '';
        let imageElement = '';

        if (isLoading) {
            renderedContent = this._loadingIndicator();
        } else if (isUser) {
            if (Array.isArray(content)) {
                content.forEach(part => {
                    if (part.type === 'image_url') {
                        imageElement = `<div class="oprel-vision-img-bubble"><img src="${part.image_url.url}"></div>`;
                    } else if (part.type === 'text') {
                        renderedContent = this._escapeUserMessage(part.text);
                    }
                });
            } else {
                renderedContent = this._escapeUserMessage(content);
            }
        } else {
            renderedContent = MessageRenderer.render(content);
        }

        msgEl.innerHTML = `
            <div class="oprel-msg-row ${isUser ? 'oprel-row-reverse' : ''}">
                <div class="oprel-avatar-container">
                    <div class="oprel-avatar ${isUser ? 'oprel-avatar-user' : 'oprel-avatar-ai p-1.5'}">
                        ${isUser
                            ? userInitials
                            : '<img src="assets/logo1.png" alt="AI" class="w-full h-full object-contain">'
                        }
                    </div>
                </div>
                <div class="oprel-msg-body ${isUser ? 'oprel-body-user' : 'oprel-body-ai'}">
                    ${imageElement}
                    ${renderedContent ? `
                        <div class="oprel-msg-content" id="${msgId}-content">
                            ${renderedContent}
                        </div>
                    ` : ''}
                </div>
            </div>
            ${!isUser && content ? this._actionsBar(msgId, content, metrics) : ''}
        `;

        container.appendChild(msgEl);

        // Syntax highlight code blocks
        if (!isUser && content) {
            MessageRenderer.highlightAll(msgEl);
        }

        this._scrollToBottom();
        return msgEl;
    }

    // ─── Loading Indicator ─────────────────────────────────────────
    static _loadingIndicator() {
        return `<div class="oprel-loading">
            <div class="oprel-loading-dot"></div>
            <div class="oprel-loading-dot"></div>
            <div class="oprel-loading-dot"></div>
        </div>`;
    }

    // ─── Escape User Message ───────────────────────────────────────
    static _escapeUserMessage(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\n/g, '<br>');
    }

    // ─── Actions Bar (outside the bubble) ──────────────────────────
    static _actionsBar(msgId, content, metrics) {
        return `<div class="oprel-actions-bar">
            <div class="oprel-actions-group">
                <button class="oprel-action-btn" onclick="Chat.copyMessage('${msgId}')" title="Copy">
                    <iconify-icon icon="solar:copy-linear" width="14"></iconify-icon>
                </button>
                <button class="oprel-action-btn" onclick="Chat.retryMessage('${msgId}')" title="Retry">
                    <iconify-icon icon="solar:refresh-linear" width="14"></iconify-icon>
                </button>
                <button class="oprel-action-btn oprel-like-btn" title="Good response">
                    <iconify-icon icon="solar:like-linear" width="14"></iconify-icon>
                </button>
                <button class="oprel-action-btn oprel-dislike-btn" title="Bad response">
                    <iconify-icon icon="solar:dislike-linear" width="14"></iconify-icon>
                </button>
            </div>
            ${metrics ? `
                <div class="oprel-tps-badge">
                    <iconify-icon icon="solar:bolt-bold" width="12"></iconify-icon>
                    <span>${metrics.tps} t/s</span>
                </div>
            ` : ''}
        </div>`;
    }

    // ─── Copy entire message ───────────────────────────────────────
    static copyMessage(msgId) {
        const el = document.getElementById(msgId);
        if (!el) return;
        const content = el.querySelector('.oprel-msg-content');
        if (!content) return;
        navigator.clipboard.writeText(content.innerText);

        // Visual feedback
        const btn = el.querySelector('.oprel-action-btn');
        if (btn) {
            btn.classList.add('oprel-copied-action');
            setTimeout(() => btn.classList.remove('oprel-copied-action'), 1500);
        }
    }

    // ─── Retry last message ────────────────────────────────────────
    static retryMessage(msgId) {
        // Remove last assistant + user from history and re-send
        if (this.state.history.length >= 2) {
            const lastUser = this.state.history[this.state.history.length - 2];
            this.state.history.pop(); // remove assistant
            this.state.history.pop(); // remove user

            // Remove the message elements
            const msgEl = document.getElementById(msgId);
            if (msgEl) {
                const prevEl = msgEl.previousElementSibling;
                if (prevEl) prevEl.remove();
                msgEl.remove();
            }

            // Re-send
            if (lastUser && lastUser.content) {
                this.elements.input.value = lastUser.content;
                this.send();
            }
        }
    }

    // ─── Scroll helper ─────────────────────────────────────────────
    static _scrollToBottom() {
        const c = this.elements.messages;
        if (c) {
            requestAnimationFrame(() => {
                c.scrollTop = c.scrollHeight;
            });
        }
    }

    // ─── Send Message (Streaming Pipeline) ─────────────────────────
    static async send() {
        const input = this.elements.input;
        if (!input || (!input.value.trim() && !this.state.selectedImage) || this.state.isGenerating) return;

        const content = input.value.trim();
        const image = this.state.selectedImage;
        const rawContent = image ? [{ type: 'image_url', image_url: { url: image } }, { type: 'text', text: content }] : content;

        input.value = '';
        input.style.height = 'auto';

        // 1. Append user message (visual only)
        this.appendMessage('user', rawContent);
        
        this.state.isGenerating = true;

        // 2. Create placeholder for streaming assistant message
        const streamMsg = this.appendMessage('assistant', '');
        const contentEl = streamMsg.querySelector('.oprel-msg-content');

        // Vision logic: Show "Analyzing image" if we have an image
        if (image && contentEl) {
            contentEl.innerHTML = `
                <div class="flex items-center gap-2 text-blue-400 font-bold text-xs mb-2">
                    <iconify-icon icon="solar:magnifer-bold" class="animate-pulse"></iconify-icon>
                    <span>Analyzing image...</span>
                </div>
                <div class="opacity-30">...</div>
            `;
        }

        this.clearImage();

        // Prepare context
        const messages = [
            { role: 'system', content: this.SYSTEM_PROMPT },
            ...this.state.history,
            { role: 'user', content: rawContent }
        ];

        // Streaming state
        let buffer = '';
        let tokenCount = 0;
        let startTime = null;
        let lastRenderTime = 0;
        const RENDER_INTERVAL = 50; // ms

        try {
            // 3. Stream tokens from API
            await API.chatCompletionStream({
                model: this.state.modelId || 'qwen2.5-7b-instruct',
                conversation_id: this.state.conversationId,
                messages: messages,
                max_tokens: image ? 24576 : 4096,  // Vision models need full context
                onConversationId: (id) => {
                    const isNew = this.state.conversationId !== id;
                    this.state.conversationId = id;
                    localStorage.setItem('oprel_last_conversation_id', id);
                    if (isNew && typeof Sidebar !== 'undefined') {
                        Sidebar.fetchHistory();
                    }
                }
            }, (token) => {
                if (!startTime) startTime = Date.now();
                buffer += token;
                tokenCount++;

                const now = Date.now();
                if (now - lastRenderTime > RENDER_INTERVAL) {
                    contentEl.innerHTML = MessageRenderer.render(buffer);
                    lastRenderTime = now;
                    this._scrollToBottom();
                }
            });

            // 4. Final render
            const endTime = Date.now();
            const durationSec = startTime ? (endTime - startTime) / 1000 : 0;
            const tps = durationSec > 0 ? (tokenCount / durationSec).toFixed(1) : '0';

            streamMsg.remove();
            this.appendMessage('assistant', buffer, { tps });

            // 5. Update history
            this.state.history.push({ role: 'user', content: rawContent });
            this.state.history.push({ role: 'assistant', content: buffer });

        } catch (e) {
            console.error('Chat stream error:', e);
            if (contentEl) {
                contentEl.innerHTML = `<div class="oprel-error">
                    <iconify-icon icon="solar:danger-triangle-bold" width="16"></iconify-icon>
                    <span>Generation failed: ${e.message}</span>
                </div>`;
            }
        } finally {
            this.state.isGenerating = false;
        }
    }
}

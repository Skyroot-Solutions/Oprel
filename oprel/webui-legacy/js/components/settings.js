class Settings {
    static state = {
        isOpen: false,
        config: {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            max_tokens: 4096,
            system_instruction: [
                'You are a helpful, expert AI assistant. Always format your responses in clean, structured Markdown.',
                '',
                'FORMATTING RULES (follow strictly):',
                '- Always wrap code in fenced code blocks with the language tag (e.g. ```python)',
                '- Use tables for comparisons or structured data',
                '- Use > blockquotes for notes or important callouts',
                '- Keep responses concise and well-organized'
            ].join('\n')
        }
    };

    static async init() {
        await this._loadFromServer();
        this._bindEvents();
        this._updateUI();
    }

    static toggle() {
        this.state.isOpen = !this.state.isOpen;
        const modal = document.getElementById('settings-modal');
        if (modal) {
            modal.classList.toggle('hidden', !this.state.isOpen);
        }
        if (this.state.isOpen) {
            this._updateUI();
        }
    }

    static async save() {
        const config = {
            temperature: parseFloat(document.getElementById('settings-temp').value),
            top_p: parseFloat(document.getElementById('settings-top-p').value),
            top_k: parseInt(document.getElementById('settings-top-k').value),
            repeat_penalty: parseFloat(document.getElementById('settings-penalty').value),
            max_tokens: parseInt(document.getElementById('settings-tokens').value),
            system_instruction: document.getElementById('settings-system-prompt').value
        };

        try {
            // Save to server
            await API.saveSettings(config);
            this.state.config = config;
            
            // Update Chat system prompt
            if (typeof Chat !== 'undefined') {
                Chat.SYSTEM_PROMPT = config.system_instruction;
            }

            if (typeof App !== 'undefined') {
                App.showToast('Generation settings updated successfully', 'success');
            }

            this.toggle();
        } catch (e) {
            console.error("Failed to save settings to server:", e);
            if (typeof App !== 'undefined') {
                App.showToast('Error saving settings to server', 'error');
            }
        }
    }

    static async resetToDefaults() {
        const defaults = {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            max_tokens: 4096,
            system_instruction: [
                'You are a helpful, expert AI assistant. Always format your responses in clean, structured Markdown.',
                '',
                'FORMATTING RULES (follow strictly):',
                '- Always wrap code in fenced code blocks with the language tag (e.g. ```python)',
                '- Use tables for comparisons or structured data',
                '- Use > blockquotes for notes or important callouts',
                '- Keep responses concise and well-organized'
            ].join('\n')
        };
        
        try {
            await API.saveSettings(defaults);
            this.state.config = defaults;
            this._updateUI();
            
            if (typeof Chat !== 'undefined') {
                Chat.SYSTEM_PROMPT = defaults.system_instruction;
            }
        } catch (e) {
            console.error("Failed to reset settings:", e);
        }
    }

    static async _loadFromServer() {
        try {
            const saved = await API.fetchSettings();
            // If server has custom instruction (not null/default), use it
            if (saved && saved.system_instruction) {
                this.state.config = { ...this.state.config, ...saved };
            } else {
                // Fallback for first-time migration from localStorage
                const local = localStorage.getItem('oprel_settings');
                if (local) {
                    const parsed = JSON.parse(local);
                    this.state.config = { ...this.state.config, ...parsed };
                    // Optionally push to server once
                    await API.saveSettings(this.state.config);
                }
            }
        } catch (e) {
            console.error("Failed to fetch settings from server", e);
        }
        
        // Push system prompt to Chat
        if (typeof Chat !== 'undefined') {
            Chat.SYSTEM_PROMPT = this.state.config.system_instruction;
        }
    }

    static _updateUI() {
        const c = this.state.config;
        
        // Sliders
        const fields = [
            { id: 'settings-temp', val: c.temperature, label: 'settings-temp-val' },
            { id: 'settings-top-p', val: c.top_p, label: 'settings-top-p-val' },
            { id: 'settings-top-k', val: c.top_k, label: 'settings-top-k-val' },
            { id: 'settings-penalty', val: c.repeat_penalty, label: 'settings-penalty-val' },
            { id: 'settings-tokens', val: c.max_tokens, label: 'settings-tokens-val' }
        ];

        fields.forEach(f => {
            const el = document.getElementById(f.id);
            const label = document.getElementById(f.label);
            if (el) el.value = f.val;
            if (label) label.innerText = f.val;
        });

        // Textarea
        const promptEl = document.getElementById('settings-system-prompt');
        if (promptEl) promptEl.value = c.system_instruction;
    }

    static _bindEvents() {
        const mappings = [
            { id: 'settings-temp', label: 'settings-temp-val' },
            { id: 'settings-top-p', label: 'settings-top-p-val' },
            { id: 'settings-top-k', label: 'settings-top-k-val' },
            { id: 'settings-penalty', label: 'settings-penalty-val' },
            { id: 'settings-tokens', label: 'settings-tokens-val' }
        ];

        mappings.forEach(m => {
            const el = document.getElementById(m.id);
            const label = document.getElementById(m.label);
            if (el && label) {
                el.addEventListener('input', () => {
                    label.innerText = el.value;
                });
            }
        });
    }
}

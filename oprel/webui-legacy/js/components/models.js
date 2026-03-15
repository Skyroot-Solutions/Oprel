class Models {
    static state = {
        list: [],      // Local models (downloaded)
        registry: {},  // Official models from SDK
        metrics: {},   // System hardware metrics
        current: null,
        initialized: false
    };

    static async fetch() {
        try {
            // Fetch local, registry, and system metrics
            const [localModels, registry, metrics] = await Promise.all([
                API.fetchModels(),
                API.fetchRegistryModels(),
                API.getMetrics()
            ]);

            this.state.list = localModels || [];
            this.state.registry = registry || { categories: {}, models: {} };
            this.state.metrics = metrics || {};

            // Check for loaded model
            const loaded = this.state.list.find(m => m.loaded);
            if (loaded) {
                this.setActive(loaded);
            } else if (this.state.list.length > 0 && !this.state.current) {
                this.setActive(this.state.list[0]);
            } else if (!this.state.current && registry.models?.['text-generation']) {
                const firstId = Object.keys(registry.models['text-generation'])[0];
                this.setActive({ model_id: firstId, name: firstId });
            }

            this.render();
            this.state.initialized = true;
        } catch (e) {
            console.error('Failed to fetch models:', e);
            this.render();
        }
    }

    static getCompatibility(sizeGB) {
        if (!sizeGB) return { status: 'unknown', text: 'Size Unknown', class: 'badge-hybrid' };
        
        const metrics = this.state.metrics;
        const vramGB = (metrics.vram_total_mb || 0) / 1024;
        const ramAvailableGB = (metrics.ram_total_gb || 0) * 0.6; // 60% of RAM
        
        const totalCapacity = vramGB + ramAvailableGB;
        
        if (sizeGB <= vramGB * 0.9) { // 10% buffer for OS/UI
            return { status: 'vram', text: 'Compatible - VRAM Only', class: 'badge-compatible' };
        } else if (sizeGB <= totalCapacity) {
            return { status: 'hybrid', text: 'Compatible - Hybrid Offload', class: 'badge-hybrid' };
        } else {
            return { status: 'incompatible', text: 'Heavy - Memory Overload', class: 'badge-incompatible' };
        }
    }

    static estimateSize(alias) {
        // Rough estimations for registry models based on common parameter counts
        const aliasLower = alias.toLowerCase();
        if (aliasLower.includes('70b') || aliasLower.includes('80b')) return 42;
        if (aliasLower.includes('32b') || aliasLower.includes('27b')) return 18;
        if (aliasLower.includes('14b')) return 9;
        if (aliasLower.includes('8b') || aliasLower.includes('7b')) return 5.5;
        if (aliasLower.includes('3b') || aliasLower.includes('4b')) return 2.8;
        if (aliasLower.includes('1.5b') || aliasLower.includes('1b')) return 1.2;
        return 0;
    }

    static setActive(model) {
        this.state.current = model;
        Chat.state.modelId = model.model_id;

        const nameEl = document.getElementById('chat-model-name');
        const inputNameEl = document.getElementById('input-model-name');
        const specsEl = document.getElementById('chat-model-specs');

        const label = model.name || model.model_id;
        if (nameEl) nameEl.innerText = label;
        if (inputNameEl) inputNameEl.innerHTML = `${label} <iconify-icon icon="solar:alt-arrow-down-linear" size="10"></iconify-icon>`;
        
        if (specsEl) {
            const isDownloaded = this.state.list.some(m => m.model_id === model.model_id);
            const status = model.loaded ? 'Online' : (isDownloaded ? 'Downloaded' : 'Cloud');
            specsEl.innerText = status;
            specsEl.className = `text-[10px] ${model.loaded ? 'text-green-500' : (isDownloaded ? 'text-blue-400' : 'text-neutral-500')}`;
        }
    }

    static render() {
        this.renderLibraryList();
        this.renderHeaderDropdown();
        this.renderInputDropdown();
    }

    static renderLibraryList() {
        const listContainer = document.getElementById('library-model-list');
        if (!listContainer) return;

        listContainer.innerHTML = '';

        if (!this.state.registry.models) {
            listContainer.innerHTML = `<div class="p-8 text-center"><iconify-icon icon="solar:cloud-info-linear" class="text-3xl text-neutral-600 mb-2"></iconify-icon><p class="text-neutral-500 italic text-xs">Registry unavailable</p></div>`;
            return;
        }

        const categories = this.state.registry.categories || {};
        const modelsGrouped = this.state.registry.models || {};

        Object.keys(modelsGrouped).forEach(catId => {
            const catInfo = categories[catId] || { name: catId, icon: '📦' };
            const models = modelsGrouped[catId];
            
            if (Object.keys(models).length === 0) return;

            const section = document.createElement('div');
            section.className = 'mb-6';
            section.innerHTML = `<h3 class="px-3 text-[10px] font-bold text-neutral-500 mb-3 uppercase tracking-widest flex items-center gap-2">
                <span>${catInfo.icon || '•'}</span> ${catInfo.name}
            </h3>`;

            const ul = document.createElement('div');
            ul.className = 'space-y-1.5 px-1';

            Object.entries(models).forEach(([alias, repo_id]) => {
                const local = this.state.list.find(m => m.model_id === repo_id || m.model_id === alias);
                const isSelected = this.state.current?.model_id === alias || this.state.current?.model_id === repo_id;
                
                const btn = document.createElement('button');
                const isDownloaded = !!local;
                
                // Styling based on state
                let borderClass = 'border-transparent';
                let bgClass = 'hover:bg-white/5';
                let titleClass = isDownloaded ? 'text-white' : 'text-neutral-500';

                if (isSelected) {
                    borderClass = 'border-blue-500/50';
                    bgClass = 'bg-blue-500/10';
                } else if (isDownloaded) {
                    borderClass = 'border-white/5';
                    bgClass = 'bg-[#1a1a1a]';
                }

                btn.className = `w-full text-left p-3 rounded-xl border transition-all duration-200 group relative overflow-hidden ${borderClass} ${bgClass}`;

                const sizeGB = local?.size_gb || this.estimateSize(alias);
                const comp = this.getCompatibility(sizeGB);

                btn.innerHTML = `
                    <div class="flex justify-between items-start mb-1">
                        <div class="flex items-center gap-2">
                            <span class="font-bold text-xs ${titleClass}">${alias}</span>
                            ${local?.loaded ? '<span class="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>' : ''}
                        </div>
                        <span class="text-[9px] font-mono text-neutral-500 bg-white/5 px-1.5 py-0.5 rounded">${sizeGB ? sizeGB.toFixed(1) + 'GB' : 'Cloud'}</span>
                    </div>
                    <div class="flex items-center justify-between mt-2">
                        <div class="flex items-center gap-1.5 text-[10px]">
                            ${isDownloaded ? `
                                <span class="text-green-500 flex items-center gap-1"><iconify-icon icon="solar:check-circle-bold" size="12"></iconify-icon> Ready</span>
                            ` : `
                                <span class="text-blue-500 flex items-center gap-1"><iconify-icon icon="solar:download-bold" size="12"></iconify-icon> Available</span>
                            `}
                        </div>
                        <div class="flex gap-1">
                             <iconify-icon icon="${comp.status === 'vram' ? 'solar:bolt-circle-bold' : (comp.status === 'hybrid' ? 'solar:layers-bold' : 'solar:danger-triangle-bold')}" 
                                class="${comp.class} text-xs" title="${comp.text}"></iconify-icon>
                        </div>
                    </div>
                `;

                btn.onclick = () => this.handleModelSelect({ model_id: alias, name: alias, loaded: local?.loaded });
                ul.appendChild(btn);
            });

            section.appendChild(ul);
            listContainer.appendChild(section);
        });
    }

    static renderHeaderDropdown() {
        const dropdown = document.getElementById('header-model-dropdown');
        if (!dropdown) return;

        dropdown.innerHTML = `
            <div class="px-2 py-1.5 border-b border-white/5 mb-1 flex justify-between items-center">
                <span class="text-[10px] font-semibold text-neutral-500 uppercase tracking-wider">Downloaded Models</span>
                <span class="px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 text-[9px] font-bold">${this.state.list.length}</span>
            </div>
        `;

        const downloaded = this.state.list;
        if (downloaded.length === 0) {
            dropdown.innerHTML += `<div class="p-3 text-center text-[10px] text-neutral-600 italic">No models downloaded yet. Visit Library to add models.</div>`;
        } else {
            downloaded.forEach(m => {
                const item = document.createElement('div');
                item.className = 'flex items-center justify-between p-2 rounded-lg hover:bg-white/5 cursor-pointer group mb-0.5';
                item.innerHTML = `
                    <div class="flex items-center gap-2 overflow-hidden">
                        <iconify-icon icon="${m.loaded ? 'solar:bolt-circle-bold' : 'solar:check-circle-linear'}" 
                            class="${m.loaded ? 'text-green-500' : 'text-blue-400'} flex-shrink-0 text-sm"></iconify-icon>
                        <span class="text-xs ${m.loaded ? 'text-white' : 'text-gray-400 group-hover:text-white'} truncate">${m.name || m.model_id}</span>
                    </div>
                `;
                item.onclick = async () => {
                    this.handleModelSelect(m);
                    document.getElementById('header-model-dropdown').classList.add('hidden');
                    if (!m.loaded) {
                        try { await this.loadCurrent(); } catch(e){}
                    }
                };
                dropdown.appendChild(item);
            });
        }
        
        const more = document.createElement('div');
        more.className = 'p-2 text-center text-[10px] text-blue-400 font-semibold cursor-pointer hover:bg-blue-400/5 border-t border-white/5 mt-1 transition-colors';
        more.innerHTML = `<iconify-icon icon="solar:library-linear" class="inline-block mr-1"></iconify-icon> Open Model Library`;
        more.onclick = (e) => {
            e.stopPropagation();
            window.switchView('models');
        };
        dropdown.appendChild(more);
    }

    static isVisionModel(m) {
        const id = (m.model_id || m.name || "").toLowerCase();
        return id.includes('vision') || id.includes('vl') || id.includes('llava') || id.includes('moondream');
    }

    static renderInputDropdown() {
        const dropdown = document.getElementById('input-model-dropdown');
        if (!dropdown) return;

        dropdown.innerHTML = "";

        const downloaded = this.state.list;
        if (downloaded.length === 0) {
            dropdown.innerHTML = `<div class="p-3 text-center text-[10px] text-neutral-600">No models ready</div>`;
            return;
        }

        const textModels = downloaded.filter(m => !this.isVisionModel(m));
        const visionModels = downloaded.filter(m => this.isVisionModel(m));

        const renderGroup = (label, models) => {
            if (models.length === 0) return;
            const group = document.createElement('div');
            group.innerHTML = `<div class="px-2 py-1.5 border-b border-white/5 mb-1 text-[9px] font-bold text-neutral-500 uppercase tracking-widest">${label}</div>`;
            models.forEach(m => {
                const item = document.createElement('div');
                item.className = 'flex items-center gap-2 p-2 rounded hover:bg-white/5 cursor-pointer';
                const isCurrent = this.state.current?.model_id === m.model_id;
                item.innerHTML = `
                    <iconify-icon icon="${m.loaded ? 'solar:bolt-circle-bold' : 'solar:check-circle-linear'}" 
                        class="${m.loaded ? 'text-green-500' : 'text-blue-400'} text-xs"></iconify-icon>
                    <span class="text-xs ${isCurrent ? 'text-blue-400 font-bold' : 'text-gray-300'} truncate">${m.name || m.model_id}</span>
                    ${isCurrent ? '<iconify-icon icon="solar:check-read-linear" class="ml-auto text-blue-400" size="12"></iconify-icon>' : ''}
                `;
                item.onclick = async (e) => {
                    e.stopPropagation();
                    this.handleModelSelect(m);
                    dropdown.classList.add('hidden');
                    if (!m.loaded) {
                        try { await this.loadCurrent(); } catch(e){}
                    }
                };
                group.appendChild(item);
            });
            dropdown.appendChild(group);
        };

        renderGroup('Text Models', textModels);
        renderGroup('Text+Image Models', visionModels);
    }

    static async handleModelSelect(payload) {
        // Extract info from payload
        const model_id = payload.model_id;
        const alias = payload.name || model_id;
        
        // Find registry info
        let repo_id = model_id;
        let category = 'Unknown';
        
        if (this.state.registry.models) {
            for (const [cat, models] of Object.entries(this.state.registry.models)) {
                if (models[alias]) {
                    repo_id = models[alias];
                    category = cat;
                    break;
                }
            }
        }

        const local = this.state.list.find(m => m.model_id === repo_id || m.model_id === alias);
        const isDownloaded = !!local;

        // Set active for metadata tracking
        this.setActive({ model_id: alias, name: alias, loaded: local?.loaded });
        
        // Render details view
        this.renderModelDetails({
            alias,
            repo_id,
            category,
            isDownloaded,
            loaded: local?.loaded,
            localInfo: local
        });

        // Re-render list to show selection
        this.renderLibraryList();
    }

    static renderModelDetails(data) {
        const detailContainer = document.getElementById('model-details-view');
        if (!detailContainer) return;

        const catInfo = this.state.registry.categories?.[data.category] || { name: data.category, icon: '📦' };
        
        // Dynamic compatibility logic
        const sizeGB = data.localInfo?.size_gb || this.estimateSize(data.alias);
        const comp = this.getCompatibility(sizeGB);

        detailContainer.innerHTML = `
            <div class="max-w-4xl mx-auto space-y-8 fade-in">
                <!-- Header Section -->
                <div class="space-y-4">
                    <div class="flex items-center gap-3">
                        <h1 class="text-3xl font-bold text-white tracking-tight">${data.alias}</h1>
                        <span class="px-2.5 py-0.5 rounded-full text-[10px] font-bold border ${comp.class}">
                            ${comp.text}
                        </span>
                    </div>
                    <p class="text-neutral-500 font-mono text-xs flex items-center gap-2">
                        <iconify-icon icon="solar:link-bold"></iconify-icon>
                        ${data.repo_id}
                    </p>
                </div>

                <!-- Stats Grid -->
                <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    <div class="bg-[#1e1e1e] p-4 rounded-xl border border-white/5 space-y-2">
                        <div class="text-neutral-500 text-[10px] font-bold uppercase tracking-wider flex items-center gap-1">
                            <iconify-icon icon="solar:bolt-bold" class="text-blue-500"></iconify-icon> Requirements
                        </div>
                        <div class="meter-bar">
                            <div class="meter-fill" style="width: ${Math.min(100, (sizeGB / 24) * 100)}%"></div>
                        </div>
                        <div class="text-right text-[10px] text-white">Est. ${sizeGB ? sizeGB.toFixed(1) : '?'} GB</div>
                    </div>
                    
                    <div class="bg-[#1e1e1e] p-4 rounded-xl border border-white/5">
                        <div class="text-neutral-500 text-[10px] font-bold uppercase tracking-wider flex items-center gap-1 mb-1">
                            <iconify-icon icon="solar:database-bold"></iconify-icon> Memory Mode
                        </div>
                        <div class="text-lg font-bold text-white">${comp.status === 'vram' ? 'VRAM Only' : (comp.status === 'hybrid' ? 'Hybrid (V+R)' : 'Heavy Swap')}</div>
                        <div class="text-[10px] text-neutral-600">${comp.text}</div>
                    </div>

                    <div class="bg-[#1e1e1e] p-4 rounded-xl border border-white/5">
                        <div class="text-neutral-500 text-[10px] font-bold uppercase tracking-wider flex items-center gap-1 mb-1">
                            <iconify-icon icon="solar:playback-speed-bold"></iconify-icon> Speed (Est.)
                        </div>
                        <div class="text-lg font-bold text-white">~${comp.status === 'vram' ? '25' : (comp.status === 'hybrid' ? '8' : '1')} t/s</div>
                        <div class="text-[10px] text-neutral-600">based on your specs</div>
                    </div>

                    <div class="bg-[#1e1e1e] p-4 rounded-xl border border-white/5">
                        <div class="text-neutral-500 text-[10px] font-bold uppercase tracking-wider flex items-center gap-1 mb-1">
                            <iconify-icon icon="solar:layers-bold"></iconify-icon> Architecture
                        </div>
                        <div class="text-lg font-bold text-white">GGUF</div>
                        <div class="text-[10px] text-neutral-600">Q4_K_M (Typical)</div>
                    </div>
                </div>

                <div class="h-px bg-white/5"></div>

                <!-- Description -->
                <div class="space-y-4">
                    <h3 class="text-lg font-bold text-white">About</h3>
                    <p class="text-neutral-400 leading-relaxed text-sm">
                        This is a high-performance ${data.category} model optimized for the Oprel SDK. 
                        It features a balanced architecture that provides both accuracy and speed, making it 
                        ideal for daily assistant tasks, coding support, and creative writing.
                    </p>
                </div>

                <!-- Actions -->
                <div class="flex items-center gap-4 pt-4">
                    ${data.loaded ? `
                        <button onclick="switchView('chat')" class="px-8 py-3 bg-[#3b82f6] text-white font-bold rounded-xl hover:bg-[#2563eb] transition-all flex items-center gap-2 shadow-xl shadow-blue-500/20">
                            <iconify-icon icon="solar:chat-round-dots-bold" size="18"></iconify-icon>
                            Open Chat
                        </button>
                    ` : `
                        <button id="load-btn" onclick="${data.isDownloaded ? 'Models.loadCurrent()' : 'Models.pullCurrent()'}" class="px-8 py-3 bg-[#3b82f6] text-white font-bold rounded-xl hover:bg-[#2563eb] transition-all flex items-center gap-2 shadow-xl shadow-blue-500/20">
                            <iconify-icon icon="solar:download-bold" size="18"></iconify-icon>
                            ${data.isDownloaded ? 'Load Model' : 'Download Model'}
                        </button>
                    `}
                    
                    <button class="px-4 py-3 bg-[#1e1e1e] text-neutral-400 font-bold rounded-xl border border-white/5 hover:text-white hover:bg-[#2a2a2a] transition-all flex items-center gap-2">
                        Q4_K_M <iconify-icon icon="solar:alt-arrow-down-linear"></iconify-icon>
                    </button>
                </div>
            </div>
        `;
    }

    static async loadCurrent() {
        const model = this.state.current;
        if (!model) return;

        const btn = document.getElementById('load-btn');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<iconify-icon icon="solar:re-order-linear" class="animate-spin"></iconify-icon> Initializing...';
        }

        const specsEl = document.getElementById('chat-model-specs');
        if (specsEl) {
            specsEl.innerText = "Provisioning model...";
            specsEl.className = "text-[10px] text-blue-400 animate-pulse";
        }

        try {
            await API.loadModel(model.model_id);
            await this.fetch();
            // refreshing fetch will re-render details if we are clever, 
            // but let's just trigger it manually for instant feedback
            this.handleModelSelect(model);
        } catch (e) {
            console.error("Failed to load model:", e);
            if (btn) {
                btn.disabled = false;
                btn.innerText = 'Error: Try again';
                btn.className += ' bg-red-600 hover:bg-red-500';
            }
        }
    }

    static async pullCurrent() {
        const model = this.state.current;
        if (!model) return;

        const btn = document.getElementById('load-btn');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<iconify-icon icon="solar:cloud-download-bold" class="animate-pulse"></iconify-icon> Downloading... (Check CLI)';
        }

        try {
            await API.pullModel(model.model_id);
            // After successful pull, let's load it automatically for convenience
            await this.fetch();
            this.handleModelSelect(model);
            await this.loadCurrent();
        } catch (e) {
            console.error("Failed to download model:", e);
            if (btn) {
                btn.disabled = false;
                btn.innerText = 'Download Failed';
                btn.className += ' bg-red-600 hover:bg-red-500';
            }
        }
    }
}

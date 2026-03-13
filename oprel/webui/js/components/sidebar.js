class Sidebar {
    static async fetchHistory() {
        try {
            const conversations = await API.fetchConversations();
            this.render(conversations);
        } catch (e) {
            console.error('Failed to fetch sidebar history:', e);
            this.render([]); // Render empty state
        }
    }

    static render(conversations) {
        const historyContainer = document.getElementById('conversation-history');
        if (!historyContainer) return;

        if (!conversations || conversations.length === 0) {
            historyContainer.innerHTML = `
                <div class="h-40 flex flex-col items-center justify-center text-neutral-600 space-y-2 opacity-50">
                    <iconify-icon icon="solar:chat-round-line-linear" width="24"></iconify-icon>
                    <span class="text-[10px] italic">No conversations yet</span>
                </div>
            `;
            return;
        }

        // Simple grouping logic (Today/Earlier)
        const groups = {
            'Today': [],
            'Earlier': []
        };

        const today = new Date().toDateString();
        conversations.forEach(c => {
            const date = c.created_at ? new Date(c.created_at).toDateString() : today;
            if (date === today) groups['Today'].push(c);
            else groups['Earlier'].push(c);
        });

        historyContainer.innerHTML = '';

        Object.entries(groups).forEach(([label, chats]) => {
            if (chats.length === 0) return;

            const section = document.createElement('div');
            section.innerHTML = `<h3 class="history-header">${label}</h3>`;
            
            const list = document.createElement('div');
            list.className = 'space-y-0.5';

            chats.forEach(chat => {
                // The container item
                const item = document.createElement('div');
                item.className = 'group flex items-center px-3 py-2.5 rounded-lg hover:bg-[#2a2a2a] cursor-pointer text-gray-400 hover:text-gray-200 transition-colors relative';
                
                // Normal View
                const normalView = document.createElement('div');
                normalView.className = 'flex flex-1 items-center gap-3 min-w-0';
                normalView.innerHTML = `
                    <iconify-icon icon="solar:message-square-linear" size="16" class="shrink-0"></iconify-icon>
                    <div class="flex-1 min-w-0">
                        <div class="text-xs font-medium truncate">${chat.title || 'New Conversation'}</div>
                    </div>
                    <div class="opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-opacity">
                        <button class="rename-btn p-1 hover:bg-[#3a3a3a] rounded text-neutral-400 hover:text-white transition-colors" title="Rename">
                            <iconify-icon icon="solar:pen-bold" size="14"></iconify-icon>
                        </button>
                        <button class="delete-btn p-1 hover:bg-red-500/20 rounded text-neutral-400 hover:text-red-400 transition-colors" title="Delete">
                            <iconify-icon icon="solar:trash-bin-trash-bold" size="14"></iconify-icon>
                        </button>
                    </div>
                `;

                // Edit View (hidden by default)
                const editView = document.createElement('div');
                editView.className = 'hidden flex-1 items-center gap-2 min-w-0';
                editView.innerHTML = `
                    <iconify-icon icon="solar:message-square-linear" size="16" class="shrink-0"></iconify-icon>
                    <input type="text" class="flex-1 bg-[#1e1e1e] border border-blue-500/50 rounded px-2 py-1 text-xs text-white outline-none w-full" value="${(chat.title || '').replace(/"/g, '&quot;')}">
                    <button class="save-btn p-1 hover:bg-green-500/20 rounded text-green-500 transition-colors shrink-0" title="Save">
                        <iconify-icon icon="solar:check-circle-bold" size="14"></iconify-icon>
                    </button>
                `;

                item.appendChild(normalView);
                item.appendChild(editView);

                // Event Listeners for switching views
                const renameBtn = normalView.querySelector('.rename-btn');
                const deleteBtn = normalView.querySelector('.delete-btn');
                const saveBtn = editView.querySelector('.save-btn');
                const titleInput = editView.querySelector('input');

                renameBtn.onclick = (e) => {
                    e.stopPropagation();
                    normalView.classList.add('hidden');
                    normalView.classList.remove('flex');
                    editView.classList.remove('hidden');
                    editView.classList.add('flex');
                    titleInput.focus();
                };

                const saveTitle = async (e) => {
                    if (e) e.stopPropagation();
                    const newTitle = titleInput.value.trim();
                    if (newTitle && newTitle !== chat.title) {
                        try {
                            // Optimistic UI update
                            normalView.querySelector('.truncate').textContent = newTitle;
                            chat.title = newTitle;
                            
                            // Revert views
                            editView.classList.add('hidden');
                            editView.classList.remove('flex');
                            normalView.classList.remove('hidden');
                            normalView.classList.add('flex');

                            // API call
                            await API.renameConversation(chat.id, newTitle);
                        } catch (err) {
                            console.error('Failed to rename chat:', err);
                            normalView.querySelector('.truncate').textContent = chat.title; // revert
                        }
                    } else {
                        // Cancel
                        editView.classList.add('hidden');
                        editView.classList.remove('flex');
                        normalView.classList.remove('hidden');
                        normalView.classList.add('flex');
                        titleInput.value = chat.title || 'New Conversation';
                    }
                };

                saveBtn.onclick = saveTitle;
                
                titleInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        saveTitle();
                    } else if (e.key === 'Escape') {
                        // cancel
                        editView.classList.add('hidden');
                        editView.classList.remove('flex');
                        normalView.classList.remove('hidden');
                        normalView.classList.add('flex');
                        titleInput.value = chat.title || 'New Conversation';
                    }
                });
                
                // Prevent input clicks from loading the chat
                titleInput.onclick = (e) => e.stopPropagation();

                deleteBtn.onclick = (e) => {
                    e.stopPropagation();
                    Sidebar.showDeleteDialog(chat.id);
                };

                // Standard click
                item.onclick = () => Chat.loadConversation(chat.id);
                list.appendChild(item);
            });

            section.appendChild(list);
            historyContainer.appendChild(section);
        });
    }

    // --- Delete Modal Logic ---
    static deleteTargetId = null;

    static showDeleteDialog(id) {
        this.deleteTargetId = id;
        document.getElementById('delete-dialog').classList.remove('hidden');
    }

    static cancelDelete() {
        this.deleteTargetId = null;
        document.getElementById('delete-dialog').classList.add('hidden');
    }

    static async confirmDelete() {
        if (!this.deleteTargetId) return;
        const id = this.deleteTargetId;
        this.deleteTargetId = null;
        document.getElementById('delete-dialog').classList.add('hidden');

        try {
            await API.deleteConversation(id);
            // Also clear main chat UI if the deleted chat is currently open
            if (typeof Chat !== 'undefined' && Chat.state.conversationId === id) {
                Chat.newConversation();
            }
            this.fetchHistory(); // Refresh sidebar
        } catch (e) {
            console.error('Failed to delete chat:', e);
            alert('Failed to delete chat. Check console for details.');
        }
    }
}

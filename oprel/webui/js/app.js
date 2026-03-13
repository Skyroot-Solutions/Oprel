const App = {
    state: {
        activeView: 'chat',
        initialized: false
    },

    elements: {
        views: {
            chat: document.getElementById('view-chat'),
            models: document.getElementById('view-models'),
            dev: document.getElementById('view-dev')
        },
        buttons: {
            // Can add more
        }
    },

    async init() {
        if (this.state.initialized) return;

        // Initialize components
        Chat.init();
        Models.fetch();
        Sidebar.fetchHistory();
        Developer.fetch();

        // Check for User Profile
        await this.checkUserProfile();

        // Start polling loops
        setInterval(() => Developer.fetch(), 2000); // 2s polling
        setInterval(() => Models.fetch(), 10000); // 10s polling

        // Event listeners for basic layout (if not handled by inline onclicks)
        window.switchView = this.switchView.bind(this);

        this.state.initialized = true;
        console.log('Oprel App Initialized');

        // Check for persisted conversation
        const lastConvId = localStorage.getItem('oprel_last_conversation_id');
        if (lastConvId) {
            Chat.loadConversation(lastConvId);
        }
    },

    async checkUserProfile() {
        try {
            const user = await API.fetchUser();
            if (user) {
                this.state.user = user;
                this.updateSidebarUser(user);
            } else {
                document.getElementById('user-setup-modal').classList.remove('hidden');
            }
        } catch (err) {
            console.error('Failed to check user profile:', err);
        }
    },

    updateSidebarUser(user) {
        if (!user) return;
        const nameEl = document.getElementById('user-name');
        const roleEl = document.getElementById('user-role');
        const avatarEl = document.getElementById('user-avatar');

        if (nameEl) nameEl.textContent = user.name;
        if (roleEl) roleEl.textContent = user.role;
        if (avatarEl) avatarEl.textContent = user.initials;
    },

    async saveUserProfile() {
        const name = document.getElementById('setup-name').value.trim();
        const role = document.getElementById('setup-role').value.trim();

        if (!name || !role) {
            alert('Please enter both name and role');
            return;
        }

        try {
            const user = await API.saveUser(name, role);
            this.updateSidebarUser(user);
            document.getElementById('user-setup-modal').classList.add('hidden');
        } catch (err) {
            console.error('Failed to save user profile:', err);
            alert('Failed to save profile. Please try again.');
        }
    },

    switchView(viewName) {
        this.state.activeView = viewName;

        // Hide all
        Object.entries(this.elements.views).forEach(([name, el]) => {
            if (!el) return;
            el.classList.add('hidden');
            el.classList.remove('flex');
        });

        // Show selected
        const selected = this.elements.views[viewName];
        if (selected) {
            selected.classList.remove('hidden');
            // Restore appropriate flex class
            selected.classList.add('flex');
        }

        // Close mobile sidebar if open
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('sidebar-overlay');
        if (sidebar && !sidebar.classList.contains('-translate-x-full')) {
            sidebar.classList.add('-translate-x-full');
            if (overlay) overlay.classList.add('hidden');
        }

        if (viewName === 'chat') {
            const scroll = document.getElementById('chat-scroll');
            if (scroll) scroll.scrollTop = scroll.scrollHeight;
        }
    },

    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('sidebar-overlay');
        if (!sidebar) return;

        const isHidden = sidebar.classList.contains('-translate-x-full');
        if (isHidden) {
            sidebar.classList.remove('-translate-x-full');
            if (overlay) overlay.classList.remove('hidden');
        } else {
            sidebar.classList.add('-translate-x-full');
            if (overlay) overlay.classList.add('hidden');
        }
    },

    handleGlobalClick(e) {
        const dropdowns = ['header-model-dropdown', 'input-model-dropdown'];
        const selectors = ['model-selector-btn', 'input-model-name'];

        dropdowns.forEach((id, i) => {
            const dropdown = document.getElementById(id);
            const selector = document.getElementById(selectors[i]);
            if (dropdown && !dropdown.contains(e.target) && selector && !selector.contains(e.target)) {
                dropdown.classList.add('hidden');
            }
        });
    }
};

window.addEventListener('click', (e) => App.handleGlobalClick(e));
window.onload = () => App.init();

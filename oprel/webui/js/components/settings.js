class Settings {
    static state = {
        isOpen: false
    };

    static toggle() {
        this.state.isOpen = !this.state.isOpen;
        const modal = document.getElementById('settings-modal');
        if (modal) {
            modal.classList.toggle('hidden', !this.state.isOpen);
        }
    }

    static save() {
        // Implement save logic here
        this.toggle();
    }
}

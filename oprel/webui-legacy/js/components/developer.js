class Developer {
    static state = {
        metrics: {},
        history: {
            cpu: [],
            gpu: [],
            vram: [],
            ram: [],
            speed: []
        },
        charts: {}
    };

    static async fetch() {
        try {
            const metrics = await API.getMetrics();
            this.state.metrics = metrics;
            
            // Push to history (limit to 20 points)
            const pushHistory = (key, val) => {
                this.state.history[key].push(val || 0);
                if (this.state.history[key].length > 20) this.state.history[key].shift();
            };

            pushHistory('cpu', metrics.cpu_usage);
            pushHistory('gpu', metrics.gpu_usage);
            pushHistory('vram', metrics.vram_used_mb ? (metrics.vram_used_mb / metrics.vram_total_mb * 100) : 0);
            pushHistory('ram', metrics.ram_used_gb ? (metrics.ram_used_gb / metrics.ram_total_gb * 100) : 0);
            pushHistory('speed', metrics.generation_speed);

            this.render();
            this.updateCharts();
        } catch (e) {
            console.error('Failed to fetch metrics:', e);
        }
    }

    static createSparklineOptions(color) {
        return {
            series: [{ data: [] }],
            chart: {
                type: 'area',
                height: 60,
                sparkline: { enabled: true },
                animations: { enabled: true, easing: 'linear', dynamicAnimation: { speed: 1000 } }
            },
            stroke: { curve: 'smooth', width: 2 },
            fill: {
                type: 'gradient',
                gradient: { shadeIntensity: 1, opacityFrom: 0.4, opacityTo: 0 }
            },
            colors: [color],
            tooltip: { enabled: false }
        };
    }

    static initCharts() {
        if (this.state.charts.cpu) return;

        const ids = ['cpu', 'gpu', 'vram', 'ram', 'speed'];
        const colors = ['#3b82f6', '#a855f7', '#f59e0b', '#10b981', '#10b981'];

        ids.forEach((id, i) => {
            const el = document.getElementById(`${id}-chart`);
            if (el) {
                this.state.charts[id] = new ApexCharts(el, this.createSparklineOptions(colors[i]));
                this.state.charts[id].render();
            }
        });

        // Token Distribution Bar Chart
        const distEl = document.getElementById('distribution-chart');
        if (distEl) {
            this.state.charts.distribution = new ApexCharts(distEl, {
                series: [{ name: 'Tokens', data: [400, 300, 200, 100] }],
                chart: { type: 'bar', height: '100%', toolbar: { show: false }, background: 'transparent' },
                plotOptions: { bar: { borderRadius: 4, horizontal: true } },
                dataLabels: { enabled: false },
                colors: ['#3b82f6'],
                theme: { mode: 'dark' },
                xaxis: { categories: ['Llama 3', 'Qwen 2.5', 'Mistral', 'Phi 3'] },
                grid: { borderColor: '#ffffff05' }
            });
            this.state.charts.distribution.render();
        }

        // Session Line Chart
        const sessEl = document.getElementById('session-chart');
        if (sessEl) {
            this.state.charts.session = new ApexCharts(sessEl, {
                series: [{ name: 'Current Context', data: [30, 40, 35, 50, 49, 60, 70, 91, 125] }],
                chart: { type: 'line', height: '100%', toolbar: { show: false }, background: 'transparent' },
                stroke: { curve: 'smooth', width: 3 },
                colors: ['#3b82f6'],
                theme: { mode: 'dark' },
                xaxis: { labels: { show: false }, axisBorder: { show: false } },
                grid: { borderColor: '#ffffff05' }
            });
            this.state.charts.session.render();
        }
    }

    static updateCharts() {
        this.initCharts();
        
        ['cpu', 'gpu', 'vram', 'ram', 'speed'].forEach(id => {
            if (this.state.charts[id]) {
                this.state.charts[id].updateSeries([{ data: this.state.history[id] }]);
            }
        });
    }

    static addLog(level, message) {
        const container = document.getElementById('log-container');
        if (!container) return;

        const time = new Date().toLocaleTimeString('en-US', { hour12: false });
        const div = document.createElement('div');
        div.className = 'flex gap-3 px-1 rounded hover:bg-white/5 transition-colors text-[11px]';
        
        const colors = {
            'INFO': 'text-blue-400',
            'WARN': 'text-amber-400',
            'ERROR': 'text-red-500',
            'DEBUG': 'text-neutral-500'
        };

        div.innerHTML = `
            <span class="text-neutral-600 shrink-0 select-none">[${time}]</span>
            <span class="shrink-0 w-10 font-bold ${colors[level] || 'text-neutral-400'}">${level}</span>
            <span class="text-neutral-300 break-all">${message}</span>
        `;
        
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }

    static render() {
        const data = this.state.metrics;

        const updateText = (id, text) => {
            const el = document.getElementById(id);
            if (el) el.innerHTML = text;
        };

        updateText('cpu-card-val', (data.cpu_usage || 0).toFixed(1) + '%');
        
        if (data.gpu_name) {
            updateText('gpu-card-val', (data.gpu_usage || 0).toFixed(1) + '%');
            const vramVal = (data.vram_used_mb / 1024 || 0).toFixed(1);
            updateText('vram-card-val', `${vramVal} GB`);
        }

        const ramVal = (data.ram_used_gb || 0).toFixed(1);
        updateText('ram-card-val', `${ramVal} GB`);

        updateText('speed-card-val', (data.generation_speed || 0).toFixed(1));
    }
}

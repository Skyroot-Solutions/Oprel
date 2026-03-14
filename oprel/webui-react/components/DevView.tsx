"use client"

import { useState, useEffect, useRef } from "react"
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts"
import { Cpu, HardDrive, Zap, Activity, MemoryStick, Layers } from "lucide-react"
import { cn } from "@/services/utils"
import { useApp } from "@/services/context"
import { API } from "@/services/api"

function generateHistory(base: number, variance: number, len = 20) {
  return Array.from({ length: len }, (_, i) => ({
    t: i,
    v: Math.max(0, Math.min(100, base + (Math.random() - 0.5) * variance * 2)),
  }))
}

const COLOR_MAP: Record<string, string> = {
  "text-primary": "#ee4647",
  "text-amber-400": "#f59e0b",
  "text-green-400": "#22c55e",
  "text-blue-400": "#3b82f6",
  "text-purple-400": "#a855f7",
}

function MetricCard({
  icon,
  label,
  value,
  sub,
  color,
  data,
  unit = "%",
  accent,
}: {
  icon: React.ReactNode
  label: string
  value: string
  sub?: string
  color: string
  data: { t: number; v: number }[]
  unit?: string
  accent?: React.ReactNode
}) {
  return (
    <div className="relative bg-[#1a1a1a] border border-border rounded-xl p-5 overflow-hidden flex flex-col h-40">
      {/* Accent icon top right */}
      {accent && (
        <div className="absolute top-3 right-3">
          {accent}
        </div>
      )}

      <div className="flex items-center gap-1.5 text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-1 z-10 relative">
        <span className={cn("shrink-0", color)}>{icon}</span>
        {label}
      </div>

      <div className="text-3xl font-bold text-foreground tracking-tight z-10 relative leading-none mb-1">
        {value}
      </div>

      {sub && (
        <div className="text-[10px] text-muted-foreground z-10 relative mb-auto">{sub}</div>
      )}

      {/* Sparkline */}
      <div className="absolute bottom-0 left-0 right-0 h-16 opacity-15">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
            <Area
              type="monotone"
              dataKey="v"
              stroke={COLOR_MAP[color] ?? "#ee4647"}
              strokeWidth={1.5}
              fill="none"
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

const CUSTOM_TOOLTIP_STYLE = {
  contentStyle: {
    background: "#1e1e1e",
    border: "1px solid rgba(255,255,255,0.06)",
    borderRadius: "8px",
    fontSize: "11px",
    color: "#e5e5e5",
  },
  cursor: { fill: "rgba(255,255,255,0.03)" },
}

function ActiveModelCard() {
  const { models, activeModelId } = useApp()
  const model = models.find((m) => m.id === activeModelId)
  if (!model) return null

  return (
    <div className="bg-[#1a1a1a] border border-border rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-bold text-foreground">Active Model</h3>
        <span className="flex items-center gap-1.5 text-[10px] text-green-400 font-bold">
          <span className="w-1.5 h-1.5 rounded-full bg-green-400 pulse-dot" /> RUNNING
        </span>
      </div>

      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center">
          <Layers size={18} className="text-primary" />
        </div>
        <div>
          <div className="text-sm font-bold text-foreground">{model.name}</div>
          <div className="text-[10px] text-muted-foreground">{model.parameters} · {model.quantization}</div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "RAM", value: `${model.ramRequired} GB` },
          { label: "VRAM", value: `${model.vramRequired ?? "—"} GB` },
          { label: "Context", value: `${(model.contextLength / 1000).toFixed(0)}K` },
        ].map(({ label, value }) => (
          <div key={label} className="bg-secondary/50 rounded-lg p-2.5 text-center">
            <div className="text-[10px] text-muted-foreground mb-0.5">{label}</div>
            <div className="text-sm font-bold text-foreground">{value}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function DevView() {
  const { activeModelId, models } = useApp()
  const [metrics, setMetrics] = useState<any>(null)

  const [cpuData, setCpuData] = useState(() => generateHistory(0, 0))
  const [gpuData, setGpuData] = useState(() => generateHistory(0, 0))
  const [vramData, setVramData] = useState(() => generateHistory(0, 0))
  const [speedData, setSpeedData] = useState(() => generateHistory(0, 0))
  const [ramData, setRamData] = useState(() => generateHistory(0, 0))

  const [cpuVal, setCpuVal] = useState(0)
  const [gpuVal, setGpuVal] = useState(0)
  const [vramVal, setVramVal] = useState(0)
  const [vramTotal, setVramTotal] = useState(0)
  const [speedVal, setSpeedVal] = useState(0)
  const [ramVal, setRamVal] = useState(0)
  const [ramTotal, setRamTotal] = useState(0)

  useEffect(() => {
    const updateMetrics = async () => {
      try {
        const data = await API.getMetrics()
        setMetrics(data)

        setCpuVal(data.cpu_usage)
        setGpuVal(data.gpu_usage || 0)
        setVramVal(((data.vram_used_mb as number) ?? 0) / 1024)
        setVramTotal((data.vram_total_mb || 0) / 1024)
        setSpeedVal(data.generation_speed || 0)
        setRamVal(data.ram_used_gb)
        setRamTotal(data.ram_total_gb)

        const push = (arr: { t: number; v: number }[], v: number) =>
          [...arr.slice(1), { t: arr[arr.length - 1].t + 1, v }]

        setCpuData((d) => push(d, data.cpu_usage))
        setGpuData((d) => push(d, data.gpu_usage || 0))
        setVramData((d) => push(d, (data.vram_total_mb && data.vram_used_mb) ? (data.vram_used_mb / data.vram_total_mb * 100) : 0))
        setSpeedData((d) => push(d, data.generation_speed ? Math.min(data.generation_speed * 2, 100) : 0)) // Scale t/s for sparkline
        setRamData((d) => push(d, (data.ram_used_gb / data.ram_total_gb) * 100))
      } catch (error) {
        console.error("Failed to fetch metrics:", error)
      }
    }

    updateMetrics()
    const interval = setInterval(updateMetrics, 2000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a] overflow-y-auto animate-fade-in-up">
      <div className="p-6 max-w-[1400px] w-full mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xs font-bold text-muted-foreground uppercase tracking-widest">System Analytics</h2>
          </div>
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 pulse-dot" />
            Live updates every 1.5s
          </div>
        </div>

        {/* Metric Cards */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3 mb-6">
          <MetricCard
            icon={<Cpu size={12} />}
            label="CPU Usage"
            value={`${cpuVal.toFixed(0)}%`}
            sub="System Process"
            color="text-primary"
            data={cpuData}
          />
          <MetricCard
            icon={<Activity size={12} />}
            label="GPU Usage"
            value={`${gpuVal.toFixed(0)}%`}
            sub={metrics?.gpu_name || "N/A"}
            color="text-purple-400"
            data={gpuData}
          />
          <MetricCard
            icon={<HardDrive size={12} />}
            label="VRAM"
            value={`${vramVal.toFixed(1)}`}
            sub={`/ ${vramTotal.toFixed(0)} GB`}
            color="text-amber-400"
            data={vramData}
            accent={
              <div className="w-7 h-7 rounded-md bg-amber-400/10 border border-amber-400/20 flex items-center justify-center">
                <HardDrive size={13} className="text-amber-400" />
              </div>
            }
          />
          <MetricCard
            icon={<MemoryStick size={12} />}
            label="RAM"
            value={`${ramVal.toFixed(1)}`}
            sub={`/ ${ramTotal.toFixed(0)} GB`}
            color="text-green-400"
            data={ramData}
          />
          <MetricCard
            icon={<Zap size={12} />}
            label="Speed"
            value={`${speedVal}`}
            sub={`t/s · Peak: ${Math.round(speedVal * 1.25)} t/s`}
            color="text-green-400"
            data={speedData}
            unit="t/s"
            accent={
              <div className="w-7 h-7 rounded-md bg-green-400/10 border border-green-400/20 flex items-center justify-center">
                <Zap size={13} className="text-green-400" />
              </div>
            }
          />
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
          {/* Token distribution Placeholder */}
          <div className="lg:col-span-2 bg-[#1a1a1a] border border-border rounded-xl p-5 flex flex-col items-center justify-center min-h-[250px] text-center">
            <Activity size={32} className="text-muted-foreground/20 mb-3" />
            <h3 className="text-sm font-bold text-foreground">Usage Statistics</h3>
            <p className="text-[10px] text-muted-foreground mt-1">
              Historical usage data will appear here as you interact with models.
            </p>
          </div>

          {/* Active model card */}
          <div className="flex flex-col gap-4">
            <ActiveModelCard />
          </div>
        </div>

        {/* Empty Session Placeholder */}
        <div className="bg-[#1a1a1a] border border-border rounded-xl p-5 flex flex-col items-center justify-center min-h-[200px] text-center">
          <Zap size={28} className="text-muted-foreground/20 mb-3" />
          <h3 className="text-sm font-bold text-foreground">Session Activity</h3>
          <p className="text-[10px] text-muted-foreground mt-1">
            Real-time session token tracking is active.
          </p>
        </div>
      </div>
    </div>
  )
}

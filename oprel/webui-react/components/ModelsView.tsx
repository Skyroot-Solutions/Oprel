"use client"

import { useState } from "react"
import {
  Search,
  Download,
  Play,
  CheckCircle2,
  Layers,
  HardDrive,
  Tag,
  Zap,
  MemoryStick,
  FileCode,
  MessageSquarePlus,
  Box,
  Cpu,
} from "lucide-react"
import { cn } from "@/services/utils"
import { useApp } from "@/services/context"
import { API } from "@/services/api"
import type { AIModel } from "@/services/data"

function CompatBadge({ compat }: { compat: AIModel["compatibility"] }) {
  const map = {
    compatible: { label: "Compatible", class: "bg-green-500/10 text-green-400 border-green-500/20" },
    hybrid: { label: "Partial", class: "bg-amber-500/10 text-amber-400 border-amber-500/20" },
    incompatible: { label: "Incompatible", class: "bg-destructive/10 text-destructive border-destructive/20" },
  }
  const { label, class: cls } = map[compat]
  return (
    <span className={cn("px-2 py-0.5 rounded-md text-[10px] font-bold border", cls)}>
      {label}
    </span>
  )
}

function StatusBadge({ status }: { status: AIModel["status"] }) {
  if (status === "loaded")
    return (
      <span className="flex items-center gap-1 text-[10px] font-bold text-green-400">
        <span className="w-1.5 h-1.5 rounded-full bg-green-400 pulse-dot" /> LOADED
      </span>
    )
  if (status === "downloading")
    return <span className="text-[10px] font-bold text-amber-400">DOWNLOADING</span>
  if (status === "available")
    return <span className="text-[10px] font-bold text-blue-400 truncate max-w-[60px]">READY</span>
  return <span className="text-[10px] font-bold text-muted-foreground uppercase opacity-50">REGISTRY</span>
}

function ModelListItem({
  model,
  isSelected,
  onSelect,
}: {
  model: AIModel
  isSelected: boolean
  onSelect: () => void
}) {
  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-full text-left p-3 rounded-lg transition-all border",
        isSelected
          ? "bg-secondary border-primary/30 text-foreground"
          : "border-transparent hover:bg-secondary/50 text-muted-foreground hover:text-foreground"
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-foreground truncate">{model.name}</div>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-[10px] text-muted-foreground">{model.size}</span>
            <span className="text-muted-foreground/30">·</span>
            <span className="text-[10px] text-muted-foreground font-mono">{model.quantization}</span>
          </div>
        </div>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <StatusBadge status={model.status} />
          <CompatBadge compat={model.compatibility} />
        </div>
      </div>
    </button>
  )
}

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="bg-[#0f0f0f] border border-border rounded-lg p-3 flex items-center gap-3">
      <div className="w-8 h-8 rounded-md bg-secondary flex items-center justify-center shrink-0 text-muted-foreground">
        {icon}
      </div>
      <div>
        <div className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">{label}</div>
        <div className="text-sm font-bold text-foreground">{value}</div>
      </div>
    </div>
  )
}

function ModelDetailPanel({ model }: { model: AIModel }) {
  const { setActiveModelId, setCurrentView, refreshModels, setIsModelLoading } = useApp()
  const [loading, setLoading] = useState(false)

  const handleLoad = async () => {
    setLoading(true)
    setIsModelLoading(true)
    try {
      await API.loadModel(model.id)
      await refreshModels()
      setActiveModelId(model.id)
    } catch (error) {
      console.error("Failed to load model:", error)
      alert("Failed to load model: " + (error instanceof Error ? error.message : "Unknown error"))
    } finally {
      setLoading(false)
      setIsModelLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full animate-fade-in-up">
      {/* Hero */}
      <div className="p-7 border-b border-border">
        <div className="flex items-start gap-5">
          <div className="w-14 h-14 rounded-xl bg-secondary border border-border flex items-center justify-center shrink-0">
            <Box size={28} className="text-primary" />
          </div>
          <div className="flex-1 min-w-0">
            <h1 className="text-xl font-bold text-foreground leading-tight">{model.name}</h1>
            <div className="flex items-center gap-2 mt-1.5 flex-wrap">
              <span className="text-xs text-muted-foreground">{model.publisher}</span>
              <span className="text-border">·</span>
              <CompatBadge compat={model.compatibility} />
              <StatusBadge status={model.status} />
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {model.status === "loaded" ? (
              <button
                onClick={() => setCurrentView("chat")}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-all"
              >
                <MessageSquarePlus size={15} /> Chat
              </button>
            ) : (
              <button
                onClick={handleLoad}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-all disabled:opacity-60"
              >
                {loading ? (
                  <>
                    <div className="w-3 h-3 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                    Loading...
                  </>
                ) : (
                  <>
                    <Play size={14} /> Load Model
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        <p className="mt-4 text-sm text-muted-foreground leading-relaxed">{model.description}</p>

        {/* Tags */}
        <div className="flex flex-wrap gap-1.5 mt-4">
          {model.tags.map((tag) => (
            <span
              key={tag}
              className="px-2 py-0.5 rounded-md bg-secondary text-[10px] font-bold text-muted-foreground uppercase tracking-wide border border-border"
            >
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* Stats grid */}
      <div className="p-6 border-b border-border">
        <h3 className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-4">Specifications</h3>
        <div className="grid grid-cols-2 xl:grid-cols-3 gap-3">
          <StatCard icon={<Layers size={15} />} label="Parameters" value={model.parameters} />
          <StatCard icon={<HardDrive size={15} />} label="File Size" value={model.size} />
          <StatCard icon={<Cpu size={15} />} label="Context Length" value={`${(model.contextLength / 1000).toFixed(0)}K tokens`} />
          <StatCard icon={<MemoryStick size={15} />} label="RAM Required" value={`${model.ramRequired} GB`} />
          <StatCard icon={<Zap size={15} />} label="Inference Speed" value={model.speed ?? "—"} />
          <StatCard icon={<FileCode size={15} />} label="Architecture" value={model.architecture.split("For")[0]} />
        </div>
      </div>

      {/* Details */}
      <div className="p-6 grid grid-cols-2 gap-6">
        <div>
          <h3 className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-3">Model Info</h3>
          <div className="space-y-2.5">
            {[
              { label: "Publisher", value: model.publisher },
              { label: "Architecture", value: model.architecture },
              { label: "Quantization", value: model.quantization },
              { label: "License", value: model.license },
            ].map(({ label, value }) => (
              <div key={label} className="flex items-center justify-between py-2 border-b border-border/50">
                <span className="text-xs text-muted-foreground">{label}</span>
                <span className="text-xs font-semibold text-foreground">{value}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-3">Performance</h3>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-muted-foreground">RAM Usage</span>
                <span className="font-semibold text-foreground">{model.ramRequired} GB</span>
              </div>
              <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
                <div
                  className="h-full rounded-full bg-primary transition-all"
                  style={{ width: `${Math.min((model.ramRequired / 32) * 100, 100)}%` }}
                />
              </div>
            </div>
            {model.vramRequired !== undefined && (
              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-muted-foreground">VRAM</span>
                  <span className="font-semibold text-foreground">{model.vramRequired} GB</span>
                </div>
                <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
                  <div
                    className="h-full rounded-full bg-amber-500 transition-all"
                    style={{ width: `${Math.min((model.vramRequired / 24) * 100, 100)}%` }}
                  />
                </div>
              </div>
            )}
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-muted-foreground">Context</span>
                <span className="font-semibold text-foreground">{(model.contextLength / 1000).toFixed(0)}K</span>
              </div>
              <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
                <div
                  className="h-full rounded-full bg-green-500 transition-all"
                  style={{ width: `${Math.min((model.contextLength / 131072) * 100, 100)}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export function ModelsView() {
  const { models, localModels, selectedModelDetail, setSelectedModelDetail } = useApp()
  const [search, setSearch] = useState("")
  const [filterStatus, setFilterStatus] = useState<"all" | "loaded" | "available">("all")

  const filtered = (filterStatus === "available" ? localModels : models).filter((m) => {
    const matchSearch =
      m.name.toLowerCase().includes(search.toLowerCase()) ||
      m.family.toLowerCase().includes(search.toLowerCase())

    if (filterStatus === "loaded") return m.status === "loaded" && matchSearch
    return matchSearch
  })

  const families = [...new Set(models.map((m) => m.family))]

  return (
    <div className="flex h-full animate-fade-in-up">
      {/* Left panel */}
      <div className="w-[320px] shrink-0 flex flex-col border-r border-border bg-background">
        <div className="p-4 border-b border-border">
          <h2 className="text-base font-bold text-foreground mb-3">Model Library</h2>

          {/* Search */}
          <div className="relative mb-3">
            <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search models..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full bg-[#1e1e1e] border border-border rounded-lg py-2 pl-8 pr-3 text-xs text-foreground placeholder:text-muted-foreground focus:border-primary/50 transition-all"
            />
          </div>

          {/* Filter tabs */}
          <div className="flex gap-1 bg-secondary rounded-lg p-0.5">
            {(["all", "loaded", "available"] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilterStatus(f)}
                className={cn(
                  "flex-1 py-1 text-[10px] font-bold rounded-md capitalize transition-all",
                  filterStatus === f
                    ? "bg-[#1e1e1e] text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-3">
          {families.map((family) => {
            const familyModels = filtered.filter((m) => m.family === family)
            if (!familyModels.length) return null
            return (
              <div key={family}>
                <div className="px-2 py-1 text-[10px] font-bold text-muted-foreground uppercase tracking-widest">
                  {family}
                </div>
                <div className="space-y-0.5">
                  {familyModels.map((m) => (
                    <ModelListItem
                      key={m.id}
                      model={m}
                      isSelected={selectedModelDetail?.id === m.id}
                      onSelect={() => setSelectedModelDetail(m)}
                    />
                  ))}
                </div>
              </div>
            )
          })}
          {!filtered.length && (
            <div className="text-center py-10 text-xs text-muted-foreground">No models found</div>
          )}
        </div>
      </div>

      {/* Right: detail */}
      <div className="flex-1 overflow-y-auto bg-[#121212]">
        {selectedModelDetail ? (
          <ModelDetailPanel model={selectedModelDetail} />
        ) : (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-muted-foreground">
            <Box size={40} className="opacity-20" />
            <p className="text-sm">Select a model to view details</p>
          </div>
        )}
      </div>
    </div>
  )
}

"use client"

import { useState } from "react"
import { X, RotateCcw, Save, Settings2 } from "lucide-react"
import { cn } from "@/services/utils"
import { useApp } from "@/services/context"
import { DEFAULT_SETTINGS, type GenerationSettings } from "@/services/data"

function SliderField({
  label,
  value,
  min,
  max,
  step,
  color,
  format,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  color: string
  format?: (v: number) => string
  onChange: (v: number) => void
}) {
  const pct = ((value - min) / (max - min)) * 100

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">{label}</label>
        <span className={cn("text-xs font-bold font-mono", color)}>{format ? format(value) : value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-secondary"
        style={{
          background: `linear-gradient(to right, var(--primary) 0%, var(--primary) ${pct}%, rgba(255,255,255,0.08) ${pct}%, rgba(255,255,255,0.08) 100%)`,
        }}
      />
    </div>
  )
}

export function SettingsModal() {
  const { settingsOpen, setSettingsOpen, settings, saveSettings } = useApp()
  const [draft, setDraft] = useState<GenerationSettings>(settings)

  const update = (key: keyof GenerationSettings, value: number | string) =>
    setDraft((prev) => ({ ...prev, [key]: value }))

  const save = async () => {
    await saveSettings(draft)
    setSettingsOpen(false)
  }

  const reset = () => setDraft(DEFAULT_SETTINGS)

  if (!settingsOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div
        className="w-[520px] max-h-[90vh] overflow-y-auto bg-[#1e1e1e] border border-border rounded-xl shadow-2xl animate-fade-in-up"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border sticky top-0 bg-[#1e1e1e] z-10">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
              <Settings2 size={16} className="text-primary" />
            </div>
            <h3 className="text-sm font-bold text-foreground">Generation Settings</h3>
          </div>
          <button
            onClick={() => setSettingsOpen(false)}
            className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
          >
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-6">
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-5">
              <SliderField
                label="Temperature"
                value={draft.temperature}
                min={0} max={2} step={0.05}
                color="text-primary"
                format={(v) => v.toFixed(2)}
                onChange={(v) => update("temperature", v)}
              />
              <SliderField
                label="Top P"
                value={draft.topP}
                min={0} max={1} step={0.05}
                color="text-purple-400"
                format={(v) => v.toFixed(2)}
                onChange={(v) => update("topP", v)}
              />
              <SliderField
                label="Top K"
                value={draft.topK}
                min={1} max={100} step={1}
                color="text-green-400"
                onChange={(v) => update("topK", v)}
              />
            </div>

            <div className="space-y-5">
              <SliderField
                label="Max Tokens"
                value={draft.maxTokens}
                min={256} max={8192} step={256}
                color="text-amber-400"
                onChange={(v) => update("maxTokens", v)}
              />
              <SliderField
                label="Repeat Penalty"
                value={draft.repeatPenalty}
                min={1} max={2} step={0.05}
                color="text-rose-400"
                format={(v) => v.toFixed(2)}
                onChange={(v) => update("repeatPenalty", v)}
              />
              <button
                onClick={reset}
                className="w-full py-2 mt-1 border border-border rounded-lg text-[10px] font-bold text-muted-foreground hover:text-foreground hover:bg-secondary transition-all flex items-center justify-center gap-1.5"
              >
                <RotateCcw size={11} /> Reset to Defaults
              </button>
            </div>
          </div>

          {/* System prompt */}
          <div>
            <label className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider mb-2 block">
              System Prompt
            </label>
            <textarea
              value={draft.systemPrompt}
              onChange={(e) => update("systemPrompt", e.target.value)}
              rows={4}
              className="w-full bg-[#141414] border border-border rounded-lg px-4 py-3 text-sm text-foreground resize-none focus:border-primary/50 transition-all placeholder:text-muted-foreground"
              placeholder="You are a helpful AI assistant..."
            />
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-border bg-[#171717] flex items-center justify-end gap-3">
          <button
            onClick={() => setSettingsOpen(false)}
            className="px-4 py-2 text-xs font-semibold rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
          >
            Cancel
          </button>
          <button
            onClick={save}
            className="px-5 py-2 text-xs font-bold rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-all flex items-center gap-1.5 shadow-lg shadow-primary/20"
          >
            <Save size={12} /> Save Changes
          </button>
        </div>
      </div>
    </div>
  )
}

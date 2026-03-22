'use client'

import { useState, useEffect, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import {
  Settings2, Cpu, SlidersHorizontal, ChevronLeft,
  Plus, Trash2, RefreshCw, Eye, EyeOff, Check, X,
  Loader2, Zap, ChevronDown, ChevronUp, RotateCcw,
  Globe, KeyRound, Server, BookOpen, Save,
} from 'lucide-react'
import { useApp } from '@/services/context'
import {
  type ProviderConfig, type ProviderType,
  PROVIDER_PRESETS, fetchProviderModels,
} from '@/services/providers'
import { cn } from '@/services/utils'

// ─── Sidebar Tab definitions ──────────────────────────────────────────────────

const TABS = [
  { id: 'config', label: 'Generation', icon: SlidersHorizontal },
  { id: 'presets', label: 'Prompt Presets', icon: BookOpen },
  { id: 'providers', label: 'AI Providers', icon: Globe },
] as const

type TabId = (typeof TABS)[number]['id']

// ─── System Prompt Presets ────────────────────────────────────────────────────

const SYSTEM_PRESETS = [
  { label: 'General', prompt: 'You are a helpful AI assistant.' },
  {
    label: 'Coder',
    prompt:
      'You are an expert software engineer. Write clean, efficient, well-documented code. Prefer concise explanations. Always include code examples.',
  },
  {
    label: 'Diagrams',
    prompt:
      'You are a technical diagram expert. When asked to create diagrams, output valid Mermaid syntax inside ```mermaid code blocks. Always produce syntactically correct Mermaid.',
  },
  {
    label: 'Web Builder',
    prompt:
      'You are a senior frontend engineer. When asked to build UIs, produce complete, self-contained HTML files with embedded CSS and JS inside ```html code blocks.',
  },
  {
    label: 'Writer',
    prompt:
      'You are a professional writer and editor. Help craft clear, engaging, and well-structured prose. Adapt tone to the request.',
  },
  {
    label: 'Tutor',
    prompt:
      'You are a patient and thorough tutor. Explain concepts step-by-step, use analogies and examples, and check for understanding.',
  },
  {
    label: 'Analyst',
    prompt:
      'You are a data analyst. Provide structured analysis, identify trends, and support conclusions with reasoning and data.',
  },
  {
    label: 'Data / SQL',
    prompt:
      'You are a database expert specialising in SQL. Write performant, standards-compliant queries. Explain query plans when asked.',
  },
  {
    label: 'DevOps',
    prompt:
      'You are a DevOps and cloud infrastructure expert. Provide practical, secure, and scalable solutions using industry best practices.',
  },
  {
    label: 'Coach',
    prompt:
      'You are a supportive life and productivity coach. Help the user clarify goals, overcome obstacles, and build positive habits.',
  },
]

// ─── Slider helper ────────────────────────────────────────────────────────────

function SettingSlider({
  label, value, min, max, step, onChange, format,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  format?: (v: number) => string
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <label className="text-xs font-semibold text-muted-foreground tracking-widest uppercase">
          {label}
        </label>
        <span className="text-sm font-bold text-primary tabular-nums">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1.5 bg-secondary rounded-full appearance-none cursor-pointer accent-primary"
      />
    </div>
  )
}

// ─── Provider type badge ──────────────────────────────────────────────────────

function ProviderBadge({ type }: { type: ProviderType }) {
  const preset = PROVIDER_PRESETS[type]
  return (
    <span
      className="px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider"
      style={{ color: preset.color, background: `${preset.color}22` }}
    >
      {preset.name}
    </span>
  )
}

// ─── Provider Form ────────────────────────────────────────────────────────────

function ProviderForm({
  initial,
  onSave,
  onCancel,
}: {
  initial?: Partial<ProviderConfig>
  onSave: (p: ProviderConfig) => Promise<void>
  onCancel: () => void
}) {
  const isEdit = Boolean(initial?.id)
  const [type, setType] = useState<ProviderType>(initial?.type ?? 'openai')
  const [name, setName] = useState(initial?.name ?? PROVIDER_PRESETS[initial?.type ?? 'openai'].name)
  const [apiKey, setApiKey] = useState(initial?.apiKey ?? '')
  const [baseUrl, setBaseUrl] = useState(initial?.baseUrl ?? PROVIDER_PRESETS[initial?.type ?? 'openai'].baseUrl)
  const [showKey, setShowKey] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const preset = PROVIDER_PRESETS[type]

  // Update defaults when type changes (only for new providers)
  useEffect(() => {
    if (!isEdit) {
      setName(PROVIDER_PRESETS[type].name)
      setBaseUrl(PROVIDER_PRESETS[type].baseUrl)
    }
  }, [type, isEdit])

  const handleSave = async () => {
    if (!apiKey.trim() && type !== 'openai-compatible') {
      setError('API key is required')
      return
    }
    setSaving(true)
    setError('')
    try {
      const id = initial?.id ?? `${type}-${Date.now()}`
      await onSave({
        id,
        name: name.trim() || preset.name,
        type,
        apiKey: apiKey.trim(),
        baseUrl: baseUrl.trim() || preset.baseUrl,
        enabled: initial?.enabled ?? true,
        enabledModelIds: initial?.enabledModelIds ?? [],
        availableModelIds: initial?.availableModelIds ?? [],
      })
    } catch (e: any) {
      setError(e.message || 'Failed to save provider')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-4 border border-border rounded-xl p-5 bg-card/60">
      <h3 className="font-semibold text-foreground">{isEdit ? 'Edit Provider' : 'New Provider'}</h3>

      {/* Type selector */}
      {!isEdit && (
        <div>
          <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-2">
            Provider Type
          </label>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {(Object.keys(PROVIDER_PRESETS) as ProviderType[]).map(t => {
              const p = PROVIDER_PRESETS[t]
              return (
                <button
                  key={t}
                  onClick={() => setType(t)}
                  className={cn(
                    'flex flex-col items-start gap-1 px-3 py-2.5 rounded-lg border text-left transition-all text-sm',
                    type === t
                      ? 'border-primary/50 bg-primary/10 text-foreground'
                      : 'border-border bg-secondary/40 text-muted-foreground hover:border-border/80 hover:bg-secondary/60'
                  )}
                >
                  <span className="font-semibold text-xs" style={{ color: p.color }}>{p.name}</span>
                  <span className="text-[10px] leading-tight opacity-70">{p.description}</span>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Name */}
      <div>
        <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-1.5">
          Display Name
        </label>
        <input
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder={preset.name}
          className="w-full bg-secondary/60 border border-border rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
        />
      </div>

      {/* API Key */}
      <div>
        <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-1.5">
          API Key
          {preset.docsUrl && (
            <a href={preset.docsUrl} target="_blank" rel="noopener noreferrer"
              className="ml-2 normal-case text-primary/70 hover:text-primary transition-colors">
              Get key ↗
            </a>
          )}
        </label>
        <div className="relative">
          <input
            type={showKey ? 'text' : 'password'}
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            placeholder="sk-..."
            className="w-full bg-secondary/60 border border-border rounded-lg px-3 py-2 pr-10 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 font-mono"
          />
          <button
            onClick={() => setShowKey(s => !s)}
            className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
          >
            {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        </div>
      </div>

      {/* Base URL (for custom / compatible) */}
      {(type === 'openai-compatible' || type === 'nvidia' || type === 'groq' || type === 'openrouter') && (
        <div>
          <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest block mb-1.5">
            Base URL
          </label>
          <input
            value={baseUrl}
            onChange={e => setBaseUrl(e.target.value)}
            placeholder={preset.baseUrl || 'https://...'}
            className="w-full bg-secondary/60 border border-border rounded-lg px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/50 font-mono"
          />
        </div>
      )}

      {error && (
        <p className="text-destructive text-xs bg-destructive/10 border border-destructive/20 px-3 py-2 rounded-lg">
          {error}
        </p>
      )}

      <div className="flex gap-2 justify-end">
        <button
          onClick={onCancel}
          className="px-4 py-2 text-sm rounded-lg border border-border text-muted-foreground hover:bg-secondary transition-all"
        >
          Cancel
        </button>
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-4 py-2 text-sm rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-all flex items-center gap-2 disabled:opacity-60"
        >
          {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
          {isEdit ? 'Save Changes' : 'Add Provider'}
        </button>
      </div>
    </div>
  )
}

// ─── Provider Card ────────────────────────────────────────────────────────────

function ProviderCard({
  provider,
  onUpdate,
  onDelete,
}: {
  provider: ProviderConfig
  onUpdate: (p: ProviderConfig) => Promise<void>
  onDelete: (id: string) => Promise<void>
}) {
  const [expanded, setExpanded] = useState(false)
  const [fetching, setFetching] = useState(false)
  const [fetchError, setFetchError] = useState('')
  const [deleting, setDeleting] = useState(false)
  const preset = PROVIDER_PRESETS[provider.type]

  const handleFetchModels = async () => {
    setFetching(true)
    setFetchError('')
    try {
      const models = await fetchProviderModels(provider)
      await onUpdate({ ...provider, availableModelIds: models, lastFetched: new Date().toISOString() })
    } catch (e: any) {
      setFetchError(e.message || 'Failed to fetch models')
    } finally {
      setFetching(false)
    }
  }

  const toggleModel = async (modelId: string) => {
    const already = provider.enabledModelIds.includes(modelId)
    const next = already
      ? provider.enabledModelIds.filter(m => m !== modelId)
      : [...provider.enabledModelIds, modelId]
    await onUpdate({ ...provider, enabledModelIds: next })
  }

  const toggleProvider = async () => {
    await onUpdate({ ...provider, enabled: !provider.enabled })
  }

  const handleDelete = async () => {
    if (!confirm(`Remove provider "${provider.name}"?`)) return
    setDeleting(true)
    try { await onDelete(provider.id) } finally { setDeleting(false) }
  }

  return (
    <div className={cn(
      'border rounded-xl overflow-hidden transition-all',
      provider.enabled ? 'border-border' : 'border-border/40 opacity-60'
    )}>
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 bg-card/60">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 text-white font-bold text-xs"
          style={{ background: preset.color }}
        >
          {provider.name.slice(0, 2).toUpperCase()}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm text-foreground truncate">{provider.name}</span>
            <ProviderBadge type={provider.type} />
          </div>
          <p className="text-xs text-muted-foreground">
            {provider.enabledModelIds?.length || 0} model{(provider.enabledModelIds?.length !== 1) ? 's' : ''} enabled
            {provider.lastFetched && (
              (() => {
                const date = new Date(provider.lastFetched);
                return isNaN(date.getTime()) ? '' : ` · fetched ${date.toLocaleDateString()}`;
              })()
            )}
          </p>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {/* Enable toggle */}
          <button
            onClick={toggleProvider}
            className={cn(
              'px-2.5 py-1 rounded-md text-xs font-medium transition-all border',
              provider.enabled
                ? 'bg-primary/10 border-primary/30 text-primary'
                : 'border-border text-muted-foreground hover:border-border'
            )}
          >
            {provider.enabled ? 'Enabled' : 'Disabled'}
          </button>
          <button
            onClick={() => setExpanded(e => !e)}
            className="w-7 h-7 rounded-lg flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
          >
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          <button
            onClick={handleDelete}
            disabled={deleting}
            className="w-7 h-7 rounded-lg flex items-center justify-center text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-all"
          >
            {deleting ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={13} />}
          </button>
        </div>
      </div>

      {/* Model list */}
      {expanded && (
        <div className="px-4 py-4 border-t border-border/50 bg-background/30 space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground font-medium">
              {provider.availableModelIds.length > 0
                ? `${provider.availableModelIds.length} models available — toggle ones to enable`
                : 'No models fetched yet'}
            </p>
            <button
              onClick={handleFetchModels}
              disabled={fetching}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border border-border hover:bg-secondary text-muted-foreground hover:text-foreground transition-all disabled:opacity-60"
            >
              {fetching ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
              {fetching ? 'Fetching…' : 'Fetch Models'}
            </button>
          </div>

          {fetchError && (
            <p className="text-destructive text-xs bg-destructive/10 border border-destructive/20 px-3 py-2 rounded-lg">
              {fetchError}
            </p>
          )}

          {provider.availableModelIds.length > 0 && (
            <div className="max-h-64 overflow-y-auto space-y-1 pr-1">
              {provider.availableModelIds.map(modelId => {
                const enabled = provider.enabledModelIds.includes(modelId)
                return (
                  <label
                    key={modelId}
                    className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-secondary/60 cursor-pointer transition-all group"
                  >
                    <div className={cn(
                      'w-4 h-4 rounded border flex items-center justify-center shrink-0 transition-all',
                      enabled
                        ? 'bg-primary border-primary'
                        : 'border-border group-hover:border-primary/50'
                    )}>
                      {enabled && <Check size={10} className="text-primary-foreground" />}
                    </div>
                    <input
                      type="checkbox" checked={enabled} onChange={() => toggleModel(modelId)}
                      className="sr-only"
                    />
                    <span className="text-xs font-mono text-foreground/80 truncate flex-1">{modelId}</span>
                    {enabled && (
                      <span
                        className="text-[9px] font-bold px-1.5 py-0.5 rounded uppercase"
                        style={{ color: preset.color, background: `${preset.color}22` }}
                      >
                        Active
                      </span>
                    )}
                  </label>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Main Settings Page ───────────────────────────────────────────────────────

export default function SettingsPage() {
  const router = useRouter()
  const {
    settings, setSettings, saveSettings,
    providers, saveProvider, removeProvider,
  } = useApp()
  const [activeTab, setActiveTab] = useState<TabId>('config')
  const [localSettings, setLocalSettings] = useState(settings)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [savedFeedback, setSavedFeedback] = useState(false)
  const [showAddProvider, setShowAddProvider] = useState(false)

  // Sync when settings load from server
  useEffect(() => { setLocalSettings(settings) }, [settings])

  const updateSetting = <K extends keyof typeof localSettings>(key: K, value: typeof localSettings[K]) => {
    setLocalSettings(prev => ({ ...prev, [key]: value }))
    setDirty(true)
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      await saveSettings(localSettings)
      setDirty(false)
      setSavedFeedback(true)
      setTimeout(() => setSavedFeedback(false), 2000)
    } finally {
      setSaving(false)
    }
  }

  const handleReset = () => {
    const defaults = {
      temperature: 0.7, topP: 0.9, topK: 40, maxTokens: 4096, repeatPenalty: 1.1,
      systemPrompt: 'You are a helpful AI assistant.',
    }
    setLocalSettings(prev => ({ ...prev, ...defaults }))
    setDirty(true)
  }

  return (
    <div className="flex h-screen w-full bg-background overflow-hidden">
      {/* ── Left sidebar ── */}
      <nav className="w-56 shrink-0 flex flex-col border-r border-border bg-card/40 px-3 py-4 gap-1">
        {/* Back */}
        <button
          onClick={() => router.push('/')}
          className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-secondary rounded-lg transition-all mb-3"
        >
          <ChevronLeft size={15} />
          <span>Back</span>
        </button>

        <p className="text-[10px] font-bold text-muted-foreground/60 uppercase tracking-widest px-3 mb-1">
          Settings
        </p>

        {TABS.map(tab => {
          const Icon = tab.icon
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                'flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm font-medium transition-all text-left',
                activeTab === tab.id
                  ? 'bg-primary/10 text-primary border border-primary/20'
                  : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
              )}
            >
              <Icon size={15} />
              {tab.label}
            </button>
          )
        })}
      </nav>

      {/* ── Content area ── */}
      <main className="flex-1 min-w-0 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-6 py-8 space-y-6">

          {/* ── Generation Config ── */}
          {activeTab === 'config' && (
            <>
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-xl font-bold text-foreground">Generation Settings</h1>
                  <p className="text-sm text-muted-foreground mt-0.5">
                    Controls how the model generates responses
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={handleReset}
                    className="flex items-center gap-1.5 px-3 py-2 text-sm rounded-lg border border-border text-muted-foreground hover:bg-secondary transition-all"
                  >
                    <RotateCcw size={13} /> Reset
                  </button>
                  <button
                    onClick={handleSave}
                    disabled={!dirty || saving}
                    className={cn(
                      'flex items-center gap-1.5 px-4 py-2 text-sm rounded-lg transition-all font-medium',
                      dirty
                        ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                        : 'bg-secondary text-muted-foreground cursor-not-allowed'
                    )}
                  >
                    {saving ? <Loader2 size={13} className="animate-spin" /> :
                      savedFeedback ? <Check size={13} /> : <Save size={13} />}
                    {savedFeedback ? 'Saved!' : 'Save'}
                  </button>
                </div>
              </div>

              <div className="space-y-6 bg-card/50 border border-border rounded-xl p-5">
                <SettingSlider
                  label="Temperature" value={localSettings.temperature} min={0} max={2} step={0.05}
                  onChange={v => updateSetting('temperature', v)}
                  format={v => v.toFixed(2)}
                />
                <SettingSlider
                  label="Top P" value={localSettings.topP} min={0} max={1} step={0.05}
                  onChange={v => updateSetting('topP', v)}
                  format={v => v.toFixed(2)}
                />
                <SettingSlider
                  label="Top K" value={localSettings.topK} min={1} max={100} step={1}
                  onChange={v => updateSetting('topK', v)}
                />
                <SettingSlider
                  label="Max Tokens" value={localSettings.maxTokens} min={256} max={32768} step={256}
                  onChange={v => updateSetting('maxTokens', v)}
                  format={v => v.toLocaleString()}
                />
                <SettingSlider
                  label="Repeat Penalty" value={localSettings.repeatPenalty} min={1} max={2} step={0.05}
                  onChange={v => updateSetting('repeatPenalty', v)}
                  format={v => v.toFixed(2)}
                />
              </div>
            </>
          )}

          {/* ── Prompt Presets ── */}
          {activeTab === 'presets' && (
            <>
              <div>
                <h1 className="text-xl font-bold text-foreground">Prompt Presets</h1>
                <p className="text-sm text-muted-foreground mt-0.5">
                  Choose a preset to instantly set the system prompt, or write your own below
                </p>
              </div>

              {/* Preset grid */}
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {SYSTEM_PRESETS.map(p => (
                  <button
                    key={p.label}
                    onClick={() => {
                      updateSetting('systemPrompt', p.prompt)
                    }}
                    className={cn(
                      'text-left px-3 py-2.5 rounded-xl border text-sm font-medium transition-all',
                      localSettings.systemPrompt === p.prompt
                        ? 'border-primary/50 bg-primary/10 text-primary'
                        : 'border-border bg-card/50 text-muted-foreground hover:border-border/80 hover:bg-secondary/60 hover:text-foreground'
                    )}
                  >
                    {p.label}
                  </button>
                ))}
              </div>

              {/* Editable textarea */}
              <div className="space-y-2">
                <label className="text-xs font-semibold text-muted-foreground uppercase tracking-widest">
                  System Prompt
                </label>
                <textarea
                  value={localSettings.systemPrompt}
                  onChange={e => updateSetting('systemPrompt', e.target.value)}
                  rows={8}
                  className="w-full bg-secondary/40 border border-border rounded-xl px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/40 resize-y font-mono leading-relaxed"
                  placeholder="You are a helpful AI assistant..."
                />
                <p className="text-xs text-muted-foreground">
                  Injected as the system message for every new conversation.
                </p>
              </div>

              <button
                onClick={handleSave}
                disabled={!dirty || saving}
                className={cn(
                  'flex items-center gap-1.5 px-4 py-2 text-sm rounded-lg transition-all font-medium',
                  dirty
                    ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                    : 'bg-secondary text-muted-foreground cursor-not-allowed'
                )}
              >
                {saving ? <Loader2 size={13} className="animate-spin" /> :
                  savedFeedback ? <Check size={13} /> : <Save size={13} />}
                {savedFeedback ? 'Saved!' : 'Save Prompt'}
              </button>
            </>
          )}

          {/* ── AI Providers ── */}
          {activeTab === 'providers' && (
            <>
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-xl font-bold text-foreground">AI Providers</h1>
                  <p className="text-sm text-muted-foreground mt-0.5">
                    Connect external providers — models appear in the model selector
                  </p>
                </div>
                <button
                  onClick={() => setShowAddProvider(p => !p)}
                  className="flex items-center gap-1.5 px-3 py-2 text-sm rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-all"
                >
                  <Plus size={14} />
                  Add Provider
                </button>
              </div>

              {/* Add provider form */}
              {showAddProvider && (
                <ProviderForm
                  onSave={async p => { await saveProvider(p); setShowAddProvider(false) }}
                  onCancel={() => setShowAddProvider(false)}
                />
              )}

              {/* Provider list */}
              <div className="space-y-3">
                {providers.length === 0 && !showAddProvider && (
                  <div className="flex flex-col items-center justify-center py-16 gap-3 text-center border border-dashed border-border/50 rounded-xl">
                    <Globe size={32} className="text-muted-foreground/40" />
                    <p className="text-sm text-muted-foreground">No providers configured yet</p>
                    <button
                      onClick={() => setShowAddProvider(true)}
                      className="text-xs text-primary hover:underline"
                    >
                      Add your first provider
                    </button>
                  </div>
                )}
                {providers.map(p => (
                  <ProviderCard
                    key={p.id}
                    provider={p}
                    onUpdate={saveProvider}
                    onDelete={removeProvider}
                  />
                ))}
              </div>

              {providers.length > 0 && (
                <p className="text-xs text-muted-foreground bg-secondary/40 rounded-lg px-4 py-3 border border-border/50">
                  <strong className="text-foreground">Tip:</strong> Enable individual models by expanding a provider and fetching its model list. Enabled models appear in the model selector with a provider tag.
                </p>
              )}
            </>
          )}

        </div>
      </main>
    </div>
  )
}

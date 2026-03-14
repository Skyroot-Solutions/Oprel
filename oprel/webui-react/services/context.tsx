"use client"

import { createContext, useContext, useState, useCallback, useEffect, type ReactNode } from "react"
import { DEFAULT_SETTINGS, type View, type Conversation, type ChatMessage, type AIModel, type GenerationSettings } from "@/services/data"
import { API, Model, Conversation as ApiConversation } from "@/services/api"

interface AppContextType {
  // Navigation
  currentView: View
  setCurrentView: (view: View) => void

  // Conversations
  conversations: Conversation[]
  activeConversationId: string | null
  setActiveConversationId: (id: string | null) => void
  createConversation: () => void
  deleteConversation: (id: string) => void
  addMessage: (conversationId: string, message: ChatMessage) => void
  setConversationMessages: (conversationId: string, messages: ChatMessage[]) => void
  linkConversation: (tempId: string, realId: string) => void
  refreshConversations: () => Promise<void>

  // Models
  models: AIModel[]          // All models (merged)
  localModels: AIModel[]     // Downloaded/on-disk models only
  activeModelId: string
  setActiveModelId: (id: string) => void
  selectedModelDetail: AIModel | null
  setSelectedModelDetail: (model: AIModel | null) => void
  refreshModels: () => Promise<void>

  // Settings
  settings: GenerationSettings
  setSettings: (s: GenerationSettings) => void
  saveSettings: (s: GenerationSettings) => Promise<void>
  settingsOpen: boolean
  setSettingsOpen: (v: boolean) => void

  // UI
  isGenerating: boolean
  setIsGenerating: (v: boolean) => void
  isModelLoading: boolean
  setIsModelLoading: (v: boolean) => void
}

const AppContext = createContext<AppContextType | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [currentView, setCurrentView] = useState<View>("chat")
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null)
  const [models, setModels] = useState<AIModel[]>([])
  const [localModels, setLocalModels] = useState<AIModel[]>([])
  const [activeModelId, setActiveModelId] = useState<string>("")
  const [selectedModelDetail, setSelectedModelDetail] = useState<AIModel | null>(null)
  const [settings, setSettings] = useState<GenerationSettings>(DEFAULT_SETTINGS)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isModelLoading, setIsModelLoading] = useState(false)

  const mapApiModels = (apiModels: Model[]): AIModel[] => {
    return apiModels.map(m => ({
      id: m.model_id,
      name: m.name || m.model_id,
      family: m.model_id.split('/')[0],
      size: m.size_gb ? `${m.size_gb.toFixed(1)} GB` : "Unknown",
      quantization: m.quantization || "Unknown",
      contextLength: 0, // Not provided by simple list
      ramRequired: 0,
      status: m.loaded ? "loaded" : "available",
      tags: [],
      description: "",
      publisher: m.model_id.split('/')[0],
      architecture: m.backend || "llama.cpp",
      parameters: "Unknown",
      license: "Unknown",
      compatibility: "compatible",
      speed: m.status === 'loaded' ? 'Active' : undefined
    }));
  };

  const refreshModels = useCallback(async () => {
    try {
      const [apiModels, registryData] = await Promise.all([
        API.fetchModels(),
        API.fetchRegistryModels()
      ]);

      const imageModelKeywords = [
        'flux', 'stable-diffusion', 'sdxl', 'aura', 'deepfloyd',
        'kolors', 'mochi', 'cogvideo', 'hunyuan', 'diffusion',
        'pixart', 'playground', 'black-forest-labs'
      ];

      const localMapped = mapApiModels(apiModels);
      const filteredLocal = localMapped.filter(m => !imageModelKeywords.some(kw => m.id.toLowerCase().includes(kw)));
      setLocalModels(filteredLocal);

      // Map registry models
      const registryModels: AIModel[] = [];
      Object.entries(registryData.models).forEach(([category, catModels]: [string, any]) => {
        Object.entries(catModels).forEach(([alias, repoId]: [string, any]) => {
          // Skip if already in local
          if (localMapped.some(m => m.id === repoId)) return;

          // Skip image generation models
          if (imageModelKeywords.some(kw => repoId.toLowerCase().includes(kw) || alias.toLowerCase().includes(kw))) return;

          registryModels.push({
            id: repoId,
            name: alias,
            family: repoId.split('/')[0],
            size: "—",
            quantization: "Multiple quants",
            contextLength: 0,
            ramRequired: 0,
            status: "available",
            tags: [category],
            description: `${category} model available in Oprel Registry`,
            publisher: repoId.split('/')[0],
            architecture: "llama.cpp",
            parameters: "Unknown",
            license: "Unknown",
            compatibility: "compatible"
          });
        });
      });

      const merged = [...filteredLocal, ...registryModels];
      setModels(merged);

      // Auto-select loaded model
      const loaded = localMapped.find(m => m.status === 'loaded');
      if (loaded && !activeModelId) {
        setActiveModelId(loaded.id);
        setSelectedModelDetail(loaded);
      } else if (localMapped.length > 0 && !activeModelId) {
        setActiveModelId(localMapped[0].id);
        setSelectedModelDetail(localMapped[0]);
      }
    } catch (error) {
      console.error("Failed to fetch models:", error);
    }
  }, [activeModelId]);

  const refreshConversations = useCallback(async () => {
    try {
      const apiConvs = await API.fetchConversations();
      setConversations(prev => {
        // Build map of existing localized messages
        const messageMap = new Map(prev.map(c => [c.id, c.messages]));

        const updated = apiConvs.map(c => {
          return {
            id: c.id,
            title: c.title || "New Chat",
            messages: messageMap.get(c.id) || [],
            createdAt: new Date(c.created_at),
            modelId: c.model_id
          };
        });

        // Keep temp conversations that aren't yet on server
        const temps = prev.filter(c => c.id.startsWith('temp-'));
        return [...temps, ...updated];
      });
    } catch (error) {
      console.error("Failed to fetch conversations:", error);
    }
  }, []);

  const refreshSettings = useCallback(async () => {
    try {
      const apiSettings = await API.fetchSettings();
      setSettings({
        temperature: apiSettings.temperature,
        topP: apiSettings.top_p,
        topK: apiSettings.top_k,
        maxTokens: apiSettings.max_tokens,
        repeatPenalty: apiSettings.repeat_penalty,
        systemPrompt: apiSettings.system_instruction || DEFAULT_SETTINGS.systemPrompt
      });
    } catch (error) {
      console.error("Failed to fetch settings:", error);
    }
  }, []);

  useEffect(() => {
    refreshModels();
    refreshConversations();
    refreshSettings();
  }, [refreshModels, refreshConversations, refreshSettings]);

  const createConversation = useCallback(() => {
    const newId = `temp-${Date.now()}`;
    const newConv: Conversation = {
      id: newId,
      title: "New Chat",
      messages: [],
      createdAt: new Date(),
      modelId: activeModelId,
    };
    setConversations((prev) => [newConv, ...prev]);
    setActiveConversationId(newId);
    setCurrentView("chat");
  }, [activeModelId]);

  const deleteConversation = useCallback(async (id: string) => {
    try {
      if (!id.startsWith('temp-')) {
        await API.deleteConversation(id);
      }
      setConversations((prev) => prev.filter((c) => c.id !== id));
      setActiveConversationId((prev) => (prev === id ? null : prev));
    } catch (error) {
      console.error("Failed to delete conversation:", error);
    }
  }, []);

  const saveSettingsToServer = useCallback(async (newSettings: GenerationSettings) => {
    try {
      await API.saveSettings({
        temperature: newSettings.temperature,
        top_p: newSettings.topP,
        top_k: newSettings.topK,
        max_tokens: newSettings.maxTokens,
        repeat_penalty: newSettings.repeatPenalty,
        system_instruction: newSettings.systemPrompt
      });
      setSettings(newSettings);
    } catch (error) {
      console.error("Failed to save settings:", error);
    }
  }, []);

  const addMessage = useCallback((conversationId: string, message: ChatMessage) => {
    setConversations((prev) => {
      const exists = prev.some(c => c.id === conversationId);
      if (!exists) {
        // Create new session entry if it doesn't exist (e.g. for first message)
        const newConv: Conversation = {
          id: conversationId,
          title: message.content.slice(0, 50) + (message.content.length > 50 ? "..." : ""),
          messages: [message],
          createdAt: new Date(),
          modelId: activeModelId,
        };
        return [newConv, ...prev];
      }

      return prev.map((c) => {
        if (c.id !== conversationId) return c
        const updated = { ...c, messages: [...c.messages, message] }
        if ((!c.title || c.title === "New Chat") && message.role === "user") {
          updated.title = message.content.slice(0, 50) + (message.content.length > 50 ? "..." : "")
        }
        return updated
      })
    })
  }, [activeModelId])

  const linkConversation = useCallback((tempId: string, realId: string) => {
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== tempId) return c
        return { ...c, id: realId }
      })
    )
    setActiveConversationId(realId)
  }, [])

  const setConversationMessages = useCallback((conversationId: string, messages: ChatMessage[]) => {
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== conversationId) return c
        return { ...c, messages }
      })
    )
  }, [])

  return (
    <AppContext.Provider
      value={{
        currentView,
        setCurrentView,
        conversations,
        activeConversationId,
        setActiveConversationId,
        createConversation,
        deleteConversation,
        addMessage,
        linkConversation,
        refreshConversations,
        models,
        activeModelId,
        setActiveModelId,
        selectedModelDetail,
        setSelectedModelDetail,
        refreshModels,
        setConversationMessages,
        localModels,
        settings,
        setSettings,
        saveSettings: saveSettingsToServer,
        settingsOpen,
        setSettingsOpen,
        isGenerating,
        setIsGenerating,
        isModelLoading,
        setIsModelLoading,
      }}
    >
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error("useApp must be used within AppProvider")
  return ctx
}

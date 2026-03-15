"use client"

import { useEffect, useRef, useState, useCallback, useMemo } from "react"
import {
  Send,
  ChevronDown,
  Camera,
  Zap,
  Brain,
  Bot,
  Copy,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  Sparkles,
} from "lucide-react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { cn } from "@/services/utils"
import { useApp } from "@/services/context"
import { API, ChatMessage as ApiChatMessage } from "@/services/api"
import type { ChatMessage, Conversation, AIModel } from "@/services/data"
import { useToast } from "@/components/ui/use-toast"

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 py-2">
      <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground typing-dot" />
      <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground typing-dot" />
      <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground typing-dot" />
    </div>
  )
}

function getContentText(content: string | any[]): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map(part => {
        if (typeof part === 'string') return part;
        if (part.type === 'text') return part.text;
        return '';
      })
      .join('');
  }
  return '';
}

function renderImages(content: string | any[]) {
  if (typeof content === 'string') return null;
  if (Array.isArray(content)) {
    const images = content.filter(part => part.type === 'image_url');
    if (images.length === 0) return null;
    return (
      <div className="flex flex-wrap gap-2 mt-2 mb-2">
        {images.map((img, i) => (
          <img
            key={i}
            src={img.image_url.url}
            className="max-w-[400px] max-h-[300px] rounded-lg border border-border object-contain bg-black/20"
            alt="Uploaded content"
          />
        ))}
      </div>
    );
  }
  return null;
}
function isVisionModel(model: AIModel | undefined) {
  if (!model) return false;
  const visionKeywords = ['vl', 'vision', 'llava', 'qwen2-vl', 'qwen2.5-vl', 'pixtral'];
  return visionKeywords.some(kw => model.id.toLowerCase().includes(kw) || model.name.toLowerCase().includes(kw));
}

function ThinkingBlock({ content, renderers }: { content: string, renderers: any }) {
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <div className={cn(
      "my-4 rounded-xl bg-secondary/20 border border-primary/10 overflow-hidden transition-all duration-300",
      isExpanded ? "pb-4" : "pb-0"
    )}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-4 py-3 text-primary/70 hover:text-primary hover:bg-primary/5 transition-all group"
      >
        <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-widest">
          <Brain size={13} className={cn("transition-transform duration-500", isExpanded && "rotate-[360deg]")} />
          Thinking Process
        </div>
        <ChevronDown size={14} className={cn("transition-transform duration-300", isExpanded && "rotate-180")} />
      </button>

      {isExpanded && (
        <div className="px-4 text-muted-foreground italic text-sm border-t border-primary/5 pt-3 animate-in fade-in slide-in-from-top-2 duration-300">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={renderers}>{content}</ReactMarkdown>
        </div>
      )}
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user"
  const [copied, setCopied] = useState(false)

  const copyContent = () => {
    const text = getContentText(message.content);
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const renderers = useMemo(() => ({
    code({ node, inline, className, children, ...props }: any) {
      const match = /language-(\w+)/.exec(className || '')
      return !inline && match ? (
        <div className="my-3 rounded-lg overflow-hidden border border-border bg-[#0a0a0a]">
          <div className="flex items-center justify-between px-4 py-2 bg-[#141414] border-b border-border">
            <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">
              {match[1]}
            </span>
            <button
              onClick={() => navigator.clipboard.writeText(String(children).replace(/\n$/, ''))}
              className="text-[10px] text-muted-foreground hover:text-foreground flex items-center gap-1 transition-colors"
            >
              <Copy size={11} /> Copy
            </button>
          </div>
          <pre className="p-4 text-xs font-mono text-[#e5e5e5] overflow-x-auto leading-relaxed">
            <code className={className} {...props}>
              {children}
            </code>
          </pre>
        </div>
      ) : (
        <code className={cn("bg-secondary/50 px-1 rounded", className)} {...props}>
          {children}
        </code>
      )
    }
  }), []);

  // Handle thinking tags
  const processedContent = useMemo(() => {
    const rawText = getContentText(message.content);
    const images = renderImages(message.content);

    if (!rawText) return images;

    // Robust extraction similar to legacy bridge
    let thinking = "";
    let cleaned = rawText;

    const startIdx = rawText.indexOf("<think>");
    if (startIdx !== -1) {
      const endIdx = rawText.indexOf("</think>", startIdx + 7);
      if (endIdx !== -1) {
        thinking = rawText.substring(startIdx + 7, endIdx).trim();
        cleaned = rawText.substring(0, startIdx).trim() + "\n\n" + rawText.substring(endIdx + 8).trim();
      } else {
        // Ongoing thinking
        thinking = rawText.substring(startIdx + 7).trim();
        cleaned = rawText.substring(0, startIdx).trim();
      }
    }

    return (
      <div className="oprel-response-container">
        {images}
        {thinking && <ThinkingBlock content={thinking} renderers={renderers} />}
        {cleaned && (
          <div className="prose prose-sm prose-invert max-w-none text-foreground leading-relaxed">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={renderers}>{cleaned}</ReactMarkdown>
          </div>
        )}
      </div>
    );
  }, [message.content, renderers]);

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[85%] px-4 py-3 rounded-2xl rounded-br-sm bg-primary/10 border border-primary/20 text-foreground text-sm leading-relaxed whitespace-pre overflow-x-auto">
          {renderImages(message.content)}
          {getContentText(message.content)}
        </div>
      </div>
    )
  }

  return (
    <div className="mb-6 group">
      <div className="flex items-start gap-3 max-w-[780px]">
        <div className="w-7 h-7 rounded-full overflow-hidden shrink-0 mt-0.5 border border-border bg-secondary">
          <img src="/gui/logo.png" alt="AI" className="w-full h-full object-cover" />
        </div>
        <div className="flex-1 min-w-0">
          {processedContent}
        </div>
      </div>
      {/* Action bar */}
      <div className="flex items-center gap-1 mt-2 ml-10 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={copyContent}
          className={cn(
            "p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary transition-all text-xs flex items-center gap-1",
            copied && "text-green-500"
          )}
          title="Copy"
        >
          <Copy size={12} />
          {copied && <span className="text-[10px]">Copied</span>}
        </button>
        <button className="p-1.5 rounded-md text-muted-foreground hover:text-primary hover:bg-secondary transition-all" title="Good response">
          <ThumbsUp size={12} />
        </button>
        <button className="p-1.5 rounded-md text-muted-foreground hover:text-destructive hover:bg-secondary transition-all" title="Bad response">
          <ThumbsDown size={12} />
        </button>
        <button className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary transition-all" title="Regenerate">
          <RotateCcw size={12} />
        </button>
        {message.tps && (
          <span className="ml-2 text-[10px] font-mono text-muted-foreground/50">
            {message.tps} t/s
          </span>
        )}
      </div>
    </div>
  )
}

export function ChatView() {
  const {
    conversations,
    activeConversationId,
    setActiveConversationId,
    addMessage,
    linkConversation,
    refreshConversations,
    models,
    localModels,
    activeModelId,
    setActiveModelId,
    isGenerating,
    setIsGenerating,
    isModelLoading,
    setIsModelLoading,
    refreshModels,
    settings,
    setConversationMessages,
  } = useApp()

  const [input, setInput] = useState("")
  const [thinking, setThinking] = useState(false)
  const [modelDropdown, setModelDropdown] = useState(false)
  const [showTyping, setShowTyping] = useState(false)
  const [streamingMessage, setStreamingMessage] = useState<string | null>(null)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [selectedImageName, setSelectedImageName] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { toast } = useToast()
  const [isHistoryLoading, setIsHistoryLoading] = useState(() => {
    // Initial state: true if we have an ID that needs loading
    return !!activeConversationId && !activeConversationId.startsWith('temp-');
  })

  const [prevActiveId, setPrevActiveId] = useState(activeConversationId);
  if (activeConversationId !== prevActiveId) {
    setPrevActiveId(activeConversationId);

    // Optimization: Only show loading screen if we don't have this conversation's messages in state yet
    const hasMessages = conversations.some(c => c.id === activeConversationId && c.messages.length > 0);

    if (activeConversationId && !activeConversationId.startsWith('temp-') && !hasMessages) {
      setIsHistoryLoading(true);
    } else {
      setIsHistoryLoading(false);
    }
  }

  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const activeConv = useMemo(() => conversations.find((c) => c.id === activeConversationId), [conversations, activeConversationId]);
  const activeModel = useMemo(() => models.find((m) => m.id === activeModelId) || models.find(m => m.status === 'loaded'), [models, activeModelId]);
  // Use a ref so sendMessage always has latest conversation without being in deps
  const activeConvRef = useRef(activeConv);
  useEffect(() => { activeConvRef.current = activeConv; }, [activeConv]);

  // Load conversation history - only trigger when activeConversationId changes
  const loadedConvIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (!activeConversationId || activeConversationId.startsWith('temp-')) {
      setIsHistoryLoading(false);
      return;
    }
    // Already loaded this convo's history
    const conv = conversations.find(c => c.id === activeConversationId);
    if (conv && conv.messages.length > 0) {
      setIsHistoryLoading(false);
      return;
    }
    // Don't reload if we already fetched this
    if (loadedConvIdRef.current === activeConversationId) {
      setIsHistoryLoading(false);
      return;
    }

    loadedConvIdRef.current = activeConversationId;
    setIsHistoryLoading(true);
    API.getConversation(activeConversationId).then(data => {
      // Backend returns a list of messages directly
      const messages = Array.isArray(data) ? data : (data as any).history || [];
      const history: ChatMessage[] = messages.map((m: any, i: number) => ({
        id: `${activeConversationId}-${i}`,
        role: m.role,
        content: m.content,
        timestamp: new Date(),
      }));
      setConversationMessages(activeConversationId, history);
    })
      .catch(err => {
        console.error(`[ChatView] API Fetch error for ${activeConversationId}:`, err);
        loadedConvIdRef.current = null;
      })
      .finally(() => {
        setIsHistoryLoading(false);
      });
    // Only depend on the ID changing, not on conversations array (to avoid loop)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeConversationId]);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!isVisionModel(activeModel)) {
      toast({
        title: "Incompatible Model",
        description: "The current model does not support images. Please switch to a vision model.",
        variant: "destructive",
      });
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      if (event.target?.result) {
        setSelectedImage(event.target.result as string);
        setSelectedImageName(file.name);
      }
    };
    reader.readAsDataURL(file);
  };

  const clearImage = () => {
    setSelectedImage(null);
    setSelectedImageName(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [activeConv?.messages, showTyping, streamingMessage, selectedImage])

  const sendMessage = useCallback(async () => {
    if ((!input.trim() && !selectedImage) || isGenerating) return;

    const currentInput = input.trim();
    const currentImage = selectedImage;

    // Resolve target conversation ID (auto-create if none)
    let convId = activeConversationId;
    if (!convId) {
      convId = `temp-${Date.now()}`;
      setActiveConversationId(convId);
    }
    const finalConvId = convId;

    const userMsgId = `u-${Date.now()}`;
    const userContent = currentImage
      ? [{ type: "image_url", image_url: { url: currentImage } }, { type: "text", text: currentInput }]
      : currentInput;

    const userMsg: ChatMessage = {
      id: userMsgId,
      role: "user",
      content: userContent,
      timestamp: new Date(),
    };

    // Capture history
    const currentConv = activeConvRef.current;
    const history = (currentConv?.messages || []).map(m => ({ role: m.role, content: m.content }));
    const contextMessages = [
      ...(settings.systemPrompt ? [{ role: "system", content: settings.systemPrompt }] : []),
      ...history,
      { role: "user", content: userContent }
    ];

    addMessage(finalConvId, userMsg);
    setInput("");
    clearImage();
    setIsGenerating(true);
    setShowTyping(true);

    let currentResponse = "";
    let effectiveConvId = finalConvId;

    try {
      await API.chatCompletionStream(
        {
          model: activeModelId,
          messages: contextMessages,
          conversation_id: finalConvId.startsWith('temp-') ? undefined : finalConvId,
          thinking: thinking,
          temperature: settings.temperature,
          max_tokens: settings.maxTokens,
          top_p: settings.topP,
          top_k: settings.topK,
          repeat_penalty: settings.repeatPenalty,
        },
        (token) => {
          setShowTyping(false);
          currentResponse += token;
          setStreamingMessage(currentResponse);
        },
        (newId) => {
          if (finalConvId.startsWith('temp-')) {
            linkConversation(finalConvId, newId);
            effectiveConvId = newId;
            refreshConversations();
          }
        }
      );

      const aiMsg: ChatMessage = {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: currentResponse,
        timestamp: new Date(),
      };

      addMessage(effectiveConvId, aiMsg);
    } catch (error) {
      console.error("Generation failed:", error);
      addMessage(effectiveConvId, {
        id: `err-${Date.now()}`,
        role: "assistant",
        content: "Error: " + (error instanceof Error ? error.message : "Unknown error occurred"),
        timestamp: new Date(),
      });
    } finally {
      setIsGenerating(false);
      setShowTyping(false);
      setStreamingMessage(null);
    }
  }, [input, selectedImage, isGenerating, activeConversationId, activeModelId, thinking, settings, addMessage, setIsGenerating, setActiveConversationId, refreshConversations]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const tokenCount = Math.ceil(input.length / 4)

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <header className="h-14 border-b border-border flex items-center justify-between px-5 shrink-0 bg-background/95 backdrop-blur-sm sticky top-0 z-20">
        <div className="flex items-center gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-foreground">
                {isModelLoading ? "Initializing Model..." : (activeModel?.name ?? "No model")}
              </span>
              <span className={cn(
                "w-2 h-2 rounded-full pulse-dot",
                isModelLoading ? "bg-amber-500" : "bg-green-500"
              )} />
            </div>
            <div className="text-[10px] text-muted-foreground font-medium">
              {isModelLoading ? "Loading into VRAM..." : `Active · ${activeModel?.speed ?? "—"}`}
            </div>
          </div>
        </div>

        <div className="relative">
          <button
            onClick={() => setModelDropdown((v) => !v)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border text-xs font-semibold text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
          >
            <Zap size={12} className="text-primary" />
            Switch Model
            <ChevronDown size={12} />
          </button>
          {modelDropdown && (
            <>
              <div className="fixed inset-0 z-40" onClick={() => setModelDropdown(false)} />
              <div className="absolute right-0 top-10 w-64 bg-[#1e1e1e] border border-border rounded-xl shadow-2xl p-2 z-50 animate-fade-in-up">
                {localModels.length > 0 ? localModels.map((m) => (
                  <button
                    key={m.id}
                    onClick={async () => {
                      setModelDropdown(false)
                      if (m.id !== activeModelId) {
                        setIsModelLoading(true)
                        try {
                          setActiveModelId(m.id)
                          await API.loadModel(m.id)
                          await refreshModels()
                        } catch (err) {
                          console.error("Failed to switch model:", err)
                        } finally {
                          setIsModelLoading(false)
                        }
                      }
                    }}
                    className={cn(
                      "w-full text-left px-3 py-2 rounded-lg text-xs transition-all flex items-center justify-between",
                      m.id === activeModelId
                        ? "bg-primary/10 text-primary font-bold"
                        : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                    )}
                  >
                    <div>
                      <div className="font-semibold">{m.name}</div>
                      <div className="text-[10px] opacity-60">{m.size} · {m.quantization}</div>
                    </div>
                    {m.status === "loaded" && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded-md bg-green-500/10 text-green-500 font-bold">LOADED</span>
                    )}
                  </button>
                )) : (
                  <div className="p-4 text-center">
                    <p className="text-[10px] text-muted-foreground">No downloaded models.</p>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </header>

      {/* Messages area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-6">
        <div className="max-w-[780px] mx-auto">
          {isHistoryLoading ? (
            <div className="flex items-center justify-center h-full min-h-[400px]">
              <div className="flex flex-col items-center gap-2">
                <div className="w-8 h-8 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
                <span className="text-xs text-muted-foreground font-medium">Loading conversation...</span>
              </div>
            </div>
          ) : (!activeConv && !isHistoryLoading) || (activeConv && activeConv.messages.length === 0 && !streamingMessage && !isHistoryLoading) ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-center gap-4">
              <div className="w-14 h-14 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
                <Sparkles size={28} className="text-primary" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-foreground mb-1">Start a conversation</h2>
                <p className="text-sm text-muted-foreground">Ask anything to {activeModel?.name}</p>
              </div>
              <div className="flex flex-wrap gap-2 justify-center mt-2">
                {["Explain transformers", "Write a React hook", "Optimize SQL query", "Debug my code"].map((s) => (
                  <button
                    key={s}
                    onClick={() => setInput(s)}
                    className="px-3 py-1.5 rounded-lg border border-border text-xs text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {activeConv?.messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}
              {streamingMessage && (
                <MessageBubble message={{ id: 'streaming', role: 'assistant', content: streamingMessage, timestamp: new Date() }} />
              )}
              {showTyping && (
                <div className="flex items-start gap-3 mb-6">
                  <div className="w-7 h-7 rounded-full overflow-hidden shrink-0 mt-0.5 border border-border bg-secondary">
                    <img src="/gui/logo.png" alt="AI" className="w-full h-full object-cover" />
                  </div>
                  <div className="flex-1">
                    <TypingIndicator />
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Input area */}
      <div className="px-5 pb-5 pt-2 bg-gradient-to-t from-background via-background to-transparent shrink-0">
        <div className="max-w-[780px] mx-auto">
          {selectedImage && (
            <div className="mb-2 p-2 rounded-xl bg-secondary/50 border border-border flex items-center gap-3 animate-in fade-in slide-in-from-bottom-2 duration-200">
              <div className="relative group/img">
                <img src={selectedImage} className="w-12 h-12 rounded-lg object-cover border border-border" alt="Selected" />
                <button
                  onClick={clearImage}
                  className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-destructive text-white flex items-center justify-center opacity-0 group-hover/img:opacity-100 transition-opacity"
                >
                  <span className="text-[10px]">&times;</span>
                </button>
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-[10px] font-bold text-foreground truncate">{selectedImageName}</div>
                <div className="text-[9px] text-muted-foreground">Ready to analyze</div>
              </div>
              <button onClick={clearImage} className="text-[10px] font-bold text-muted-foreground hover:text-foreground px-2">Remove</button>
            </div>
          )}

          <div
            className={cn(
              "bg-[#1e1e1e] border border-border rounded-xl overflow-hidden transition-all",
              "focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/20"
            )}
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message..."
              rows={1}
              className="w-full bg-transparent text-foreground px-4 pt-4 pb-2 text-sm resize-none focus:outline-none placeholder:text-muted-foreground min-h-[52px] max-h-[200px]"
            />

            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="image/*"
              onChange={handleImageSelect}
            />

            <div className="flex items-center justify-between px-3 pb-3 pt-1">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className={cn(
                    "p-1.5 text-muted-foreground hover:text-foreground hover:bg-secondary rounded-lg transition-all",
                    selectedImage && "text-primary bg-primary/10"
                  )}
                  title="Attach image"
                >
                  <Camera size={16} />
                </button>
                <div className="h-4 w-px bg-border mx-1" />

                {/* Mode toggle */}
                <div className="flex items-center bg-[#171717] border border-border rounded-full p-0.5 gap-0.5">
                  <button
                    onClick={() => setThinking(false)}
                    className={cn(
                      "px-3 py-1 rounded-full text-[10px] font-bold transition-all",
                      !thinking ? "bg-secondary text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    Fast
                  </button>
                  <button
                    onClick={() => setThinking(true)}
                    className={cn(
                      "px-3 py-1 rounded-full text-[10px] font-bold transition-all flex items-center gap-1",
                      thinking ? "bg-secondary text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    <Brain size={10} />
                    Thinking
                  </button>
                </div>

                <div className="h-4 w-px bg-border mx-1" />
                <span className="text-[10px] font-mono text-muted-foreground">{tokenCount} / {settings.maxTokens}</span>
              </div>

              <button
                onClick={sendMessage}
                disabled={!input.trim() || isGenerating}
                className={cn(
                  "w-9 h-9 rounded-lg flex items-center justify-center transition-all",
                  input.trim() && !isGenerating
                    ? "bg-primary text-primary-foreground hover:bg-primary/90 shadow-lg shadow-primary/20"
                    : "bg-secondary text-muted-foreground cursor-not-allowed"
                )}
              >
                <Send size={15} />
              </button>
            </div>
          </div>

          <div className="text-center mt-2 text-[10px] text-muted-foreground">
            Using{" "}
            <span
              onClick={() => setModelDropdown(true)}
              className="text-muted-foreground hover:text-primary cursor-pointer transition-colors font-medium"
            >
              {activeModel?.name}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

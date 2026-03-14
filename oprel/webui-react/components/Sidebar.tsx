"use client"

import { useState, useEffect } from "react"
import {
  MessageSquarePlus,
  Box,
  BarChart2,
  Settings,
  Search,
  Trash2,
  ChevronRight,
  Cpu,
  Bot,
} from "lucide-react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { cn } from "@/services/utils"
import { useApp } from "@/services/context"
import { API } from "@/services/api"
import type { Conversation } from "@/services/data"

function groupConversations(convs: Conversation[]) {
  const now = new Date()
  const today: Conversation[] = []
  const yesterday: Conversation[] = []
  const older: Conversation[] = []

  convs.forEach((c) => {
    const diff = now.getTime() - c.createdAt.getTime()
    const dayMs = 86400000
    if (diff < dayMs) today.push(c)
    else if (diff < dayMs * 2) yesterday.push(c)
    else older.push(c)
  })

  return { today, yesterday, older }
}

export function Sidebar() {
  const {
    currentView,
    setCurrentView,
    conversations,
    activeConversationId,
    setActiveConversationId,
    createConversation,
    deleteConversation,
    settingsOpen,
    setSettingsOpen,
    activeModelId,
    models,
  } = useApp()

  const pathname = usePathname()
  const router = useRouter()

  const [search, setSearch] = useState("")
  const [deleteId, setDeleteId] = useState<string | null>(null)

  const activeModel = models.find((m) => m.id === activeModelId)
  const filtered = conversations.filter((c) =>
    c.title.toLowerCase().includes(search.toLowerCase())
  )
  const groups = groupConversations(filtered)

  function ConvGroup({ label, items }: { label: string; items: Conversation[] }) {
    if (!items.length) return null
    return (
      <div>
        <div className="px-3 mb-1 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
          {label}
        </div>
        <div className="space-y-0.5">
          {items.map((conv) => (
            <div
              key={conv.id}
              className={cn(
                "group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-all text-sm",
                activeConversationId === conv.id && pathname === "/"
                  ? "bg-secondary text-foreground"
                  : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
              )}
              onClick={() => {
                setActiveConversationId(conv.id)
                if (pathname !== "/") router.push("/")
              }}
            >
              <MessageSquarePlus size={13} className="shrink-0 opacity-60" />
              <span className="flex-1 truncate text-xs">{conv.title}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  setDeleteId(conv.id)
                }}
                className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:text-destructive transition-all"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>
      </div>
    )
  }

  const [user, setUser] = useState<{ name: string; role: string; initials: string }>({
    name: "User",
    role: "Developer",
    initials: "U",
  })

  useEffect(() => {
    API.fetchUser()
      .then((data) => {
        const initials = data.name
          .split(" ")
          .map((n) => n[0])
          .join("")
          .toUpperCase()
        setUser({ ...data, initials })
      })
      .catch((err) => console.error("Failed to fetch user:", err))
  }, [])

  return (
    <>
      <aside className="w-[260px] shrink-0 flex flex-col h-full bg-[#171717] border-r border-border">
        {/* Logo */}
        <div className="p-4 pb-3 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg overflow-hidden shrink-0">
            <img src="/gui/logo.png" alt="Oprel" className="w-full h-full object-cover" />
          </div>
          <span className="font-bold text-sm tracking-tight text-foreground">OPREL STUDIO</span>
        </div>

        {/* New Chat Button */}
        <div className="px-3 pb-3 space-y-2">
          <button
            onClick={() => {
              createConversation()
              if (pathname !== "/") router.push("/")
            }}
            className="w-full flex items-center justify-between px-4 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-semibold hover:bg-primary/90 transition-all"
          >
            <span>New Chat</span>
            <MessageSquarePlus size={16} />
          </button>

          {/* Nav buttons */}
          <div className="grid grid-cols-2 gap-2">
            <Link
              href="/models"
              className={cn(
                "flex flex-col items-center gap-1.5 py-3 rounded-lg border border-border text-xs font-semibold transition-all text-center",
                pathname === "/models"
                  ? "bg-secondary text-foreground border-border"
                  : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
              )}
            >
              <Box size={18} />
              <span className="text-[10px]">Models</span>
            </Link>
            <Link
              href="/dev"
              className={cn(
                "flex flex-col items-center gap-1.5 py-3 rounded-lg border border-border text-xs font-semibold transition-all text-center",
                pathname === "/dev"
                  ? "bg-secondary text-foreground border-border"
                  : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
              )}
            >
              <BarChart2 size={18} />
              <span className="text-[10px]">Dev</span>
            </Link>
          </div>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto px-2 py-2 space-y-4">
          <ConvGroup label="Today" items={groups.today} />
          <ConvGroup label="Yesterday" items={groups.yesterday} />
          <ConvGroup label="Older" items={groups.older} />
          {filtered.length === 0 && (
            <div className="text-center py-8 text-muted-foreground text-xs">No chats found</div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-border p-3 space-y-3">
          {/* Search */}
          <div className="relative">
            <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search chats..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full bg-[#1e1e1e] border border-border rounded-lg py-2 pl-8 pr-3 text-xs text-foreground placeholder:text-muted-foreground focus:border-primary/50 transition-all"
            />
          </div>

          {/* Active model pill */}
          {/* <div className="flex items-center gap-2 px-2 py-2 rounded-lg bg-secondary/50">
            <div className="w-7 h-7 rounded-md bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0">
              <Cpu size={13} className="text-primary" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-[11px] font-semibold text-foreground truncate">
                {activeModel?.name.split(" ").slice(0, 3).join(" ")}
              </div>
              <div className="text-[10px] text-muted-foreground">
                {activeModel?.size} · {activeModel?.quantization}
              </div>
            </div>
            <div className="w-2 h-2 rounded-full bg-green-500 pulse-dot shrink-0" />
          </div> */}

          {/* User / Settings */}
          <div className="flex items-center gap-2 px-1">
            <div className="w-7 h-7 rounded-lg bg-primary/20 flex items-center justify-center text-[10px] font-bold text-primary shrink-0">
              {user.initials}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-[11px] font-semibold text-foreground">{user.name}</div>
              <div className="text-[10px] text-muted-foreground">{user.role}</div>
            </div>
            <button
              onClick={() => setSettingsOpen(true)}
              className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary transition-all"
            >
              <Settings size={15} />
            </button>
          </div>
        </div>
      </aside>

      {/* Delete confirm dialog */}
      {deleteId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-[#1e1e1e] border border-border rounded-xl p-6 w-[360px] shadow-2xl animate-fade-in-up">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-9 h-9 rounded-lg bg-destructive/10 flex items-center justify-center">
                <Trash2 size={16} className="text-destructive" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-foreground">Delete Chat</h3>
                <p className="text-xs text-muted-foreground">This action cannot be undone</p>
              </div>
            </div>
            <div className="flex gap-2 justify-end mt-5">
              <button
                onClick={() => setDeleteId(null)}
                className="px-4 py-2 text-xs font-semibold rounded-lg bg-secondary text-foreground hover:bg-secondary/80 transition-all"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  deleteConversation(deleteId)
                  setDeleteId(null)
                }}
                className="px-4 py-2 text-xs font-semibold rounded-lg bg-destructive text-white hover:bg-destructive/90 transition-all flex items-center gap-1.5"
              >
                <Trash2 size={12} /> Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

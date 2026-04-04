"use client"

import { useState, useRef } from "react"
import { ChatView } from "@/components/ChatView"
import { Sidebar } from "@/components/Sidebar"
import { SettingsModal } from "@/components/SettingsModal"

export default function ChatPage() {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      <Sidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(v => !v)} />
      <main className="flex-1 min-w-0 overflow-hidden relative flex flex-col">
        <ChatView sidebarOpen={sidebarOpen} onToggleSidebar={() => setSidebarOpen(v => !v)} />
      </main>
      <SettingsModal />
    </div>
  )
}

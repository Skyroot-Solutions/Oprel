"use client"

import { useState } from "react"
import { ChatView } from "@/components/ChatView"
import { Sidebar } from "@/components/Sidebar"
import { SettingsModal } from "@/components/SettingsModal"
import { ModelsView } from "@/components/ModelsView"
import { DevView } from "@/components/DevView"
import { KnowledgeView } from "@/components/KnowledgeView"
import { useApp } from "@/services/context"

export default function ChatPage() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const { currentView } = useApp()

  const renderView = () => {
    switch (currentView) {
      case "chat":
        return <ChatView sidebarOpen={sidebarOpen} onToggleSidebar={() => setSidebarOpen(v => !v)} />
      case "models":
        return <ModelsView />
      case "dev":
        return <DevView />
      case "knowledge":
        return <KnowledgeView />
      default:
        return <ChatView sidebarOpen={sidebarOpen} onToggleSidebar={() => setSidebarOpen(v => !v)} />
    }
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      <Sidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(v => !v)} />
      <main className="flex-1 min-w-0 overflow-hidden relative flex flex-col">
        {renderView()}
      </main>
      <SettingsModal />
    </div>
  )
}

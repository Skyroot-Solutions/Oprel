"use client"

import { DevView } from "@/components/DevView"
import { Sidebar } from "@/components/Sidebar"
import { SettingsModal } from "@/components/SettingsModal"

export default function DevPage() {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      <Sidebar />
      <main className="flex-1 min-w-0 overflow-hidden relative flex flex-col">
        <DevView />
      </main>
      <SettingsModal />
    </div>
  )
}

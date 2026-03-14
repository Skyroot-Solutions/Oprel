import type { Metadata, Viewport } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { AppProvider } from '@/services/context'
import { Toaster } from '@/components/ui/toaster'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Oprel Studio',
  description: 'Local AI model runner — chat, manage and analyze LLMs',
}

export const viewport: Viewport = {
  themeColor: '#0f0f0f',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans antialiased bg-background text-foreground h-full overflow-hidden">
        <AppProvider>
          {children}
          <Toaster />
        </AppProvider>
      </body>
    </html>
  )
}

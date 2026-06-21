'use client'

import { useEffect, useRef, useState, type ChangeEvent, type Dispatch, type SetStateAction } from 'react'
import { Brain, CheckCircle2, ChevronDown, ChevronUp, UploadCloud, Trash2, Sparkles, Send, Loader2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Switch } from '@/components/ui/switch'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { sendChatMessage, uploadDocument, clearKnowledgeBase } from '@/lib/api'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  thoughtProcess?: string
}

const PORTFOLIO_LIMIT = 50000
const MODEL_OPTIONS = [
  { id: 'llama-3.3-70b-versatile', label: 'Llama 3.3 70B', dot: 'bg-green-500' },
  { id: 'llama-3.1-8b-instant', label: 'Llama 3.1 8B', dot: 'bg-blue-500' },
  { id: 'qwen/qwen3-32b', label: 'Qwen 3 32B', dot: 'bg-amber-500' },
  { id: 'openai/gpt-oss-120b', label: 'GPT-OSS 120B', dot: 'bg-violet-500' },
] as const

type SetState<T> = Dispatch<SetStateAction<T>>

interface SidebarProps {
  ragEnabled: boolean
  setRagEnabled: SetState<boolean>
  selectedModel: string
  setSelectedModel: SetState<string>
  currentDeviceTokens: number
  deviceId: string
  onKnowledgeBaseCleared: () => void
}

interface ChatAreaProps {
  ragEnabled: boolean
  selectedModel: string
  currentDeviceTokens: number
  setCurrentDeviceTokens: SetState<number>
  deviceId: string
  resetSignal: number
}

export function Sidebar({
  ragEnabled,
  setRagEnabled,
  selectedModel,
  setSelectedModel,
  currentDeviceTokens,
  deviceId,
  onKnowledgeBaseCleared,
}: SidebarProps) {
  const [uploading, setUploading] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const usagePercentage = currentDeviceTokens / PORTFOLIO_LIMIT

  const handleFileSelect = async (file: File) => {
    setUploading(true)
    try {
      if (!deviceId) {
        throw new Error('Missing device ID. Refresh and try again.')
      }
      await uploadDocument(file, deviceId)
      setUploadedFileName(file.name)
    } catch (error) {
      console.error('Upload failed:', error)
    } finally {
      setUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const handleClearKnowledgeBase = async () => {
    setClearing(true)
    try {
      if (!deviceId) {
        throw new Error('Missing device ID. Refresh and try again.')
      }
      await clearKnowledgeBase(deviceId)
      localStorage.removeItem('chat_history')
      setUploadedFileName(null)
      onKnowledgeBaseCleared()
    } catch (error) {
      console.error('Clear failed:', error)
    } finally {
      setClearing(false)
    }
  }

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  return (
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-72 flex-col border-r border-sidebar-border bg-sidebar">
      {/* App Title */}
      <div className="flex items-center gap-3 border-b border-sidebar-border px-5 py-5">
        <div className="flex size-9 items-center justify-center rounded-lg bg-primary/10">
          <Sparkles className="size-5 text-primary" />
        </div>
        <div>
          <h1 className="text-sm font-semibold text-sidebar-foreground">Multimodal RAG Assistant</h1>
          <p className="text-xs text-muted-foreground">Built by Anas</p>
        </div>
      </div>

      {/* Controls Section */}
      <div className="flex-1 space-y-6 overflow-y-auto px-5 py-5">
        {/* Model Switcher */}
        <div className="space-y-2">
          <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Model
          </label>
          <Select value={selectedModel} onValueChange={setSelectedModel}>
            <SelectTrigger className="w-full bg-sidebar-accent border-sidebar-border">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              {MODEL_OPTIONS.map((model) => (
                <SelectItem key={model.id} value={model.id}>
                  <div className="flex items-center gap-2">
                    <div className={`size-2 rounded-full ${model.dot}`} />
                    {model.label}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* RAG Mode Toggle */}
        <div className="space-y-2">
          <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Knowledge Base
          </label>
          <div className="flex items-center justify-between rounded-lg border border-sidebar-border bg-sidebar-accent px-4 py-3">
            <div className="flex items-center gap-2">
              <Brain className="size-4 text-primary" />
              <span className="text-sm font-medium text-sidebar-foreground">RAG Mode</span>
            </div>
            <Switch checked={ragEnabled} onCheckedChange={setRagEnabled} />
          </div>
          {ragEnabled && (
            <p className="text-xs text-muted-foreground">
              Retrieval augmented generation is active
            </p>
          )}
        </div>

        {/* File Ingestion */}
        <div className="space-y-2">
          <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Document Upload
          </label>
          <input
            ref={fileInputRef}
            type="file"
            onChange={handleFileInputChange}
            className="hidden"
          />
          <div
            onClick={() => fileInputRef.current?.click()}
            className="group cursor-pointer rounded-lg border-2 border-dashed border-sidebar-border bg-sidebar-accent/50 px-4 py-6 text-center transition-colors hover:border-primary/50 hover:bg-sidebar-accent"
          >
            {uploading ? (
              <Loader2 className="mx-auto size-8 animate-spin text-primary" />
            ) : uploadedFileName ? (
              <CheckCircle2 className="mx-auto size-8 text-green-500" />
            ) : (
              <UploadCloud className="mx-auto size-8 text-muted-foreground transition-colors group-hover:text-primary" />
            )}
            <p className="mt-2 text-sm font-medium text-sidebar-foreground">
              {uploading ? 'Uploading...' : uploadedFileName ?? 'Upload Document'}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {uploadedFileName ? 'Document indexed and ready for RAG' : 'Any file format'}
            </p>
          </div>
        </div>

        {/* Clear Button */}
        <Button
          variant="outline"
          onClick={handleClearKnowledgeBase}
          disabled={clearing}
          className="w-full border-destructive/50 text-destructive hover:bg-destructive/10 hover:text-destructive"
        >
          {clearing ? (
            <Loader2 className="size-4 animate-spin mr-2" />
          ) : (
            <Trash2 className="size-4" />
          )}
          {clearing ? 'Clearing...' : 'Clear Knowledge Base'}
        </Button>
      </div>

      {/* Telemetry Dashboard */}
      <div className="border-t border-sidebar-border px-5 py-5">
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Device Usage
            </span>
            <span className="text-xs font-mono text-primary">{Math.round(usagePercentage * 100)}%</span>
          </div>
          <Progress value={Math.min(usagePercentage * 100, 100)} className="h-2" />
          <p className="text-xs text-muted-foreground">
            <span className="font-mono text-sidebar-foreground">{currentDeviceTokens.toLocaleString()}</span> / 50,000 Tokens
          </p>
          {usagePercentage >= 0.85 && usagePercentage < 1.0 && (
            <p className="text-xs text-amber-600 font-medium">⚠️ Approaching limit</p>
          )}
          {usagePercentage >= 1.0 && (
            <p className="text-xs text-destructive font-medium">🚨 Limit reached</p>
          )}
        </div>
      </div>
    </aside>
  )
}

export function ChatArea({
  ragEnabled,
  selectedModel,
  currentDeviceTokens,
  setCurrentDeviceTokens,
  deviceId,
  resetSignal,
}: ChatAreaProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [thoughtsOpen, setThoughtsOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const hasInitializedResetRef = useRef(false)

  useEffect(() => {
    const savedChat = localStorage.getItem('chat_history')
    if (!savedChat) return

    try {
      const parsed = JSON.parse(savedChat) as Message[]
      if (Array.isArray(parsed)) {
        setMessages(parsed)
      }
    } catch {
      localStorage.removeItem('chat_history')
    }
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    localStorage.setItem('chat_history', JSON.stringify(messages))
  }, [messages])

  useEffect(() => {
    if (!hasInitializedResetRef.current) {
      hasInitializedResetRef.current = true
      return
    }
    setMessages([])
  }, [resetSignal])

  const saveTokens = (tokens: number) => {
    const newTotal = currentDeviceTokens + tokens
    setCurrentDeviceTokens(newTotal)
    localStorage.setItem('deviceTokens', newTotal.toString())
  }

  const handleSendMessage = async (e?: { preventDefault?: () => void; key?: string }) => {
    if (e?.key && e.key !== 'Enter') return
    if (loading || !inputValue.trim()) return

    e?.preventDefault?.()

    if (currentDeviceTokens >= PORTFOLIO_LIMIT) {
      setError('🚨 Demo limit reached! You have exhausted the 50,000 token limit for this session.')
      return
    }
    if (!deviceId) {
      setError('Device ID is not ready yet. Please wait a second and try again.')
      return
    }

    const userMessage = inputValue.trim()
    setInputValue('')
    setError(null)
    setLoading(true)

    setMessages((prev) => [...prev, {
      id: Date.now().toString(),
      role: 'user',
      content: userMessage,
    }])

    try {
      const response = await sendChatMessage(
        userMessage,
        selectedModel,
        ragEnabled,
        currentDeviceTokens,
        deviceId
      )

      const messageId = Date.now().toString()
      setMessages((prev) => [...prev, {
        id: messageId,
        role: 'assistant',
        content: response.text,
        thoughtProcess: response.thought_process,
      }])

      saveTokens(response.tokens_used_this_turn)
    } catch (err: unknown) {
      const errorMsg = String((err as { message?: string })?.message || err)
      if (errorMsg.includes('429') || errorMsg.toLowerCase().includes('rate limit')) {
        setError('⚠️ Backend API limit reached! The provider is at capacity. Try a different model.')
      } else {
        setError(`Error: ${errorMsg}`)
      }
    } finally {
      setLoading(false)
      requestAnimationFrame(() => {
        inputRef.current?.focus()
      })
    }
  }

  const isLimitReached = currentDeviceTokens >= PORTFOLIO_LIMIT

  return (
    <main className="ml-72 flex h-screen flex-col">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-border px-6 py-4">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Multimodal Chat</h2>
          <p className="text-sm text-muted-foreground">Interact with your documents using AI</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="size-2 animate-pulse rounded-full bg-green-500" />
          <span className="text-xs text-muted-foreground">Connected</span>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="border-b border-destructive/20 bg-destructive/10 px-6 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto max-w-3xl space-y-6">
          {messages.length === 0 && (
            <div className="flex h-full items-center justify-center">
              <div className="text-center">
                <Brain className="mx-auto size-12 text-muted-foreground/50 mb-3" />
                <p className="text-muted-foreground">
                  Start a conversation or upload documents to begin.
                </p>
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.role === 'user' ? (
                <div className="max-w-md rounded-2xl rounded-br-md bg-secondary px-4 py-3">
                  <div className="prose prose-invert text-sm text-secondary-foreground">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                </div>
              ) : (
                <div className="max-w-2xl space-y-3">
                  {msg.thoughtProcess && (
                    <Collapsible open={thoughtsOpen} onOpenChange={setThoughtsOpen}>
                      <CollapsibleTrigger asChild>
                        <button className="flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted">
                          <Brain className="size-4" />
                          AI Thought Process
                          {thoughtsOpen ? (
                            <ChevronUp className="size-4" />
                          ) : (
                            <ChevronDown className="size-4" />
                          )}
                        </button>
                      </CollapsibleTrigger>
                      <CollapsibleContent className="mt-2">
                        <div className="rounded-lg border border-border bg-muted/50 px-4 py-3">
                          <p className="font-mono text-xs text-muted-foreground leading-relaxed whitespace-pre-wrap">
                            {msg.thoughtProcess}
                          </p>
                        </div>
                      </CollapsibleContent>
                    </Collapsible>
                  )}
                  <div className="rounded-2xl rounded-bl-md border border-border bg-card px-4 py-3">
                    <div className="prose prose-invert text-sm leading-relaxed text-card-foreground">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="max-w-2xl space-y-3">
                <div className="flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm text-muted-foreground">
                  <Loader2 className="size-4 animate-spin" />
                  <span>Thinking...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="sticky bottom-0 border-t border-border bg-background px-6 py-4">
        <div className="mx-auto max-w-3xl">
          {isLimitReached && (
            <div className="mb-3 rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-2">
              <p className="text-sm text-destructive font-medium">
                🚨 Portfolio Demo Limit Reached! To explore further, let&apos;s schedule an interview.
              </p>
            </div>
          )}
          <div className="flex items-center gap-3 rounded-xl border border-border bg-card px-4 py-3">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  handleSendMessage(e)
                }
              }}
              placeholder={isLimitReached ? 'Demo limit reached.' : 'Message the Multimodal Engine...'}
              disabled={loading || isLimitReached}
              className="flex-1 bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none disabled:opacity-50"
            />
            <Button
              size="icon"
              onClick={() => {
                void handleSendMessage()
              }}
              disabled={loading || isLimitReached || !inputValue.trim()}
              className="size-9 shrink-0"
            >
              {loading ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Send className="size-4" />
              )}
              <span className="sr-only">Send message</span>
            </Button>
          </div>
          <p className="mt-2 text-center text-xs text-muted-foreground">
            Press Enter to send
          </p>
        </div>
      </div>
    </main>
  )
}

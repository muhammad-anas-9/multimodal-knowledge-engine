'use client'

import { useEffect, useState } from 'react'
import { Sidebar, ChatArea } from '@/components/nexus-dashboard'

export default function Home() {
  const [selectedModel, setSelectedModel] = useState('llama-3.3-70b-versatile')
  const [ragEnabled, setRagEnabled] = useState(true)
  const [currentDeviceTokens, setCurrentDeviceTokens] = useState(0)
  const [resetSignal, setResetSignal] = useState(0)
  const [deviceId, setDeviceId] = useState('')

  useEffect(() => {
    const saved = localStorage.getItem('deviceTokens')
    if (saved) {
      setCurrentDeviceTokens(parseInt(saved, 10))
    }
  }, [])

  useEffect(() => {
    const existingDeviceId = localStorage.getItem('device_id')
    if (existingDeviceId) {
      setDeviceId(existingDeviceId)
      return
    }

    const generatedDeviceId =
      typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID()
        : Math.random().toString(36).substring(2, 15)
    localStorage.setItem('device_id', generatedDeviceId)
    setDeviceId(generatedDeviceId)
  }, [])

  return (
    <div className="min-h-screen bg-background">
      <Sidebar
        ragEnabled={ragEnabled}
        setRagEnabled={setRagEnabled}
        selectedModel={selectedModel}
        setSelectedModel={setSelectedModel}
        currentDeviceTokens={currentDeviceTokens}
        deviceId={deviceId}
        onKnowledgeBaseCleared={() => setResetSignal((prev) => prev + 1)}
      />
      <ChatArea
        ragEnabled={ragEnabled}
        selectedModel={selectedModel}
        currentDeviceTokens={currentDeviceTokens}
        setCurrentDeviceTokens={setCurrentDeviceTokens}
        deviceId={deviceId}
        resetSignal={resetSignal}
      />
    </div>
  )
}

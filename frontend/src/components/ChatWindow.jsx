import React, { useEffect, useRef, useState } from 'react'
import MessageBubble from './MessageBubble'
import InputBox from './InputBox'

const USE_STREAMING = true

export default function ChatWindow() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const chatEndRef = useRef(null)

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async (text) => {
    if (!text.trim()) return

    const userMsg = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])

    if (USE_STREAMING) {
      setIsLoading(true)
      const botMsg = { role: 'assistant', content: '' }
      setMessages(prev => [...prev, botMsg])

      const url = `/api/stream?message=${encodeURIComponent(text)}`
      const es = new EventSource(url)

      es.addEventListener('message', (e) => {
        const token = e.data
        setMessages(prev => {
          const updated = [...prev]
          const lastIndex = updated.length - 1
          if (lastIndex >= 0 && updated[lastIndex].role === 'assistant') {
            updated[lastIndex] = {
              ...updated[lastIndex],
              content: updated[lastIndex].content + token,
            }
          }
          return updated
        })
      })

      const closeStream = () => {
        setIsLoading(false)
        es.close()
      }

      es.addEventListener('end', closeStream)
      es.addEventListener('error', () => {
        closeStream()
      })
      return
    }

    try {
      setIsLoading(true)
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })
      if (!res.ok) throw new Error('Request failed')
      const data = await res.json()
      const botMsg = { role: 'assistant', content: data.reply }
      setMessages(prev => [...prev, botMsg])
    } catch (e) {
      const botMsg = { role: 'assistant', content: 'Error: failed to get response.' }
      setMessages(prev => [...prev, botMsg])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="chat-window">
      <div className="messages" role="log" aria-live="polite">
        {messages.map((m, idx) => (
          <MessageBubble key={idx} role={m.role} content={m.content} />
        ))}
        <div ref={chatEndRef} />
      </div>
      {isLoading && (
        <div className="loading">Assistant is typing…</div>
      )}
      <InputBox onSend={handleSend} disabled={isLoading} />
    </div>
  )
}
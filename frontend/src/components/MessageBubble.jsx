import React from 'react'

export default function MessageBubble({ role, content }) {
  const isUser = role === 'user'
  return (
    <div className={`bubble ${isUser ? 'user' : 'assistant'}`}>
      <div className="bubble-role">{isUser ? 'You' : 'Assistant'}</div>
      <div className="bubble-content">{content}</div>
    </div>
  )
}
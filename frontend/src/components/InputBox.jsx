import React, { useState } from 'react'

export default function InputBox({ onSend, disabled }) {
  const [text, setText] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (disabled) return
    onSend?.(text)
    setText('')
  }

  return (
    <form className="input-box" onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="Type your message..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={disabled}
      />
      <button type="submit" disabled={disabled || !text.trim()}>Send</button>
    </form>
  )
}
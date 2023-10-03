import { useEffect, useState } from 'react'

type CodeProps = {
  code: string
  language: string
}

export default function Code({ code, language }: CodeProps) {
  const [highlightedCode, setHighlightedCode] = useState<string | null>(null)

  useEffect(() => {
    const fetchHighlightedCode = async () => {
      const res = await fetch('/api/highlight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, language }),
      })
      const data = await res.json()
      setHighlightedCode(data.highlightedCode)
    }

    fetchHighlightedCode()
  }, [code, language])

  return (
    <pre className="code">
      <code
        dangerouslySetInnerHTML={{ __html: highlightedCode ?? code }}
      ></code>
    </pre>
  )
}

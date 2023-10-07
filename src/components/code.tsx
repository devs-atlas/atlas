type CodeProps = {
  code: string[]
}

export default function Code({ code }: CodeProps) {
  const [highlightedCode, output] = code
  return (
    <pre className="code">
      <code dangerouslySetInnerHTML={{ __html: highlightedCode }}></code>
      {output && <div className="output">{output}</div>}
    </pre>
  )
}

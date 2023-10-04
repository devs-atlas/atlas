type CodeProps = {
  code: string
}

export default function Code({ code }: CodeProps) {
  return (
    <pre className="code">
      <code dangerouslySetInnerHTML={{ __html: code }}></code>
    </pre>
  )
}

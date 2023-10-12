import styles from './Code.module.css'

type CodeProps = {
  code: string[]
}

export default function Code({ code }: CodeProps) {
  const [highlightedCode, output] = code
  return (
    <pre>
      <div className={styles.code}>
        <code dangerouslySetInnerHTML={{ __html: highlightedCode }}></code>
        {output && <div className={styles.output}>{output}</div>}
      </div>
    </pre>
  )
}

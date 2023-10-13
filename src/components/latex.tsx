import { MathJax, MathJaxContext } from 'better-react-mathjax'
import { ReactNode } from 'react'

type LatexProps = {
  children: ReactNode
  inline?: boolean
}

export function Latex({ children, inline }: LatexProps) {
  return (
    <div style={{ fontSize: '25px' }}>
      <MathJaxContext>
        <MathJax inline={inline}>{`\\(${children}\\)`}</MathJax>
      </MathJaxContext>
    </div>
  )
}

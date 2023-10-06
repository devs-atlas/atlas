import { MathJax } from 'better-react-mathjax'
import { ReactNode } from 'react'

type LatexProps = {
  children: ReactNode
  inline?: boolean
}

export function Latex({ children, inline }: LatexProps) {
  return (
    <div className="latex">
      <MathJax inline={inline}>{`\\(${children}\\)`}</MathJax>
    </div>
  )
}

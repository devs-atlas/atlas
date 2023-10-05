import { ReactNode } from 'react'
import { MathJax } from 'better-react-mathjax'

type LatexProps = {
  children: ReactNode
}

export function Latex({ children }: LatexProps) {
  return (
    <div className="latex">
      <MathJax>{`\\(${children}\\)`}</MathJax>
    </div>
  )
}

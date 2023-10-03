import { ReactNode } from 'react'
import MathJax from 'react-mathjax2'

type LatexProps = {
  children: ReactNode
}

export function Latex({ children }: LatexProps) {
  return (
    <div className="latex">
      <MathJax.Context input="tex">{children}</MathJax.Context>
    </div>
  )
}

export const Eq = MathJax.Node

import { ReactNode } from 'react'

type Props = {
  children: ReactNode
}

export default function PostLayout({ children }: Props) {
  return <>{children}</>
}

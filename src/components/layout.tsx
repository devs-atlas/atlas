import { ReactNode } from 'react'
import Navbar from './navbar'

type Props = {
  children: ReactNode
}

export default function Layout({ children }: Props) {
  return (
    <main>
      <Navbar />
      {children}
    </main>
  )
}
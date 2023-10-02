import Image from 'next/image'
import Link from 'next/link'
import styles from './Navbar.module.css'
import { raleway } from '~/styles/fonts'

export default function Navbar() {
  return (
    <ul className={`${styles.navbar} ${raleway.className}`}>
      <li className={styles.navItem}>
        <div className={styles.icon}>
          <Image
            priority
            src="/icon.svg"
            alt="atlas icon"
            width={20}
            height={20}
          />
          <Link href="/">atlas</Link>
        </div>
      </li>
      <li className={styles.navItem}>
        <Link href="/about">about</Link>
      </li>
    </ul>
  )
}

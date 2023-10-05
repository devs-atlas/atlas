import Image from 'next/image'
import Link from 'next/link'
import { fragment } from '~/styles/fonts'
import styles from './Navbar.module.css'

export default function Navbar() {
  return (
    <ul className={`${styles.navbar} ${fragment.className}`}>
      <li className={styles.navItem}>
        <div className={styles.icon}>
          <div className="aspect-ratio-box">
            <Image
              src="/icon.webp"
              alt="atlas icon"
              width={50}
              height={50}
              priority={true}
            />
          </div>
          <Link href="/">atlas</Link>
        </div>
      </li>
      <li className={styles.navItem}>
        <Link href="/about">about</Link>
      </li>
    </ul>
  )
}

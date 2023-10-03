import styles from './VertSeparator.module.css'

export default function Separator() {
  return (
    <div className={styles.vertSeparator}>
      <div className={`${styles.circle} ${styles.top}`}></div>
      <div className={`${styles.circle} ${styles.bottom}`}></div>
    </div>
  )
}

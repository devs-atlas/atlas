import styles from './Separator.module.css'

const Circles = ({ numCircles }: { numCircles: number }) => {
  return (
    <>
      {Array.from({ length: numCircles }).map((_, i) => (
        <div
          key={i}
          className={styles.circle}
          // @ts-ignore
          style={{ '--circle-index': i / (numCircles - 1) }}
        ></div>
      ))}
    </>
  )
}

type SeparatorProps = {
  numCircles: number
  width: string
}
export const Separator = ({ numCircles, width }: SeparatorProps) => {
  return (
    <div className={styles.separator} style={{ width: width }}>
      <Circles numCircles={numCircles} />
    </div>
  )
}

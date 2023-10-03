import Date from '~/components/date'
import Layout from '~/components/layout'

import Link from 'next/link'
import { Fragment } from 'react'
import type { PostMeta } from '~/lib/posts'
import { getAllPostMeta } from '~/lib/posts'
import { manrope } from '~/styles/fonts'
import styles from './Meta.module.css'

function Meta({ id, title, description, date, image }: PostMeta) {
  return (
    <Link href={`/posts/${id}`}>
      <div className={`${styles.meta} ${manrope.className}`}>
        <div className={styles.metaContainer}>
          <h2 className={styles.metaTitle}>{title}</h2>
          <Date className={styles.metaDate} dateString={date} />
          <p className={styles.metaDescription}>{description}</p>
        </div>
        {image}
      </div>
    </Link>
  )
}

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

const Separator = ({ numCircles }: { numCircles: number }) => {
  return (
    <div className={styles.separator}>
      <Circles numCircles={numCircles} />
    </div>
  )
}

const allPostMeta = getAllPostMeta()

export default function Home() {
  return (
    <Layout>
      {allPostMeta.map(({ meta }, index) => (
        <Fragment key={meta.title}>
          <Meta {...meta} />
          {index < allPostMeta.length - 1 && <Separator numCircles={5} />}
        </Fragment>
      ))}
    </Layout>
  )
}

import Date from '~/components/date'
import Layout from '~/components/layout'

import Link from 'next/link'
import { Fragment } from 'react'
import { Separator } from '~/components/separator'
import type { PostMeta } from '~/lib/posts'
import { getAllPostMeta } from '~/lib/posts'
import { fragment } from '~/styles/fonts'
import styles from './Meta.module.css'

function Meta({ id, title, description, date, image }: PostMeta) {
  return (
    <Link href={`/posts/${id}`}>
      <div className={`${styles.meta} ${fragment.className}`}>
        <div className={styles.metaContainer}>
          <div className={styles.metaTitle}>{title}</div>
          <Date className={styles.metaDate} dateString={date} />
          <div className={styles.metaDescription}>{description}</div>
        </div>
        {image}
      </div>
    </Link>
  )
}

export default function Home() {
  const allPostMeta = getAllPostMeta()

  return (
    <Layout>
      {allPostMeta.map(({ meta }, index) => (
        <Fragment key={meta.title}>
          <Meta {...meta} />
          {index < allPostMeta.length - 1 && (
            <Separator numCircles={5} width="50%" />
          )}
        </Fragment>
      ))}
    </Layout>
  )
}

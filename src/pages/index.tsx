import Date from '~/components/date'
import Layout from '~/components/layout'

import Link from 'next/link'
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
          <h3 className={styles.metaDate}>{date}</h3>
          <Date dateString={date} />
          <p className={styles.metaDescription}>{description}</p>
        </div>
        {image}
      </div>
    </Link>
  )
}

export default function Home() {
  return (
    <Layout>
      {getAllPostMeta().map(({ meta }) => (
        <Meta key={meta.title} {...meta} />
      ))}
    </Layout>
  )
}

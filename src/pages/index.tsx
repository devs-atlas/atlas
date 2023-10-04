import Date from '~/components/date'
import Layout from '~/components/layout'

import Link from 'next/link'
import type { PostMeta } from '~/lib/posts'
import { postMeta } from '~/lib/posts'
import { fragment } from '~/styles/fonts'
import styles from './Meta.module.css'
import { Separator } from '~/components/separator'

type MetaProps = {
  meta: PostMeta
}

function Meta({ meta }: MetaProps) {
  return (
    <Link href={`/posts/${meta.id}`}>
      <div className={`${styles.meta} ${fragment.className}`}>
        <div className={styles.metaContainer}>
          <div className={styles.metaTitle}>{meta.title}</div>
          <Date className={styles.metaDate} dateString={meta.date} />
          <div className={styles.metaDescription}>{meta.description}</div>
        </div>
        {meta.image}
      </div>
    </Link>
  )
}

export default function Home() {
  return (
    <Layout>
      <Meta meta={postMeta.get('vqgan')!} />
      {/* <Separator numCircles={5} width="50%" /> */}
    </Layout>
  )
}

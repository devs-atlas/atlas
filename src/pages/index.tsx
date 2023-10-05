import Date from '~/components/date'
import Layout from '~/components/layout'

import Image from 'next/image'
import Link from 'next/link'
import type { PostMeta } from '~/lib/posts'
import { postMeta } from '~/lib/posts'
import { fragment } from '~/styles/fonts'
import styles from './Meta.module.css'

type MetaProps = {
  meta: PostMeta
}

function Meta({ meta }: MetaProps) {
  const { src, width, height, alt } = meta.image
  return (
    <Link href={`/posts/${meta.id}`}>
      <div className={`${styles.meta} ${fragment.className}`}>
        <div className={styles.metaContainer}>
          <div className={styles.metaTitle}>{meta.title}</div>
          <Date className={styles.metaDate} dateString={meta.date} />
          <div className={styles.metaDescription}>{meta.description}</div>
        </div>
        <Image
          src={src}
          width={width}
          height={height}
          alt={alt}
          style={{ width: 'auto', height: 'auto' }}
          priority={true}
        />
      </div>
    </Link>
  )
}

type HomeProps = {
  meta: PostMeta
}

export default function Home({ meta }: HomeProps) {
  return (
    <Layout>
      <Meta meta={meta} />
      {/* <Separator numCircles={5} width="50%" /> */}
    </Layout>
  )
}

export async function getServerSideProps() {
  const meta = postMeta.get('vqgan')

  return { props: { meta } }
}

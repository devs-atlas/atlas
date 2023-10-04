import dynamic from 'next/dynamic'
import Layout from './layout'
import type { PostMeta, Snippets } from '~/lib/posts'

type PostProps = {
  meta: PostMeta
  snippets: Snippets
}

const VQgan = dynamic(() => import('../../posts/vqgan'), {
  ssr: false,
})

export default function PostLayout({ meta, snippets }: PostProps) {
  return (
    <Layout>
      {meta.id === 'vqgan' && <VQgan meta={meta} snippets={snippets} />}
    </Layout>
  )
}

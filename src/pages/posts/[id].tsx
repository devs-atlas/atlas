import PostLayout from '~/components/postlayout'
import { getAllPostIds, postMapping } from '~/lib/posts'

import _404 from '~/pages/404'

type PostProps = { id: string }

export default function Post({ id }: PostProps) {
  const post = postMapping.get(id)

  if (!post) {
    return <_404 />
  }

  return <PostLayout post={post!.post} />
}

type StaticProps = { params: { id: string } }

export async function getStaticProps({ params }: StaticProps) {
  return {
    props: {
      id: params.id,
    },
  }
}
export async function getStaticPaths() {
  const paths = getAllPostIds()

  return {
    paths,
    fallback: false,
  }
}

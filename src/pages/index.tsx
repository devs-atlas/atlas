import Layout from '~/components/layout'

import { getAllPostMeta } from '~/lib/posts'

type MetaProps = {
  title: string
  description: string
  date: string
}

function PostMeta({ title, description, date }: MetaProps) {
  return (
    <>
      {title}
      {description}
      {date}
    </>
  )
}

export default function Home() {
  return (
    <Layout>
      {getAllPostMeta().map(({ meta }) => (
        <PostMeta key={meta.title} {...meta} />
      ))}
    </Layout>
  )
}

import Layout from './layout'

type Props = {
  post: JSX.Element
}

export default function PostLayout({ post }: Props) {
  return <Layout>{post}</Layout>
}

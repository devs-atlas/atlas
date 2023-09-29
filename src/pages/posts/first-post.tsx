import { ReactElement } from 'react'
import Layout from '~/components/layout'
import PostLayout from '~/components/postlayout'

const Post = () => <>hi from post </>

Post.getLayout = (page: ReactElement) => (
  <Layout>
    <PostLayout>{page}</PostLayout>
  </Layout>
)

export default Post

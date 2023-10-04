import { readFileSync, readdirSync } from 'fs'
import path from 'path'
import PostLayout from '~/components/postlayout'
import { highlight } from '~/lib/highlight'
import { Snippets, getAllPostIds, postMeta } from '~/lib/posts'
import _404 from '~/pages/404'

type PostProps = {
  id: string
  snippets: Snippets
}

export default function Post({ id, snippets }: PostProps) {
  const meta = postMeta.get(id)

  if (!meta) {
    return <_404 />
  }

  return <PostLayout meta={meta} snippets={snippets} />
}

const snippetDir = path.join(process.cwd(), 'snippets')

type StaticProps = { params: { id: string } }

export async function getStaticProps({ params }: StaticProps) {
  const snippets: Snippets = {}

  const snippetFiles = `${snippetDir}/${params.id}`

  readdirSync(snippetFiles).forEach((snippetFile) => {
    const snippetContents = readFileSync(
      `${snippetFiles}/${snippetFile}`
    ).toString('utf-8')

    snippets[snippetFile] = highlight(snippetContents)
  })

  return {
    props: {
      id: params.id,
      snippets,
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

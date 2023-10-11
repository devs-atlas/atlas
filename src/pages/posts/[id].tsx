import { readFileSync, readdirSync } from 'fs'
import path from 'path'
import PostLayout from '~/components/postlayout'
import { highlight } from '~/lib/highlight'
import type { Snippets } from '~/lib/posts'
import { getAllPostIds, postMeta } from '~/lib/posts'
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

type StaticProps = { params: { id: string } }

export async function getStaticProps({ params }: StaticProps) {
  const snippets: Snippets = {}

  const snippetFiles = path.join(process.cwd(), 'posts', params.id, 'snippets')

  readdirSync(snippetFiles).forEach((snippetFile) => {
    const [snippetContents, output] = readFileSync(
      `${snippetFiles}/${snippetFile}`
    )
      .toString('utf-8')
      .split('# SNIPPET #')
      .map((e) => e.trim())

    const highlightedCode = highlight(snippetContents)

    snippets[snippetFile] = [highlightedCode, output || '']
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
    fallback: true,
  }
}

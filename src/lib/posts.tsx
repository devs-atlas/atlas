export type PostMeta = {
  title: string
  id: string
  description: string
  date: string
  image: ImageProps
}

type ImageProps = {
  src: string
  width: number
  height: number
  alt: string
}

export type Snippets = {
  [key: string]: string[]
}

export const postMeta = new Map<string, PostMeta>()
postMeta.set('vqgan', {
  title: 'GPT for All',
  id: 'vqgan',
  description:
    'A fresh look',
  date: '2023-02-10',
  image: {
    src: '/posts/vqgan/preview.png',
    width: 300,
    height: 200,
    alt: 'placeholder',
  } satisfies ImageProps,
})

export function getAllPostIds() {
  return Array.from(Object.keys(postMeta)).map((id) => ({
    params: { id },
  }))
}

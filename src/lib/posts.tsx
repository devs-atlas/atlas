import Image from 'next/image'

export type PostMeta = {
  title: string
  id: string
  description: string
  date: string
  image: JSX.Element
}

export type Snippets = {
  [key: string]: string
}

export const postMeta = new Map<string, PostMeta>()
postMeta.set('vqgan', {
  title: 'VQ-GAN From the Bottom Up',
  id: 'vqgan',
  description:
    'A foundational, scalable, generative architecture for unstructured data.',
  date: '2023-02-10',
  image: (
    <Image
      src="/posts/vqgan/preview.png"
      width={300}
      height={200}
      alt="placeholder"
      style={{ width: 'auto', height: 'auto' }}
      priority
    />
  ),
})

export function getAllPostIds() {
  return Array.from(Object.keys(postMeta)).map((id) => ({
    params: { id },
  }))
}

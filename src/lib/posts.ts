import { VQGan, meta } from '~/posts/vqgan'

export const postMapping = new Map([
  [
    'vqgan',
    {
      post: VQGan,
      meta,
    },
  ],
])

export function getAllPostMeta() {
  return Array.from(postMapping.keys()).map((id) => ({
    meta: postMapping.get(id)!.meta,
  }))
}

export function getAllPostIds() {
  return Array.from(postMapping.keys()).map((id) => ({
    params: { id },
  }))
}

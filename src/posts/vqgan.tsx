import Image from 'next/image'
import { garamond } from '~/styles/fonts'

function VQGan() {
  return (
    <div className={garamond.className}>
      text<p>more text</p>
    </div>
  )
}

const meta = {
  title: 'VQ-GAN From the Bottom Up',
  description:
    'A foundational, scalable, generative architecture for unstructured data',
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
}

export { VQGan, meta }

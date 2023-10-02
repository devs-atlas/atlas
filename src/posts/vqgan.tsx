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
  date: '2023-10-2',
}

export { VQGan, meta }

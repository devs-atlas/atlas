import Image from 'next/image'
import { garamond } from '~/styles/fonts'

function VQGan2() {
  return (
    <div className={garamond.className}>
      text<p>more texte</p>
    </div>
  )
}

const meta2 = {
  title: 'hellsd;kfbnsd;jkfn',
  description: 'kill me',
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

export { VQGan2, meta2 }

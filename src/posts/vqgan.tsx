import Image from 'next/image'
import Code from '~/components/code'
import Date from '~/components/date'
import { Eq, Latex } from '~/components/latex'
import { Separator } from '~/components/separator'
import type { PostMeta } from '~/lib/posts'
import { fragment, garamond } from '~/styles/fonts'

const integral = `f(x) = \\int_{-\\infty}^\\infty\\hat f(\\xi)\\,e^{2\\pi i\\xi x}\\,d\\xi`
const snippets = new Map([
  // TODO: better sol'n
  [
    'get_batch',
    `def get_batch(data, context_length, batch_size):
    ix = torch.randint(0,len(data)-context_length-1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+context_length], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+1+context_length], dtype=torch.long) for i in ix])
    return x,y # y is one-step right shifted version of x, same size

get_batch(train_data,64,32)[0].shape #(B,T)
`,
  ],
])

export function VQGan() {
  return (
    <div className="postContainer">
      <div className="headerContainer">
        <div className={`${fragment.className} headerTitle`}>{meta.title}</div>
        <Date
          className={`${fragment.className} headerDate`}
          dateString={meta.date}
        />
        <div className={`${fragment.className} headerDescription`}>
          {meta.description}
        </div>
      </div>
      <Separator numCircles={5} width="100%" />
      <div className={garamond.className}>
        <h1>An Incredibly Sophisticated Subtitle</h1>
        <h2>An Even Better Sub-Subtitle</h2>
        <p>
          It&apos;s better to first understand the concept of language modeling
          generally rather than just in the transformer/GPT context--from now on
          I&apos;ll just call it GPT since it is the most used variant of the
          broader transformer architectures. For those who don&apos;t know, GPT
          and transformer are basically the same thing. Although none have come
          close to GPT in terms of actually realizing its potential at a large
          scale, the theoretical language modeling framework is more general
          than just the transformer. If you want to have a better intuition for
          both why GPTs work so well and understanding developments in the
          future, thinking of it&apos;s components at different levels of
          abstraction is really helpful. Language models take in a sequence of
          words as context and use that to auto-regressively generate a
          convincing continuation of the sentence. The sequence of words come
          from some huge training corpus, sampled randomly. Auto-regressive
          means that we have some initial context in the form of a sequence of
          words and we use that context to predict the next word. Then, we add
          that word back onto the original sentence and feed it back into the
          language model and have it try to generate the next word with the new
          sentence as input. That&apos;s what ChatGPT does. ChatGPT is
          performing a mapping from `sequence of words &rarr; word`. You can
          think of it like any other function `f` that takes in an input `x` and
          maps it to `y`, but a much, much longer formula.
        </p>
        <Latex>
          <Eq>{integral}</Eq>
        </Latex>
        <p>
          But ChatGPT is already trained. When the model is first learning to do
          the mapping(training), it is actually doing a `sequence of
          words(x)&rarr;sequence of words(y)` mapping, not just `sequence of
          words(x) &rarr; word(y)`
        </p>
        <Code code={snippets.get('get_batch') ?? ''} language={'python'} />
        <p>More Machine learning bullshit with a code block above.</p>
        <p>More Machine learning bullshit with a code block above.</p>
        <p>More Machine learning bullshit with a code block above.</p>
        <p>More Machine learning bullshit with a code block above.</p>
      </div>
    </div>
  )
}

export const meta: PostMeta = {
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
}

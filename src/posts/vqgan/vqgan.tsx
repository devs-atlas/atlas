import Image from 'next/image'
import {Latex} from '../../components/latex'
import Code from '~/components/code'
import Date from '~/components/date'
import { Separator } from '~/components/separator'
import type { PostMeta, Snippets } from '~/lib/posts'
import useInlineCodeStyling from '~/lib/useInlineCodeStyling'
import styles from '~/styles/Post.module.css'
import { fragment, garamond } from '~/styles/fonts'
import SelfAttentionDiagram from './diagrams/SelfAttentionDiagram'

type PostProps = {
  meta: PostMeta
  snippets: Snippets
}

export default function VQGan({ meta, snippets }: PostProps) {
  useInlineCodeStyling()

  return (
    <div className="postContainer">
      <div className="headerContainer">
        <div className={`${fragment.className} ${styles.headerTitle}`}>
          {meta.title}
        </div>
        <Date
          className={`${fragment.className} ${styles.headerDate}`}
          dateString={meta.date}
        />
        <div className={`${fragment.className} ${styles.headerDescription}`}>
          {meta.description}
        </div>
        <Separator numCircles={5} width="100%" />
      </div>
      <div className={garamond.className}>
  
      </div>
      {/*<SelfAttentionDiagram />*/}
      <div className={garamond.className}>
        <p>
          Even though the transformer is used to generate images in VQ-GAN, it's
          helpful to understand it on its own. Transformers have by-far the most
          quality material online in the NLP domain, so first presenting it in
          this context is a worthwhile move. It is much more difficult to
          understand transformers in the image domain and transfer that
          understanding to the NLP domain than vice versa.
        </p>
        <h1>Basics of Language Modeling</h1>
        <p>
          You should first understand the concept of language modeling generally
          rather than just in the GPT context. Although no other language models
          have come close to GPTs in terms of performing well at a large scale,
          it's helpful to undestand that the theoretical language modeling
          framework is more general than just the transformer. Thinking of
          architectures at different levels of abstraction depending on the
          context is helpful. For now, don't think about GPTs, think language
          models.
        </p>
        <h2>Auto-regressive Generation</h2>
        <p>
          
          During generation(like when you use ChatGPT), language models take in a sequence
          of words as context and use that to auto-regressively generate a convincing
          continuation of the sentence. Auto-regressive means that we have some initial
          context in the form of a sequence of
          words and we use that context to predict the next word.

          
          We add that word back onto
          the original sentence and feed it back into the
          language model and have it try to generate the next word with the new sentence as input.

          This is the "generating" side in the image below.

          GPTs are performing a mapping from `sequence of words -&gt; word`.
          You can think of it like any other function `f` that takes in an input `x` and maps it to `y`, but a much,
          much longer formula.
        </p>
        <h2>Training</h2>
        <p>
          But ChatGPT is already trained. When the model is learning to generate sentences(training), it performs a `sequence of words(x)-&gt;sequence of words(y)`
          mapping, not just `sequence of words(x) -&gt; word(y)`.

          Let's say we have the sentence `Hi my name is John`. When we break it into a sequence to be fed into the
          language model, the string is split into `["Hi", "my", "name", "is", "John"]`.
          </p>
          <p>
          The input sequence for one training sample is `["Hi", "my", "name", "is"]` and the output sequence is `["my", "name", "is", "John"]`.

          `x` and `y` are the same length, so it's really like mapping `Hi-&gt;my`,
          `my-&gt;name`, `name-&gt;is`, and so on. This is an essential insight, because that's what GPTs are doing at their core. 
          They are mapping each word in a sequence to the next word in the sequence directly. 
        </p>
        <p>
          GPTs are set-up in a clever way such that, at the position of `Hi`, the model can only make it's 
          prediction using the previous context(just `Hi` in this case). `my` 
          cannot interact with `name`, it only has access to `Hi my`.
        </p>
        <Image
          className="language-img"
          src="/posts/vqgan/nopeeking.svg"
          alt="Graph of word embeddings versus dimensions"
          width={500}
          height={500}
        /> 
        <p>
          As a result, when we pass in one training sentence into a GPT, it's like the model is
          outputting several next-word predictions all at once. `Hi` is doing its best to figure out `my`
          The model is making the next-sequence predictions at once. This is a lot more
          efficient than just comparing the last generated token since we can now compare
          the prediction at every single place in the sentence to the true value
          instead of only the last.
        </p>
        <p>
          When I said previously that language model do a `sequence of
          words-&gt;word` mapping when generating new sentences, that wasn't
          entirely true. It's still doing `sequence of words(x)-&gt;sequence of
          words(y)`, except we only care about the last word in the output
          sequence `y`. That's because the prior information about the models
          predictions for the shifted-over version aren't needed in the context
          of auto-regressive sampling.
        </p>
        <h3>Recap</h3>
        <Image
          className="language-img"
          src="/posts/vqgan/language.svg"
          alt="Graph of word embeddings versus dimensions"
          width={500}
          height={500}
        /> 
        <p>
          The transformer has two different phases: training and
          generation. When training, the model is mapping
          `sequence-&gt;sequence` and comparing it's predictions for each word
          to the true shifted over version of the input at every single
          position. In order to make these predictions for each word in the
          sequence, the model uses itself and the words before it. At the
          position `name` in `Hi my name is John`, the model is mapping `name`
          to `is` using the context of `Hi my name`.
        </p>
        <p>
          So now you understand how a transformer works at a high level and how
          it predicts the next word for each word in the sequence while only
          using itself and the words before it. I've left a lot of details up in
          the air so far, but most of the questions that you have about this
          will hopefully be solved as we try to answer the following questions:
        </p>
        <ol>
          <li>How is text represented as numbers by the model?</li>
          <li>Why can't the model look forward to just cheat the answer?</li>
          <li>
            How does the model include previous words(not just itself) as
            context?
          </li>
        </ol>
        <p>
          I'll start with the first and tackle the next one after, since 1 is a
          more `essential/fundamental` question to language modeling generally,
          while two and three are more GPT-specific.
        </p>
        <h1>Text to numbers and back</h1>
        <p>
          I've mentioned mapping words to other words so far, but obviously this
          is a big abstraction. No deep learning model can actually take in a
          string directly as input and process that through a neural network.
          Instead, we need an internal representation for language. The
          solution? Tokenization and embeddings.
        </p>
        <Image
          className="language-img"
          src="/posts/vqgan/texttonumbersnoshape.svg"
          alt="Graph of word embeddings versus dimensions"
          width={500}
          height={500}
        />
        <h2>Tokenization</h2>
        <p>
          Tokenization refers to transforming an input string into a discrete sequence of word-tokens.
          Typically, tokenization is a two-step process:
          <ol>
            <li>Build a vocabulary from the initial training corpus</li>
            <li>Use that vocabulary to break up the input string into a sequence of tokens</li>
          </ol>
          
          </p>
        <p>
          Before breaking up a sentence into a sequence of tokens, we need to identify
          the list of possible tokens, or our "vocabulary". There are many ways to do this:
          <ul>
            <li>Character level - get all unique characters from training corpus and assign each an integer</li>
            <li>Sub-word level - use an algorithm like Byte-Pair Tokenization to identify common sub-words</li>
            <li>Sentence level - split entire sentences into tokens using another pre-trained model</li>
          </ul>
        </p>
        <p>
          In practice, sub-word tokenization is the most common, and for good reason. Characters are too granular for a good
          discrete-representation of language. Think about it this way: `a` can be used in so many different contexts, whereas `apple`
          has a much more definite atomized meaning. On the other end, sentence-level tokenization is just too coarse-grained for language modeling tasks.
        </p>
        <p>
          I'll use a character-level tokenizer (will explain later) in the
          beginning and then switch it to BPE later on.
        </p>
        <h2>Embedding</h2>
        <Image
          className="language-img"
          src="/posts/vqgan/embedding.svg"
          alt="Graph of word embeddings versus dimensions"
          width={500}
          height={500}
        />
        <p>
          Tokenization converts strings to integers, yet these integers are not
          semantically meaningful. There's no information about the meaning of
          the word encoded in the integer(i.e. `flower=123` and `cybertruck= 124`).
          We could let the model operate on single integers, but then we only have
          one number to represent each word. This is not enough information to work with per word.
          
          Embeddings address this by assigning each integer(that refers to a sub-word token) in the
          vocabulary to an n-dimensional vector known as its embedding. The data
          structure that handles this assignment is called a lookup table. You
          take a string, map each string to an integer using sub-word
          tokenization, and then map that integer to the n-dimensional vector.

        </p>
        <p>
          The vectors in the lookup-table are adjusted during training. As a result, their positions in vector space begin
          to reflect their semantic meaning(look at the image above). `Germany` is 10 away from `Berlin` and `Tokyo` is 10 away from `Japan`.
          `Germany-Berlin=Capital` and `Japan-Tokyo=Capital`. Woah.
        </p>
        <h2>Unembedding</h2>
        <p>
          Unembedding is a fundamentally different process than embedding. There's no way of doing a reverse lookup table.
          We need to use a more flexible vector to integer model: a position-wise linear layer. If you don't get what
          "position-wise" means, I will explain. Our embedded sequence is composed of a
          list of n-dimensional vectors. These vectors each refer to an independent token in the sequence.
          When we use a position-wise linear layer, it means that each token vector
          in the sequence of token vectors is processed by the same linear
          layer. Therefore, the linear layer isn't a `sequence-&gt;sequence`
          linear map, but a `word-&gt;word` linear map applied individually
          across tokens. Remember when I said that when a GPT is doing a
          `sequence(x)-&gt;sequence(y)` mapping, it's really like each token in
          `x` is being mapped to the `y` in the same position. The position-wise
          layer is the reason for this. Now I can answer question 2 from above.
          "Why can't the model look forward to cheat the answer?" It's simply
          because there is no mechanism to do so. The embedding operates on each
          token independently, and the unembedding acts on each token vector
          independently. The ability to use context from before was an add-on
          made by the transformer.
        </p>
        <p>
          At this point, assuming you understand everything explained to this
          point, you've got a really good foundational intuition. With that
          groundwork laid, I now feel comfortable integrating some math notation
          into it.
        </p>
        <h1>Understanding the flow of data in terms of changes of shapes</h1>
        <p>In this section, I'll introduce you to a style of mathematical notation that makes reasoning about/debugging
          deep learning models drastically easier. 
        </p>
        <p>
          The input to our model is not actually a single sequence of text; it's
          a batch made of several, independent sequences of text. Each sequence
          in the batch is processed separately by the model, but batching is
          much more computationally efficient. I'll call the number of sequences
          in each batch `B`, although this is sometimes called `batch_size`.
          Each sequence in the batch is made up of a certain number of words.
          I'll call the number of tokens in each sequence `T`, although you
          might see it as `context_length`. When the input has only been
          tokenized(not yet embedded), the input is a batch of sequences of
          integers with shape `(B,T)`. Remember how I said that we could convert
          this representation to an equivalent version that uses one-hot
          encoding instead? If we used that representation, all the `T` integers
          in the sequence would be converted to a vector with the length
          `vocab_size`, meaning it's `(B,T,vocab_size)`. You can think of the
          non one-hot encoded version as `(B,T,1)`(the 1 is just a single
          integer) if that helps make the comparison clearer.
        </p>
        <p>
          When we embed it, we map each of those integers to a `C`-dimensional
          vector. `C` is also sometimes known as the `embed_dim`. Once embedded,
          the shape is `(B,T,C)`.
        </p>
        <p>
          In code, I will use embed_dim and context length, but I think the
          `(B,T,C)` convention is very good short-hand for comments and einsum
          notation, which I'll use later.
        </p>
        <Image
          className="language-img"
          src="/posts/vqgan/texttonumbers.svg"
          alt="Graph of word embeddings versus dimensions"
          width={500}
          height={500}
        />
        <h3>Recap</h3>
        <p>
          We have a bunch of text in our training dataset. We get text in
          batches and tokenize, resulting in an input with shape `(B,T)`.
          Converting this to a one-hot encoded representation yields a
          `(B,T,vocab_size)` input. After embedding, the input becomes
          `(B,T,C)`. Then comes the unembedding, which maps the `C`-dimensional
          vector back to `vocab_size`. Since this is the only functionality
          needed to go from input to output, we can start by building a simpler
          language model that goes from tokenization to embedding to
          unembedding.
        </p>
        <h1>Coding it up</h1>
        <p>
          First let's build the character-level tokenizer and load some data in
          (tiny-shakespeare). I'll explain each line in comments next to the
          code.
        </p>
        <Code code={snippets['open.py']} />
        <p>Get all the unique characters in data with `set`</p>
        <Code code={snippets['vocab.py']} />
        <p>
          Create tokenization mapping from integer to character and character to
          integer.
        </p>
        <Code code={snippets['encode.py']} />
        <Code code={snippets['tokenize.py']} />
        <p>Create a dataloader function from raw string. 
            1. Get batch_size random start index
            2. `x` runs from the `index` to `index+context_length`
            3. `y` runs from `index+1` to `index+1+context_length` -&gt; the shift</p>
        <Code code={snippets['get_batch.py']} />
        <h2>Tiniest Language Model</h2>
        <p>
          Now that we've got the necessary ingredients for the simplest possible
          embed to unembed language model, let's build it...
        </p>
        <Code code={snippets['language_model.py']} />
        <p>... and train</p>
        <Code code={snippets['losses_1.py']} />
        {/* TODO: show image in between */}
        <p>... and generate</p>
        <Code code={snippets['generate_simple.py']} />
        {/* TODO: add pre code showing output here */}
        <p>
          An interesting
          property well-known in <a href="https://transformer-circuits.pub/2021/framework/index.html">mechanistic interpretability</a>
           is that our simple language model(`embed-&gt;unembed`) approximates bigram
          statistics. Since the model is simply mapping one word to one word
          with no other context, the best it can theoretically do is to emulate
          bigram statistics. Bi-gram statistics are the literal frequency counts
          of each one word to one word mapping present in the dataset. Of
          course, it doesn't behave exactly the same; we intentionally cause
          some variability by probabilistically sampling instead of directly
          taking the highest component.
        </p>
        <h1>What's a GPT?</h1>
        <p>
          What we've just built is not a GPT, but it's close. The unique GPT
          components are located in between the embedding and unembedding layer.
          We've got the outer shell of a language model, but the inner
          components(what makes GPT special) will allow for reading prior
          context. Thankfully, these components aren't too hard to wrap your
          head around given the framework that I've laid out so far. Once
          embedded, the `(B,T,C)` input is fed through a series of residually
          stacked "transformer layers". Each transformer layer is identical in
          architecture. These transformers layers are "residual" because they do
          not replace the signal like `new_x = transformer_layer(x)`, but `x = x
          + transformer_layer(x)`. Inside the transformer layer are two
          components, an attention layer and an MLP layer. First, the attention
          layer is added onto the original, `x = x + attention(x)`, and then `x
          = x + mlp(x)`. If you expand this out, it looks like `x = x +
          attention(x) + mlp(x + attention(x))`. The original input persists
          through each transformer layer but is having information read and
          wrote to it in each layer. You can think of this as a sort of data
          highway. The embedded input `(B,T,C)` is fed into the transformer
          layers and it's values are repeatedly read and wrote to until it
          reaches the unembedding. Recall how, in our simple
          `embedding-&gt;unembedding` model, each token is being directly mapped
          from current to next token. Transformer layers add in the ability to
          mix in context from previous words as well. Specifically, the
          attention layer is the only operation that allows for information to
          mix from previous words because the MLP is another position-wise
          network. The concept that I've just described is the residual stream,
          a key insight of mechanistic interpretability. With this insight, the
          behaviour and interpretability of the model becomes much clearer. All
          operations on the residual stream are linear (direct addition), so
          information has the ability to persist in certain "linear subspaces"
          of the model throughout many layers. Consequently, layers can
          communicate and perform functions across each other.
        </p>

        <h2>Attention</h2>
        <p>
          In GPTs, attention acts as the primary mechanism for contextual understanding.
          It facilitates token-to-token information exchange crucial for predicting subsequent tokens.

          There are two functionalities that we want to have when moving context between tokens with the purpose of
          improving understanding throughout the residual stream.
        </p>
        <ol>
          <li>Read information from previous tokens(without also looking ahead to cheat)</li>
          <li>Write information from previous tokens into the residual stream</li>
        </ol>
        <h2>How to read</h2>
        <p>
          Reading information is similar to a data-retrieval process.
          We want each token to look to its left and consider which previous words are relevant to the understanding of
          the current token in some way. In the end, this should look like a probability distribution(all in between 0 and
          1, sum to 1) over all the tokens previous and including the current token. These probabilities reflect what
          proportion of the relevant information from the other token that we want to copy into our token.</p>
          
        <p>
          To do so, we want each token in the sequence, including itself, to "broadcast" a certain vector out that is
          supposed to represent its meaning that is relevant to the current token. It's called `K`. The current token
          then "broadcasts" its own vector to establish what information it's looking for, `Q`. At every position, every
          other token is deciding how relevant it is to the current token in this process. Attention uses the dot-product
          between the broadcasted `Q` vector and all corresponding `K` vectors to obtain a relevancy score value and then
          softmaxes to get probabilities. The entire reading process is summed up in the attention table. Each row corresponds
          to how much of the other token vectors we want to copy into the current.
        </p>
        <p>
          So the square grid that I referred to is actually a real thing -- the attention pattern. The attention pattern, `A`, is responsible for all the reading operations from other tokens. As a result, this is where we implemement the masking operation so our model can only look in the past. All that's needed is just a lower-triangular mask.
          The way you get that singular relevancy number between the `Q` and `K` vectors is by performing a dot-product comparison between the broadcasted vectors of all the other tokens in the sequence to the current token. Both `Q` and `K` have shape `(B,T,C)`, so to get the proper behavior of multiplication between each feature vector, the dot product is performed with the tranpose of `K`. `Q(B,T,C) @ K(B,C,T)--&gt;A(B,T,T)`. This is the square grid of tokens I wasreferring to. We can perform this kind of multiplication in PyTorch and the batch dimension is ignored so each query token vector is being multiplied by every other key token vector.
          The linear layers take the function of aligning to whatever operation they're put in. In this NLP, and more broadly attention context, the operation is that the `Q` vector should be what information we want from the other tokens, and the `K` vector should be what information the other vector has. Dot-producting them together along the feature vector dimension gives you the similarity. We can use this as our token reading operation.
        </p>
        <p>We implement it in code form here</p>
        <Code code={snippets['self_attention.py']} />
        <h2>From One to Multiple Heads</h2>
        <p>
          In this setup, each attention layer is responsible for one read and
          write operation. This is an inefficient use of compute because you can
          reduce the dimensionality of the `QKV` matrices and still achieve good
          performance. The way to resolve this is to split up each of the `QKV` matrices
          into `num_heads` heads, and apply the attention operation separately
          for each head. `Q` has shape `(B,T,C)`, so if we want four heads, we
          make the shape `(B,T,4,C//4)`. I'll call `C//4` the `head_dim` and
          substitute `4` for num_heads from now on. Then, we can just apply the
          same attention operation on each head separately and combine them at
          the end. For the attention pattern `A`, this means rearranging the `Q`
          and `K`(which have shape `(B,T,num_heads,head_dim)`) into
          `(B,num_heads,T,head_dim)`. To perform the scaled-dot product
          operation, you can just perform a matrix multiplication along the
          `head_dim` in the same way we did it along the `embed_dim` before. The
          output is `(B,num_heads,T,T)`.
        </p>
        <Code code={snippets['multihead_attention.py']} />
        <h2>MLP</h2>
        <p>
          From a mechanistic interpretability perspective, the MLP in the
          transformer is still somewhat mystifying. Though powerful, its reason
          for existence is less certain than the attention mechanism. MLP layers
          seem to improve the performance of the model, but they're not as
          direly necessary as attention. Regardless, they're importance to
          understand and implement. Just like everything else in the
          transformer, the MLP transformation is applied positon-wise to each
          token. There is no information movement between tokens, only better
          understanding the current token. The MLP takes an input `(B,T,C)`,
          expands it through a linear layer with `N` times the dimensionality of
          the embedding_dim, and then shrinks it back down from `N*embed_dim` to
          `embed_dim`. Increasing the hidden layer allows the model to learn a
          more complex behavior. After being fed through the expansion and
          unexpansion, the input is fed through an activation function.
        </p>
        <Code code={snippets['mlp.py']} />
        <h2>TransformerLayer</h2>
        <p>
          To further abstract the transformer layer so it's simpler in the actual GPT class, I put all the logic for the residual
          (attention + MLP) in here. That way, I can just run the input through a list of TransformerLayers. </p>
        <Code code={snippets['transformer_layer.py']}/>
        <h2>Positional Embedding</h2>
        <Image
          className="language-img"
          src="/posts/vqgan/positionalembedding.svg"
          alt="Graph of word embeddings versus dimensions"
          width={500}
          height={500}
        />
        <p>
          OK, last thing. Since we're including prior context, we need some way
          of telling the model *where* it's copying information from when doing
          attention. Attention does not have any notion of position inherintly.
          There are many more complex solutions out there but learned positional
          embeddings will do for our purposes. Similar to how we assigned an
          integer to each word in our vocabulary, we can also assign an integer
          to each position for all positions from 0 to the max context length.
          We can do this by creating an `nn.Embedding` with `embedding_dim`
          equal to the main `embedding_dim` and the number of embeddings equal
          to the max context length and an embedding dimension. Then, you can
          just add the positional embeddings to the main embeddings.
        </p>
        <Code code={snippets['positional_embedding.py']} />
        <h2>The long awaited GPT</h2>
        <p>
          Everything is finally built. Embedding, unembedding, MLP, multi-head
          attention, positional embedding. All that's left is to actually put it
          into this class. This could should be pretty simple if you've followed
          up until this point.
        </p>
      </div>
      <Code code={snippets['gpt.py']} />
      <div className={garamond.className}>
        <p>Let's use the same training loop as before but use a small GPT.</p>
        {/* TODO: missing plt.plot(losses screenshot) */}
        <Code code={snippets['losses_2.py']} />
        <Code code={snippets['generate_sequence.py']} />
        <p>It might not look great, but compare that to the output of an untrained model. It's clearly picking up on things
          like the rarity of certain special characters and commonality of newlines.</p>
        <Code code={snippets['random_output.py']} />
        {/* TODO: find clean way to display the last output*/}
        <h1>Boom</h1>
        <p>
          That's all there is to GPTs! With the intuition and technical
          knowledge you've got now, you should be able to understand new
          modifications to the architecture with relative ease. I'll go into
          some of these paradigms in another post.
        </p>
      </div>
    </div>
  )
}

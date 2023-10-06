import { useEffect } from 'react'
import Code from '~/components/code'
import Date from '~/components/date'
import { Separator } from '~/components/separator'
import type { PostMeta, Snippets } from '~/lib/posts'
import { fragment, garamond } from '~/styles/fonts'

type PostProps = {
  meta: PostMeta
  snippets: Snippets
}

export default function VQGan({ meta, snippets }: PostProps) {
  // TODO: extract to hook
  useEffect(() => {
    const elements = document.querySelectorAll('.post-content') // Adjust the selector as needed

    elements.forEach((el) => {
      el.innerHTML = el.innerHTML.replace(
        /`([^`]+)`/g,
        '<span class="inline-code">$1</span>'
      )
    })
  }, [])

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
      <div className={`${garamond.className} post-content`}>
        <p>
          This tutorial is part one of a three(maybe four) part series on
          VQ-GAN, a generative image model.
        </p>
        <p>
          Even though the transformer is used to generate images in VQ-GAN, it's
          helpful to understand it on its own. Transformers have by-far the most
          quality material online in the NLP domain, so first presenting it in
          this context is a worthwhile move. It is much more difficult to
          understand transformers in the image domain and transfer that
          understanding to the NLP domain than vice versa.
        </p>
        <h1>Highest-level Overview</h1>
        <p>
          Transformer=GPT in this tutorial. They have nuanced differences in
          reality, but in common parlance they mean the same thing. I'll use GPT
          in this article.
        </p>
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
        <p>
          Language models take in a sequence of words as context and use that to
          auto-regressively generate a convincing continuation of the sentence.
          The sequence of words come from some huge training corpus, sampled
          randomly. Auto-regressive means that we have some initial context in
          the form of a sequence of words and we use that context to predict the
          next word. Then, we add that word back onto the original sentence and
          feed it back into the language model and have it try to generate the
          next word with the new sentence as input. That's what ChatGPT does.
          ChatGPT is performing a mapping from `sequence of words --&gt; word`.
          You can think of it like any other function `f` that takes in an input
          `x` and maps it to `y`, but a much, much longer formula.
        </p>
        <p>
          But ChatGPT is already trained. When the model is first learning to do
          the mapping(training), it is actually doing a `sequence of
          words(x)--&gt;sequence of words(y)` mapping, not just `sequence of
          words(x) --&gt; word(y)`. Let's say we have the full sentence -- `Hi
          my name is John` and when we break it into a sequence to be fed into
          the language model, it becomes `["Hi", "my", "name", "is", "John"]`.
          The input sequence for one training sample would be `["Hi", "my",
          "name", "is"]` and the output sequence would be `["my", "name", "is",
          "John"]`. Notice how the input `x` and output `y` are the same length,
          so it's really like we're one-to-one mapping `Hi-&gt;my`,
          `my-&gt;name`, `name-&gt;is`, and so on. This is an essential insight,
          because that's what GPTs are doing at their core. At this point, you
          might be a bit confused(like I was). "The input `x` already has the
          word `my` in it, so why can't the transformer just automatically tell
          that name is coming next because it's also in the input? Can't it just
          directly use that word in the input to make the prediction for
          `Hi-&gt;my`?" The answer is no. GPTs are set-up in a clever way such
          that, at the position of `Hi`, the model can only make it's prediction
          using the previous context, only `Hi` in this case. Similarly, `my`
          cannot interact with `name`, it only has access to `Hi my`. As a
          result, when we pass in one training sentence into a GPT, it's like
          the model is outputting several auto-regressive predictions in each
          `sequence-&gt;sequence` mapping. The model is making the next-sequence
          predictions at once. This is a lot more efficient than just comparing
          the last generated token since we can now compare the prediction at
          every single place in the sentence to the true value instead of only
          the last.
        </p>
        <p>
          When I said previously that language model do a `sequence of
          words--&gt;word` mapping when generating new sentences, that wasn't
          entirely true. It's still doing `sequence of words(x)-&gt;sequence of
          words(y)`, except we only care about the last word in the output
          sequence `y`. That's because the prior information about the models
          predictions for the shifted-over version aren't needed in the context
          of auto-regressive sampling. This is just an unfortunate consequence
          of the big gains we get in performance during training, since it could
          be faster.
        </p>
        <p>
          Recap: The transformer has two different phases: training and
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
        <h2>Text to numbers and back</h2>
        <p>
          I've mentioned mapping words to other words so far, but obviously this
          is a big abstraction. No deep learning model can actually take in a
          string directly as input and process that through a neural network.
          Instead, we need an internal representation for language. The
          solution? Tokenization and embeddings.
        </p>
        <h2>Step 1: Tokenization</h2>
        <p>
          Tokenization is the step of converting "Hi my name is John" to ["Hi",
          "my", "name", "is" , "John"]. One way we could do this is to take all
          the words in the entire training dataset and assign them a unique
          integer identifier. This is like a "vocabulary" that we can use to
          convert a string in the dataset to a list of integers. If "Hi" had
          integer index 741, "my" 129, and "name" 860, "Hi my name" would become
          [741, 129, 860]. Once again, this is a simplification. Real models
          don't just get all the words because this would make the vocabularies
          way too large. In reality, a vocabulary is typically built from
          sub-word chunks using an algorithm like Byte-Pair Encoding(BPE). A
          word like "ensures" might be split up into "en", "sure", "s". It's
          also possible that "ensure" appears so frequently in the training
          dataset that it gets its own integer index for the entire word.
          Choosing a suitable tokenization method is project-specific and is
          often very messy(integers, for example, are not handled well by BPE).
        </p>
        <p>
          This sentence representation(list of integers) is one of two versions.
          It's more visually interpretable but less computation-friendly than
          the other representation which uses one-hot encoding. In one-hot
          encoding, each token is represented by a vector where all elements are
          zeros except for a single '1' at the index corresponding to the
          token's integer identifier from the vocabulary. Using our previous
          values(Hi=741, my=129, name=860), the sentence would not be a list of
          integers, but a list of one-hot encoded vectors. The first vector in
          the list would refer to "Hi", so there would be a '0' everywhere
          except at the index 741 in the vector. The length of each vector is
          the length of our "vocabulary". You'll often see this called
          `vocab_size`.
        </p>
        <p>
          The one-hot encoded representation is what we'll actually use as the
          input and output to our model, so it's important that you understand
          it well. I'll get more into the details of that later on, though.
        </p>
        <p>
          I'll use a character-level tokenizer (will explain later) in the
          beginning and then switch it to BPE later on.
        </p>
        <h2>Step 2: Embedding</h2>
        <p>
          Tokenization converts strings to integers, yet these integers are not
          semantically meaningful. There's no information about the meaning of
          the word encoded in the integer("flower" may be 123 and "cybertruck"
          124). Ideally, the model would learn something about what these words
          mean during the training procedure and their numerical values
          pertained to some semantic meaning. Embeddings address this by
          assigning each integer(that refers to a sub-word token) in the
          vocabulary to an n-dimensional vector known as its **embedding**. The
          structure that handles this assignment is called a lookup table. You
          take a string, map each string to an integer using sub-word
          tokenization, and then map that integer to the n-dimensional vector.
        </p>
        <p>
          Recap: Tokenization converts from a string to a list of integers.
          Embedding converts from a list of integers to a list of vectors.
        </p>
        <h2>Step 3: Unembedding</h2>
        <p>
          Unembedding is a different process from embedding, despite the name.
          Obviously, it can't simply do a reverse-embedding since we don't
          expect the output of the model to be made of vectors of the exact same
          values as the input. They will be changed a bit. Instead, we use
          something called a position-wise linear layer. If you don't get what a
          linear layer is, I suggest you watch 3blue1browns series on neural
          networks. If you don't get what "position-wise" means, I will explain.
          Our embedded sequence is composed of a list of n-dimensional vectors.
          These vectors each refer to an independent token in the sequence. When
          we use a position-wise linear layer, it means that each token vector
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
          made by the transformer, not the other way around(the ability to look
          ahead for context was removed).
        </p>
        <p>
          At this point, assuming you understand everything explained to this
          point, you've got a really good foundational intuition. With that
          groundwork laid, I now feel comfortable integrating some math notation
          into it.
        </p>
        <h2>Understanding the flow of data in terms of changes of shapes</h2>
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
        <p>
          Recap: We have a bunch of text in our training dataset. We get text in
          batches and tokenize, resulting in an input with shape `(B,T)`.
          Converting this to a one-hot encoded representation yields a
          `(B,T,vocab_size)` input. After embedding, the input becomes
          `(B,T,C)`. Then comes the unembedding, which maps the `C`-dimensional
          vector back to `vocab_size`. Since this is the only functionality
          needed to go from input to output, we can start by building a simpler
          language model that goes from tokenization to embedding to
          unembedding.
        </p>
        <p>
          First let's build the character-level tokenizer and load some data in
          (tiny-shakespeare). I'll explain each line in comments next to the
          code.
        </p>
        <Code code={snippets['open.py']} />
        <Code code={snippets['vocab.py']} />
        <Code code={snippets['encode.py']} />
        <Code code={snippets['tokenize.py']} />
        <Code code={snippets['get_batch.py']} />
        <p>
          Now that we've got data loaded in, it's time to embed. I could use
          `nn.Embedding`, but I'm going to make my own. Not exactly "on my own",
          because I'm going to call `torch.nn.functional.embedding` instead
          which allows us to actually create the embedding matrix ourselves.
          It's helpful to see it like that.
        </p>
        <Code code={snippets['embedding.py']} />
        <h2>Tiniest Language Model</h2>
        <Code code={snippets['language_model.py']} />
        <Code code={snippets['losses_1.py']} />
        {/* TODO: show image in between */}
        <Code code={snippets['word_embeddings.py']} />
        <p>We can also make it generate text</p>
        <Code code={snippets['text.py']} />
        {/* TODO: add pre code showing output here */}
        <p>
          Mechanistic interpretability is the study of deep neural networks and
          their workings. It's application to GPTs has been pioneered by
          Anthropic in their "Transformer Circuits" series. I highly, highly
          recommend reading that after going through this post.
        </p>
        <Code code={snippets['self_attention.py']} />
        <h2>From One to Multiple Heads</h2>
        <p>
          In this setup, each attention layer is responsible for one read and
          write operation. This is an inefficient use of compute because you can
          reduce the dimensionality of the `QKV` matrices and still achieve good
          performance. The way to resolve this is to split up the `QKV` matrices
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
        <h2>Positional Embedding</h2>
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
        <h2>The long awaited GPT</h2>
        <p>
          Everything is finally built. Embedding, unembedding, MLP, multi-head
          attention, positional embedding. All that's left is to actually put it
          into this class. This could should be pretty simple if you've followed
          up until this point.
        </p>
        <Code code={snippets['gpt.py']} />
        <p>Let's use the same training loop as before but use a small GPT.</p>
        {/* TODO: missing plt.plot(losses screenshot) */}
        <Code code={snippets['losses_2.py']} />
        <Code code={snippets['generate_sequence.py']} />
        {/* TODO: find clean way to display the last output*/}
        <h2>Boom</h2>
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

class LanguageModel(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()

        self.embedding = Embedding(embed_dim, vocab_size)
        self.unembedding = Unembedding(embed_dim, vocab_size)

    def forward(self, x):
        return self.unembedding(self.embedding(x))

class Embedding(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(vocab_size, embed_dim))

    def forward(self, x, padding_idx=None):
        # expecting (B,T)
        return F.embedding(x, self.matrix, padding_idx=padding_idx)


class Unembedding(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.unembedder = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        # expecting (B,T,C)
        return self.unembedder(x)

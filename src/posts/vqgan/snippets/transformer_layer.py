class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(TransformerLayer, self).__init__()

        self.self_attention = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.self_attention(
            self.norm1(x)
        )  # residual connection with "pre-norm"
        x = x + self.mlp(self.norm2(x))

        return x
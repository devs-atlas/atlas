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


class GPT(LanguageModel):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        num_layers,
        max_context_length,
        vocab_size,
    ):
        super().__init__(embedding_dim, vocab_size)
        self.transformer_layers = nn.ModuleList(
            (
                TransformerLayer(embedding_dim, num_heads)
                for _ in range(num_layers)
            )
        )
        self.pos_embedding = nn.Embedding(
            max_context_length, embedding_dim
        ).to("mps")
        self.init_drop = nn.Dropout(0.05)
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        emb_x = self.embedding(x)

        B, T, C = emb_x.shape
        pos_emb_idx = torch.arange(0, T).to("mps")
        pos_emb_x = self.pos_embedding(pos_emb_idx)

        x = self.init_drop(pos_emb_x + emb_x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        x = self.final_norm(x)
        logits = self.unembedding(x)

        return logits

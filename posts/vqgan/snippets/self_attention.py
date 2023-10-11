class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.scaling_dim = torch.sqrt(
            torch.tensor([self.embedding_dim], dtype=torch.float32)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        A = q @ k.transpose(-1, -2) / self.scaling_dim
        A = F.softmax(A, dim=-1)
        mask = torch.triu(
            torch.ones((T, T), device=x.device), diagonal=1
        ).bool()  # Upper triangular mask
        A = A.masked_fill(mask, -1e9)  # Use masked_fill for masking

        return A @ v

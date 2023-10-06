class MLP(nn.Module):
    def __init__(self, embedding_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.fc2 = nn.Linear(embedding_dim * 4, embedding_dim)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(F.gelu(x))
        x = self.dropout(x)
        return x

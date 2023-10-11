embedding_dim = 16
num_heads = 4
num_transformer_layers = 5
context_length = 256

lm = GPT(
    embedding_dim,
    num_heads,
    num_transformer_layers,
    context_length,
    vocab_size,
).to("mps")
optimizer = torch.optim.Adam(lm.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
losses = []

for step in range(5000):
    x, y = get_batch(train_data, 256, 64)
    x = x.to("mps")
    y = y.to("mps")

    pred_logits = lm(x)
    loss = criterion(
        pred_logits.view(-1, vocab_size).float(), y.view(-1)
    ).requires_grad_(True)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.cpu().detach().numpy())
    if step % 100 == 0:
        print(f"Step {step}, Loss {loss.item()}")

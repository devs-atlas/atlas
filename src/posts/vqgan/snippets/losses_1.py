lm = LanguageModel(2, vocab_size)
optimizer = torch.optim.Adam(lm.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
losses = []

for step in range(10000):
    x, y = get_batch(train_data, 1, 64)
    pred_logits = lm(x)
    loss = criterion(
        pred_logits.view(-1, vocab_size).float(), y.view(-1)
    ).requires_grad_(True)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.detach().numpy())
    if step % 10 == 0:
        print(f"Step {step}, Loss {loss.item()}")

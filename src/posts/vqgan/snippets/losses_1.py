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

# SNIPPET #
Step 0, Loss 4.480882167816162
Step 1000, Loss 3.8772220611572266
Step 2000, Loss 3.561326742172241
Step 3000, Loss 3.372966766357422
Step 4000, Loss 2.8575117588043213
Step 5000, Loss 3.0783464908599854
Step 6000, Loss 3.2430574893951416
Step 7000, Loss 3.1417269706726074
Step 8000, Loss 3.090763568878174
Step 9000, Loss 3.076702117919922
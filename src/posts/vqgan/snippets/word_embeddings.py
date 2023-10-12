# Extract the first two dimensions of the embedding matrix
x = lm.embedding.matrix[:, 0].detach().numpy()
y = lm.embedding.matrix[:, 1].detach().numpy()


plt.figure(figsize=(12, 12))

# Create scatter plot
plt.scatter(x, y)

# Annotate each point with its corresponding word
for i, word in enumerate(vocab):
    plt.annotate(word, (x[i], y[i]))

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Word Embeddings")
plt.show()

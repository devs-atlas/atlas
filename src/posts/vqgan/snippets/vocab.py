# set returns only the unique set of chars, then we convert to a list so we can sort
vocab = sorted(list(set(data)))
vocab_size = len(vocab)

len(vocab), vocab[:10]
# SNIPPET #
(65, ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3'])

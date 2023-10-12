def get_batch(data, context_length, batch_size):
    ix = torch.randint(0, len(data) - context_length - 1, (batch_size,))
    x = torch.stack(
        [
            torch.tensor(data[i : i + context_length], dtype=torch.long)
            for i in ix
        ]
    )
    y = torch.stack(
        [
            torch.tensor(
                data[i + 1 : i + 1 + context_length], dtype=torch.long
            )
            for i in ix
        ]
    )
    return x, y  # y is one-step right shifted version of x, same size


get_batch(train_data, 64, 32)[0].shape  # (B,T)

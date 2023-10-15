b_s = 3
x, _ = get_batch(train_data, 32, b_s)
token_tensors = x[:, -1].view(b_s, -1, 1)
context = x
for i in range(50):
    relevant_logit = lm(context)[:, -1, :]
    probs = F.softmax(relevant_logit, dim=-1)
    idx = torch.multinomial(probs, 1, replacement=True)
    token_tensors = torch.cat((token_tensors, idx.unsqueeze(-1)), dim=-1)
    x = x[:, 1:].view(b_s, -1)
    context = torch.cat((x, idx), dim=-1)

print(
    "".join(
        decode(list(token_tensors[0].flatten().detach().numpy()))
    )
)

# SNIPPET #
fesn!atbRFhed wfrnt o soaq:A
Wle he,oI
Wud deis
MVS
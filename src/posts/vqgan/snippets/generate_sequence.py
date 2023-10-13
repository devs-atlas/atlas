def generate_sequence(
    lm,
    train_data,
    batch_size=1,
    seq_length=256,
    context_length=256,
    temperature=1.0,
):
    # Get initial context
    initial_context, _ = get_batch(train_data, 256, batch_size)
    initial_context = initial_context.to("mps")

    generated_tokens = initial_context[:, -1].unsqueeze(-1)
    context = initial_context

    for _ in range(seq_length):
        if context_length and context.size(1) > context_length:
            context = context[:, -context_length:]

        # Fetch the relevant logits from the model
        logits = lm(context)[:, -1, :] / temperature
        # scale by temperature and compute probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample a token based on the probabilities
        sampled_token = torch.multinomial(probs, 1)

        # Append the token and update context
        generated_tokens = torch.cat((generated_tokens, sampled_token), dim=-1)
        context = torch.cat((context[:, 1:], sampled_token), dim=-1)

    # Convert tokens to string
    generated_sequence = "".join(
        decode(list(generated_tokens[0].flatten().cpu().detach().numpy()))
    )

    return generated_sequence


# Usage
generated_text = generate_sequence(lm, train_data, temperature=1)
print(generated_text)

# SNIPPET #
ntusr oh alr?hrw  ernrre Tretstb sbd
T i e o e unOe
otelN hlme udWwWorte he m,eTigr  t f
? ooe vdo

A?bdoiSomnuedor tnrFlodA oepo uy,loc e vneoy i mi trleuord
onooc.unl,taoTroeld nod;cTaacdduo aa
Nrog
cwilloaa
Cerr reai hrtlVdtoib:Wgtr
N!?orrrmlrhdo edenilo

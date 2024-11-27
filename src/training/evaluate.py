import torch as th


def evaluate(model, x, y):
    model.eval()
    x = th.from_numpy(pad_zero([x], max_len=MAX_LENGTH)).long().to(device)
    y = th.from_numpy(pad_zero([y], max_len=MAX_LENGTH)).long().to(device)
    decoder_outputs = model(x, y)
    _, topi = decoder_outputs.topk(1)
    decoded_ids = topi.squeeze()
    decoded_words = []
    for idx in decoded_ids:
        decoded_words.append(dataset.index2word[idx.item()])
    return "".join(decoded_words)
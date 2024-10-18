import torch as th


def evaluate(model, x, y):
    model.eval()
    x = th.from_numpy((pad_zero([x], max_len=MAX_LENGTH)).long())
"""Synthetic tasks for Module 05. Each has a known right answer, so results can be checked."""
import torch

N_CLASSES = 4


def make_recall_batch(batch_size, seq_len, generator, n_classes=4, noise_dim=2):
    """Delayed recall from Module 04. Step 0 carries a class label as a one-hot vector in the first
    n_classes features; every later step is noise. The label is read out at the last step."""
    labels = torch.randint(0, n_classes, (batch_size,), generator=generator)
    x = torch.randn(batch_size, seq_len, n_classes + noise_dim, generator=generator) * 0.5
    x[:, :, :n_classes] = 0.0
    x[torch.arange(batch_size), 0, labels] = 1.0
    return x, labels


def make_order_batch(batch_size, seq_len, generator, vocab=10):
    """Order task. Every sequence contains the marker tokens 1 and 2 exactly once, the rest is
    random filler from 3..vocab-1. The label is 1 if marker 1 comes before marker 2, else 0.
    The two classes contain exactly the same tokens, so only the order separates them."""
    tokens = torch.randint(3, vocab, (batch_size, seq_len), generator=generator)
    pos = torch.stack([torch.randperm(seq_len, generator=generator)[:2] for _ in range(batch_size)])
    rows = torch.arange(batch_size)
    tokens[rows, pos[:, 0]] = 1
    tokens[rows, pos[:, 1]] = 2
    labels = (pos[:, 0] < pos[:, 1]).long()
    return tokens, labels


def make_neighbour_batch(batch_size, seq_len, generator, vocab=6):
    """Neighbour task. Predict, at every position i, (token[i-1] + token[i+1]) mod vocab, with the
    sequence treated as a ring. Answering needs two separate lookups, one to the left and one to the right."""
    tokens = torch.randint(0, vocab, (batch_size, seq_len), generator=generator)
    targets = (torch.roll(tokens, 1, dims=1) + torch.roll(tokens, -1, dims=1)) % vocab
    return tokens, targets

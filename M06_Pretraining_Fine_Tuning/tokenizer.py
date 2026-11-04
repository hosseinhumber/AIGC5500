"""Byte-pair encoding from scratch for Module 06.

The corpus is a dictionary {word: count}. Each word is a tuple of symbols that
starts as single characters plus an end-of-word marker. Training repeatedly
merges the most frequent adjacent pair.
"""
from collections import Counter

END = "</w>"


def word_to_symbols(word):
    return tuple(word) + (END,)


def pair_counts(vocab):
    """Count adjacent symbol pairs, weighted by how often each word occurs."""
    counts = Counter()
    for symbols, freq in vocab.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] += freq
    return counts


def apply_merge(vocab, pair):
    merged = pair[0] + pair[1]
    new_vocab = {}
    for symbols, freq in vocab.items():
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        new_vocab[tuple(out)] = new_vocab.get(tuple(out), 0) + freq
    return new_vocab


def train_bpe(word_counts, n_merges):
    """Return the list of merges as (pair, count_at_the_time).

    Ties are broken by taking the largest (count, pair) so the result is
    deterministic and does not depend on dictionary order.
    """
    vocab = {word_to_symbols(w): c for w, c in word_counts.items()}
    merges = []
    for _ in range(n_merges):
        counts = pair_counts(vocab)
        if not counts:
            break
        pair = max(counts, key=lambda p: (counts[p], p))
        merges.append((pair, counts[pair]))
        vocab = apply_merge(vocab, pair)
    return merges


def encode_word(word, merges):
    """Apply the learned merges, in the order they were learned, to a new word."""
    symbols = word_to_symbols(word)
    for pair, _ in merges:
        merged = pair[0] + pair[1]
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = tuple(out)
    return list(symbols)


def word_counts_from_text(text):
    return dict(Counter(text.lower().split()))

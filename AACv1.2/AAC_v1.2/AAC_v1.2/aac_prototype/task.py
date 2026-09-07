"""
Synthetic associative-recall benchmark.

A sequence presents n_pairs distinct (KEY_i, VAL_j) associations, each
separated by filler noise tokens, followed by a long filler gap, then a
run of QUERY_i tokens that must be answered with the associated VAL_j.
This is the standard stress test for architectures that claim long-range,
selective memory (the filler gap is deliberately made much longer than
the fast recurrent state's practical receptive field) -- exactly the
setting AAC's persistent associative memory is meant to help with.

Token layout (all disjoint id ranges):
    KEY_i     id = i                          for i in [0, n_ids)
    VAL_j     id = n_ids + j                  for j in [0, n_ids)
    QUERY_i   id = 2*n_ids + i                 for i in [0, n_ids)
    FILLER_f  id = 3*n_ids + f                 for f in [0, n_filler_vocab)
"""
import numpy as np

KEY, VAL, QUERY, FILLER = 0, 1, 2, 3


def decode_token(tok_id, n_ids):
    if tok_id < n_ids:
        return KEY, tok_id
    if tok_id < 2 * n_ids:
        return VAL, tok_id - n_ids
    if tok_id < 3 * n_ids:
        return QUERY, tok_id - 2 * n_ids
    return FILLER, tok_id - 3 * n_ids


class AssociativeRecallTask:
    def __init__(self, n_ids=16, n_filler_vocab=40, seed=0):
        self.n_ids = n_ids
        self.n_filler_vocab = n_filler_vocab
        self.vocab_size = 3 * n_ids + n_filler_vocab
        self.rng = np.random.default_rng(seed)

    def sample_episode(self, n_pairs=5, filler_between=(3, 8), gap_len=(40, 60),
                        n_queries=5):
        rng = self.rng
        ids = rng.choice(self.n_ids, size=max(n_pairs, n_queries), replace=False)
        pair_ids = ids[:n_pairs]
        # values are an independent random assignment (id -> value id)
        val_of = {i: int(rng.integers(0, self.n_ids)) for i in pair_ids}

        seq, kinds = [], []

        def emit(kind, ident):
            if kind == KEY:
                seq.append(ident)
            elif kind == VAL:
                seq.append(self.n_ids + ident)
            elif kind == QUERY:
                seq.append(2 * self.n_ids + ident)
            else:
                seq.append(3 * self.n_ids + ident)
            kinds.append(kind)

        for i in pair_ids:
            emit(KEY, int(i))
            emit(VAL, val_of[int(i)])
            for _ in range(rng.integers(*filler_between)):
                emit(FILLER, int(rng.integers(0, self.n_filler_vocab)))

        for _ in range(rng.integers(*gap_len)):
            emit(FILLER, int(rng.integers(0, self.n_filler_vocab)))

        query_ids = rng.choice(pair_ids, size=n_queries, replace=True)
        query_positions, query_labels = [], []
        for i in query_ids:
            emit(QUERY, int(i))
            query_positions.append(len(seq) - 1)
            query_labels.append(val_of[int(i)])
            for _ in range(rng.integers(1, 4)):
                emit(FILLER, int(rng.integers(0, self.n_filler_vocab)))

        return {
            "tokens": np.array(seq, dtype=int),
            "kinds": np.array(kinds, dtype=int),
            "query_positions": query_positions,
            "query_labels": query_labels,
        }

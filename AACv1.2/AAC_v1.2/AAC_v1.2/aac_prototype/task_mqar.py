"""MQAR task (shared vocab, no privileged key/query alignment)."""
import numpy as np


class MQARTask:
    def __init__(self, n_vocab=64, seed=0):
        self.n_vocab = n_vocab
        self.vocab_size = n_vocab
        self.rng = np.random.default_rng(seed)

    def sample_episode(self, n_pairs=6, filler_between=(2, 6), gap_len=(20, 40),
                        n_queries=6):
        rng = self.rng
        chosen = rng.choice(self.n_vocab, size=2 * n_pairs, replace=False)
        keys, vals = chosen[:n_pairs], chosen[n_pairs:]
        val_of = {int(k): int(v) for k, v in zip(keys, vals)}

        seq, targets = [], {}
        key_positions = []

        def emit_filler(n):
            for _ in range(n):
                seq.append(int(rng.integers(0, self.n_vocab)))

        for k in keys:
            key_positions.append(len(seq))
            seq.append(int(k))
            seq.append(val_of[int(k)])
            emit_filler(int(rng.integers(*filler_between)))

        emit_filler(int(rng.integers(*gap_len)))

        query_keys = rng.choice(keys, size=n_queries, replace=True)
        for k in query_keys:
            seq.append(int(k))
            targets[len(seq) - 1] = val_of[int(k)]
            emit_filler(int(rng.integers(1, 4)))

        return {
            "tokens": np.array(seq, dtype=int),
            "query_positions": list(targets.keys()),
            "query_labels": list(targets.values()),
            "key_positions": key_positions,
        }

"""
Controller (AAC spec sections 7, 8, 18, 19, 20).

The single learned utility estimator U_hat = f_phi(e_t, S_t, T_t, P_t) is
implemented as a small MLP (see aac/mlp.py). Its scalar output drives three
downstream decisions, all following the value-of-computation principle

    a_t = argmax_a [ E(delta_L | a) - lambda_C C(a) - lambda_M M(a) ]

by comparing a learned expected-benefit signal to a fixed cost:

  - write into temporary trace  (section 8):  gate on U_hat
  - promote trace -> persistent (section 9):  gate on accumulated evidence
    in the trace, itself fed by U_hat at each visit (see trace.py)
  - adaptive reasoning steps    (section 20):  extra refinement passes are
    taken while the output distribution's entropy (a proxy for uncertainty
    u_t) exceeds a threshold and budget remains, i.e. while the estimated
    marginal benefit of another pass plausibly exceeds lambda_C.
"""
import numpy as np
from .mlp import MLP


class Controller:
    def __init__(self, d_in, d_hidden=32, write_threshold=0.5,
                 lambda_c=0.02, max_reasoning_steps=3, seed=0):
        self.utility_net = MLP(d_in, d_hidden, 1, head="sigmoid", seed=seed)
        self.write_threshold = write_threshold
        self.lambda_c = lambda_c
        self.max_reasoning_steps = max_reasoning_steps

        # bookkeeping for the printed run report
        self.n_read_attempts = 0
        self.n_write_gated_in = 0
        self.n_write_gated_out = 0
        self.total_reasoning_steps = 0
        self.total_events = 0

    def estimate_utility(self, features, train=False):
        """features: (batch, d_in) -> (batch,) utility in [0,1]."""
        u = self.utility_net.forward(features, train=train)[:, 0]
        return u

    def train_utility(self, features, proxy_labels):
        """proxy_labels: (batch,) in {0,1}, 1 = 'this event is worth
        remembering' (see task.py for how the synthetic task supplies this
        supervision signal -- a stand-in for the true counterfactual U(e_t)
        of section 7, which is not cheaply observable online)."""
        y = self.utility_net.forward(features, train=True)
        self.utility_net.backward_sigmoid_bce(proxy_labels[:, None])

    def gate_write(self, utility_scalar):
        """Value-of-computation write decision (section 8/19): write only
        if expected benefit U_hat exceeds the fixed compute+storage cost."""
        self.total_events += 1
        should_write = utility_scalar > (self.write_threshold + self.lambda_c)
        if should_write:
            self.n_write_gated_in += 1
        else:
            self.n_write_gated_out += 1
        return should_write

    def n_queries(self, uncertainty, n_max=3, kappa=3.0):
        """Multi-query fan-out (section 13): n_t = min(n_max, ceil(kappa*u_t))."""
        return int(min(n_max, np.ceil(kappa * uncertainty)))

    def reasoning_budget(self, entropy, entropy_threshold=0.6):
        """Adaptive reasoning steps (section 20): keep taking extra passes
        while uncertainty is high and budget remains."""
        steps = 0
        e = entropy
        while e > entropy_threshold and steps < self.max_reasoning_steps:
            steps += 1
            e *= 0.6  # each refinement pass is assumed to reduce uncertainty
        self.total_reasoning_steps += steps
        return steps

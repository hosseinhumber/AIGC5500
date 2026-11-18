"""The multi-armed bandit: one state, several actions, and the exploration problem in its purest form."""
import numpy as np


class Bandit:
    """k arms with Gaussian rewards. The true means are drawn once, from a standard normal; each pull adds unit noise."""
    def __init__(self, k=10, seed=0):
        rng = np.random.default_rng(seed)
        self.k = k
        self.means = rng.normal(0, 1, k)
        self.best = self.means.max()
        self.rng = np.random.default_rng(seed + 10_000)

    def pull(self, a):
        return self.means[a] + self.rng.normal()


def run_bandit(bandit, strategy, steps=1000, eps=0.1, c=2.0, optimistic=0.0, tau=0.2):
    """Returns the per-step regret (best mean minus the mean of the arm chosen) and the pulls of each arm.
    strategy: 'greedy' (use `optimistic` as the starting estimate), 'epsilon', 'ucb' or 'softmax'."""
    k = bandit.k
    n = np.zeros(k)
    est = np.full(k, optimistic if strategy == "greedy" else 0.0)
    regret = np.zeros(steps)
    rng = np.random.default_rng(1)
    for t in range(steps):
        if strategy == "greedy":
            a = int(np.argmax(est + 1e-9 * rng.random(k)))
        elif strategy == "epsilon":
            a = int(rng.integers(k)) if rng.random() < eps else int(np.argmax(est))
        elif strategy == "ucb":
            a = t if t < k else int(np.argmax(est + c * np.sqrt(np.log(t + 1) / np.maximum(n, 1))))
        elif strategy == "softmax":
            p = np.exp((est - est.max()) / tau); p /= p.sum()
            a = int(rng.choice(k, p=p))
        else:
            raise ValueError(strategy)
        r = bandit.pull(a)
        n[a] += 1
        est[a] += (r - est[a]) / n[a]
        regret[t] = bandit.best - bandit.means[a]
    return regret, n

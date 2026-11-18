"""A tiny, exactly solvable setting for preference learning: reward model, KL-regularized RL, and DPO.
There are several prompts and a fixed menu of candidate responses for each. A 'policy' is a table of
probabilities over responses for each prompt, defined by logits."""
import numpy as np
import torch
import torch.nn.functional as F


class PreferenceWorld:
    """n_prompts prompts, n_resp responses each. The true reward of a response is drawn from a standard normal.
    A labeller prefers response i over j with probability sigmoid(r_i - r_j) (the Bradley-Terry model)."""
    def __init__(self, n_prompts=6, n_resp=8, seed=0):
        rng = np.random.default_rng(seed)
        self.n_prompts, self.n_resp = n_prompts, n_resp
        self.true_r = torch.as_tensor(rng.normal(0, 1, (n_prompts, n_resp)), dtype=torch.float32)
        self.rng = rng

    def sample_pairs(self, n):
        """Return prompt, winner and loser indices for n labelled comparisons."""
        p = self.rng.integers(0, self.n_prompts, n)
        i = self.rng.integers(0, self.n_resp, n)
        j = (i + self.rng.integers(1, self.n_resp, n)) % self.n_resp
        pi = torch.sigmoid(self.true_r[p, i] - self.true_r[p, j]).numpy()
        i_wins = self.rng.random(n) < pi
        w = np.where(i_wins, i, j); l = np.where(i_wins, j, i)
        return torch.as_tensor(p), torch.as_tensor(w), torch.as_tensor(l)


def fit_reward_model(world, p, w, l, steps=400, lr=0.1, l2=0.0):
    """Fit a reward table by maximizing the Bradley-Terry likelihood of the labelled preferences."""
    r = torch.zeros(world.n_prompts, world.n_resp, requires_grad=True)
    opt = torch.optim.Adam([r], lr=lr)
    for _ in range(steps):
        loss = -F.logsigmoid(r[p, w] - r[p, l]).mean() + l2 * (r ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return r.detach()


def kl_regularized_optimum(ref_logits, reward, beta):
    """The policy that maximizes E[reward] - beta KL(policy || reference): pi* is proportional to ref * exp(reward / beta)."""
    return torch.softmax(ref_logits + reward / beta, dim=-1)


def expected(policy, table):
    return (policy * table).sum(-1).mean().item()


def kl(policy, ref):
    return (policy * (policy.clamp_min(1e-12).log() - ref.clamp_min(1e-12).log())).sum(-1).mean().item()


def dpo_loss(logits, ref_logits, p, w, l, beta):
    """-log sigmoid( beta [ (log pi(w) - log ref(w)) - (log pi(l) - log ref(l)) ] ), averaged over pairs."""
    lp = torch.log_softmax(logits, -1); lr_ = torch.log_softmax(ref_logits, -1)
    margin = beta * ((lp[p, w] - lr_[p, w]) - (lp[p, l] - lr_[p, l]))
    return -F.logsigmoid(margin).mean()


def train_dpo(ref_logits, p, w, l, beta=0.1, steps=500, lr=0.05):
    logits = ref_logits.clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        loss = dpo_loss(logits, ref_logits, p, w, l, beta)
        opt.zero_grad(); loss.backward(); opt.step()
    return logits.detach()


def reinforce_to_optimum(ref_logits, reward, beta, steps=2000, lr=0.05):
    """Maximize E[reward] - beta KL by gradient ascent on the exact objective (no sampling noise), for comparison."""
    logits = ref_logits.clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        pol = torch.softmax(logits, -1)
        obj = (pol * reward).sum(-1).mean() - beta * (pol * (pol.clamp_min(1e-12).log() - torch.log_softmax(ref_logits, -1))).sum(-1).mean()
        opt.zero_grad(); (-obj).backward(); opt.step()
    return logits.detach()

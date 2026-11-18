"""Checks that the Module 09 modules behave as the learning content says.
Run from this folder:  python test_modules.py   (about a minute on a CPU)"""
import numpy as np
import torch

torch.set_num_threads(1)
from cartpole import CartPole
from mdp import (GridMDP, evaluate_policy, q_from_v, value_iteration, greedy_policy, discounted_return,
                  q_learning, exact_objective, reinforce_gradient_estimate)
from bandit import Bandit, run_bandit
from dqn import Replay, train_dqn
from pg import ppo_clip_objective, returns_to_go, train_ppo
from rlhf import (PreferenceWorld, fit_reward_model, kl_regularized_optimum, expected, kl, train_dpo,
                  reinforce_to_optimum)

# 1. CartPole: a random policy falls quickly, a simple feedback rule balances for the full 200 steps.
env, lens = CartPole(0), []
for _ in range(100):
    env.reset(); n = 0
    while True:
        _, _, te, tr = env.step(int(np.random.default_rng(n).integers(2))); n += 1
        if te or tr: break
    lens.append(n)
assert 12 < np.mean(lens) < 40
env = CartPole(1); s = env.reset(); n = 0
while True:
    s, _, te, tr = env.step(int(s[2] + 0.5 * s[3] > 0)); n += 1
    if te or tr: break
assert n == 200

# 2. Returns and the Bellman equations.
assert abs(discounted_return([1, 1, 1], 0.5) - 1.75) < 1e-12
assert np.allclose(returns_to_go([1, 1, 1], 0.5), [1.75, 1.5, 1.0])
m = GridMDP()
assert np.allclose(m.P.sum(2), 1.0)
pi = np.full((m.nS, 4), 0.25)
v = evaluate_policy(m, pi)
P_pi, r_pi = np.einsum("sa,sat->st", pi, m.P), np.einsum("sa,sa->s", pi, m.R)
assert np.abs(v - (r_pi + m.gamma * P_pi @ v)).max() < 1e-10                  # Bellman expectation equation
vs, iters = value_iteration(m)
assert np.abs(vs - q_from_v(m, vs).max(1)).max() < 1e-9                      # Bellman optimality equation
assert np.abs(evaluate_policy(m, greedy_policy(q_from_v(m, vs))) - vs).max() < 1e-8
assert (vs >= v - 1e-12).all() and vs[m.idx[(3, 0)]] > v[m.idx[(3, 0)]] + 0.3

# 3. Exploration: optimism and an explicit exploration bonus beat plain greedy on the bandit.
tot = {k: np.mean([run_bandit(Bandit(10, s), st, 500, **kw)[0].sum() for s in range(60)])
       for k, (st, kw) in {"greedy": ("greedy", {}), "optimistic": ("greedy", {"optimistic": 5.0}),
                            "ucb": ("ucb", {})}.items()}
assert tot["optimistic"] < tot["greedy"] and tot["ucb"] < tot["greedy"]

# 4. Tabular Q-learning with some exploration reaches a good policy; with none it does worse.
s0 = m.idx[(3, 0)]
def mean_value(eps): return np.mean([evaluate_policy(m, greedy_policy(q_learning(m, steps=50000, eps=eps, seed=k)[0]))[s0]
                                     for k in range(5)])
v1, v0 = mean_value(0.1), mean_value(0.0)
assert v1 > 0.38 and v1 > v0 + 0.2

# 5. The policy-gradient estimator agrees with the exact gradient, with and without a baseline.
rng = np.random.default_rng(0)
theta = rng.normal(0, 0.5, (m.nS, 4))
th = torch.tensor(theta, requires_grad=True)
exact_objective(m, th, s0).backward(); g_exact = th.grad.numpy()
var = {}
for base in (False, True):
    g = reinforce_gradient_estimate(m, theta, s0, 8000, rng, baseline=base)
    est = g.mean(0)
    assert (est * g_exact).sum() / np.linalg.norm(est) / np.linalg.norm(g_exact) > 0.99
    var[base] = g.reshape(len(g), -1).var(0).sum()
assert var[True] < var[False]

# 6. The PPO clipped objective takes the pessimistic branch.
r = torch.tensor([1.5, 1.5, 0.5, 0.5]); a = torch.tensor([1.0, -1.0, 1.0, -1.0])
assert torch.allclose(ppo_clip_objective(r, a, 0.2), torch.tensor([1.2, -1.5, 0.5, -0.8]))

# 7. The replay buffer is a ring, and DQN runs end to end.
buf = Replay(3)
for k in range(5): buf.add(np.full(4, k), k, float(k), np.full(4, k + 1), 0.0)
assert sorted(buf.a.tolist()) == [2, 3, 4] and buf.n == 3
ep = train_dqn(0, steps=1500, warmup=200)
assert len(ep) > 5 and all(1 <= e[1] <= 200 for e in ep)
log, stats = train_ppo(0, steps=2048)
assert len(log) > 10 and len(stats) == 2

# 8. Preference learning: the KL-regularized optimum is the maximizer, DPO reaches it, and a reward model learns the order.
w = PreferenceWorld(6, 8, 0)
ref = torch.zeros(6, 8); beta = 0.5
star = kl_regularized_optimum(ref, w.true_r, beta)
def objective(pol): return expected(pol, w.true_r) - beta * kl(pol, torch.softmax(ref, -1))
for _ in range(20):
    pert = torch.softmax(torch.log(star) + 0.1 * torch.randn(6, 8), -1)
    assert objective(pert) < objective(star)
assert (torch.softmax(reinforce_to_optimum(ref, w.true_r, beta), -1) - star).abs().max() < 1e-3
p, wi, lo = w.sample_pairs(5000)
rm = fit_reward_model(w, p, wi, lo, steps=300)
via_rm = kl_regularized_optimum(ref, rm, beta)
dpo = torch.softmax(train_dpo(ref, p, wi, lo, beta=beta, steps=400), -1)
assert (dpo - via_rm).abs().max() < 0.01
assert abs(expected(dpo, w.true_r) - expected(star, w.true_r)) < 0.1
print("all checks passed")

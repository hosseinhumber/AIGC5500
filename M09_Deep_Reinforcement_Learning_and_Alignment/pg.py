"""Policy-gradient agents for CartPole: REINFORCE (with or without a baseline) and PPO with the clipped objective."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cartpole import CartPole


def mlp(i, o, hidden=64):
    return nn.Sequential(nn.Linear(i, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, o))


def returns_to_go(rewards, gamma):
    g, out = 0.0, np.zeros(len(rewards), np.float32)
    for i in reversed(range(len(rewards))):
        g = rewards[i] + gamma * g
        out[i] = g
    return out


def ppo_clip_objective(ratio, adv, eps=0.2):
    """The PPO surrogate for each sample: min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A). Higher is better."""
    return torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)


def train_reinforce(seed=0, steps=30000, baseline=True, gamma=0.99, lr=3e-3, episodes_per_update=8, max_steps=200):
    """Vanilla policy gradient. Each update uses whole episodes, so its signal is the return-to-go.
    With baseline=True the return is centred by a learned state-value function."""
    torch.manual_seed(seed)
    env = CartPole(seed, max_steps)
    pi, vf = mlp(4, 2), mlp(4, 1)
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    opt_v = torch.optim.Adam(vf.parameters(), lr=1e-2)
    t, log, grad_var = 0, [], []
    while t < steps:
        S, A, G, ep_ends = [], [], [], []
        for _ in range(episodes_per_update):
            s, obs, acts, rews = env.reset(), [], [], []
            while True:
                with torch.no_grad():
                    a = int(torch.distributions.Categorical(logits=pi(torch.as_tensor(s)[None])).sample())
                s2, r, term, trunc = env.step(a)
                obs.append(s); acts.append(a); rews.append(r); s = s2; t += 1
                if term or trunc:
                    break
            S += obs; A += acts; G += list(returns_to_go(rews, gamma))
            log.append((t, float(sum(rews))))
        S, A, G = torch.as_tensor(np.array(S)), torch.as_tensor(A), torch.as_tensor(np.array(G))
        if baseline:
            v = vf(S)[:, 0]
            adv = (G - v.detach())
            loss_v = F.mse_loss(v, G)
            opt_v.zero_grad(); loss_v.backward(); opt_v.step()
        else:
            adv = G
        adv = adv / (adv.std() + 1e-8)          # same rescaling in both cases; only the baseline differs
        logp = torch.distributions.Categorical(logits=pi(S)).log_prob(A)
        loss = -(logp * adv).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return log


def train_ppo(seed=0, steps=30000, rollout=1024, epochs=6, minibatch=128, clip=0.2, gamma=0.99, lam=0.95,
              lr=3e-4, max_steps=200, ent=0.0, use_clip=True):
    torch.manual_seed(seed)
    env = CartPole(seed, max_steps)
    pi, vf = mlp(4, 2), mlp(4, 1)
    opt = torch.optim.Adam(list(pi.parameters()) + list(vf.parameters()), lr=lr)
    s, ep_ret, log, t = env.reset(), 0.0, [], 0
    stats = []
    while t < steps:
        S = np.zeros((rollout, 4), np.float32); A = np.zeros(rollout, np.int64)
        R = np.zeros(rollout, np.float32); D = np.zeros(rollout, np.float32)
        NV = np.zeros(rollout, np.float32); LP = np.zeros(rollout, np.float32); V = np.zeros(rollout, np.float32)
        cut = np.zeros(rollout, np.float32)
        for i in range(rollout):
            st = torch.as_tensor(s)[None]
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=pi(st)); a = dist.sample()
                V[i] = vf(st)[0, 0]; LP[i] = dist.log_prob(a)
            s2, r, term, trunc = env.step(int(a))
            S[i], A[i], R[i] = s, int(a), r
            D[i] = float(term); cut[i] = float(term or trunc)
            with torch.no_grad():
                NV[i] = vf(torch.as_tensor(s2)[None])[0, 0]
            ep_ret += r; t += 1
            if term or trunc:
                log.append((t, ep_ret)); ep_ret = 0.0; s = env.reset()
            else:
                s = s2
        adv = np.zeros(rollout, np.float32); last = 0.0
        for i in reversed(range(rollout)):
            delta = R[i] + gamma * (1 - D[i]) * NV[i] - V[i]
            last = delta + gamma * lam * (1 - cut[i]) * last
            adv[i] = last
        ret = adv + V
        S_, A_, LP_, ADV, RET = map(torch.as_tensor, (S, A, LP, adv, ret))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        clipfrac = []
        for _ in range(epochs):
            perm = torch.randperm(rollout)
            for k in range(0, rollout, minibatch):
                idx = perm[k:k + minibatch]
                dist = torch.distributions.Categorical(logits=pi(S_[idx]))
                ratio = torch.exp(dist.log_prob(A_[idx]) - LP_[idx])
                surr = ppo_clip_objective(ratio, ADV[idx], clip) if use_clip else ratio * ADV[idx]
                loss = -surr.mean() + 0.5 * F.mse_loss(vf(S_[idx])[:, 0], RET[idx]) - ent * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(pi.parameters()) + list(vf.parameters()), 0.5)
                opt.step()
                clipfrac.append(float(((ratio - 1).abs() > clip).float().mean()))
        stats.append((t, float(np.mean(clipfrac))))
    return log, stats

"""Deep Q-network with a replay buffer and a target network, each of which can be switched off."""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cartpole import CartPole


class QNet(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class Replay:
    """A ring buffer of transitions. With capacity equal to the batch size, a 'batch' is just the latest
    transitions in the order they happened, which is what learning without replay means."""
    def __init__(self, capacity, obs_dim=4):
        self.cap, self.n, self.i = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.d = np.zeros(capacity, np.float32)

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.n, size=batch)
        return [torch.as_tensor(x[idx]) for x in (self.s, self.a, self.r, self.s2, self.d)]


def train_dqn(seed=0, steps=20000, replay=True, target=True, gamma=0.99, lr=1e-3, batch=64,
              buffer=20000, warmup=500, target_every=250, eps_start=1.0, eps_end=0.05, eps_decay=8000,
              max_steps=200, log_every=None):
    """Returns the list of (env step at the end of the episode, episode return).
    replay=False: the buffer holds only the latest `batch` transitions, so every update uses consecutive,
        strongly correlated experience that is then thrown away.
    target=False: the bootstrap target is computed with the network that is being trained."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = CartPole(seed, max_steps)
    q = QNet()
    q_tgt = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = Replay(buffer if replay else batch)
    warm = warmup if replay else batch
    s = env.reset()
    ep_ret, episodes, q_log = 0.0, [], []
    for t in range(1, steps + 1):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * t / eps_decay)
        if rng.random() < eps:
            a = int(rng.integers(2))
        else:
            with torch.no_grad():
                a = int(q(torch.as_tensor(s)[None]).argmax())
        s2, r, term, trunc = env.step(a)
        buf.add(s, a, r, s2, float(term))      # time-limit truncation is not a true terminal state
        ep_ret += r
        s = s2
        if term or trunc:
            episodes.append((t, ep_ret)); ep_ret = 0.0; s = env.reset()
        if t >= warm:
            bs, ba, br, bs2, bd = buf.sample(batch, rng) if replay else \
                [torch.as_tensor(x[:batch]) for x in (buf.s, buf.a, buf.r, buf.s2, buf.d)]
            if not replay and batch > 1:    # order the ring buffer so the batch really is the latest transitions
                order = [(buf.i + k) % buf.cap for k in range(buf.cap)]
                bs, ba, br, bs2, bd = [torch.as_tensor(x[order]) for x in (buf.s, buf.a, buf.r, buf.s2, buf.d)]
            with torch.no_grad():
                net = q_tgt if target else q
                y = br + gamma * (1 - bd) * net(bs2).max(1).values
            pred = q(bs).gather(1, ba[:, None])[:, 0]
            loss = F.smooth_l1_loss(pred, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            if target and t % target_every == 0:
                q_tgt.load_state_dict(q.state_dict())
            if log_every and t % log_every == 0:
                with torch.no_grad():
                    q_log.append((t, float(q(bs).max(1).values.mean())))
    return (episodes, q_log) if log_every else episodes


def smooth_curve(episodes, steps, grid=500, window=2000):
    """Mean return of the episodes that ended in the last `window` steps, evaluated every `grid` steps."""
    ts = np.array([e[0] for e in episodes]); rs = np.array([e[1] for e in episodes])
    out = []
    for g in range(grid, steps + 1, grid):
        m = (ts > g - window) & (ts <= g)
        out.append(rs[m].mean() if m.any() else np.nan)
    return np.array(out)

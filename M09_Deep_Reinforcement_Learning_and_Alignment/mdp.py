"""A small tabular Markov decision process (a 4 by 4 grid with slipping) and exact solvers for it."""
import numpy as np

ACTIONS = ["up", "right", "down", "left"]
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class GridMDP:
    """Rows 0 to 3, columns 0 to 3. The goal (+1) is the top-right corner, the pit (-1) is the cell below it.
    Both are terminal. A wall cell sits at (1, 1). Every other step costs `step_cost`.
    With probability `slip` the agent moves at a right angle to the chosen direction (half each way)."""
    def __init__(self, slip=0.2, step_cost=0.0, gamma=0.9):
        self.n_rows = self.n_cols = 4
        self.goal, self.pit, self.wall = (0, 3), (1, 3), (1, 1)
        self.slip, self.step_cost, self.gamma = slip, step_cost, gamma
        self.cells = [(r, c) for r in range(4) for c in range(4) if (r, c) != self.wall]
        self.idx = {cell: i for i, cell in enumerate(self.cells)}
        self.nS, self.nA = len(self.cells), 4
        self.terminal = np.array([c in (self.goal, self.pit) for c in self.cells])
        self.P = np.zeros((self.nS, self.nA, self.nS))
        self.R = np.zeros((self.nS, self.nA))
        for cell in self.cells:
            s = self.idx[cell]
            if self.terminal[s]:
                self.P[s, :, s] = 1.0        # absorbing, no further reward
                continue
            for a in range(self.nA):
                for a2, pr in ((a, 1 - slip), ((a + 1) % 4, slip / 2), ((a - 1) % 4, slip / 2)):
                    nxt = self._move(cell, a2)
                    self.P[s, a, self.idx[nxt]] += pr
                    reward = 1.0 if nxt == self.goal else -1.0 if nxt == self.pit else step_cost
                    self.R[s, a] += pr * reward

    def _move(self, cell, a):
        r, c = cell[0] + MOVES[a][0], cell[1] + MOVES[a][1]
        if not (0 <= r < 4 and 0 <= c < 4) or (r, c) == self.wall:
            return cell
        return (r, c)

    def sample_step(self, s, a, rng):
        s2 = rng.choice(self.nS, p=self.P[s, a])
        return s2, self.R[s, a]


def evaluate_policy(mdp, pi):
    """Solve the Bellman expectation equations v = r_pi + gamma P_pi v exactly. pi has shape (nS, nA)."""
    P_pi = np.einsum("sa,sat->st", pi, mdp.P)
    r_pi = np.einsum("sa,sa->s", pi, mdp.R)
    return np.linalg.solve(np.eye(mdp.nS) - mdp.gamma * P_pi, r_pi)


def q_from_v(mdp, v):
    return mdp.R + mdp.gamma * mdp.P @ v


def value_iteration(mdp, tol=1e-10, max_iter=10000):
    v = np.zeros(mdp.nS)
    for k in range(1, max_iter + 1):
        v_new = q_from_v(mdp, v).max(1)
        done = np.abs(v_new - v).max() < tol
        v = v_new
        if done:
            break
    return v, k


def greedy_policy(q):
    pi = np.zeros_like(q)
    pi[np.arange(len(q)), q.argmax(1)] = 1.0
    return pi


def discounted_return(rewards, gamma):
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
    return g


def rollout_return(mdp, pi, s0, rng, max_steps=200):
    s, rewards = s0, []
    for _ in range(max_steps):
        if mdp.terminal[s]:
            break
        a = rng.choice(mdp.nA, p=pi[s])
        s, r = mdp.sample_step(s, a, rng)
        rewards.append(r)
    return discounted_return(rewards, mdp.gamma), len(rewards)


def q_learning(mdp, steps=20000, eps=0.1, alpha=0.1, seed=0, start=None, max_ep_len=100, optimistic=0.0):
    """Tabular Q-learning with an epsilon-greedy behaviour policy. Returns the Q table and how many
    episodes reached the goal. Every episode starts in `start` (default: the bottom-left corner)."""
    rng = np.random.default_rng(seed)
    start = mdp.idx[(3, 0)] if start is None else start
    Q = np.full((mdp.nS, mdp.nA), optimistic)
    Q[mdp.terminal] = 0.0
    s, ep_len, goals = start, 0, 0
    goal_state = mdp.idx[mdp.goal]
    for _ in range(steps):
        a = int(rng.integers(mdp.nA)) if rng.random() < eps else int(np.argmax(Q[s] + 1e-9 * rng.random(mdp.nA)))
        s2, r = mdp.sample_step(s, a, rng)
        Q[s, a] += alpha * (r + mdp.gamma * Q[s2].max() - Q[s, a])
        ep_len += 1
        if mdp.terminal[s2] or ep_len >= max_ep_len:
            goals += int(s2 == goal_state)
            s, ep_len = start, 0
        else:
            s = s2
    return Q, goals


def softmax_policy(theta):
    import torch
    return torch.softmax(theta, dim=1)


def exact_objective(mdp, theta, s0):
    """J(theta) = v_pi(s0) for a softmax policy with logits theta (nS by nA), computed exactly and differentiably."""
    import torch
    pi = torch.softmax(theta, dim=1)
    P = torch.as_tensor(mdp.P, dtype=torch.float64); R = torch.as_tensor(mdp.R, dtype=torch.float64)
    P_pi = torch.einsum("sa,sat->st", pi, P); r_pi = (pi * R).sum(1)
    v = torch.linalg.solve(torch.eye(mdp.nS, dtype=torch.float64) - mdp.gamma * P_pi, r_pi)
    return v[s0]


def reinforce_gradient_estimate(mdp, theta, s0, n_episodes, rng, baseline=False, max_steps=100):
    """Monte Carlo policy-gradient estimate: average over episodes of sum_t gamma^t grad log pi(a_t|s_t) (G_t - b).
    With baseline=True, b is the exact state value v_pi(s_t), which depends on the state and not on the action."""
    pi = np.exp(theta - theta.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True)
    v = evaluate_policy(mdp, pi) if baseline else np.zeros(mdp.nS)
    grads = np.zeros((n_episodes,) + theta.shape)
    for e in range(n_episodes):
        s, traj = s0, []
        for _ in range(max_steps):
            if mdp.terminal[s]:
                break
            a = rng.choice(mdp.nA, p=pi[s]); s2, r = mdp.sample_step(s, a, rng)
            traj.append((s, a, r)); s = s2
        G = 0.0
        for t in reversed(range(len(traj))):
            s_t, a_t, r_t = traj[t]
            G = r_t + mdp.gamma * G
            g = -pi[s_t].copy(); g[a_t] += 1.0          # gradient of log softmax with respect to the logits
            grads[e, s_t] += mdp.gamma ** t * g * (G - v[s_t])
    return grads

"""CartPole in plain numpy: the same physics as the classic control task, with no dependencies."""
import math
import numpy as np


class CartPole:
    """State: cart position, cart velocity, pole angle, pole angular velocity.
    Actions: 0 pushes the cart left, 1 pushes it right. Reward is +1 for every step the pole stays up.
    An episode ends when the pole tilts past 12 degrees, the cart leaves the track, or max_steps is reached."""
    n_actions = 2
    obs_dim = 4

    def __init__(self, seed=0, max_steps=200):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.g, self.mc, self.mp, self.l, self.force, self.tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
        self.theta_lim = 12 * 2 * math.pi / 360
        self.x_lim = 2.4

    def reset(self):
        self.s = self.rng.uniform(-0.05, 0.05, size=4)
        self.t = 0
        return self.s.astype(np.float32)

    def step(self, a):
        x, xd, th, thd = self.s
        f = self.force if a == 1 else -self.force
        ct, st = math.cos(th), math.sin(th)
        tm = self.mc + self.mp
        pml = self.mp * self.l
        tmp = (f + pml * thd ** 2 * st) / tm
        tha = (self.g * st - ct * tmp) / (self.l * (4.0 / 3.0 - self.mp * ct ** 2 / tm))
        xa = tmp - pml * tha * ct / tm
        x, xd = x + self.tau * xd, xd + self.tau * xa
        th, thd = th + self.tau * thd, thd + self.tau * tha
        self.s = np.array([x, xd, th, thd])
        self.t += 1
        terminated = bool(abs(x) > self.x_lim or abs(th) > self.theta_lim)
        truncated = self.t >= self.max_steps
        return self.s.astype(np.float32), 1.0, terminated, truncated

"""Checks that the Module 08 modules behave as the learning content says.
Run from this folder:  python test_modules.py   (about a minute on a CPU)"""
import math
import torch

from data import load_data
from toy import ring_of_gaussians, grid_of_gaussians, mode_coverage, grid_coverage
from gan import Generator, Discriminator, gradient_penalty, train_gan, sample_gan
from diffusion import Schedule, q_sample, Denoiser, train_diffusion, sample_ddpm, sample_ddim
from translate import split_halves, join_halves, train_translator, translate
import torch.nn.functional as F

torch.manual_seed(0)

# 1. Real samples from the toy datasets cover every mode and sit near them.
X8, _ = ring_of_gaussians(4000)
assert mode_coverage(X8)[0] == 8 and mode_coverage(X8)[1] > 0.99
X25, _ = grid_of_gaussians(5000)
assert grid_coverage(X25)[0] == 25 and grid_coverage(X25)[1] > 0.99

# 2. The gradient penalty is zero for a critic whose gradient norm is exactly one, and positive otherwise.
class Linear1(torch.nn.Module):
    def __init__(self, scale):
        super().__init__(); self.w = torch.tensor([0.6, 0.8]) * scale
    def forward(self, x, c=None): return x @ self.w
real, fake = torch.randn(64, 2), torch.randn(64, 2)
gp1, n1 = gradient_penalty(Linear1(1.0), real, fake)
gp3, n3 = gradient_penalty(Linear1(3.0), real, fake)
assert abs(n1 - 1.0) < 1e-5 and gp1.item() < 1e-8 and abs(n3 - 3.0) < 1e-5 and gp3.item() > 3.9

# 3. When the discriminator is sure a sample is fake, the original generator loss sends almost no gradient and the non-saturating loss sends a lot.
logit = torch.tensor([-12.0], requires_grad=True)
g_min, = torch.autograd.grad(-F.binary_cross_entropy_with_logits(logit, torch.zeros(1)), logit)
g_non, = torch.autograd.grad(F.binary_cross_entropy_with_logits(logit, torch.ones(1)), logit)
assert abs(g_min.item()) < 1e-5 and abs(g_non.item()) > 0.99

# 4. The closed-form forward process matches adding noise one step at a time.
s = Schedule(100)
x = torch.full((200000, 1), 1.5)
for t in range(60):
    x = s.alphas[t].sqrt() * x + s.betas[t].sqrt() * torch.randn_like(x)
xt, _ = q_sample(s, torch.full((200000, 1), 1.5), torch.full((200000,), 59))
assert abs(x.mean() - xt.mean()) < 0.01 and abs(x.std() - xt.std()) < 0.01
assert s.alpha_bar[-1].sqrt() < 0.1

# 5. The denoiser learns: the noise-prediction loss falls, and the samplers return the right shapes and are deterministic given a seed.
m = Denoiser(d_data=2, d_hidden=64, T=100)
losses = train_diffusion(m, s, X8, steps=600, lr=2e-3)
assert sum(losses[-50:]) < sum(losses[:50])
a = sample_ddpm(m, s, 50, 2, seed=1); b = sample_ddpm(m, s, 50, 2, seed=1)
assert a.shape == (50, 2) and torch.equal(a, b)
c = sample_ddim(m, s, 50, 2, n_steps=10, seed=1); d = sample_ddim(m, s, 50, 2, n_steps=10, seed=1)
assert c.shape == (50, 2) and torch.equal(c, d)

# 6. A short GAN run produces samples of the right shape and moves the generator towards the data.
G, D = Generator(d_noise=8, d_out=2), Discriminator()
before = sample_gan(G, 500, seed=0).norm(dim=1).mean()
train_gan(G, D, X8, objective="nonsat", steps=800, seed=0)
after = sample_gan(G, 500, seed=0).norm(dim=1).mean()
assert abs(after - 2.0) < abs(before - 2.0)

# 7. The conditional generator takes a label and a different label changes the output.
Gc = Generator(d_noise=4, d_out=64, n_classes=10, out_act="sigmoid")
z = torch.randn(3, 4)
o1 = Gc(z, F.one_hot(torch.tensor([1, 1, 1]), 10).float()); o2 = Gc(z, F.one_hot(torch.tensor([2, 2, 2]), 10).float())
assert o1.shape == (3, 64) and not torch.allclose(o1, o2)

# 8. Translation: the halves split and rejoin, and a short training run lowers the error to the truth.
Xtr, ytr, Xte, yte = load_data()
top, bottom = split_halves(Xte)
assert top.shape == (359, 32) and torch.equal(join_halves(top, bottom), Xte)
G0 = train_translator(Xtr, steps=1, seed=0)
G1 = train_translator(Xtr, steps=300, seed=0)
assert (translate(G1, top) - bottom).abs().mean() < (translate(G0, top) - bottom).abs().mean()
print("all Module 08 checks passed")

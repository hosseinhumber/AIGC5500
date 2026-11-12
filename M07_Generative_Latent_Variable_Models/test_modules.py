"""Checks that the Module 07 modules behave as the learning content says.
Run from this folder:  python test_modules.py   (about a minute on a CPU)"""
import math
import numpy as np
import torch

from data import load_data, one_hot
from autoencoder import AE, train_ae, reconstruction_mse, pca_mse
from vae import VAE, kl_to_standard_normal, elbo_terms, train_vae, iwae_bound, active_units
from latent import interpolate_latent, interpolate_pixels, sample_prior, sample_conditional
from metrics import frechet_distance, precision_recall, nearest_train_distance, train_feature_net, get_features

torch.manual_seed(0)
Xtr, ytr, Xte, yte = load_data()
assert Xtr.shape == (1438, 64) and Xte.shape == (359, 64)

# 1. Closed-form KL agrees with a Monte Carlo estimate.
g = torch.Generator().manual_seed(0)
mu, logvar = torch.randn(3, 4, generator=g), 0.5 * torch.randn(3, 4, generator=g)
closed = kl_to_standard_normal(mu, logvar)
std = torch.exp(0.5 * logvar)
z = mu + std * torch.randn(200000, 3, 4, generator=g)
log_q = (-0.5 * ((z - mu) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)
log_p = (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi)).sum(-1)
assert torch.allclose(closed, (log_q - log_p).mean(0), atol=0.05)

# 2. A linear autoencoder spans the same error as PCA at the same code size.
a = AE(d_latent=4, linear=True); train_ae(a, Xtr, epochs=400, lr=5e-3)
assert abs(reconstruction_mse(a, Xte) - pca_mse(Xtr, Xte, 4)) / pca_mse(Xtr, Xte, 4) < 0.02

# 3. The reparameterization trick lets reconstruction gradients reach the encoder.
v = VAE(d_latent=4)
rec, kl = elbo_terms(v, Xtr[:64])
v.zero_grad(); rec.mean().backward()
assert v.mu.weight.grad.abs().sum() > 0

# 4. A trained VAE beats an untrained one on the ELBO, and the importance-weighted bound is at least the ELBO.
v = VAE(d_latent=8); before = iwae_bound(v, Xte, K=1).mean().item()
train_vae(v, Xtr, epochs=60)
e1, e10 = iwae_bound(v, Xte, K=1, seed=1).mean().item(), iwae_bound(v, Xte, K=10, seed=1).mean().item()
assert e1 > before and e10 > e1 - 0.05

# 5. Very large beta collapses the posterior onto the prior.
c = VAE(d_latent=8); train_vae(c, Xtr, epochs=80, beta=4.0)
assert active_units(c, Xtr) == 0

# 6. Conditional sampling and interpolation return the right shapes.
cv = VAE(d_latent=4, n_classes=10); train_vae(cv, Xtr, y_onehot=one_hot(ytr), epochs=10)
assert sample_conditional(cv, 3, 5).shape == (5, 64) and sample_prior(v, 7).shape == (7, 64)
assert interpolate_latent(v, Xte[0], Xte[1], steps=8).shape == (8, 64) and interpolate_pixels(Xte[0], Xte[1], steps=8).shape == (8, 64)

# 7. Metrics: identical sets are close, a constant image has no recall, a copy is at distance zero from its source.
net = train_feature_net(Xtr, ytr, epochs=40)
ftr, fte = get_features(net, Xtr), get_features(net, Xte)
assert frechet_distance(ftr, fte) < 10
const = get_features(net, Xtr.mean(0, keepdim=True).repeat(359, 1))
assert precision_recall(fte, const)[1] < 0.1
assert frechet_distance(fte, const) > 5 * frechet_distance(ftr, fte)
assert nearest_train_distance(Xtr[:5], Xtr).max().item() < 1e-6
print("all Module 07 checks passed")

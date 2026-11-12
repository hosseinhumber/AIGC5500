# Module 07: Generative Models I, Latent Variable Models

Five notebooks (run in order) and the Python modules they were built from. CPU only, no downloads.

| File | Purpose |
|---|---|
| `data.py` | The 8 by 8 digits dataset: split, scaling to [0, 1], one-hot labels |
| `autoencoder.py` | `AE` (linear or with a hidden layer), training with optional input noise, reconstruction error, PCA baseline |
| `vae.py` | `VAE` (optionally conditional), closed-form KL, ELBO terms, training with a KL weight and annealing, importance-weighted bound, active units |
| `latent.py` | Interpolation in latent and pixel space, decoding a grid, sampling from the prior and per class |
| `metrics.py` | Small feature network, Fréchet distance, precision and recall in feature space, nearest-training-image distance |
| `test_modules.py` | Checks the modules against the claims in the learning content |

| Notebook | Covers |
|---|---|
| Part 1, Autoencoders and the bottleneck | Code size against error, a linear autoencoder against PCA, the nonlinear gain, denoising |
| Part 2, From autoencoder to VAE | Prior samples from an autoencoder and a VAE, closed-form KL, the ELBO identity on an exactly solvable model, reparameterization against the score-function estimator |
| Part 3, Training a VAE | The KL weight, posterior collapse, active units, whether annealing helps |
| Part 4, The latent space and conditioning | A 2-D code, interpolation measured against a pixel cross-fade, one-dimension sweeps, a conditional VAE |
| Part 5, Evaluating generative models | The importance-weighted bound, Fréchet distance and its dependence on sample size, precision and recall, a model that copies, reconstruction against generation |

The five notebooks are self-contained: the code they run is already inside them, and they were executed so the outputs you see are real. The `.py` files are the source the notebooks were built from, plus a test suite. You can ignore them unless you want to import them into your own work.

The experiments use small networks on 8 by 8 digit images, so every number can be reproduced. They show mechanics, not how large image models behave. Two results are reported as they came out and may differ from what you expect: latent interpolation shows no measurable advantage over a pixel cross-fade at this resolution (Part 4), and KL annealing does not rescue a model whose weight on the KL term is too high (Part 3).

Setup: `pip install torch numpy matplotlib scikit-learn scipy jupyter`, then `python test_modules.py`. The test suite takes about a minute. Executing all five notebooks takes roughly 15 minutes on two CPU cores, with Part 3 the longest.

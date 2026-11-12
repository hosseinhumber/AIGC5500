# Module 08: Generative Models II, Adversarial and Diffusion Models

Five notebooks (run in order) and the Python modules they were built from. CPU only, no downloads.

| File | Purpose |
|---|---|
| `data.py` | The 8 by 8 digits dataset: split, scaling to [0, 1], one-hot labels |
| `toy.py` | Two-dimensional toy data (a ring of 8 Gaussians, a 5 by 5 grid of 25) and mode-coverage measures |
| `gan.py` | Generator and discriminator (optionally conditional), three training objectives (original, non-saturating, WGAN-GP), gradient penalty, sampling |
| `diffusion.py` | Noise schedule, closed-form forward process, noise-prediction network, ancestral (DDPM) and deterministic (DDIM) samplers, label conditioning |
| `translate.py` | Paired image-to-image translation on digits (top half to bottom half): regression with L1 or squared error, with or without an adversarial term |
| `vae.py` | The Module 07 VAE, used as the comparison baseline in Part 5 |
| `metrics.py` | Small feature network, Fréchet distance, precision and recall in feature space, nearest-training-image distance |
| `test_modules.py` | Checks the modules against the claims in the learning content |

| Notebook | Covers |
|---|---|
| Part 1, The adversarial game | The best discriminator against its formula, the value of the game as a Jensen-Shannon distance, saturation of the original generator loss, a first GAN on eight Gaussians |
| Part 2, Instability and mode collapse | Original against non-saturating loss on 25 modes, mode collapse on digits, WGAN-GP and a check of its gradient penalty |
| Part 3, Conditional GANs and translation | A label-conditioned WGAN-GP checked with a classifier, fixed noise with changing label, paired translation with squared error, L1 and an adversarial term |
| Part 4, Denoising diffusion | The forward process against its closed form, training and sampling on eight Gaussians and on digits, sampling with fewer network evaluations, label conditioning |
| Part 5, Choosing between generative models | VAE, GAN and diffusion on the same data with the same scoring: Fréchet distance, precision, recall, copy check, class mix, training and sampling cost |

The five notebooks are self-contained: the code they run is already inside them, and they were executed so the outputs you see are real. The `.py` files are the source the notebooks were built from, plus a test suite. You can ignore them unless you want to import them into your own work.

The experiments use small networks on 8 by 8 digit images and on two-dimensional toy data, so every number can be reproduced. They show mechanics, not how large image models behave. Three results are reported as they came out and may differ from what you expect: the original and non-saturating generator losses cover the same number of modes on the 25-mode grid (Part 2), the WGAN-GP that prevents collapse on digits covers fewer modes than either on that grid at the same training budget (Part 2), and deterministic samplers with 5 to 50 evaluations score better than the 100-step ancestral sampler on the same network, where the two differ in step count and in sampler at once (Part 4).

Setup: `pip install torch numpy matplotlib scikit-learn scipy jupyter`, then `python test_modules.py`. The test suite takes about a minute. Executing all five notebooks takes roughly 10 minutes on two CPU cores.

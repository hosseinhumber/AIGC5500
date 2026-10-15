# Module 05: Attention and the Transformer

Four notebooks (run in order) and the Python modules they were built from. CPU only, no downloads.

| File | Purpose |
|---|---|
| `attention.py` | Scaled dot-product attention and multi-head attention from scratch |
| `positional.py` | Sinusoidal and learned positional encodings |
| `transformer.py` | Encoder block, decoder block, small sequence classifier |
| `vit.py` | Patch embedding and a tiny Vision Transformer |
| `tasks.py` | Synthetic tasks with known answers |
| `test_modules.py` | Checks the modules against PyTorch |
| `lab05_starter.py` | Starter for Lab 05 |

The four notebooks are self-contained: the code they run is already inside them. The `.py` files are the source the notebooks were built from, plus a test suite. You can ignore them unless you want to import them into your own work.

Setup: `pip install torch numpy matplotlib scikit-learn jupyter`, then `python test_modules.py`.

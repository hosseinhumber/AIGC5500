# Module 06: Pretraining, Transfer and Parameter-Efficient Fine-Tuning

Five notebooks (run in order) and the Python modules they were built from. CPU only, no downloads.

| File | Purpose |
|---|---|
| `tokenizer.py` | Byte-pair encoding from scratch |
| `masks.py` | Full, causal and cross attention masks, and a small self-attention layer |
| `budget.py` | Compute (6ND) and training-memory arithmetic |
| `lora.py` | `LoRALinear`, merging, and a helper that adds LoRA to an MLP |
| `quant.py` | Absmax quantization (whole tensor or blocks) and a simplified normal-float code |
| `tasks.py` | The digits adaptation task: pretraining, three shifts, adaptation loop |
| `test_modules.py` | Checks the modules against the numbers in the learning content |

| Notebook | Covers |
|---|---|
| Part 1, Tokenization and self-supervised pretraining | BPE, token counts, next-token and contrastive losses |
| Part 2, Three families of pretrained Transformers | Masks, why the future must be hidden, generation, scoring answer tokens only |
| Part 3, Scale and the memory bill | 6ND, 16 bytes per parameter measured, a 7 billion parameter model under three methods |
| Part 4, LoRA | Low-rank update, merging, adaptation experiment, singular values of the update |
| Part 5, Quantization and QLoRA | Absmax, outliers and blocks, normal-float, QLoRA in miniature |

The five notebooks are self-contained: the code they run is already inside them, and they were executed so the outputs you see are real. The `.py` files are the source the notebooks were built from, plus a test suite. You can ignore them unless you want to import them into your own work.

The experiments use a small network on 8 by 8 digit images, so they run in seconds and every number can be reproduced. They show mechanics, not how a model with billions of parameters behaves. Parts 4 and 5 take about two minutes in total on a laptop CPU.

Setup: `pip install torch numpy matplotlib scikit-learn jupyter`, then `python test_modules.py`.

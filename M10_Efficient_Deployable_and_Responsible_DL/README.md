# Module 10: Efficient, Deployable and Responsible Deep Learning

Five notebooks (run in order) and the Python modules they were built from. CPU only, no downloads: the data are the scikit-learn 8 by 8 digits and a synthetic lending data set.

| File | Purpose |
|---|---|
| `data.py` | The digits split and the synthetic lending data (known truth, biased history, a proxy feature) |
| `compress.py` | Distillation, global magnitude pruning, structured pruning, sparse CSR inference, timing |
| `quant.py` | Uniform quantization from scratch: per-tensor and per-channel, real int8 inference, quantization-aware training, outlier channel |
| `infer.py` | Tiny decoder with a key-value cache, batching service times, a dynamic-batching server simulation, ONNX export |
| `xai.py` | Small CNN, saliency, Grad-CAM, occlusion, randomization and deletion tests, exact Shapley values, calibration, corruptions, max-softmax AUROC |
| `audit.py` | Group rates, fairness gaps, proxy leakage, group thresholds, model card generator |
| `test_modules.py` | Checks the modules against claims in the learning content (about 40 seconds) |

| Notebook | Covers |
|---|---|
| Part 1, Distillation and pruning | A 32-unit student against a 301,066-parameter teacher with 5% of the labels, the temperature, unstructured and structured pruning, and whether sparsity is faster |
| Part 2, Quantization | Rounding error by bit width, real int8 inference, post-training against quantization-aware training, an outlier channel, model size |
| Part 3, Inference optimization | KV cache against recomputation, batching, a latency-budget simulation, ONNX export and ONNX Runtime |
| Part 4, Interpretability, calibration and robustness | Saliency, Grad-CAM and occlusion with a randomization and a deletion test, Shapley values, ECE and temperature scaling, corruptions, shift, max-softmax |
| Part 5, Bias audit and model cards | A lending audit by group, the proxy, three fixes, a generated model card |

The notebooks were executed, so the outputs are real. Accuracy and sizes reproduce exactly. **Timings and memory depend on your machine** (they were measured on a two-core cloud machine, one thread), so expect the pattern and not the decimals. Several results are reported as they came out: temperature made no difference to distillation, distilling on the 71 labelled images was worse than labels alone, a sparse matrix product helped but a dense product on a pruned network did not, and the learned networks are much larger than the digits need, so every cliff is later than it would be on a tight network.

Setup: `pip install torch numpy scikit-learn matplotlib jupyter onnx onnxruntime`, then `python test_modules.py`. Executing all five notebooks takes about 7 minutes on two CPU cores.

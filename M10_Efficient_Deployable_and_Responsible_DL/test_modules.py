"""Checks the Module 10 modules against claims in the learning content. Runs in about 40 seconds."""
import copy, numpy as np, torch
from data import *; from compress import *; from quant import *; from infer import *; from xai import *; from audit import *
torch.set_num_threads(1)
Xtr, ytr, Xte, yte = load_digits_split(0)
assert Xtr.shape == (1438, 64) and Xte.shape == (359, 64)
base = train(mlp([64, 512, 512, 10]), Xtr, ytr, epochs=30, seed=0)
assert count_params(base) == 301066 and accuracy(base, Xte, yte) > 0.97
# pruning keeps the masked weights at zero and structured pruning builds a smaller dense network
m = copy.deepcopy(base); mk = magnitude_masks(m, 0.9); apply_masks(m, mk); assert abs(1 - count_nonzero(m) / count_nonzero(base) - 0.9) < 0.01
assert count_params(structured_prune(base, 0.25)) < 30000
# quantization: error doubles per bit, int8 integer inference matches float, STE trains
w = base[0].weight.detach()
assert weight_error(w, 6, False) > 2 * weight_error(w, 8, False) and weight_error(w, 8, True) < weight_error(w, 8, False)
q = int8_inference(base, Xte, Xtr[:256]); assert (q.argmax(1) == base(Xte).argmax(1)).float().mean() > 0.99
assert model_bytes(base, 8, True) < model_bytes(base, 32) / 3.5
o = add_outlier_channel(base); assert torch.allclose(o(Xte), base(Xte), atol=1e-3)
# kv cache gives the same tokens; the simulator is arrival limited when batching helps
d = TinyDecoder().eval(); pr = torch.arange(8)[None] % 100
assert bool((generate(d, pr, 24, True) == generate(d, pr, 24, False)).all())
# explanations and calibration
assert abs(exact_shapley(lambda z: z @ np.array([1., 2, 3, 0, 0]), np.ones(5), np.zeros(5))[0].sum() - 6) < 1e-9
over = overfit_mlp(Xtr[:200], ytr[:200]); lg = over(Xte).detach(); T = fit_temperature(over(Xtr[200:600]).detach(), ytr[200:600])
p = torch.softmax(lg, 1).numpy(); p2 = torch.softmax(lg / T, 1).numpy()
assert ece(p2, yte.numpy()) < ece(p, yte.numpy()) and (lg.argmax(1) == (lg / T).argmax(1)).all()
# audit: the proxy carries the group and removing it narrows the gap
D = lending_data(8000, 0); X, g, yh, yt = D["X"], D["group"], D["y_hist"], D["y_true"]
f = fit_logistic(X[:5000], yh[:5000]); r = group_rates((f(X[5000:]) > .5).astype(int), yt[5000:], g[5000:])
assert gaps(r)["disparate_impact_ratio"] < 0.6 and proxy_leakage(X, g, 3)[0] > 0.85
f2 = fit_logistic(X[:5000][:, [0, 1, 2, 4]], yh[:5000]); r2 = group_rates((f2(X[5000:][:, [0, 1, 2, 4]]) > .5).astype(int), yt[5000:], g[5000:])
assert gaps(r2)["demographic_parity_gap"] < 0.05
print("all checks passed")

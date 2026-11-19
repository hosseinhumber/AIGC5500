"""Inference optimization: batching, a key-value cache for a tiny decoder, a serving simulation, and ONNX export."""
import math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from compress import timeit


# ---------------------------------------------------------------- tiny decoder-only transformer with an optional KV cache
class TinyDecoder(nn.Module):
    def __init__(self, vocab=100, d=64, heads=4, layers=2, max_len=512, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.d, self.h, self.L = d, heads, layers
        self.emb = nn.Embedding(vocab, d); self.pos = nn.Embedding(max_len, d)
        self.qkv = nn.ModuleList([nn.Linear(d, 3 * d) for _ in range(layers)])
        self.out = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.ff1 = nn.ModuleList([nn.Linear(d, 4 * d) for _ in range(layers)])
        self.ff2 = nn.ModuleList([nn.Linear(4 * d, d) for _ in range(layers)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, tokens, cache=None):
        """tokens: (B, T_new). cache: list of (K, V) per layer or None. Returns logits for the new positions and the cache."""
        B, T = tokens.shape
        past = 0 if cache is None else cache[0][0].shape[2]
        x = self.emb(tokens) + self.pos(torch.arange(past, past + T))
        new_cache = []
        for i in range(self.L):
            h = self.ln1[i](x)
            q, k, v = self.qkv[i](h).split(self.d, dim=-1)
            sp = lambda t: t.view(B, T, self.h, self.d // self.h).transpose(1, 2)
            q, k, v = sp(q), sp(k), sp(v)
            if cache is not None:
                k = torch.cat([cache[i][0], k], 2); v = torch.cat([cache[i][1], v], 2)
            new_cache.append((k, v))
            att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d // self.h)
            S = k.shape[2]
            mask = torch.arange(S)[None, :] > (past + torch.arange(T))[:, None]
            att = att.masked_fill(mask, float("-inf")).softmax(-1)
            x = x + self.out[i]((att @ v).transpose(1, 2).reshape(B, T, self.d))
            x = x + self.ff2[i](F.gelu(self.ff1[i](self.ln2[i](x))))
        return self.head(x), new_cache


@torch.no_grad()
def generate(model, prompt, n_new, use_cache=True):
    """Greedy decoding. Without the cache the whole sequence is run again at every step."""
    tokens = prompt.clone(); cache = None
    for _ in range(n_new):
        if use_cache:
            logits, cache = model(tokens[:, -1:] if cache is not None else tokens, cache)
        else:
            logits, _ = model(tokens)
        tokens = torch.cat([tokens, logits[:, -1].argmax(-1, keepdim=True)], 1)
    return tokens


def cache_bytes(model, seq_len, batch=1):
    return 2 * model.L * batch * seq_len * model.d * 4


def exp_kv_cache(lengths=(32, 64, 128, 256), prompt_len=8, repeats=3):
    m = TinyDecoder().eval()
    prompt = torch.arange(prompt_len)[None] % 100
    rows = []
    for n in lengths:
        a = generate(m, prompt, n, True); b = generate(m, prompt, n, False)
        same = bool((a == b).all())
        t_c = timeit(lambda: generate(m, prompt, n, True), repeats=repeats, warmup=1)
        t_n = timeit(lambda: generate(m, prompt, n, False), repeats=repeats, warmup=1)
        rows.append(dict(new_tokens=n, with_cache_s=t_c, without_cache_s=t_n, same_output=same,
                         cache_kb=cache_bytes(m, prompt_len + n) / 1024))
    return rows


# ---------------------------------------------------------------- batching and a latency budget
def service_time_table(model, X, batches=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
    """Measured seconds to run one batch of each size."""
    out = {}
    with torch.no_grad():
        for b in batches:
            xb = X[:b]
            out[b] = timeit(lambda: model(xb), repeats=100, warmup=10)
    return out


def simulate_server(service, rate, max_batch, max_wait, n=4000, seed=0):
    """Dynamic batching. Requests arrive as a Poisson process at `rate` per second. The server takes whatever is waiting,
    up to `max_batch`, once the oldest request has waited `max_wait` seconds (or the batch is full). `service` maps batch size
    to measured seconds. Returns (p50, p95, throughput)."""
    rng = np.random.default_rng(seed)
    arr = np.cumsum(rng.exponential(1.0 / rate, n))
    sizes = sorted(service)
    def st(b):
        s = min(x for x in sizes if x >= b) if b <= sizes[-1] else sizes[-1]
        return service[s]
    lat = np.zeros(n); t = 0.0; i = 0
    while i < n:
        t = max(t, arr[i])                                          # the server is free and a request is waiting
        fill = arr[i + max_batch - 1] if i + max_batch - 1 < n else np.inf
        start = max(t, min(arr[i] + max_wait, fill))               # leave at the deadline, or when the batch is full
        j = min(i + max_batch, int(np.searchsorted(arr, start, side="right")))
        j = max(j, i + 1)
        finish = start + st(j - i)
        lat[i:j] = finish - arr[i:j]
        t = finish; i = j
    return float(np.percentile(lat, 50)), float(np.percentile(lat, 95)), n / (t - arr[0])


# ---------------------------------------------------------------- ONNX
def export_onnx(model, path, in_dim=64):
    model.eval()
    torch.onnx.export(model, torch.zeros(1, in_dim), path, input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    return os.path.getsize(path)


def ort_session(path):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

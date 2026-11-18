"""Regenerates the curves quoted in the learning content (same seeds as the notebooks) and saves them to figdata.json."""
import json, time, numpy as np, torch
from multiprocessing import Pool
torch.set_num_threads(1)
from bandit import Bandit, run_bandit
from dqn import train_dqn, smooth_curve
from pg import train_reinforce, train_ppo
from rlhf import *
STEPS = 30000
CFG = {"full": dict(), "window of 64": dict(replay=False), "online (batch 1)": dict(replay=False, batch=1),
       "replay, batch 1": dict(batch=1), "no target": dict(target=False)}
def dqn_job(a):
    torch.set_num_threads(1); n, s = a
    ep, ql = train_dqn(s, steps=STEPS, log_every=500, **CFG[n])
    return n, s, smooth_curve(ep, STEPS).tolist(), ql
def pg_job(a):
    torch.set_num_threads(1); n, s = a
    if n == "REINFORCE, no baseline": c = smooth_curve(train_reinforce(s, STEPS, baseline=False), STEPS)
    elif n == "REINFORCE, baseline": c = smooth_curve(train_reinforce(s, STEPS, baseline=True), STEPS)
    elif n == "PPO": c = smooth_curve(train_ppo(s, STEPS)[0], STEPS)
    elif n == "PPO aggressive, clipped": c = smooth_curve(train_ppo(s, STEPS, lr=3e-3, epochs=20)[0], STEPS)
    elif n == "PPO aggressive, no clip": c = smooth_curve(train_ppo(s, STEPS, lr=3e-3, epochs=20, use_clip=False)[0], STEPS)
    return n, s, c.tolist()
if __name__ == "__main__":
    out = {}
    t0 = time.time()
    strat = {"greedy (start at 0)": ("greedy", {}), "greedy, optimistic start 5": ("greedy", {"optimistic": 5.0}),
             "epsilon-greedy, eps 0.01": ("epsilon", {"eps": 0.01}), "epsilon-greedy, eps 0.1": ("epsilon", {"eps": 0.1}),
             "epsilon-greedy, eps 0.3": ("epsilon", {"eps": 0.3}), "UCB, c = 2": ("ucb", {"c": 2.0}), "softmax, tau = 0.2": ("softmax", {"tau": 0.2})}
    out["bandit"] = {}
    for name, (st, kw) in strat.items():
        reg = np.array([run_bandit(Bandit(10, s), st, 1000, **kw)[0] for s in range(200)])
        out["bandit"][name] = {"curve": reg.cumsum(1).mean(0)[9::10].tolist(), "total": float(reg.sum(1).mean())}
    print("bandit", time.time() - t0, flush=True)
    # preference learning
    def wp(seed, n):
        w = PreferenceWorld(6, 8, seed); p, wi, lo = w.sample_pairs(n); return w, p, wi, lo
    out["rlhf"] = {}
    for n in (50, 200, 5000):
        rows = []
        for b in (10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01):
            tr, px, ks = [], [], []
            for seed in range(20):
                w, p, wi, lo = wp(seed, n); rm = fit_reward_model(w, p, wi, lo, steps=300); ref = torch.zeros(6, 8)
                pol = kl_regularized_optimum(ref, rm, b)
                tr.append(expected(pol, w.true_r)); px.append(expected(pol, rm)); ks.append(kl(pol, torch.softmax(ref, -1)))
            rows.append([b, float(np.mean(ks)), float(np.mean(px)), float(np.mean(tr))])
        out["rlhf"][str(n)] = rows
    out["exploit"] = {}
    for bonus in (0.0, 1.0):
        rows = []
        for b in (10.0, 3.0, 1.0, 0.5, 0.3, 0.1, 0.03):
            tr, px, ks = [], [], []
            for seed in range(20):
                w, p, wi, lo = wp(seed, 5000); rm = fit_reward_model(w, p, wi, lo, steps=300)
                rm2 = rm.clone(); rm2[torch.arange(6), w.true_r.argmin(1)] = rm.max(1).values + bonus
                ref = torch.zeros(6, 8); pol = kl_regularized_optimum(ref, rm2, b)
                tr.append(expected(pol, w.true_r)); px.append(expected(pol, rm2)); ks.append(kl(pol, torch.softmax(ref, -1)))
            rows.append([b, float(np.mean(ks)), float(np.mean(px)), float(np.mean(tr))])
        out["exploit"][str(bonus)] = rows
    print("rlhf", time.time() - t0, flush=True)
    jobs = [(n, s) for n in CFG for s in range(3)]
    with Pool(2) as p: res = p.map(dqn_job, jobs)
    out["dqn"] = {n: {"curves": [r[2] for r in res if r[0] == n], "q": [[x[1] for x in r[3]] for r in res if r[0] == n]} for n in CFG}
    print("dqn", time.time() - t0, flush=True)
    names = ["REINFORCE, no baseline", "REINFORCE, baseline", "PPO", "PPO aggressive, clipped", "PPO aggressive, no clip"]
    with Pool(2) as p: res = p.map(pg_job, [(n, s) for n in names for s in range(3)])
    out["pg"] = {n: [r[2] for r in res if r[0] == n] for n in names}
    json.dump(out, open("figdata.json", "w"))
    print("done", time.time() - t0, flush=True)

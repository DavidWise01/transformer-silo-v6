#!/usr/bin/env python3
"""Honesty anchor for v6. Gradient-checks the hand-written expert backprop
(analytic vs numerical) so the training is real, and checks the cascade + the
calibration diagnostics behave, INCLUDING the instrument's core claim: the
calibrated (telegraphed) task really has higher confidence-vs-correctness AUC
than the miscalibrated (entangled) one. Run: python selftest.py"""
from __future__ import annotations
import numpy as np
from model import (init_expert, loss_and_grad, predict, encode,
                   expected_calibration_error, confidence_auc, cascade)
from tasks import make_world, silo_input, plain_input, D, G, N, K

fails = 0
def check(name, cond, extra=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))
    if not cond: fails += 1


# ---- 1) gradient check the expert backprop (the anchor) ----
print("== gradient check: expert backprop ==")
world = make_world(seed=1)
P = init_expert(D, 8, G, seed=3)
X, w = silo_input(world, np.array([0, 1, 5, 5, 9, 9, 13, 2]))
y = 2
_, g = loss_and_grad(P, X, y, w)
worst, wk = 0.0, None
for k in P:
    ng = np.zeros_like(P[k]); flat = P[k].reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + 1e-6; lp = loss_and_grad(P, X, y, w)[0]
        flat[i] = old - 1e-6; lm = loss_and_grad(P, X, y, w)[0]
        flat[i] = old
        ng.reshape(-1)[i] = (lp - lm) / 2e-6
    rel = np.abs(g[k] - ng).max() / (np.abs(ng).max() + 1e-12)
    if rel > worst: worst, wk = rel, k
check("analytic == numerical (< 1e-5)", worst < 1e-5, f"max rel err {worst:.2e} at {wk}")

# ---- 2) predict + calibration diagnostics ----
print("\n== calibration diagnostics ==")
pred, conf = predict(P, X, w)
check("confidence in [1/G, 1] and pred valid", 1.0/G - 1e-9 <= conf <= 1.0 and 0 <= pred < G, f"pred={pred} conf={conf:.3f}")
# ECE: perfectly calibrated (conf==accuracy per bin) -> ~0; confidently wrong -> high
rng = np.random.default_rng(0)
c_cal = rng.uniform(0, 1, 4000); ok_cal = (rng.uniform(0, 1, 4000) < c_cal).astype(int)  # P(correct)=conf
c_bad = np.full(2000, 0.95); ok_bad = np.zeros(2000, int)                                 # 95% sure, always wrong
check("ECE ~ 0 for a calibrated set", expected_calibration_error(c_cal, ok_cal) < 0.05,
      f"ECE={expected_calibration_error(c_cal, ok_cal):.3f}")
check("ECE high for a confidently-wrong set", expected_calibration_error(c_bad, ok_bad) > 0.9,
      f"ECE={expected_calibration_error(c_bad, ok_bad):.3f}")
# AUC: confidence that perfectly ranks correct-above-wrong -> 1.0; no signal -> ~0.5
check("conf AUC = 1.0 when confidence ranks right>wrong",
      abs(confidence_auc([.9, .8, .7, .6], [1, 1, 0, 0]) - 1.0) < 1e-9)
check("conf AUC ~ 0.5 when confidence is uninformative",
      abs(confidence_auc([.7, .7, .7, .7], [1, 0, 1, 0]) - 0.5) < 1e-9)

# ---- 3) cascade cost invariants ----
print("\n== cascade invariants ==")
toks = [np.array([0, 4, 8, 12, 1, 5, 9, 13]), np.array([2, 2, 6, 6, 10, 10, 14, 14])]
si = [silo_input(world, t) for t in toks]; pi = [plain_input(world, t) for t in toks]
lo = cascade(P, si, P, pi, tau=0.0, silo_pairs=K*K, plain_pairs=N*N)   # never escalate
hi = cascade(P, si, P, pi, tau=1.01, silo_pairs=K*K, plain_pairs=N*N)  # always escalate
check("tau=0 -> all silo, cost K^2 each", all(o[2] == "silo" and o[3] == K*K for o in lo))
check("tau>1 -> all escalate, cost K^2+N^2 each", all(o[2] == "plain" and o[3] == K*K + N*N for o in hi))
escs = [sum(o[2] == "plain" for o in cascade(P, si, P, pi, tau=t, silo_pairs=K*K, plain_pairs=N*N))
        for t in (0.0, 0.3, 0.6, 1.01)]
check("escalation is monotonic non-decreasing in tau", all(escs[i] <= escs[i+1] for i in range(len(escs)-1)), f"{escs}")

# ---- 4) the instrument's CORE claim: calibration differs by task ----
print("\n== core claim: calibrated task has higher conf-AUC (quick train) ==")
from train import run_task
w0 = make_world(seed=0)
ent = run_task(w0, "entangled", 0)
tel = run_task(w0, "telegraphed", 0)
check("telegraphed silo is better-calibrated than entangled (higher AUC, lower ECE)",
      tel["silo_conf_auc"] > ent["silo_conf_auc"] and tel["silo_ece"] < ent["silo_ece"],
      f"AUC tel={tel['silo_conf_auc']:.2f} > ent={ent['silo_conf_auc']:.2f}; ECE tel={tel['silo_ece']:.2f} < ent={ent['silo_ece']:.2f}")
check("telegraphed has real headroom, entangled has ~none",
      tel["gap"] > 0.15 and ent["gap"] < 0.05, f"gap tel={tel['gap']:.2f} ent={ent['gap']:.2f}")
check("cue-free cascade closes most of the telegraphed gap under plain's cost",
      tel["recovery_frac"] is not None and tel["recovery_frac"] > 0.8
      and tel["operating_point"]["pairs"] < tel["plain_only"]["pairs"],
      f"recovery={tel['recovery_frac']}, pairs={tel['operating_point']['pairs']} < {tel['plain_only']['pairs']}")

print("\n" + ("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED"))
raise SystemExit(1 if fails else 0)

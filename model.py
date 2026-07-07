#!/usr/bin/env python3
"""v6 model: a CONFIDENCE-GATED CASCADE over two independent experts.

No learned router this time (that was v5). Instead: run the cheap SILO expert
first; if its top-class confidence clears a threshold tau, keep its answer and
pay only K^2 attention; otherwise ESCALATE to the PLAIN expert and pay the full
(K^2 + N^2). The routing signal is the silo's OWN confidence -- no regime cue.

The whole point of v6 is that this only works when the silo is CALIBRATED: when
it is uncertain on exactly the examples it gets wrong. When the silo is instead
CONFIDENTLY WRONG, its confidence carries no signal and the cascade is blind.
We measure calibration directly (ECE) and show the cascade succeed and fail.

Both experts are ordinary single-head classifiers over a shared-shape encoder;
all backprop is hand-written and gradient-checked in selftest.py.
"""
from __future__ import annotations
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def init_expert(d, h, C, seed=0):
    rng = np.random.default_rng(seed)
    s = 0.2
    def w(*shape): return rng.standard_normal(shape) * s
    return {
        "Wq": w(d, d), "Wk": w(d, d), "Wv": w(d, d), "Wo": w(d, d),
        "W1": w(d, h), "b1": np.zeros(h), "W2": w(h, d), "b2": np.zeros(d),
        "Wc": w(d, C), "bc": np.zeros(C),
    }


# ---------- encoder: (X, w) -> pooled vector (identical to v3-v5, gradient-checked) ----------
def encode(P, X, w=None):
    n, d = X.shape
    if w is None:
        w = np.ones(n)
    wn = w / w.sum()
    Q, K, V = X @ P["Wq"], X @ P["Wk"], X @ P["Wv"]
    S = (Q @ K.T) / np.sqrt(d)
    A = softmax(S, axis=1)
    Ctx = A @ V
    Attn = Ctx @ P["Wo"]
    Z1 = X + Attn
    Hpre = Z1 @ P["W1"] + P["b1"]
    H = np.maximum(0.0, Hpre)
    M = H @ P["W2"] + P["b2"]
    Z2 = Z1 + M
    p = (Z2 * wn[:, None]).sum(axis=0)
    return p, dict(X=X, K=K, V=V, Q=Q, A=A, Ctx=Ctx, Z1=Z1, Hpre=Hpre, H=H, wn=wn, d=d)


def encode_backward(P, cache, dp):
    g = {}
    dZ2 = np.outer(cache["wn"], dp)
    dZ1 = dZ2.copy()
    dM = dZ2
    g["W2"] = cache["H"].T @ dM
    g["b2"] = dM.sum(axis=0)
    dH = dM @ P["W2"].T
    dHpre = dH * (cache["Hpre"] > 0)
    g["W1"] = cache["Z1"].T @ dHpre
    g["b1"] = dHpre.sum(axis=0)
    dZ1 += dHpre @ P["W1"].T
    dAttn = dZ1
    g["Wo"] = cache["Ctx"].T @ dAttn
    dCtx = dAttn @ P["Wo"].T
    dA = dCtx @ cache["V"].T
    dV = cache["A"].T @ dCtx
    dS = cache["A"] * (dA - (dA * cache["A"]).sum(axis=1, keepdims=True))
    dS /= np.sqrt(cache["d"])
    dQ = dS @ cache["K"]
    dK = dS.T @ cache["Q"]
    g["Wq"] = cache["X"].T @ dQ
    g["Wk"] = cache["X"].T @ dK
    g["Wv"] = cache["X"].T @ dV
    return g


# ---------- one expert: forward, loss+grad, predict-with-confidence ----------
def expert_logits(P, X, w=None):
    p, c = encode(P, X, w)
    return p @ P["Wc"] + P["bc"], p, c


def loss_and_grad(P, X, y, w=None):
    logits, p, c = expert_logits(P, X, w)
    probs = softmax(logits)
    loss = -np.log(probs[y] + 1e-12)
    dl = probs.copy(); dl[y] -= 1.0
    g = {k: np.zeros_like(v) for k, v in P.items()}
    g["Wc"] = np.outer(p, dl); g["bc"] = dl
    for k, v in encode_backward(P, c, P["Wc"] @ dl).items():
        g[k] += v
    return loss, g


def predict(P, X, w=None):
    """Return (pred_class, confidence) where confidence = top softmax prob."""
    logits, _, _ = expert_logits(P, X, w)
    probs = softmax(logits)
    return int(np.argmax(probs)), float(probs.max())


# ---------- the cascade + honest diagnostics ----------
def cascade(Psilo, silo_in, Pplain, plain_in, tau, silo_pairs, plain_pairs):
    """Silo first; escalate to plain if silo confidence < tau. Returns a per-example
    list of (pred, confidence, which, pairs). Escalation pays BOTH experts' attention
    (you already ran the silo), so an escalated example costs silo_pairs+plain_pairs."""
    out = []
    for sX, sw, pX, pw in _pairs(silo_in, plain_in):
        sp, sc = predict(Psilo, sX, sw)
        if sc >= tau:
            out.append((sp, sc, "silo", silo_pairs))
        else:
            pp, _ = predict(Pplain, pX, pw)
            out.append((pp, sc, "plain", silo_pairs + plain_pairs))
    return out


def _pairs(silo_in, plain_in):
    for (sX, sw), (pX, pw) in zip(silo_in, plain_in):
        yield sX, sw, pX, pw


def expected_calibration_error(confs, corrects, n_bins=10):
    """ECE: average |confidence - accuracy| over confidence bins. A calibrated model
    has ECE ~ 0 (its stated confidence matches how often it is right). A confidently
    wrong model has high ECE -- and that is exactly when a confidence cascade fails."""
    confs = np.asarray(confs, float); corrects = np.asarray(corrects, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (confs > lo) & (confs <= hi) if b > 0 else (confs >= lo) & (confs <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(confs[m].mean() - corrects[m].mean())
    return float(ece)


def confidence_auc(confs, corrects):
    """AUROC of confidence as a detector of 'this prediction is correct'. 0.5 = the
    confidence tells you nothing about correctness (the trap); ~1.0 = confidence
    perfectly ranks right-above-wrong (a cascade can route on it)."""
    confs = np.asarray(confs, float); corrects = np.asarray(corrects, bool)
    pos = confs[corrects]; neg = confs[~corrects]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUROC (ties = 0.5)
    wins = 0.0
    for a in pos:
        wins += np.sum(a > neg) + 0.5 * np.sum(a == neg)
    return float(wins / (len(pos) * len(neg)))

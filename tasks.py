#!/usr/bin/env python3
"""Two CUE-FREE mixed-regime tasks for the v6 cascade. Neither has a regime cue
token -- the cascade must decide from the silo's confidence alone. The tasks
differ ONLY in whether the cheap silo's uncertainty lines up with its errors:

  ENTANGLED  -- bag and order examples are drawn from the SAME content
                distribution (uniform random groups). The silo, an orderless
                content view, sees identical inputs for both regimes, so on an
                order example it confidently emits the plurality -- CONFIDENTLY
                WRONG. Its confidence tells you nothing about correctness.

  TELEGRAPHED -- bag examples have a CLEAR content plurality (silo confident and
                right); order examples have FLAT content (a G-way tie), so the
                silo is genuinely UNCERTAIN exactly when it is about to be wrong
                (it can't read the first token). Difficulty telegraphs the regime.

Same cascade, opposite outcomes -- the difference is calibration. This is a
CONSTRUCTED contrast to isolate WHEN a confidence cascade works, not a claim that
real tasks look like either one.
"""
from __future__ import annotations
import numpy as np

D = 6          # embedding width
G = 4          # latent groups (= classes)
V = 16         # vocabulary (4 tokens per group)
N = 8          # tokens per bag
K = 4          # silo intents


def make_world(seed=0):
    rng = np.random.default_rng(seed)
    protos = rng.standard_normal((G, D)) * 2.2
    tok_group = np.array([g for g in range(G) for _ in range(V // G)])
    E = np.stack([protos[tok_group[t]] + rng.standard_normal(D) * 0.35 for t in range(V)])
    Ppos = rng.standard_normal((N, D)) * 0.5
    return {"protos": protos, "tok_group": tok_group, "E": E, "Ppos": Ppos, "seed": seed}


# ---------- k-means silo front-end (deterministic, farthest-point seed) ----------
def _seed_centroids(vs, k):
    chosen = [vs[0].copy()]
    while len(chosen) < k:
        d = np.min([np.sum((vs - c) ** 2, axis=1) for c in chosen], axis=0)
        chosen.append(vs[int(np.argmax(d))].copy())
    return np.stack(chosen)


def centrifuge(vs, k, max_spins=25):
    cen = _seed_centroids(vs, k)
    assign = np.argmin(((vs[:, None, :] - cen[None, :, :]) ** 2).sum(-1), axis=1)
    for _ in range(max_spins):
        for j in range(k):
            members = vs[assign == j]
            if len(members):
                cen[j] = members.mean(axis=0)
        new = np.argmin(((vs[:, None, :] - cen[None, :, :]) ** 2).sum(-1), axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
    sizes = np.array([max(1, int((assign == j).sum())) for j in range(k)], dtype=float)
    return cen, sizes


# ---------- cue-free inputs ----------
def silo_input(world, tokens):
    """Content silo: k-means on token embeddings -> K intents. Orderless, cue-free."""
    return centrifuge(world["E"][tokens], K)


def plain_input(world, tokens):
    """Plain expert: tokens + positions. Order-aware, cue-free (no regime marker)."""
    X = world["E"][tokens] + world["Ppos"][:len(tokens)]
    return X, np.ones(len(tokens))


def _tok_of_group(world):
    tg = world["tok_group"]
    return [np.where(tg == g)[0] for g in range(G)]


def _label(groups, r):
    if r == 0:   # bag -> plurality (order-insensitive)
        return int(np.argmax(np.bincount(groups, minlength=G)))
    return int(groups[0])   # order -> first token's group (order-sensitive)


def _entangled_sample(world, n, seed):
    rng = np.random.default_rng(seed)
    tg = _tok_of_group(world)
    ex = []
    for _ in range(n):
        r = int(rng.integers(0, 2))
        groups = rng.integers(0, G, size=N)              # SAME dist for both regimes
        tokens = np.array([rng.choice(tg[g]) for g in groups])
        ex.append((r, tokens, _label(groups, r)))
    return ex


def _telegraphed_sample(world, n, seed):
    rng = np.random.default_rng(seed)
    tg = _tok_of_group(world)
    ex = []
    for _ in range(n):
        r = int(rng.integers(0, 2))
        if r == 0:                                        # bag: CLEAR plurality
            dom = int(rng.integers(0, G))
            groups = np.where(rng.random(N) < 0.75, dom, rng.integers(0, G, size=N))
        else:                                             # order: FLAT content (a tie)
            groups = np.repeat(np.arange(G), N // G)      # exactly N/G of each group
            rng.shuffle(groups)
        tokens = np.array([rng.choice(tg[g]) for g in groups])
        ex.append((r, tokens, _label(groups, r)))
    return ex


def dataset(world, task, n_train=1200, n_test=600, seed=200):
    sampler = {"entangled": _entangled_sample, "telegraphed": _telegraphed_sample}[task]
    return {"task": task, "train": sampler(world, n_train, seed),
            "test": sampler(world, n_test, seed + 1), "n_classes": G, "chance": 1.0 / G}

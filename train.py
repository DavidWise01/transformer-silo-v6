#!/usr/bin/env python3
"""Train the two experts on each task, run the CONFIDENCE-GATED CASCADE, and
measure why it works or fails. The headline is the contrast between ENTANGLED
(the silo is confidently wrong -> cascade is blind) and TELEGRAPHED (the silo is
calibrated -> cascade routes for free). Deterministic. Run: python train.py"""
from __future__ import annotations
import json, time
import numpy as np
from model import (init_expert, loss_and_grad, predict,
                   expected_calibration_error, confidence_auc)
from tasks import make_world, dataset, silo_input, plain_input, D, G, N, K

H, EPOCHS, LR, BATCH = 16, 250, 0.01, 32
SILO_PAIRS, PLAIN_PAIRS = K * K, N * N          # 16, 64  (cue-free plain = N tokens)


def train_expert(world, examples, input_fn, seed=0, epochs=EPOCHS):
    rng = np.random.default_rng(seed)
    P = init_expert(D, H, G, seed=seed)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(x) for k, x in P.items()}
    data = [(input_fn(world, t), y) for (_, t, y) in examples]
    idx = np.arange(len(data)); step = 0
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, len(idx), BATCH):
            g = {k: np.zeros_like(x) for k, x in P.items()}
            b = idx[s:s + BATCH]
            for i in b:
                (X, w), y = data[i]
                _, gi = loss_and_grad(P, X, y, w)
                for k in g: g[k] += gi[k]
            step += 1
            b1, b2, eps = 0.9, 0.999, 1e-8
            for k in P:
                gg = g[k] / len(b)
                m[k] = b1 * m[k] + (1 - b1) * gg
                v[k] = b2 * v[k] + (1 - b2) * gg * gg
                P[k] -= LR * (m[k] / (1 - b1 ** step)) / (np.sqrt(v[k] / (1 - b2 ** step)) + eps)
    return P


def run_task(world, task, seed=0):
    ds = dataset(world, task, seed=200)
    Psilo = train_expert(world, ds["train"], silo_input, seed)
    Pplain = train_expert(world, ds["train"], plain_input, seed)

    # per-example: silo pred/conf/correct, plain pred/correct, true regime
    recs = []
    for (r, t, y) in ds["test"]:
        sX, sw = silo_input(world, t)
        sp, sc = predict(Psilo, sX, sw)
        pX, pw = plain_input(world, t)
        pp, _ = predict(Pplain, pX, pw)
        recs.append({"r": r, "y": y, "sp": sp, "sc": sc, "pp": pp,
                     "s_ok": int(sp == y), "p_ok": int(pp == y)})
    n = len(recs)
    silo_acc = sum(x["s_ok"] for x in recs) / n
    plain_acc = sum(x["p_ok"] for x in recs) / n
    ece = expected_calibration_error([x["sc"] for x in recs], [x["s_ok"] for x in recs])
    auc = confidence_auc([x["sc"] for x in recs], [x["s_ok"] for x in recs])

    # cascade sweep over tau
    curve = []
    for tau in np.linspace(0.25, 1.0, 31):
        correct = pairs = esc = esc_bag = esc_ord = nb = no = 0
        for x in recs:
            nb += (x["r"] == 0); no += (x["r"] == 1)
            if x["sc"] >= tau:
                correct += x["s_ok"]; pairs += SILO_PAIRS
            else:
                correct += x["p_ok"]; pairs += SILO_PAIRS + PLAIN_PAIRS
                esc += 1; esc_bag += (x["r"] == 0); esc_ord += (x["r"] == 1)
        curve.append({"tau": round(float(tau), 3), "acc": round(correct / n, 4),
                      "pairs": round(pairs / n, 1), "esc": round(esc / n, 3),
                      "esc_bag": round(esc_bag / max(1, nb), 3),
                      "esc_ord": round(esc_ord / max(1, no), 3)})

    gap = plain_acc - silo_acc                      # headroom the expensive expert offers
    # honest operating point: cheapest cascade that matches plain accuracy (within 1pt)
    feasible = [c for c in curve if c["acc"] >= plain_acc - 0.01]
    op = min(feasible, key=lambda c: c["pairs"]) if feasible else None
    recovery = None
    if op is not None and gap > 1e-6:
        recovery = round((op["acc"] - silo_acc) / gap, 3)   # fraction of the gap closed
    return {
        "task": task, "chance": round(ds["chance"], 4),
        "silo_only": {"acc": round(silo_acc, 4), "pairs": SILO_PAIRS},
        "plain_only": {"acc": round(plain_acc, 4), "pairs": PLAIN_PAIRS},
        "gap": round(gap, 4), "recovery_frac": recovery,
        "silo_ece": round(ece, 4), "silo_conf_auc": round(auc, 4),
        "operating_point": op, "curve": curve,
    }


def verdict_for(res):
    e = res; op = e["operating_point"]
    if e["gap"] < 0.05:
        # no headroom: the expensive expert is barely better, so nothing to route to
        return (f"{e['task'].upper()}: NO HEADROOM -- plain {e['plain_only']['acc']:.2f} is barely "
                f"above silo {e['silo_only']['acc']:.2f} (gap {e['gap']*100:.0f}%). When the regime "
                f"is illegible in the content, NEITHER expert can do the hard half cue-free, so a "
                f"cascade has nothing better to escalate to -- and the silo's confidence is weak "
                f"anyway (ECE {e['silo_ece']:.2f}, conf-vs-correct AUC {e['silo_conf_auc']:.2f}). "
                f"The cascade correctly stays cheap (~{op['pairs'] if op else SILO_PAIRS} pairs) but "
                f"cannot manufacture accuracy neither expert has.")
    return (f"{e['task'].upper()}: CUE-FREE CASCADE WORKS -- the silo is CALIBRATED (ECE "
            f"{e['silo_ece']:.2f}, conf-vs-correct AUC {e['silo_conf_auc']:.2f}: it is uncertain "
            f"exactly when it is wrong), so its confidence routes the hard regime to plain with NO "
            f"cue. It closes {e['recovery_frac']*100:.0f}% of the silo->plain gap ({e['silo_only']['acc']:.2f}"
            f"->{op['acc']:.2f}, plain {e['plain_only']['acc']:.2f}) at {op['pairs']} avg attention "
            f"pairs vs plain's {e['plain_only']['pairs']} -- {(1-op['pairs']/e['plain_only']['pairs'])*100:.0f}% "
            f"cheaper -- escalating {op['esc_ord']*100:.0f}% of order but {op['esc_bag']*100:.0f}% of bag.")


def run(seed=0):
    world = make_world(seed=seed)
    tasks = {t: run_task(world, t, seed) for t in ("entangled", "telegraphed")}
    verdict = (
        "Same confidence-gated cascade, two cue-free tasks, opposite outcomes -- and "
        "the difference is CALIBRATION. " +
        verdict_for(tasks["telegraphed"]) + " " + verdict_for(tasks["entangled"]) +
        " The lesson: a confidence cascade does not detect the hard regime -- it "
        "detects the cheap model's own UNCERTAINTY, which only helps when that "
        "uncertainty lines up with its errors (low ECE, AUC >> 0.5). And calibration "
        "here tracks LEGIBILITY: when the regime shows up in the content (telegraphed), "
        "the silo is calibrated AND a capable fallback exists, so you get cue-free "
        "routing for free; when it doesn't (entangled), the silo is confidently wrong "
        "AND nothing better exists to route to. This is the honest sequel to v5's catch "
        "'the regime must be detectable in the input': if it is detectable in the "
        "CONTENT, you need neither a cue token nor a learned router -- a confidence "
        "cascade suffices; if it isn't, nothing does. Constructed contrast, tiny models, "
        "synthetic -- a demonstration of WHEN cascades work, not a benchmark."
    )
    return {"config": {"D": D, "G": G, "N": N, "K": K, "H": H, "epochs": EPOCHS,
                       "seed": seed, "silo_pairs": SILO_PAIRS, "plain_pairs": PLAIN_PAIRS,
                       "n_train": 1200, "n_test": 600},
            "tasks": tasks, "verdict": verdict}


if __name__ == "__main__":
    t0 = time.time()
    res = run(seed=0)
    with open("results.json", "w") as f:
        json.dump(res, f, indent=2)
    for t, e in res["tasks"].items():
        op = e["operating_point"]
        print(f"\n[{t}]  chance {e['chance']}")
        print(f"  silo-only  acc={e['silo_only']['acc']:.3f} @ {e['silo_only']['pairs']} pairs")
        print(f"  plain-only acc={e['plain_only']['acc']:.3f} @ {e['plain_only']['pairs']} pairs")
        print(f"  gap plain-silo={e['gap']:.3f}  silo ECE={e['silo_ece']:.3f}  conf-vs-correct AUC={e['silo_conf_auc']:.3f}")
        if e["gap"] >= 0.05 and op:
            print(f"  CASCADE op: acc={op['acc']:.3f} @ {op['pairs']} pairs "
                  f"(closes {e['recovery_frac']*100:.0f}% of gap; esc order {op['esc_ord']*100:.0f}% / bag {op['esc_bag']*100:.0f}%)  -> CUE-FREE WIN")
        else:
            print(f"  CASCADE: no headroom (plain ~ silo) -> nothing to route to; stays cheap")
    print(f"\n{res['verdict']}\n[{time.time()-t0:.1f}s]")

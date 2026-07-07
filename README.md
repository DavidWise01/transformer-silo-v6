# transformer-silo v6 — the cascade & the confidence trap

[v5](https://davidwise01.github.io/transformer-silo-v5/) routed with a regime
**cue**, and its honest catch was: *a router can only route because the regime is
detectable in the input.* v6 tests the obvious escape hatch — **skip the cue.**
Run **Dave's cheap silo first**; escalate to the plain expert **only when the
silo is unsure.** The silo's own confidence is the router. No cue, no learned gate.

**Does it work?** Only when the silo is **calibrated** — uncertain *exactly* when
it's wrong. That single condition is the whole story, and v6 shows it by running
the same cascade on two cue-free tasks that differ only in whether the hard
regime is legible in the content.

## The cascade

Every example runs the cheap silo (`K²=16` attention pairs). If the silo's
top-class probability clears a threshold `τ`, keep its answer. Otherwise escalate
to the order-aware plain expert (`+N²=64`, so an escalated example costs `80`).
The routing signal is the silo's confidence — nothing else.

## The two tasks (both cue-free, seed 0, 1200 train / 600 test)

- **telegraphed** — bag examples have a **clear content plurality** (silo confident
  and right); order examples have **flat content** (a G-way tie), so the silo is
  genuinely uncertain *exactly when* it can't read the first token.
- **entangled** — bag and order examples are drawn from the **same content
  distribution**, so the silo sees identical inputs for both regimes and
  confidently emits the plurality on the order ones. **Confidently wrong.**

| task | silo only | plain only | **cascade** | silo ECE | conf-AUC | outcome |
|------|-----------|------------|-------------|----------|----------|---------|
| **telegraphed** | 0.625 @ 16 | 0.923 @ 64 | **0.922 @ 45.3** | 0.05 | 0.90 | cue-free win |
| **entangled** | 0.677 @ 16 | 0.688 @ 64 | 0.680 @ 18.5 | 0.12 | 0.59 | no headroom |

## What the numbers mean

- **`silo ECE`** (expected calibration error) — the average gap between how
  confident the silo is and how often it's actually right. `0` = perfectly
  calibrated. Entangled's **0.12** means it's *confidently wrong*; telegraphed's
  **0.05** means it knows what it doesn't know.
- **`conf-AUC`** — the AUROC of the silo's confidence used as a detector of "this
  answer is correct." **0.5 = blind** (confidence tells you nothing). Entangled
  **0.59** is near-blind; telegraphed **0.90** ranks right-above-wrong. *A cascade
  routes on this number, so it inherits exactly the silo's calibration.*
- **telegraphed → cue-free win.** The cascade closes **~100%** of the silo→plain
  gap (0.625 → 0.922, plain 0.923) at **45.3 attention pairs vs plain's 64 —
  ~29% cheaper** — escalating **90% of order** examples but **0% of bag.** The
  confidence found the hard regime with no cue, because the silo was calibrated.
- **entangled → no headroom.** Plain (0.688) barely beats silo (0.677): when the
  regime is illegible, *neither* expert can do the hard half cue-free, so the
  cascade has nothing better to escalate to — and confidence is near-blind anyway.
  It correctly stays cheap but can't manufacture accuracy neither expert has.

## Is that good or bad?

Both results are **good** — because the point of an honest instrument is to map
the boundary, not to win. The **telegraphed** result is a genuine positive: you
can route cue-free and cheaply *when the cheap model is calibrated.* The
**entangled** result is the matching negative that keeps you from over-claiming
it: confidence-routing is **not** magic — it detects the cheap model's *own
uncertainty*, which only helps when that uncertainty lines up with its errors.

This is the honest sequel to v5's catch. v5: *the regime must be detectable in
the input.* v6: if it's detectable **in the content**, you need neither a cue
token nor a learned router — a confidence cascade suffices; if it isn't, **nothing
does**, and no amount of confidence-watching changes that. **Confidence ≠
competence.**

## Verify first

```bash
python selftest.py    # gradient-checks the expert backprop (<1e-5) + calibration/cascade
                      # invariants + asserts the core finding (telegraphed better-calibrated)
python train.py       # retrain both experts on both tasks, run the cascade -> results.json
```

The **gradient check is the honesty anchor**: the experts really train by gradient
descent (analytic grads match numerical `< 1e-5`), and the selftest asserts the
load-bearing claim — the telegraphed silo genuinely is better-calibrated (AUC 0.90
vs 0.59) and the cue-free cascade genuinely closes the gap under plain's cost.

## Files

| File | Role |
|------|------|
| `model.py` | encoder + two experts + the cascade + ECE / conf-AUC; hand-written, gradient-checked backward |
| `tasks.py` | the world + the entangled & telegraphed cue-free tasks |
| `train.py` | Adam training of both experts on both tasks → `results.json` |
| `selftest.py` | gradient check + calibration/cascade invariants + the core-claim assertion |
| `results.json` | the trained results the page reports |
| `index.html` | the cascade diagram + the two-task contrast + the curves + the verdict |

The hexalogy: [v1](https://davidwise01.github.io/transformer-silo/) build ·
[v2](https://davidwise01.github.io/transformer-silo-v2/) measure ·
[v3](https://davidwise01.github.io/transformer-silo-v3/) train one silo ·
[v4](https://davidwise01.github.io/transformer-silo-v4/) two in parallel ·
[v5](https://davidwise01.github.io/transformer-silo-v5/) route with a cue ·
v6 route on confidence.

---
David Lee Wise / ROOT0 / TriPod LLC · CC-BY-ND-4.0

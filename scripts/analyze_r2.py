#!/usr/bin/env python
"""Round-2 analysis, executed exactly as pre-registered in PLAN.md.

PRIMARY (pre-registered): mean paired per-problem inflation difference (A - B),
  95% bootstrap CI, 10k resamples, seed 0, computed on generations that did NOT
  stop on a loop (four-way-clean subset: all four configs non-looped).
CO-PRIMARY (pre-registered): loop rate per arm.
SECONDARY (exploratory, labelled): median difference, Wilcoxon signed-rank,
  and the same primary estimator computed WITH looped generations included, to
  quantify how much of the pilot's effect was loop-collapse.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import norm_ans, extract_boxed, marker_count

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_r2")
ARMS = {"A_distilled": ("r1d-bf16", "r1d-3bit"), "B_RL": ("dsr-bf16", "dsr-3bit")}


def load(tag):
    return {r["idx"]: r for r in (json.loads(l) for l in open(os.path.join(RES, f"{tag}.jsonl")))}


def boot_ci(x, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), size=(n, len(x)))
    means = np.asarray(x)[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    d = {a: (load(f16), load(f3)) for a, (f16, f3) in ARMS.items()}
    out = []
    P = out.append

    P("=" * 78)
    P("ROUND 2 ANALYSIS  (N=500 MATH-500, 2 models x 2 precisions, single seed)")
    P("=" * 78)

    # ---------- descriptive, per configuration ----------
    P("\n[1] PER-CONFIGURATION SUMMARY")
    for a, (m16, m3) in d.items():
        for lab, m in (("BF16", m16), ("3-bit", m3)):
            rs = list(m.values())
            acc = np.mean([norm_ans(extract_boxed(r["text"])) == norm_ans(r["gold"]) for r in rs])
            P(f"  {a:12s} {lab:6s} n={len(rs)} acc={acc:.3f} "
              f"loop={np.mean([r['loop_stopped'] for r in rs]):.3f} "
              f"cap={np.mean([r['truncated'] for r in rs]):.3f} "
              f"mean_tok={np.mean([r['gen_tokens'] for r in rs]):.0f}")

    # ---------- CO-PRIMARY: loop rate ----------
    P("\n[2] CO-PRIMARY ENDPOINT - loop rate (pre-registered)")
    lr = {}
    for a, (m16, m3) in d.items():
        l16 = np.mean([r["loop_stopped"] for r in m16.values()])
        l3 = np.mean([r["loop_stopped"] for r in m3.values()])
        lr[a] = (l16, l3)
        P(f"  {a:12s} BF16 {l16:.3f} -> 3-bit {l3:.3f}   (delta {l3-l16:+.3f})")
    dA, dB = lr["A_distilled"][1], lr["B_RL"][1]
    se = np.sqrt(dA * (1 - dA) / 500 + dB * (1 - dB) / 500)
    P(f"  3-bit loop-rate difference A-B = {dA-dB:+.3f}  SE={se:.4f}  z={(dA-dB)/se:.1f}"
      f"  95% CI [{dA-dB-1.96*se:+.3f}, {dA-dB+1.96*se:+.3f}]")

    # ---------- PRIMARY: paired inflation difference, four-way clean ----------
    P("\n[3] PRIMARY ENDPOINT - paired inflation difference A-B (pre-registered)")
    a16, a3 = d["A_distilled"]; b16, b3 = d["B_RL"]
    allidx = sorted(set(a16) & set(a3) & set(b16) & set(b3))
    clean = [i for i in allidx if not (a16[i]["loop_stopped"] or a3[i]["loop_stopped"]
                                       or b16[i]["loop_stopped"] or b3[i]["loop_stopped"])]
    P(f"  four-way-clean subset: n={len(clean)} of {len(allidx)} problems "
      f"({100*len(clean)/len(allidx):.0f}%)")
    infA = np.array([a3[i]["gen_tokens"] - a16[i]["gen_tokens"] for i in clean], float)
    infB = np.array([b3[i]["gen_tokens"] - b16[i]["gen_tokens"] for i in clean], float)
    diff = infA - infB
    lo, hi = boot_ci(diff)
    P(f"  arm A inflation: mean {infA.mean():+.0f}  median {np.median(infA):+.0f}  sd {infA.std(ddof=1):.0f}")
    P(f"  arm B inflation: mean {infB.mean():+.0f}  median {np.median(infB):+.0f}  sd {infB.std(ddof=1):.0f}")
    P(f"  >>> PAIRED DIFFERENCE (A-B): mean {diff.mean():+.0f} tokens")
    P(f"  >>> 95% bootstrap CI (10k resamples, seed 0): [{lo:+.0f}, {hi:+.0f}]")
    verdict = ("EXCLUDES zero -> effect supported" if (lo > 0 or hi < 0)
               else "INCLUDES zero -> NULL RESULT under the pre-registered rule")
    P(f"  >>> {verdict}")
    P(f"  >>> sign vs pilot (pilot A-B was +919, A inflating MORE): "
      f"{'SAME sign' if diff.mean() > 0 else 'REVERSED sign'}")
    P(f"  problems where A inflates more: {int((diff>0).sum())}/{len(diff)}")

    # ---------- SECONDARY (exploratory, labelled) ----------
    P("\n[4] SECONDARY / EXPLORATORY (not pre-registered - labelled as such)")
    P(f"  median paired difference: {np.median(diff):+.0f} tokens")
    try:
        from scipy.stats import wilcoxon
        st, p = wilcoxon(infA, infB)
        P(f"  Wilcoxon signed-rank (rank-based, robust to the heavy tail): W={st:.0f}, p={p:.4g}")
    except Exception as e:
        P(f"  Wilcoxon unavailable: {e}")
    # same estimator WITH loops included -> how much of the pilot effect was loop-collapse
    infA_all = np.array([a3[i]["gen_tokens"] - a16[i]["gen_tokens"] for i in allidx], float)
    infB_all = np.array([b3[i]["gen_tokens"] - b16[i]["gen_tokens"] for i in allidx], float)
    dall = infA_all - infB_all
    lo2, hi2 = boot_ci(dall)
    P(f"  WITH looped generations included (pilot-style metric, n={len(allidx)}):")
    P(f"     arm A {infA_all.mean():+.0f}, arm B {infB_all.mean():+.0f}, "
      f"difference {dall.mean():+.0f}, 95% CI [{lo2:+.0f}, {hi2:+.0f}]")
    P("     -> comparison of [3] vs this line isolates how much of the pilot's")
    P("        inflation effect was loop-collapse rather than longer reasoning.")

    # ---------- marker density ----------
    P("\n[5] HESITATION-MARKER DENSITY (per 1k tokens, four-way-clean subset)")
    for a, (m16, m3) in d.items():
        m1 = np.mean([1000 * marker_count(m16[i]["text"]) / max(1, m16[i]["gen_tokens"]) for i in clean])
        m2 = np.mean([1000 * marker_count(m3[i]["text"]) / max(1, m3[i]["gen_tokens"]) for i in clean])
        P(f"  {a:12s} BF16 {m1:.2f} -> 3-bit {m2:.2f}  (delta {m2-m1:+.2f})")

    # ---------- accuracy ----------
    P("\n[6] ACCURACY (all 500 per config)")
    for a, (m16, m3) in d.items():
        f = lambda m: np.mean([norm_ans(extract_boxed(r["text"])) == norm_ans(r["gold"]) for r in m.values()])
        P(f"  {a:12s} {f(m16):.3f} -> {f(m3):.3f}  (delta {f(m3)-f(m16):+.3f})")

    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(RES, "summary_r2.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(f"\n[saved] {os.path.join(RES,'summary_r2.txt')}")


if __name__ == "__main__":
    main()

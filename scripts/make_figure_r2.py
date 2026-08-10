#!/usr/bin/env python
"""Figure 1 for the Round-2 (N=500) result, built from raw per-problem JSONL.

Reuses analyze.py's extractor / marker counter so figure and tables share a code
path. Panels: (A) accuracy, (B) loop rate [co-primary, significant],
(C) hesitation-marker density, (D) paired inflation difference [primary, NULL].
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import norm_ans, extract_boxed, marker_count

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_r2")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "papers", "arxiv"))
ARMS = {"A: distilled": ("r1d-bf16", "r1d-3bit"), "B: +RL": ("dsr-bf16", "dsr-3bit")}
COL = {"BF16": "#4C72B0", "3-bit": "#C44E52"}


def load(tag):
    return {r["idx"]: r for r in (json.loads(l) for l in open(os.path.join(RES, f"{tag}.jsonl")))}


def boot_ci(x, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    m = np.asarray(x)[rng.randint(0, len(x), size=(n, len(x)))].mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    d = {a: (load(f16), load(f3)) for a, (f16, f3) in ARMS.items()}
    arms = list(ARMS)
    a16, a3 = d[arms[0]]; b16, b3 = d[arms[1]]
    idx = sorted(set(a16) & set(a3) & set(b16) & set(b3))
    clean = [i for i in idx if not (a16[i]["loop_stopped"] or a3[i]["loop_stopped"]
                                    or b16[i]["loop_stopped"] or b3[i]["loop_stopped"])]

    fig, ax = plt.subplots(1, 4, figsize=(13.4, 3.4))
    x = np.arange(2); w = 0.36

    def bars(axis, fn, ylab, subset=None):
        for j, prec in enumerate(["BF16", "3-bit"]):
            vals = [fn(d[a][j], subset) for a in arms]
            axis.bar(x + (j - .5) * w, vals, w, label=prec, color=COL[prec],
                     edgecolor="black", linewidth=.5)
            for xi, v in zip(x + (j - .5) * w, vals):
                axis.text(xi, v, f"{v:.2f}" if v < 10 else f"{v:.1f}",
                          ha="center", va="bottom", fontsize=7.5)
        axis.set_xticks(x); axis.set_xticklabels(arms, fontsize=9)
        axis.set_ylabel(ylab, fontsize=9); axis.tick_params(labelsize=8)
        axis.spines[["top", "right"]].set_visible(False); axis.margins(y=.2)

    acc = lambda m, s: np.mean([norm_ans(extract_boxed(m[i]["text"])) == norm_ans(m[i]["gold"])
                                for i in (s or m)])
    loop = lambda m, s: np.mean([m[i]["loop_stopped"] for i in (s or m)])
    mark = lambda m, s: np.mean([1000 * marker_count(m[i]["text"]) / max(1, m[i]["gen_tokens"])
                                 for i in (s or m)])

    bars(ax[0], acc, "Accuracy")
    bars(ax[1], loop, "Degenerate loop rate")
    bars(ax[2], mark, "Hesitation markers\nper 1k tokens", clean)
    ax[0].legend(fontsize=8, frameon=False, loc="lower left")

    # Panel D: the primary endpoint - a null
    infA = np.array([a3[i]["gen_tokens"] - a16[i]["gen_tokens"] for i in clean], float)
    infB = np.array([b3[i]["gen_tokens"] - b16[i]["gen_tokens"] for i in clean], float)
    diff = infA - infB
    lo, hi = boot_ci(diff)
    axd = ax[3]
    XL = 8000  # display window; outliers beyond are counted in the annotation
    n_out = int((np.abs(diff) > XL).sum())
    axd.hist(np.clip(diff, -XL, XL), bins=np.linspace(-XL, XL, 33),
             color="#999999", edgecolor="black", linewidth=.4)
    top = axd.get_ylim()[1]
    axd.set_ylim(0, top * 1.35)
    axd.axvline(0, color="black", lw=1, ls=":")
    axd.axvline(diff.mean(), color="#C44E52", lw=2)
    axd.hlines(top * 1.12, lo, hi, color="#C44E52", lw=3)
    axd.plot([lo, hi], [top * 1.12] * 2, "|", color="#C44E52", ms=7, mew=2)
    axd.text(0, top * 1.20,
             f"mean {diff.mean():+.0f}   95% CI [{lo:+.0f}, {hi:+.0f}]  (n.s.)",
             ha="center", va="bottom", fontsize=7.5, color="#C44E52")
    axd.set_xlabel("Paired inflation difference (A $-$ B), tokens", fontsize=8.5)
    axd.set_ylabel(f"Problems (n={len(clean)})", fontsize=9)
    axd.tick_params(labelsize=8); axd.spines[["top", "right"]].set_visible(False)
    if n_out:
        axd.text(.99, .02, f"{n_out} outliers clipped to $\\pm${XL//1000}k",
                 transform=axd.transAxes, ha="right", va="bottom", fontsize=6.5, color="#555555")

    for a_, l in zip(ax, "ABCD"):
        a_.set_title(l, loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"fig1_r2.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight"); print("wrote", p)

    print(f"\nplotted values (n_clean={len(clean)}):")
    for a in arms:
        m16, m3 = d[a]
        print(f"  {a:14s} acc {acc(m16,None):.3f}->{acc(m3,None):.3f} | loop {loop(m16,None):.3f}->{loop(m3,None):.3f}"
              f" | mark {mark(m16,clean):.2f}->{mark(m3,clean):.2f}")
    print(f"  paired diff mean {diff.mean():+.0f}  CI [{lo:+.0f}, {hi:+.0f}]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build Figure 1 for the paper directly from the raw per-problem JSONL results.

Reuses analyze.py's extractor / marker counter / loop detector so the figure and
the paper's Table 1 are computed by identical code paths. No numbers are hard-coded.

Usage:  python make_figure.py            (writes ../../papers/arxiv/fig1_drift.pdf + .png)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import norm_ans, extract_boxed, marker_count, loop_detect

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "papers", "arxiv"))

ARMS = {
    "A: distilled (R1-Distill-Qwen-1.5B)": ("r1d-bf16.jsonl", "r1d-3bit.jsonl"),
    "B: +RL (DeepScaleR-1.5B)":            ("dsr-bf16.jsonl", "dsr-3bit.jsonl"),
}
COL = {"BF16": "#4C72B0", "3-bit": "#C44E52"}


def load(fname):
    """Return per-problem dicts, keyed by idx, with metrics recomputed from raw text."""
    out = {}
    for line in open(os.path.join(RES, fname)):
        r = json.loads(line)
        text = r["text"]
        correct = norm_ans(extract_boxed(text)) == norm_ans(r["gold"])
        out[r["idx"]] = dict(
            tokens=r["gen_tokens"],
            truncated=bool(r["truncated"]),
            correct=bool(correct),
            markers_per_1k=1000.0 * marker_count(text) / max(1, r["gen_tokens"]),
            loop=bool(loop_detect(text)),
        )
    return out


def agg(d, key):
    return float(np.mean([v[key] for v in d.values()]))


def main():
    data = {a: {"BF16": load(f16), "3-bit": load(f3)} for a, (f16, f3) in ARMS.items()}
    arms = list(ARMS)
    idxs = sorted(set(data[arms[0]]["BF16"]))

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.3))
    x = np.arange(len(arms))
    w = 0.36

    # --- Panels A-C: paired bars, BF16 vs 3-bit ---------------------------------
    panels = [
        ("accuracy",       "Accuracy",              lambda d: agg(d, "correct")),
        ("markers_per_1k", "Hesitation markers\nper 1k tokens", lambda d: agg(d, "markers_per_1k")),
        ("loop",           "Degenerate loop rate",  lambda d: agg(d, "loop")),
    ]
    for ax, (_, ylab, fn) in zip(axes[:3], panels):
        for j, prec in enumerate(["BF16", "3-bit"]):
            vals = [fn(data[a][prec]) for a in arms]
            ax.bar(x + (j - 0.5) * w, vals, w, label=prec, color=COL[prec],
                   edgecolor="black", linewidth=0.5)
            for xi, v in zip(x + (j - 0.5) * w, vals):
                ax.text(xi, v, f"{v:.2f}" if v < 10 else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["A: distilled", "B: +RL"], fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.18)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")

    # --- Panel D: paired per-problem token inflation ----------------------------
    ax = axes[3]
    infl = {a: np.array([data[a]["3-bit"][i]["tokens"] - data[a]["BF16"][i]["tokens"]
                         for i in idxs], dtype=float) for a in arms}
    trunc = {a: np.array([data[a]["3-bit"][i]["truncated"] for i in idxs]) for a in arms}

    for j, a in enumerate(arms):
        v = infl[a]
        jit = (np.random.RandomState(0).rand(len(v)) - 0.5) * 0.16
        ax.scatter(np.full(len(v), j) + jit, v, s=14,
                   c=np.where(trunc[a], "#C44E52", "#888888"),
                   alpha=0.75, linewidths=0)
        ax.hlines(v.mean(), j - 0.22, j + 0.22, color="black", lw=2, zorder=3)
        ax.text(j, v.mean(), f"  mean {v.mean():.0f}", fontsize=8, va="bottom", ha="left")
    ax.axhline(0, color="black", lw=0.7, ls=":")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(["A: distilled", "B: +RL"], fontsize=9)
    ax.set_ylabel("Per-problem token inflation\n(3-bit $-$ BF16)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter([], [], s=14, c="#C44E52", label="hit 8192-token cap")
    ax.scatter([], [], s=14, c="#888888", label="completed")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    for ax, lab in zip(axes, "ABCD"):
        ax.set_title(lab, loc="left", fontweight="bold", fontsize=11)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"fig1_drift.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("wrote", p)

    # --- echo the plotted numbers so they can be diffed against summary.txt -----
    print("\n--- values plotted (recomputed from raw JSONL) ---")
    for a in arms:
        for prec in ["BF16", "3-bit"]:
            d = data[a][prec]
            print(f"{a:38s} {prec:6s} acc={agg(d,'correct'):.3f} "
                  f"mark/1k={agg(d,'markers_per_1k'):.2f} loop={agg(d,'loop'):.3f} "
                  f"mean_tok={agg(d,'tokens'):.1f} trunc={agg(d,'truncated'):.3f}")
        print(f"{a:38s} mean per-problem inflation = {infl[a].mean():.1f} tok")
    print(f"paired mean difference (A - B) = {(infl[arms[0]] - infl[arms[1]]).mean():.1f} tok")


if __name__ == "__main__":
    main()

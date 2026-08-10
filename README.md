# Quantization-Induced Reasoning Drift — experiment artifacts

Complete artifacts for the paper *"Quantization-Induced Reasoning Drift: A Survey of Overthinking and Chain-of-Thought Inflation in Quantized Long-CoT Models, with a Pre-Registered Controlled Experiment"* (Gagandeep Singh).

**Headline: the pre-registered primary endpoint is a NULL**, with sign opposite to the hypothesis. That result, and the reasons an earlier 50-problem pilot reported the opposite, are the substance of the paper. Everything needed to check it is here.

## The question

At matched low-bit quantization, does a purely **distilled** reasoning model degrade differently from the **same model after RL post-training**?

- **Arm A** — `DeepSeek-R1-Distill-Qwen-1.5B` (pure SFT distillation)
- **Arm B** — `DeepScaleR-1.5B-Preview` (the *same base weights*, further trained with RL)

This is a strict post-training increment, not two pipelines that merely share a base model.

## Results at a glance (N = 500 MATH-500 problems per configuration, 2,000 generations)

| Metric (BF16 → 3-bit) | Arm A (distilled) | Arm B (+RL) |
|---|---|---|
| Accuracy | 0.776 → 0.302 (−47.4 pts) | 0.838 → 0.462 (−37.6 pts) |
| Loop rate (20-gram ×4) | 0.020 → **0.598** | 0.002 → **0.348** |
| Hesitation markers / 1K tokens† | 11.07 → 15.70 (+4.63) | 12.70 → **24.64 (+11.94)** |
| Token inflation† | +1,174 (median +628) | +1,587 (median +798) |

† on the four-way-clean subset (n = 162; no configuration looped)

**Primary endpoint (pre-registered):** paired inflation difference A − B = **−412 tokens, 95% bootstrap CI [−997, +115]** — includes zero. **H1 is not supported**, and the sign is opposite to both the hypothesis and the earlier pilot (+919). Wilcoxon signed-rank agrees (W = 5780, p = 0.17).

**Co-primary endpoint:** loop-rate difference +0.250 (SE 0.031, z = 8.2, 95% CI [+0.190, +0.310]).

## Two caveats we consider essential

**1. The loop-rate result is definition-dependent, and does not survive a stricter definition.**

| Loop definition (full-text scan) | Arm A | Arm B | Difference |
|---|---|---|---|
| 10-gram ×3 | 0.786 | 0.636 | +0.150 |
| **20-gram ×4 (pre-registered)** | **0.604** | **0.362** | **+0.242** |
| 30-gram ×4 | 0.196 | 0.130 | +0.066 |
| 40-gram ×8 | 0.058 | 0.056 | +0.002 |
| **30-gram ×20** ([Pipis et al.](https://arxiv.org/abs/2512.12895)) | **0.030** | **0.042** | **−0.012** |

Under the criterion used by the closest prior work, the effect **vanishes and marginally reverses**. The claim is therefore about **short-period** repetition only.

**2. Aggregate CoT-token-inflation metrics conflate two different behaviours.** With a fixed token cap and no loop handling, a collapsed generation contributes the maximum possible token count — so a metric intended to measure deliberation instead measures collapse. This is exactly what produced the spurious significant result in our 50-problem pilot (`results_pilot/`), which is included here so the error can be inspected rather than taken on trust.

## Layout

```
PLAN.md                  pre-registration: hypothesis, endpoints, decision rule,
                         oracle checks — plus the Round-2 design revision, with
                         its own disclosure of what was known when it was written
scripts/
  run_eval.py            pilot runner (8K cap, no loop stopping)
  run_eval2.py           Round-2 runner (16K cap, online loop-aware stopping)
  analyze.py             shared extractor / marker counter / loop detector
  analyze_r2.py          pre-registered Round-2 analysis
  make_figure.py         pilot figure
  make_figure_r2.py      Round-2 figure (Figure 1 in the paper)
  run_all.sh, run_all_r2.sh
results_pilot/           50-problem pilot: 200 generations + summary
results_round2/          500-problem run: 2,000 generations + summary
                         + loop_sensitivity.txt
```

Each JSONL line is one generation: `idx`, `unique_id`, `level`, `gold`, `gen_tokens`, `gen_seconds`, `truncated`, `loop_stopped`, `loop_at_token`, `max_tokens`, `text`. Problem statements are **not** redistributed — load them from [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500).

## Reproducing

Hardware used: Apple M5 Max, 36 GB, `mlx-lm` 0.31.3. Total generation time **15.5 h** for 2,000 generations (28.0 s/generation).

```bash
python -m venv .venv && source .venv/bin/activate
pip install mlx-lm datasets numpy matplotlib scipy

# quantize (3.501 bits/weight, group size 64) — verify bpw before running
mlx_lm.convert --hf-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  -q --q-bits 3 --q-group-size 64 --mlx-path models/r1d-1.5b-3bit

./scripts/run_all_r2.sh          # ~15.5 h, resumable, per-item checkpointing
python scripts/analyze_r2.py     # reproduces results_round2/summary_r2.txt
python scripts/make_figure_r2.py # reproduces Figure 1
```

The analysis and figure scripts import the same extractor, marker counter, and loop detector from `analyze.py`, so tables and figures cannot silently diverge. Bootstrap CIs use 10,000 resamples with `seed=0` and are deterministic.

Note that generation is stochastic (T = 0.6, top-p 0.95, **single seed, not fixed**), so a rerun will not reproduce per-problem token counts exactly. Aggregate rates should land close; during this run, per-50-problem accuracy blocks varied between 0.16 and 0.48, which is itself a documented finding about how unstable 50-problem estimates are.

## Limitations

(i) MLX RTN group quantization, not AWQ/GPTQ; (ii) arm B is distilled-*then*-RL, not never-distilled, so the confound is narrowed rather than broken; (iii) single seed; (iv) one benchmark, mid-difficulty — the ordering may invert on harder benchmarks such as AIME; (v) normalized-string answer matching without symbolic equivalence; (vi) loop-stopping costs an estimated 2–4% of correct answers (audited by replay); (vii) the four-way-clean subset is 32% of problems, so the primary endpoint conditions on a plausibly easier subsample; (viii) 1.5B scale only; (ix) the loop-rate result is definition-dependent (above).

## License

Code: MIT. Generations and derived data: CC BY 4.0. Model weights are not redistributed.

## Citation

Preprint forthcoming; this README will be updated with the arXiv identifier once available.

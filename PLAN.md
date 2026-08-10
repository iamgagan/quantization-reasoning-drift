# Experiment: Does RL post-training change quantization-induced overthinking? (Pilot)

## Motivation
The survey (`papers/quantization-reasoning-drift.md`, §6.2) identified a missing controlled experiment: matched models with different training provenance (distillation vs. RL) under identical quantization, measured with overthinking-specific metrics — not just accuracy. No paper in the literature runs this.

## Honest scope note (read first)
A truly "never-distilled" small open reasoning model may not exist (Qwen3 small models use strong-to-weak distillation; QwQ-32B is RL-native but 32B is out of local budget). The cleanest tractable contrast at 1.5B is:
- **Arm A (distilled):** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` — pure SFT distillation from R1.
- **Arm B (distilled + RL):** `agentica-org/DeepScaleR-1.5B-Preview` — the *same* base model further trained with RL (GRPO).

So this pilot tests a *weaker but well-defined* hypothesis: **H1: RL post-training on top of distillation reduces quantization-induced overthinking amplification relative to pure distillation, at matched scale and quantization.** It does NOT test "distillation vs. native training" in the pure sense; that limitation will be stated in any writeup.

## Design
- **Precisions:** BF16 (reference) vs. 3-bit MLX group quantization (aggressive; RTN-style group quant, group size 64 default). NOTE: MLX quantization is not AWQ/GPTQ — effects may differ from the papers; RTN-style is generally *worse* than AWQ, so 3-bit effects should be at least as visible. Disclosed as a scheme difference. 4-bit optional extension if time permits.
- **Benchmark:** MATH-500 (HuggingFaceH4/MATH-500), first N=50 problems (fixed slice, seed-free, reproducible).
- **Decoding:** temperature 0.6, top-p 0.95 (matches arXiv:2606.00206), max_tokens 8192.
- **Configs:** 2 models × 2 precisions = 4 runs × 50 problems.

## Metrics (defined before implementation)
1. **Accuracy**: normalized boxed-answer exact match.
2. **Mean CoT length** (generated tokens) and **CTIR** = (len_q − len_bf16)/len_bf16 per model.
3. **Overthinking-marker rate**: occurrences of marker set {wait, but, alternatively, however, hmm, maybe, perhaps, actually, let me check, let me verify, re-check, reconsider} per 1K generated tokens (case-insensitive, word-boundary).
4. **Truncation rate**: fraction of generations hitting max_tokens (proxy for runaway/loop).
5. **Answer-in-trace-but-wrong rate** (overthinking-error proxy): among incorrect finals, fraction where the normalized gold answer string appears in the trace.
6. **Loop rate**: fraction of generations containing a 20-gram repeated ≥4 times within any 1024-token window (per arXiv:2606.02011's detector, word-level approximation).

## Oracle / pre-implementation checks
- [ ] Answer extractor validated on ≥5 hand-checked MATH-500 items (boxed forms, fractions, latex).
- [ ] Marker counter validated on a synthetic string with known counts.
- [ ] Quantized model file size ≈ expected (3-bit ≈ ~0.7–0.9 GB for 1.5B) and produces coherent text on a smoke prompt.
- [ ] Smoke run: 2 problems per config completes end-to-end and writes JSONL.

## Success criteria for the pilot (what would count as signal)
- Direction check: does 3-bit quantization increase CoT length / marker rate / truncation in Arm A (replicating the literature qualitatively under MLX RTN)?
- The novel comparison: is the *amplification* (Δ from BF16 to 3-bit) smaller in Arm B than Arm A on CoT length and marker rate? Report with per-problem paired statistics; N=50 is a pilot — wide uncertainty expected, no strong claims.

## Runtime budget
~4 configs × 50 problems × ≲8K tokens; estimated several hours on M5 Max. Runs incrementally checkpoint to `results/*.jsonl`; partial results are analyzable.

## Ledger
- [x] venv (python 3.12, uv)
- [x] install mlx-lm 0.31.3, datasets
- [x] download + quantize models — all 4 variants consistent: bf16 3.3G each; 3-bit 753M each @ 3.501 bpw (note: DeepScaleR first converted at fp32/4.0bpw by default; deleted and redone with --dtype bfloat16 for parity)
- [x] oracle checks — all PASS (extractor 5/5 incl. documented numeric-equivalence limitation `12.0`≠`12`; marker counter exact; loop detector both directions)
- [x] smoke run — 2 items, coherent output, ~270 tok/s on 3-bit
- [x] full run launched: `run_all.sh` via nohup, PID 86986, log `results/run_all.log`, order: r1d-3bit → dsr-3bit → r1d-bf16 → dsr-bf16, N=50, max_tokens=8192, T=0.6, top-p=0.95. Early observation at 10/50: majority of r1d-3bit generations hit the 8192 cap (truncation signal).
- [x] full run complete (4 configs × 50, finished 20:34 EDT)
- [x] analysis + writeup: paper §7 added; results/summary.txt saved

## Known limitations to disclose in writeup
- MLX group quantization (RTN-style) ≠ AWQ/GPTQ used in the literature.
- Arm B is distilled+RL, not never-distilled; tests the RL-post-training variant of the hypothesis.
- N=50, single seed, one benchmark — pilot-scale; no strong statistical claims.
- Answer matching is normalized string equality (no symbolic equivalence); accuracy is approximate but bias applies equally to all 4 configs.
- max_tokens=8192 caps the observable CoT inflation (papers used up to 32K); truncation rate partially absorbs what would otherwise appear as longer traces.

---

# Round 2 (pre-registered, NOT YET RUN)

Written after the pilot, before any Round-2 generation. Status of every number below: **planned, not measured.**

## What is actually weak in the pilot (ranked by how much it threatens the claim)

| # | Defect | Evidence it matters | Fix |
|---|---|---|---|
| 1 | **Truncation censoring** | 64% (arm A) and 32% (arm B) of 3-bit generations hit the 8192-token cap. Token inflation is a *lower bound*, and "accuracy" partly measures "did it finish", not "did it reason correctly". This confounds the headline metric. | Raise cap to 32768. Report the uncensored inflation and the fraction still censored. |
| 2 | **Underpowered** | Paired difference d = 0.284 (mean 919, sd 3239, n=50) → 95% CI [19, 1790] barely excludes 0. Power analysis: **n ≈ 98 for 80% power, n ≈ 131 for 90%** at the observed effect size. | Run all 500 MATH-500 problems (~4× the 90%-power requirement, and removes the "first 50" selection concern). |
| 3 | **Single seed** | No estimate of run-to-run variance; the whole effect could be seed noise. | 3 seeds per configuration; report between-seed spread alongside the paired CI. |
| 4 | **Answer matching is string equality** | `norm_ans` cannot see that $0.5$ and $\frac{1}{2}$ agree. Biases accuracy *down* by an unknown amount, possibly unequally across arms. | Add `math_verify` / SymPy equivalence; report both matchers so the pilot's numbers stay comparable. |
| 5 | **RTN group quantization only** | The literature's effects are reported for AWQ/GPTQ; MLX group quantization is round-to-nearest. Generalization across quantizers is assumed, not shown. | Add a GPTQ or AWQ 3-bit arm (needs a CUDA box; not runnable on this machine). |
| 6 | **Arm B is distilled+RL, not never-distilled** | The confound is only *partially* broken: both arms saw distillation. | Add a natively-RL-trained arm if a genuinely non-distilled small reasoning model becomes available. Currently blocked — no such open model identified. |

Defects 1–4 are runnable locally. 5–6 are not, and should stay disclosed limitations.

## Measured cost basis (from the pilot, not estimated)

Pilot: **85.9 min wall for 200 generations** on Apple M5 Max / mlx-lm 0.31.3 — 111 tok/s at BF16, ~245 tok/s at 3-bit, 25.8 s/generation averaged.

Projected Round-2 cost (**projection, not a measurement** — and biased low, because raising the cap lets currently-truncated generations run longer):

| Configuration | Generations | Projected wall @ 8K cap | With 32K cap (rough) |
|---|---|---|---|
| 150 problems × 4 configs × 1 seed | 600 | ~4.3 h | ~7–9 h |
| 500 problems × 4 configs × 1 seed | 2,000 | ~14.3 h | ~24–30 h |
| 500 problems × 4 configs × 3 seeds | 6,000 | ~43 h | ~3–4 days |

## Recommended Round 2 (the smallest run that fixes the real problems)

**500 MATH-500 problems × 2 models × 2 precisions × 1 seed, cap 32768, dual answer-matcher.** Fixes defects 1, 2, and 4 — the three that actually threaten the claim — and defers seeds (3) to a Round 3 only if the effect survives. Roughly a day of local compute.

## Pre-registered hypothesis and stopping rule (fixed before running)

- **H1 (unchanged):** at matched quantization, the purely distilled arm shows greater quantization-induced token inflation than the RL arm.
- **Primary endpoint:** mean paired per-problem inflation difference (A − B), 95% bootstrap CI, 10k resamples, seed 0 — same estimator as the pilot.
- **Decision rule, committed in advance:**
  - CI excludes 0 with the same sign → upgrade the paper's language from "suggestive" to "supported at N=500, single seed".
  - CI includes 0 → **report the null**, and revise §7 to say the pilot effect did not survive scaling. This is the outcome the pilot's weak CI makes genuinely plausible and it must be published either way.
  - Sign flips → report as a failed replication of our own pilot.
- **No metric will be added, dropped, or redefined after seeing Round-2 results.** Any post-hoc analysis will be labeled exploratory.
- The pilot numbers stay in the paper regardless; Round 2 is reported alongside, not as a replacement.

## Round-2 design REVISION (made after a 3-problem smoke test, before any Round-2 outcome data)

**Disclosure:** this revision was made after inspecting generations for MATH-500 problems 0-2 on arm A 3-bit. That is outcome data on 3 problems. It was inspected for *mechanism*, not for the primary endpoint, and the primary endpoint's estimator is unchanged. Recorded here rather than presented as if pre-planned.

**What the smoke test showed.** Running arm A 3-bit with the planned 32768-token cap:
- problem 0 → 30,750 tokens in 164 s; problem 1 → 32,768 tokens (hit cap) in 175 s.
- Inspecting the 30,750-token generation: repetition begins around 25-30% of the way through, and the final ~70% is literal degenerate loop (tail is `. [0. [0. [0. ...` repeated).

**Why this kills planned fix #1 ("raise the cap to 32K").** Raising the cap does not uncensor a longer *reasoning* trace; it lets a degenerate loop run 4x longer. CTIR would then largely measure *the cap we chose*, not the model's reasoning. The pilot's 8192 cap was censoring, but uncensoring it naively makes the headline metric less meaningful, not more.

**Revised design:**
1. Cap = **16384** (2x pilot), not 32768.
2. **Online loop-aware early stopping** (`run_eval2.py`): generation halts once a 20-gram repeats >=4 times in the trailing 600-word window - the same loop definition `analyze.py` already used offline, so online and offline agree by construction. Records `loop_stopped` and `loop_at_token`.
3. **Primary endpoint (unchanged estimator):** mean paired per-problem inflation difference (A - B), 95% bootstrap CI, 10k resamples, seed 0 - but computed on generations that did **not** stop on a loop, so it measures reasoning length rather than loop duration.
4. **Co-primary endpoint (promoted from secondary):** loop rate per arm. The smoke test makes clear this is the dominant mechanism for the distilled arm, not a side observation.
5. `loop_at_token` gives a new, cheap measurement the pilot could not make: *when* the collapse starts.

**Oracle checks on the online detector (run before launch, all passed):** no fire on clean prose; fires on a synthetic loop; no fire on short text; agrees with the offline detector on the smoke generation; on the pilot's 50 arm-A BF16 generations, online and offline disagree once (idx 12), where offline flags a mid-text divisor enumeration and online correctly does not fire - the generation ends with a valid `\boxed{}` answer. The online rule is therefore conservative: it does not cut healthy generations short.

**Token accounting verified comparable to the pilot:** streamed token count vs `len(tokenizer.encode(text))` differs by <=0.06% on the smoke generations.

**Revised cost (measured on 3 problems, so a rough basis):** ~17 s/problem for 3-bit arms with early stopping, vs 164-175 s without. Projected full run 500 problems x 4 configs: **~10-16 h**, down from the ~24-30 h projected for the naive 32K design - and the resulting numbers are more interpretable.

**Unchanged:** hypothesis H1, the decision rule (including the obligation to report a null), and the commitment not to redefine metrics after seeing Round-2 outcomes.

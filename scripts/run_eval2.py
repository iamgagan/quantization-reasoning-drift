#!/usr/bin/env python
"""Round-2 runner: MATH-500 through an MLX model with ONLINE LOOP DETECTION.

Difference from run_eval.py (the pilot runner):
  * cap raised 8192 -> 16384 (configurable)
  * generation stops early when a degenerate repetition loop is detected, so
    "CoT length" measures reasoning rather than how long the loop ran before
    hitting an arbitrary cap.

Records, per problem: token count, whether it stopped on a loop, where the loop
started, and whether it hit the hard cap. Checkpoints per item; resumable.
"""
import argparse, json, os, time
from collections import Counter

# --- online loop detector -----------------------------------------------------
# Same shape as analyze.py's loop_detect (20-grams, >=4 repeats) so offline and
# online definitions agree; applied to a trailing window for cost.
NGRAM, REPS, WINDOW_WORDS = 20, 4, 600


def looping(words):
    if len(words) < NGRAM * REPS:
        return False
    w = words[-WINDOW_WORDS:]
    g = Counter(tuple(w[i:i + NGRAM]) for i in range(len(w) - NGRAM))
    return bool(g) and g.most_common(1)[0][1] >= REPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--check-every", type=int, default=256, help="tokens between loop checks")
    ap.add_argument("--no-loop-stop", action="store_true", help="disable early stop (ablation)")
    ap.add_argument("--out-dir", default="results_r2")
    args = ap.parse_args()

    from datasets import load_dataset
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.tag}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                pass
    print(f"[{args.tag}] resuming; {len(done)} already done", flush=True)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(args.n))
    model, tokenizer = load(args.model_path)
    sampler = make_sampler(temp=args.temp, top_p=args.top_p)
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."

    for idx in range(args.n):
        if idx in done:
            continue
        item = ds[idx]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["problem"] + suffix}],
            add_generation_prompt=True, tokenize=False)

        t0 = time.time()
        chunks, n_tok, loop_stop, loop_at = [], 0, False, None
        for resp in stream_generate(model, tokenizer, prompt=prompt,
                                    max_tokens=args.max_tokens, sampler=sampler):
            chunks.append(resp.text)
            n_tok += 1
            if (not args.no_loop_stop) and n_tok % args.check_every == 0:
                if looping("".join(chunks).split()):
                    loop_stop, loop_at = True, n_tok
                    break
        text = "".join(chunks)
        dt = time.time() - t0

        rec = {
            "idx": idx, "unique_id": item["unique_id"], "level": item["level"],
            "gold": item["answer"], "gen_tokens": n_tok, "gen_seconds": round(dt, 1),
            "truncated": (not loop_stop) and n_tok >= args.max_tokens - 8,
            "loop_stopped": loop_stop, "loop_at_token": loop_at,
            "max_tokens": args.max_tokens, "text": text,
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        flag = f" LOOP@{loop_at}" if loop_stop else (" CAP" if rec["truncated"] else "")
        print(f"[{args.tag}] {idx+1}/{args.n} tok={n_tok} {dt:.0f}s{flag}", flush=True)
    print(f"[{args.tag}] DONE", flush=True)


if __name__ == "__main__":
    main()

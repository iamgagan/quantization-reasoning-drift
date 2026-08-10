#!/usr/bin/env python
"""Run MATH-500 subset through an MLX model, checkpointing per-item results to JSONL."""
import argparse, json, os, time, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    from datasets import load_dataset
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    out_path = os.path.join(args.out_dir, f"{args.tag}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["idx"])
                except Exception:
                    pass
    print(f"[{args.tag}] resuming; {len(done)} items already done", flush=True)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(args.n))
    model, tokenizer = load(args.model_path)
    sampler = make_sampler(temp=args.temp, top_p=args.top_p)

    prompt_suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    for idx in range(args.n):
        if idx in done:
            continue
        item = ds[idx]
        messages = [{"role": "user", "content": item["problem"] + prompt_suffix}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        t0 = time.time()
        text = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens,
                        sampler=sampler, verbose=False)
        dt = time.time() - t0
        n_tok = len(tokenizer.encode(text))
        rec = {
            "idx": idx, "unique_id": item["unique_id"], "level": item["level"],
            "gold": item["answer"], "gen_tokens": n_tok, "gen_seconds": round(dt, 1),
            "truncated": n_tok >= args.max_tokens - 8, "text": text,
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{args.tag}] {idx+1}/{args.n} tokens={n_tok} time={dt:.0f}s", flush=True)
    print(f"[{args.tag}] DONE", flush=True)

if __name__ == "__main__":
    main()

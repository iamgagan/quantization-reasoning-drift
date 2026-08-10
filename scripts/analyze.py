#!/usr/bin/env python
"""Analyze run_eval.py JSONL outputs: accuracy, CoT length, CTIR, marker rate, loops, answer-in-trace."""
import json, os, re, sys, glob
from collections import Counter

MARKERS = ["wait", "but", "alternatively", "however", "hmm", "maybe", "perhaps",
           "actually", "let me check", "let me verify", "re-check", "reconsider",
           "double-check", "hold on"]

def norm_ans(s):
    if s is None: return ""
    s = s.strip()
    s = re.sub(r"\\left|\\right|\\!|\\,|\\;|\\ ", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = re.sub(r"\s+", "", s)
    s = s.strip("$")
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", s)
    s = s.rstrip(".")
    return s.lower()

def extract_boxed(text):
    # last \boxed{...} with balanced braces
    starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
    if not starts: return None
    st = starts[-1]
    depth, i = 1, st
    while i < len(text) and depth > 0:
        if text[i] == "{": depth += 1
        elif text[i] == "}": depth -= 1
        i += 1
    return text[st:i-1] if depth == 0 else None

def marker_count(text):
    t = text.lower()
    c = 0
    for m in MARKERS:
        if " " in m or "-" in m:
            c += t.count(m)
        else:
            c += len(re.findall(r"\b" + re.escape(m) + r"\b", t))
    return c

def loop_detect(text, n=20, reps=4, window=1024):
    words = text.split()
    if len(words) < n * reps: return False
    for start in range(0, max(1, len(words) - window), window // 2):
        w = words[start:start + window]
        grams = Counter(tuple(w[i:i+n]) for i in range(len(w) - n))
        if grams and grams.most_common(1)[0][1] >= reps:
            return True
    return False

def analyze(path):
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    res = []
    for r in rows:
        pred = norm_ans(extract_boxed(r["text"]))
        gold = norm_ans(r["gold"])
        correct = (pred == gold) and pred != ""
        mk = marker_count(r["text"])
        res.append({
            "idx": r["idx"], "correct": correct, "tokens": r["gen_tokens"],
            "truncated": r["truncated"], "markers": mk,
            "marker_rate": 1000.0 * mk / max(1, r["gen_tokens"]),
            "loop": loop_detect(r["text"]),
            "gold_in_trace": gold != "" and (gold in norm_ans_text(r["text"])),
        })
    acc = sum(x["correct"] for x in res) / n
    mean_tok = sum(x["tokens"] for x in res) / n
    mean_mk = sum(x["marker_rate"] for x in res) / n
    trunc = sum(x["truncated"] for x in res) / n
    loops = sum(x["loop"] for x in res) / n
    wrong = [x for x in res if not x["correct"]]
    ans_in_trace_wrong = (sum(x["gold_in_trace"] for x in wrong) / len(wrong)) if wrong else 0.0
    return {"file": os.path.basename(path), "n": n, "accuracy": round(acc, 3),
            "mean_tokens": round(mean_tok, 1), "marker_per_1k": round(mean_mk, 2),
            "truncated_frac": round(trunc, 3), "loop_frac": round(loops, 3),
            "gold_in_trace_among_wrong": round(ans_in_trace_wrong, 3)}, res

def norm_ans_text(text):
    # normalize whole text the same way for substring search
    t = re.sub(r"\\left|\\right|\\!|\\,|\\;|\\ ", "", text)
    t = t.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    t = re.sub(r"\s+", "", t)
    return t.lower()

if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob("results/*.jsonl"))
    summaries = []
    per_item = {}
    for p in paths:
        s, res = analyze(p)
        summaries.append(s)
        per_item[s["file"]] = res
    print(json.dumps(summaries, indent=2))
    # CTIR pairing: tag convention <model>-<prec>.jsonl
    by = {s["file"].replace(".jsonl", ""): s for s in summaries}
    for m in ["r1d", "dsr"]:
        b, q = by.get(f"{m}-bf16"), by.get(f"{m}-3bit")
        if b and q:
            ctir = 100.0 * (q["mean_tokens"] - b["mean_tokens"]) / b["mean_tokens"]
            dmk = q["marker_per_1k"] - b["marker_per_1k"]
            print(f"{m}: CTIR={ctir:+.1f}%  delta_marker_per_1k={dmk:+.2f}  "
                  f"acc {b['accuracy']:.3f}->{q['accuracy']:.3f}")

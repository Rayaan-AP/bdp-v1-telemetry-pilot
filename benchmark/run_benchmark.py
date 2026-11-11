#!/usr/bin/env python3
"""
BDP-v1 Telemetry Pilot
======================
A paired benchmark of dense key-value telemetry (BDP-v1) versus verbose
human-prose handoffs between three LLM agents.

Conditions
----------
prose : orchestrator writes a 150-220 word brief; strategist writes a
        150-220 word analysis; verifier writes a 150-220 word review.
bdp   : orchestrator emits a <=45-token BDP-v1 spec; strategist emits a
        <=110-token BDP-v1 packet; verifier emits a <=70-token packet.
        Each model gets an isolated <think> scratchpad with an ANTI-RUSH
        instruction: "compress the syntax, never the reasoning".

Providers
---------
Gemini Flash : any CLI command that accepts a prompt as its final argument.
               Set GEMINI_CMD, e.g.:
                 export GEMINI_CMD="gemini -m gemini-3.8-flash-high -p"
DeepSeek     : OpenAI-compatible chat completions.
Qwen         : OpenAI-compatible chat completions.

Environment
-----------
GEMINI_CMD          required, e.g. "gemini -m gemini-3.8-flash-high -p"
DEEPSEEK_API_KEY    required
DEEPSEEK_BASE_URL   default https://api.deepseek.com/v1
DEEPSEEK_MODEL      default deepseek-chat
QWEN_API_KEY        required
QWEN_BASE_URL       required (e.g. your Qwen/DashScope compatible endpoint)
QWEN_MODEL          default qwen3.8-max

Usage
-----
  python benchmark/run_benchmark.py --tasks 24 --runs 1 --workers 4

Output
------
  runs.jsonl      one record per run, including all inter-agent messages
  runs.csv        flat run table
  per_task.csv    paired prose-vs-BDP table
  summary.json    aggregate statistics
  summary.md      human-readable summary
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shlex
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOISE_PREFIXES = (
    "YOLO mode is enabled",
    "Ripgrep is not available",
    "Warning:",
    "Error when talking to Gemini",
    "An unexpected critical error",
    "[AGY-POOL",
    "[BUS]",
)
LOCK = threading.Lock()


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


GEMINI_CMD = env("GEMINI_CMD")
DS_BASE = env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
DS_KEY = env("DEEPSEEK_API_KEY")
DS_MODEL = env("DEEPSEEK_MODEL", "deepseek-chat")
QWEN_BASE = env("QWEN_BASE_URL").rstrip("/")
QWEN_KEY = env("QWEN_API_KEY")
QWEN_MODEL = env("QWEN_MODEL", "qwen3.8-max")


# --------------------------------------------------------------------------
# token estimate
# --------------------------------------------------------------------------
def token_est(text: str) -> int:
    """Deterministic token estimate: words + punctuation, x1.15.

    The same estimator is applied to both conditions, so the ratio is
    comparable. API usage is also recorded exactly for the HTTP providers.
    """
    if not text:
        return 0
    words = re.findall(r"\w+|[^\w\s]", text)
    return max(1, int(len(words) * 1.15))


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------
def chat(base: str, key: str, model: str, messages: list[dict],
         max_tokens: int = 500, timeout: int = 90) -> dict:
    if not base or not key:
        raise RuntimeError("Missing OpenAI-compatible base URL or API key")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        msg = data["choices"][0]["message"]
        text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        return {"text": text, "usage": data.get("usage", {}),
                "latency": round(time.time() - t, 3),
                "model": data.get("model", model)}
    except urllib.error.HTTPError as e:
        return {"text": "", "usage": {}, "latency": round(time.time() - t, 3),
                "error": f"HTTP {e.code}: {e.read().decode()[:400]}"}
    except Exception as e:
        return {"text": "", "usage": {}, "latency": round(time.time() - t, 3),
                "error": repr(e)}


def deepseek(messages: list[dict], max_tokens: int = 500) -> dict:
    return chat(DS_BASE, DS_KEY, DS_MODEL, messages, max_tokens)


def qwen(messages: list[dict], max_tokens: int = 500) -> dict:
    return chat(QWEN_BASE, QWEN_KEY, QWEN_MODEL, messages, max_tokens)


def clean_cli(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(NOISE_PREFIXES):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def gemini(prompt: str, timeout: int = 180) -> dict:
    if not GEMINI_CMD:
        raise RuntimeError("GEMINI_CMD is not set")
    cmd = shlex.split(GEMINI_CMD) + [prompt]
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        raw = (p.stdout or "") + "\n" + (p.stderr or "")
        text = clean_cli(raw)
        return {"text": text, "latency": round(time.time() - t, 3)}
    except subprocess.TimeoutExpired:
        return {"text": "", "latency": round(time.time() - t, 3),
                "error": "timeout"}
    except Exception as e:
        return {"text": "", "latency": round(time.time() - t, 3),
                "error": repr(e)}


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------
def fmt_options(options: dict) -> str:
    return "\n".join(f"{k}) {v}" for k, v in options.items())


def prose_prompts(task: dict) -> tuple[str, str, str, str]:
    opts = fmt_options(task["options"])
    p1 = (
        "You are the ORCHESTRATOR of a 3-agent team (orchestrator, strategist, verifier).\n"
        "Write a detailed, polite briefing to the strategist in 150-220 words of normal "
        "professional prose. Explain the task and constraints, and ask for a recommendation. "
        "Do NOT solve it yourself.\n"
        f"TASK: {task['prompt']}\nOPTIONS:\n{opts}\n"
        "Output only the briefing."
    )
    p2 = (
        "You are the STRATEGIST. Read the orchestrator's briefing and analyze the task.\n"
        "Write a detailed recommendation to the verifier in 150-220 words of normal "
        "professional prose. State your choice and explain the reasoning.\n"
        "BRIEFING:\n{brief}\n"
        f"TASK: {task['prompt']}\nOPTIONS:\n{opts}\n"
        "Output only the recommendation."
    )
    p3 = (
        "You are the VERIFIER. Read the strategist's recommendation and verify it.\n"
        "Write a detailed review to the orchestrator in 150-220 words of normal "
        "professional prose. Say whether you agree, and why.\n"
        "RECOMMENDATION:\n{analysis}\n"
        f"TASK: {task['prompt']}\nOPTIONS:\n{opts}\n"
        "Output only the review."
    )
    p4 = (
        "You are the ORCHESTRATOR. Based on the verifier's review, choose the correct option.\n"
        "Output exactly one line: FINAL: <A|B|C>\n"
        "REVIEW:\n{review}\n"
        f"TASK: {task['prompt']}\nOPTIONS:\n{opts}"
    )
    return p1, p2, p3, p4


def bdp_prompts(task: dict) -> tuple[str, str, str, str]:
    opts = fmt_options(task["options"])
    p1 = (
        "You are the ORCHESTRATOR. Compile the task into ONE BDP-v1 telemetry packet.\n"
        "Output ONLY this line (no prose, max 45 tokens):\n"
        "BDP:v1|INTENT:<short_intent>|TGT:<short_target>|REQ:<key_constraints>|GATE:<check>\n"
        f"TASK: {task['prompt']}\nOPTIONS:\n{opts}"
    )
    p2 = (
        "You are the STRATEGIST. Read the BDP-v1 spec and respond ONLY with:\n"
        "<think>[one line of reasoning]</think>\n"
        "BDP:v1|NODE:STRATEGIST|VOTE:<A|B|C>|ROOT_CAUSE:<short>|VERIFY:<short>\n"
        "Max 110 tokens. No prose.\n"
        "SPEC:\n{spec}\n"
        f"TASK: {task['prompt']}\nOPTIONS:\n{opts}"
    )
    p3 = (
        "You are the VERIFIER. Read the strategist's BDP packet and respond ONLY with:\n"
        "<think>[one line of check]</think>\n"
        "BDP:v1|NODE:VERIFIER|STAT:<PASS|FAIL>|ISSUE:<short>|CONF:<0-100>\n"
        "Max 70 tokens. No prose.\n"
        "PACKET:\n{packet}\n"
        f"TASK: {task['prompt']}\nOPTIONS:\n{opts}"
    )
    p4 = (
        "You are the ORCHESTRATOR. Based on the verifier's BDP packet, choose the correct option.\n"
        "Output exactly one line: FINAL: <A|B|C>\n"
        "PACKET:\n{verification}\n"
        f"TASK: {task['prompt']}\nOPTIONS:\n{opts}"
    )
    return p1, p2, p3, p4


def parse_final(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"FINAL\s*:\s*\**\s*([ABC])", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"^\s*\**\s*([ABC])\s*\**\s*[\).:]", text, re.M)
    if m:
        return m.group(1).upper()
    return None


def schema_ok(condition: str, msgs: dict) -> bool:
    if condition == "prose":
        return True
    spec, packet, ver = msgs["spec"], msgs["packet"], msgs["verification"]
    return (
        "BDP:v1|" in spec
        and all(k in spec for k in ("INTENT", "TGT", "REQ", "GATE"))
        and "BDP:v1|" in packet
        and "VOTE:" in packet
        and "ROOT_CAUSE" in packet
        and "BDP:v1|" in ver
        and "STAT:" in ver
        and "CONF" in ver
    )


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------
def run_one(task: dict, condition: str, run_idx: int = 0) -> dict:
    rid = f"{task['id']}-{condition}-r{run_idx}"
    errors: list[str] = []
    usage = {"orchestrator_calls": 0, "strategist": [], "verifier": []}
    t0 = time.time()

    if condition == "prose":
        p1, p2, p3, p4 = prose_prompts(task)
        r1 = gemini(p1)
        usage["orchestrator_calls"] += 1
        if r1.get("error"):
            errors.append(r1["error"])
        brief = r1["text"]
        r2 = deepseek([{"role": "user", "content": p2.format(brief=brief)}])
        usage["strategist"].append(r2.get("usage", {}))
        if r2.get("error"):
            errors.append(r2["error"])
        analysis = r2["text"]
        r3 = qwen([{"role": "user", "content": p3.format(analysis=analysis)}])
        usage["verifier"].append(r3.get("usage", {}))
        if r3.get("error"):
            errors.append(r3["error"])
        review = r3["text"]
        inter = {"brief": brief, "analysis": analysis, "review": review}
        r4 = gemini(p4.format(review=review))
        usage["orchestrator_calls"] += 1
        if r4.get("error"):
            errors.append(r4["error"])
        final_raw = r4["text"]
    else:
        p1, p2, p3, p4 = bdp_prompts(task)
        r1 = gemini(p1)
        usage["orchestrator_calls"] += 1
        if r1.get("error"):
            errors.append(r1["error"])
        spec = r1["text"]
        r2 = deepseek([{"role": "user", "content": p2.format(spec=spec)}], max_tokens=180)
        usage["strategist"].append(r2.get("usage", {}))
        if r2.get("error"):
            errors.append(r2["error"])
        packet = r2["text"]
        r3 = qwen([{"role": "user", "content": p3.format(packet=packet)}], max_tokens=120)
        usage["verifier"].append(r3.get("usage", {}))
        if r3.get("error"):
            errors.append(r3["error"])
        verification = r3["text"]
        inter = {"spec": spec, "packet": packet, "verification": verification}
        r4 = gemini(p4.format(verification=verification))
        usage["orchestrator_calls"] += 1
        if r4.get("error"):
            errors.append(r4["error"])
        final_raw = r4["text"]

    final = parse_final(final_raw)
    return {
        "run_id": rid,
        "task_id": task["id"],
        "category": task["category"],
        "condition": condition,
        "run": run_idx,
        "inter_messages": inter,
        "inter_chars": sum(len(v) for v in inter.values()),
        "inter_tokens_est": sum(token_est(v) for v in inter.values()),
        "final_raw": final_raw,
        "final_answer": final,
        "correct": bool(final == task["answer"]) if final else False,
        "answer_key": task["answer"],
        "correct_text": task["options"][task["answer"]],
        "schema_ok": schema_ok(condition, inter),
        "latency_s": round(time.time() - t0, 2),
        "api_usage": usage,
        "errors": errors,
    }


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
def summarize(records: list[dict], outdir: Path) -> dict:
    import numpy as np
    from scipy import stats

    by = {(r["task_id"], r["condition"], r["run"]): r for r in records}
    task_ids = sorted({r["task_id"] for r in records})
    reductions, prose_tok, bdp_tok = [], [], []
    prose_acc, bdp_acc = [], []
    for tid in task_ids:
        runs = sorted({r["run"] for r in records if r["task_id"] == tid})
        for run in runs:
            p = by.get((tid, "prose", run))
            b = by.get((tid, "bdp", run))
            if not p or not b or not p["inter_tokens_est"]:
                continue
            reductions.append(1 - b["inter_tokens_est"] / p["inter_tokens_est"])
            prose_tok.append(p["inter_tokens_est"])
            bdp_tok.append(b["inter_tokens_est"])
            prose_acc.append(1 if p["correct"] else 0)
            bdp_acc.append(1 if b["correct"] else 0)

    def ci(vals: list[float], n: int = 5000) -> list[float]:
        arr = np.array(vals)
        rng = np.random.default_rng(42)
        means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)]
        return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

    summary = {
        "n_pairs": len(reductions),
        "n_runs": len(records),
        "tasks": len(task_ids),
        "reduction_mean_pct": round(100 * float(np.mean(reductions)), 2) if reductions else None,
        "reduction_median_pct": round(100 * float(np.median(reductions)), 2) if reductions else None,
        "reduction_ci95": [round(100 * x, 2) for x in ci(reductions)] if reductions else None,
        "prose_tokens_mean": round(float(np.mean(prose_tok)), 1) if prose_tok else None,
        "bdp_tokens_mean": round(float(np.mean(bdp_tok)), 1) if bdp_tok else None,
        "prose_accuracy_pct": round(100 * float(np.mean(prose_acc)), 1) if prose_acc else None,
        "bdp_accuracy_pct": round(100 * float(np.mean(bdp_acc)), 1) if bdp_acc else None,
        "schema_ok_pct": round(
            100 * sum(1 for r in records if r["condition"] == "bdp" and r["schema_ok"])
            / max(1, sum(1 for r in records if r["condition"] == "bdp")), 1),
        "prose_latency_mean_s": round(float(np.mean(
            [r["latency_s"] for r in records if r["condition"] == "prose"])), 1),
        "bdp_latency_mean_s": round(float(np.mean(
            [r["latency_s"] for r in records if r["condition"] == "bdp"])), 1),
        "prose_errors": sum(1 for r in records if r["condition"] == "prose" and r["errors"]),
        "bdp_errors": sum(1 for r in records if r["condition"] == "bdp" and r["errors"]),
        "prose_correct": sum(1 for r in records if r["condition"] == "prose" and r["correct"]),
        "bdp_correct": sum(1 for r in records if r["condition"] == "bdp" and r["correct"]),
    }
    if reductions:
        summary["wilcoxon_p"] = float(stats.wilcoxon(reductions).pvalue) if len(reductions) > 5 else None

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# BDP-v1 Telemetry Pilot\n"]
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v}")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")

    with open(outdir / "per_task.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "category", "prose_tokens", "bdp_tokens",
                    "reduction_pct", "prose_correct", "bdp_correct",
                    "bdp_schema_ok", "prose_latency_s", "bdp_latency_s"])
        for tid in task_ids:
            p = by.get((tid, "prose", 0))
            b = by.get((tid, "bdp", 0))
            if not p or not b or not p["inter_tokens_est"]:
                continue
            w.writerow([tid, p["category"], p["inter_tokens_est"],
                        b["inter_tokens_est"],
                        round(100 * (1 - b["inter_tokens_est"] / p["inter_tokens_est"]), 2),
                        p["correct"], b["correct"], b["schema_ok"],
                        p["latency_s"], b["latency_s"]])

    with open(outdir / "runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "task_id", "category", "condition",
                    "inter_tokens_est", "inter_chars", "final_answer",
                    "correct", "schema_ok", "latency_s", "errors"])
        for r in sorted(records, key=lambda x: (x["task_id"], x["condition"])):
            w.writerow([r["run_id"], r["task_id"], r["category"], r["condition"],
                        r["inter_tokens_est"], r["inter_chars"], r["final_answer"],
                        r["correct"], r["schema_ok"], r["latency_s"],
                        "; ".join(r.get("errors") or [])])
    return summary


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="BDP-v1 paired telemetry pilot")
    ap.add_argument("--tasks", type=int, default=24, help="number of tasks")
    ap.add_argument("--runs", type=int, default=1, help="runs per task/condition")
    ap.add_argument("--workers", type=int, default=4, help="parallel runs")
    ap.add_argument("--out", default=str(HERE.parent / "results"), help="output directory")
    ap.add_argument("--tasks-file", default=str(HERE / "tasks.json"))
    ap.add_argument("--resume", action="store_true", help="skip run_ids already in runs.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tasks = json.loads(Path(args.tasks_file).read_text())[: args.tasks]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fpath = outdir / "runs.jsonl"

    existing = []
    done_ids = set()
    if args.resume and fpath.exists():
        for line in fpath.read_text().splitlines():
            try:
                rec = json.loads(line)
                existing.append(rec)
                done_ids.add(rec["run_id"])
            except Exception:
                pass

    wanted = []
    for task in tasks:
        for run in range(args.runs):
            for cond in ("prose", "bdp"):
                rid = f"{task['id']}-{cond}-r{run}"
                if rid not in done_ids:
                    wanted.append((task, cond, run))
    random.Random(args.seed).shuffle(wanted)

    print(f"[bench] tasks={len(tasks)} runs={len(wanted)} workers={args.workers}")
    new_records = []
    with open(fpath, "a") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, t, c, r): (t, c, r) for t, c, r in wanted}
            done = 0
            for fut in as_completed(futs):
                task, cond, run = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"run_id": f"{task['id']}-{cond}-r{run}",
                           "task_id": task["id"], "category": task["category"],
                           "condition": cond, "run": run, "errors": [repr(e)],
                           "inter_tokens_est": 0, "inter_chars": 0,
                           "correct": False, "schema_ok": False,
                           "latency_s": 0, "api_usage": {}, "inter_messages": {},
                           "final_raw": "", "final_answer": None,
                           "answer_key": task["answer"],
                           "correct_text": task["options"][task["answer"]]}
                new_records.append(rec)
                with LOCK:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                done += 1
                print(f"[{done}/{len(wanted)}] {rec['run_id']} "
                      f"tokens={rec['inter_tokens_est']} correct={rec['correct']} "
                      f"err={len(rec.get('errors') or [])}")

    summary = summarize(existing + new_records, outdir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

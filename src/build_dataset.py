"""Assemble the final dataset: download -> normalize -> curate -> mix hard negatives -> dedup -> split.

Outputs (under data/out/):
  train.jsonl / val.jsonl / test.jsonl   -- canonical examples (see format_utils)
  train_pc.jsonl                          -- prompt/completion rows for Adaption upload
  stats.json                              -- counts by source / hard-negative kind (for the model card)

Run:  python -m src.build_dataset --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List

import yaml

from . import dedup, hard_negatives, multiturn, schema_drift
from .format_utils import to_prompt_completion
from .schema_validator import validate_answer


# --------------------------------------------------------------------------------------
# Source adapters. Each returns a list of canonical examples with meta.source set.
# The field layouts of xLAM / ToolACE vary across versions, so these are defensive and
# skip rows they can't parse rather than crashing.
# --------------------------------------------------------------------------------------
def _load_json_field(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def load_xlam(repo: str, limit: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(repo, split="train")
    out: List[Dict[str, Any]] = []
    for row in ds:
        tools = _load_json_field(row.get("tools"))
        answers = _load_json_field(row.get("answers"))
        query = row.get("query")
        if not (tools and answers and query):
            continue
        calls = [{"name": a.get("name"), "arguments": a.get("arguments", {})} for a in answers]
        out.append(
            {
                "tools": _normalize_tools(tools),
                "query": query,
                "answer": {"type": "tool_call", "calls": calls},
                "meta": {"source": "xlam", "hn_kind": None},
            }
        )
        if len(out) >= limit:
            break
    print(f"[build] xLAM: loaded {len(out)}")
    return out


def load_toolace(repo: str, limit: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(repo, split="train")
    out: List[Dict[str, Any]] = []
    for row in ds:
        tools = _load_json_field(row.get("system") or row.get("tools"))
        convs = row.get("conversations")
        if not convs:
            continue
        # ToolACE stores multi-turn conversations; take single-turn tool-call pairs.
        user_msg, tool_msg = None, None
        for turn in convs:
            role = turn.get("from") or turn.get("role")
            content = turn.get("value") or turn.get("content")
            if role in ("user", "human"):
                user_msg = content
            elif role in ("assistant", "gpt", "function_call") and user_msg:
                tool_msg = content
                break
        parsed_calls = _load_json_field(tool_msg) if tool_msg else None
        if not (tools and user_msg and parsed_calls):
            continue
        if isinstance(parsed_calls, dict):
            parsed_calls = [parsed_calls]
        calls = [
            {"name": c.get("name"), "arguments": c.get("arguments", c.get("parameters", {}))}
            for c in parsed_calls
            if isinstance(c, dict) and c.get("name")
        ]
        if not calls:
            continue
        out.append(
            {
                "tools": _normalize_tools(tools),
                "query": user_msg,
                "answer": {"type": "tool_call", "calls": calls},
                "meta": {"source": "toolace", "hn_kind": None},
            }
        )
        if len(out) >= limit:
            break
    print(f"[build] ToolACE: loaded {len(out)}")
    return out


def load_toucan(repo: str, limit: int) -> List[Dict[str, Any]]:
    """Toucan-1.5M (Apache-2.0): real MCP-server tool-calling trajectories. Defensive field parsing.

    Layouts vary; we extract the first user query, the available tools, and the first assistant tool
    call. Rows we can't parse are skipped rather than crashing.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(repo, split="train", streaming=True)
    except Exception as e:  # gated / offline / renamed
        print(f"[build] Toucan skipped ({e})")
        return []
    out: List[Dict[str, Any]] = []
    for row in ds:
        tools = _load_json_field(row.get("tools") or row.get("available_tools") or row.get("functions"))
        msgs = row.get("messages") or row.get("conversations")
        if not (tools and msgs):
            continue
        user_msg, calls = None, None
        for turn in msgs:
            role = turn.get("role") or turn.get("from")
            content = turn.get("content") or turn.get("value")
            if role in ("user", "human"):
                user_msg = content
            elif role in ("assistant", "gpt"):
                tc = turn.get("tool_calls") or _load_json_field(content)
                if tc and user_msg:
                    calls = tc
                    break
        if not (user_msg and calls):
            continue
        if isinstance(calls, dict):
            calls = [calls]
        norm_calls = []
        for c in calls:
            fn = c.get("function", c) if isinstance(c, dict) else {}
            name = fn.get("name")
            if name:
                args = fn.get("arguments", fn.get("parameters", {}))
                norm_calls.append({"name": name, "arguments": _load_json_field(args) or {}})
        if not norm_calls:
            continue
        out.append(
            {
                "tools": _normalize_tools(tools),
                "query": user_msg,
                "answer": {"type": "tool_call", "calls": norm_calls},
                "meta": {"source": "toucan", "hn_kind": None},
            }
        )
        if len(out) >= limit:
            break
    print(f"[build] Toucan: loaded {len(out)}")
    return out


def _normalize_tools(tools: Any) -> List[Dict[str, Any]]:
    """Coerce various tool encodings into [{name, description, parameters}]."""
    if isinstance(tools, dict):
        tools = [tools]
    norm: List[Dict[str, Any]] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        # OpenAI style: {"type":"function","function":{...}}
        fn = t.get("function", t)
        name = fn.get("name")
        if not name:
            continue
        norm.append(
            {
                "name": name,
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return norm


# --------------------------------------------------------------------------------------
# Curation
# --------------------------------------------------------------------------------------
def curate_positives(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only schema-valid, well-formed positive examples."""
    kept = []
    for ex in examples:
        if not ex["tools"]:
            continue
        ok, _ = validate_answer(ex["answer"], ex["tools"])
        if ok:
            kept.append(ex)
    print(f"[build] curate positives: {len(examples)} -> {len(kept)} schema-valid")
    return kept


def harvest_tool_pool(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Unique tool schemas (by name) to seed the hard-negative generator."""
    seen: Dict[str, Dict[str, Any]] = {}
    for ex in examples:
        for t in ex["tools"]:
            seen.setdefault(t["name"], t)
    return list(seen.values())


def carve_novel_tools(
    positives: List[Dict[str, Any]], frac: float, seed: int
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], set]:
    """Hold out a fraction of tool NAMES; route positives that use ONLY held-out tools to a novel-test.

    Guarantees the novel-test measures generalization to tools never seen in training. Returns
    (train_pool_positives, novel_test_positives, novel_tool_names).
    """
    rng = random.Random(seed)
    all_names = sorted({t["name"] for ex in positives for t in ex["tools"]})
    rng.shuffle(all_names)
    n_novel = int(len(all_names) * frac)
    novel = set(all_names[:n_novel])

    train_pool, novel_test = [], []
    for ex in positives:
        names = {t["name"] for t in ex["tools"]}
        if names and names.issubset(novel):
            novel_test.append(ex)
        elif names & novel:
            # mixed: drop it so novel tools never leak into training context
            continue
        else:
            train_pool.append(ex)
    print(
        f"[build] novel-tool holdout: {len(novel)} tool names held out -> "
        f"{len(novel_test)} novel-test examples, {len(train_pool)} kept for train pool"
    )
    return train_pool, novel_test, novel


def split(
    examples: List[Dict[str, Any]], ratios: Dict[str, float], seed: int
) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = cfg["seed"]
    dcfg = cfg["dataset"]
    out_dir = cfg["paths"]["out_dir"]

    # 1. Load sources
    positives: List[Dict[str, Any]] = []
    positives += load_xlam(dcfg["sources"]["xlam"]["repo"], dcfg["sources"]["xlam"]["max_examples"])
    positives += load_toolace(
        dcfg["sources"]["toolace"]["repo"], dcfg["sources"]["toolace"]["max_examples"]
    )
    toucan_cfg = dcfg["sources"].get("toucan", {})
    if toucan_cfg.get("enabled"):
        positives += load_toucan(toucan_cfg["repo"], toucan_cfg["max_examples"])

    # 2. Curate + cap positives
    positives = curate_positives(positives)
    random.Random(seed).shuffle(positives)
    positives = positives[: dcfg["target_positive"]]

    # 2b. Optional quality filter (keep top fraction) — heuristic (offline) or Claude LLM-as-judge
    keep_frac = dcfg.get("quality_keep_frac", 1.0)
    if keep_frac < 1.0:
        from .quality_filter import filter_examples

        score_fn = None
        if dcfg.get("quality_judge") == "claude":
            from .claude_judge import make_claude_judge

            score_fn = make_claude_judge(model=dcfg.get("quality_judge_model", "claude-opus-4-8"))
            print("[build] quality filter: Claude LLM-as-judge")
        positives = filter_examples(positives, keep_frac, score_fn=score_fn, seed=seed)

    # 2c. Novel-tool holdout (generalization test) — carve BEFORE building the training tool pool
    novel_frac = dcfg.get("novel_tool_frac", 0.0)
    novel_test: List[Dict[str, Any]] = []
    if novel_frac > 0:
        positives, novel_test, novel_names = carve_novel_tools(positives, novel_frac, seed)

    # 3. Hard negatives + multi-turn from the TRAIN tool pool only (novel tools stay unseen)
    pool = harvest_tool_pool(positives)
    base_n = len(positives)

    hn_ratio = dcfg["hard_negative_ratio"]
    n_hard = int(base_n * hn_ratio / max(1 - hn_ratio, 1e-6))
    hard = hard_negatives.generate(pool, n_hard, cfg["hard_negatives"]["kinds"], seed=seed)
    print(f"[build] hard negatives: requested {n_hard}, got {len(hard)}")

    mt_ratio = dcfg.get("multiturn_ratio", 0.0)
    mt: List[Dict[str, Any]] = []
    if mt_ratio > 0:
        n_mt = int(base_n * mt_ratio / max(1 - mt_ratio, 1e-6))
        mt = multiturn.generate(
            pool, n_mt, cfg["multiturn"]["kinds"], seed=seed,
            long_context_size=cfg["multiturn"].get("long_context_size", 10),
        )
        print(f"[build] multi-turn: requested {n_mt}, got {len(mt)}")

    sd_ratio = dcfg.get("schema_drift_ratio", 0.0)
    sd: List[Dict[str, Any]] = []
    if sd_ratio > 0:
        n_sd = int(base_n * sd_ratio / max(1 - sd_ratio, 1e-6))
        sd = schema_drift.generate(pool, n_sd, cfg["schema_drift"]["kinds"], seed=seed)
        print(f"[build] schema-drift: requested {n_sd}, got {len(sd)}")

    # 4. Combine + dedup
    combined = positives + hard + mt + sd
    combined = dedup.dedup_all(
        combined,
        minhash_threshold=cfg["dedup"]["minhash_threshold"],
        semantic_threshold=cfg["dedup"]["semantic_threshold"],
    )

    # 5. Split, then kill cross-split leakage from train
    parts = split(combined, dcfg["splits"], seed)
    if cfg["dedup"]["check_cross_split"]:
        parts["train"] = dedup.drop_cross_split_leaks(
            parts["train"], [parts["val"], parts["test"]],
            threshold=cfg["dedup"]["semantic_threshold"],
        )

    # 6. Write canonical + prompt/completion (chat template applied lazily at train time,
    #    here we emit a template-free prompt/completion that Adaption can also consume).
    for name, rows in parts.items():
        write_jsonl(os.path.join(out_dir, f"{name}.jsonl"), rows)
    pc_rows = [to_prompt_completion(ex) for ex in parts["train"]]
    write_jsonl(os.path.join(out_dir, "train_pc.jsonl"), pc_rows)
    if novel_test:
        write_jsonl(os.path.join(out_dir, "test_novel.jsonl"), novel_test)

    # 7. Stats for the model card
    stats = {"total": sum(len(v) for v in parts.values()), "novel_test": len(novel_test)}
    for name, rows in parts.items():
        stats[name] = {
            "n": len(rows),
            "by_source": dict(Counter(r["meta"]["source"] for r in rows)),
            "by_hn_kind": dict(Counter(r["meta"].get("hn_kind") for r in rows if r["meta"].get("hn_kind"))),
            "by_mt_kind": dict(Counter(r["meta"].get("mt_kind") for r in rows if r["meta"].get("mt_kind"))),
            "by_sd_kind": dict(Counter(r["meta"].get("sd_kind") for r in rows if r["meta"].get("sd_kind"))),
        }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("[build] done:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

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
import re
from collections import Counter
from typing import Any, Dict, List, Optional

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


def _toolace_tools(system: str) -> List[Dict[str, Any]]:
    """Tools are a JSON array embedded in ToolACE's system string after '... invoke:'."""
    if not system:
        return []
    start = system.find("[{")
    if start == -1:
        return []
    try:
        arr, _ = json.JSONDecoder().raw_decode(system[start:])
        return arr if isinstance(arr, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def _toolace_calls(value: str) -> List[Dict[str, Any]]:
    """Parse ToolACE's assistant call DSL: [Name(a="x", b=1), Name2(c=[1,2])] -> [{name,arguments}]."""
    import ast

    s = (value or "").strip()
    if not (s.startswith("[") and "(" in s):
        return []
    s = s[1:-1] if s.endswith("]") else s[1:]
    calls, i, n = [], 0, len(s)
    while i < n:
        while i < n and s[i] in ", ":
            i += 1
        j = s.find("(", i)
        if j == -1:
            break
        name = s[i:j].strip()
        depth, k = 0, j
        while k < n:
            if s[k] == "(":
                depth += 1
            elif s[k] == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        argstr = s[j + 1 : k]
        args: Dict[str, Any] = {}
        try:
            node = ast.parse(f"dict({argstr})", mode="eval")
            for kw in node.body.keywords:  # type: ignore[attr-defined]
                if kw.arg:
                    args[kw.arg] = ast.literal_eval(kw.value)
        except (SyntaxError, ValueError):
            args = {}
        if name:
            calls.append({"name": name, "arguments": args})
        i = k + 1
    return calls


def _clean_toolace_dialog(text: str):
    """ToolACE flattens the whole multi-turn conversation into the user message, behind an identical
    'Role definition: ... Historical dialog data is as follows:' scaffold. That boilerplate repeats
    across hundreds of rows and tanks the intrinsic dataset-quality grade. Parse it into a proper
    (history, final_query): drop the scaffold, keep the real turns. Returns (history|None, query)."""
    if not text:
        return None, text
    marker = "Historical dialog data is as follows:"
    if marker not in text:
        return None, text.strip()
    body = text.split(marker, 1)[1]
    parts = re.split(r"\n\s*(Inquirer:|Response assistant:)\s*", "\n" + body)
    turns = []
    for i in range(1, len(parts) - 1, 2):
        role = "user" if parts[i] == "Inquirer:" else "assistant"
        content = parts[i + 1].strip()
        if content:
            turns.append({"role": role, "content": content})
    last_user = max((i for i, t in enumerate(turns) if t["role"] == "user"), default=None)
    if last_user is None or len(turns[last_user]["content"]) < 3:
        return None, text.strip()
    return (turns[:last_user] or None), turns[last_user]["content"]


def load_toolace(repo: str, limit: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(repo, split="train")
    out: List[Dict[str, Any]] = []
    for row in ds:
        tools = _toolace_tools(row.get("system", ""))
        convs = row.get("conversations")
        if not (tools and convs):
            continue
        # take the first user -> assistant(tool-call) pair (single-turn positive)
        user_msg, call_str = None, None
        for turn in convs:
            role = turn.get("from") or turn.get("role")
            content = turn.get("value") or turn.get("content")
            if role in ("user", "human"):
                user_msg = content
            elif role in ("assistant", "gpt") and user_msg:
                call_str = content
                break
        calls = _toolace_calls(call_str) if call_str else []
        if not (user_msg and calls):
            continue
        history, query = _clean_toolace_dialog(user_msg)  # strip boilerplate, recover real turns
        ex = {
            "tools": _normalize_tools(tools),
            "query": query,
            "answer": {"type": "tool_call", "calls": calls},
            "meta": {"source": "toolace", "hn_kind": None},
        }
        if history:
            ex["history"] = history
        out.append(ex)
        if len(out) >= limit:
            break
    print(f"[build] ToolACE: loaded {len(out)}")
    return out


def _toucan_row_to_example(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse one Toucan SFT row -> canonical example, or None if unparseable. Handles JSON-string
    fields and OpenAI function-wrapped tools/calls. Pure (no network) so it's unit-testable."""
    tools = row.get("tools") or row.get("available_tools") or row.get("functions")
    if isinstance(tools, str):
        tools = _load_json_field(tools)
    if isinstance(tools, list):  # unwrap {"type":"function","function":{...}}
        tools = [t.get("function", t) if isinstance(t, dict) else t for t in tools]
    msgs = row.get("messages") or row.get("conversations")
    if isinstance(msgs, str):
        msgs = _load_json_field(msgs)
    if not (tools and msgs):
        return None
    user_msg, calls = None, None
    for turn in msgs:
        if not isinstance(turn, dict):
            continue
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
        return None
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
        return None
    return {
        "tools": _normalize_tools(tools),
        "query": user_msg,
        "answer": {"type": "tool_call", "calls": norm_calls},
        "meta": {"source": "toucan", "hn_kind": None},
    }


def load_toucan(repo: str, limit: int, config: str = "SFT") -> List[Dict[str, Any]]:
    """Toucan-1.5M (Apache-2.0): real MCP-server tool-calling trajectories. Defensive field parsing.

    Toucan requires a config name (SFT / Qwen3 / OSS / Kimi-K2). SFT rows are
    {question, tools (function-wrapped), messages}; messages/tools may be JSON strings. We STREAM (so we
    only fetch ~`limit` rows, not the full multi-GB set), extract the first user query + first assistant
    tool call, and skip rows we can't parse rather than crashing.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(repo, config, split="train", streaming=True) if config \
            else load_dataset(repo, split="train", streaming=True)
    except Exception as e:  # gated / offline / renamed / missing config
        print(f"[build] Toucan skipped ({type(e).__name__}: {str(e)[:80]})")
        return []
    out: List[Dict[str, Any]] = []
    for row in ds:
        ex = _toucan_row_to_example(row)
        if ex is not None:
            out.append(ex)
        if len(out) >= limit:
            break
    print(f"[build] Toucan: loaded {len(out)}")
    return out


# ToolACE (and others) use Python-style type names; map them to JSON Schema types so validation works.
_TYPE_MAP = {
    "dict": "object", "list": "array", "tuple": "array", "int": "integer", "float": "number",
    "str": "string", "bool": "boolean", "any": "string", "none": "null",
}


def _normalize_schema_types(node: Any) -> Any:
    """Recursively rewrite non-standard JSON-Schema 'type' values (dict->object, int->integer, ...)."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                out[k] = _TYPE_MAP.get(v.lower(), v.lower() if v.lower() in
                                       {"object", "array", "integer", "number", "string", "boolean", "null"} else "string")
            else:
                out[k] = _normalize_schema_types(v)
        return out
    if isinstance(node, list):
        return [_normalize_schema_types(x) for x in node]
    return node


def _normalize_tools(tools: Any) -> List[Dict[str, Any]]:
    """Coerce various tool encodings into [{name, description, parameters}] with JSON-Schema types."""
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
        params = fn.get("parameters", {"type": "object", "properties": {}})
        params = _normalize_schema_types(params) if isinstance(params, dict) else {"type": "object", "properties": {}}
        params.setdefault("type", "object")
        norm.append({"name": name, "description": fn.get("description", ""), "parameters": params})
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


def _split_group_key(ex: Dict[str, Any], i: int) -> str:
    """Group key so matched examples land in ONE split. Multilingual twins share a `pair_id`
    (identical gold, different-language query); if one twin leaked into test while another sat in
    train the model could memorize the shared gold and inflate the held-out multilingual number —
    and the cross-split dedup guard misses it because the query TEXT differs. Ungrouped rows get a
    unique key, so they split individually exactly as before."""
    meta = ex.get("meta", {}) or {}
    pid = meta.get("pair_id")
    if pid is not None:
        return f"{meta.get('source', '')}:pair:{pid}"
    return f"solo:{i}"


def split(
    examples: List[Dict[str, Any]], ratios: Dict[str, float], seed: int
) -> Dict[str, List[Dict[str, Any]]]:
    """Group-aware split: matched twins (shared pair_id) never straddle train/val/test. Slicing is
    over GROUPS (a twin-set counts once); solo rows are their own group, so their split is unchanged.
    Deterministic: groups are keyed in example order, then the key list is shuffled with `seed`."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for i, ex in enumerate(examples):
        groups.setdefault(_split_group_key(ex, i), []).append(ex)
    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    buckets = {
        "train": keys[:n_train],
        "val": keys[n_train : n_train + n_val],
        "test": keys[n_train + n_val :],
    }
    return {name: [ex for k in ks for ex in groups[k]] for name, ks in buckets.items()}


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

    # 1. Load sources (each guarded — a gated/unavailable source is skipped, not fatal)
    positives: List[Dict[str, Any]] = []
    for name, loader, key in (
        ("xlam", load_xlam, "xlam"),
        ("toolace", load_toolace, "toolace"),
    ):
        scfg = dcfg["sources"].get(key, {})
        if scfg.get("enabled") is False:
            continue
        try:
            positives += loader(scfg["repo"], scfg["max_examples"])
        except Exception as e:
            print(f"[build] source '{name}' skipped ({type(e).__name__}: {str(e)[:120]})")
    toucan_cfg = dcfg["sources"].get("toucan", {})
    if toucan_cfg.get("enabled"):
        try:
            positives += load_toucan(toucan_cfg["repo"], toucan_cfg["max_examples"],
                                      config=toucan_cfg.get("config", "SFT"))
        except Exception as e:
            print(f"[build] source 'toucan' skipped ({type(e).__name__})")

    # execution-verified examples from deterministic tool environments (correct by construction)
    n_env = dcfg.get("env_examples", 0)
    if n_env:
        from . import envs

        mc_frac = dcfg.get("env_multicall_frac", 0.0)
        n_mc = int(round(n_env * mc_frac))
        n_single = n_env - n_mc
        env_ex = envs.generate(n_single, seed=seed)
        if n_mc > 0:
            env_ex += envs.generate_multicall(n_mc, seed=seed)
        positives += env_ex
        print(f"[build] env (execution-verified): +{len(env_ex)} ({n_mc} multi-call)")

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

    # 3. Hard negatives + multi-turn from the TRAIN tool pool only (novel tools stay unseen).
    #    Slice sizes use a SHARED denominator so realized shares == intended shares. The old code
    #    sized each slice as if it were the only additive one (n = base * r/(1-r)), which, with four
    #    additive slices, shrank every realized share below target (positives drifted to ~0.62, hard
    #    negatives to ~0.17 instead of 0.22). Here: total = base_n / positive_share, then each slice
    #    is round(total * slice_share) — arithmetic the model card can honestly quote.
    pool = harvest_tool_pool(positives)
    base_n = len(positives)

    hn_ratio = dcfg["hard_negative_ratio"]
    mt_ratio = dcfg.get("multiturn_ratio", 0.0)
    sd_ratio = dcfg.get("schema_drift_ratio", 0.0)
    positive_share = 1.0 - (hn_ratio + mt_ratio + sd_ratio)
    if positive_share <= 0:
        raise ValueError(
            f"slice ratios sum to >= 1 (hn={hn_ratio}, mt={mt_ratio}, sd={sd_ratio}); "
            "positives would be crowded out"
        )
    total_target = base_n / positive_share
    n_hard = round(total_target * hn_ratio)
    n_mt = round(total_target * mt_ratio)
    n_sd = round(total_target * sd_ratio)

    hard = hard_negatives.generate(
        pool, n_hard, cfg["hard_negatives"]["kinds"], seed=seed, positives=positives
    )
    print(f"[build] hard negatives: requested {n_hard}, got {len(hard)}")

    mt: List[Dict[str, Any]] = []
    if mt_ratio > 0:
        mt = multiturn.generate(
            pool, n_mt, cfg["multiturn"]["kinds"], seed=seed,
            long_context_size=cfg["multiturn"].get("long_context_size", 10),
        )
        print(f"[build] multi-turn: requested {n_mt}, got {len(mt)}")

    sd: List[Dict[str, Any]] = []
    if sd_ratio > 0:
        sd = schema_drift.generate(pool, n_sd, cfg["schema_drift"]["kinds"], seed=seed)
        print(f"[build] schema-drift: requested {n_sd}, got {len(sd)}")

    # multilingual slice (matched en/hi/hi-rom twins) — uses Adaptive Data's 242-language strength
    mlx: List[Dict[str, Any]] = []
    ml_n = dcfg.get("multilingual_examples", 0)
    if ml_n:
        from . import multilingual
        mlx = multilingual.generate(ml_n, seed=seed)
        print(f"[build] multilingual: +{len(mlx)} (en/hi/hi-rom)")

    # 4. Combine + dedup
    combined = positives + hard + mt + sd + mlx
    combined = dedup.dedup_all(
        combined,
        minhash_threshold=cfg["dedup"]["minhash_threshold"],
        semantic_threshold=cfg["dedup"]["semantic_threshold"],
    )

    # 4b. Decontaminate against public BFCL/ToolACE-style probes (leakage guard for the hidden-set claim)
    contamination = None
    if cfg["dedup"].get("decontaminate"):
        from . import decontaminate as _decon
        probes = _decon._load_probes(cfg["dedup"].get("decontam_probes"))
        n_before = len(combined)
        combined, dropped = _decon.decontaminate(
            combined, probes,
            ngram_threshold=cfg["dedup"].get("decontam_ngram_threshold", 0.6),
            cos_threshold=cfg["dedup"].get("decontam_cos_threshold", 0.92),
        )
        contamination = {"probes_checked": len(probes), "dropped_count": len(dropped),
                         "before": n_before, "after": len(combined),
                         "ngram_threshold": cfg["dedup"].get("decontam_ngram_threshold", 0.6),
                         "cos_threshold": cfg["dedup"].get("decontam_cos_threshold", 0.92)}
        print(f"[build] decontamination: dropped {len(dropped)} of {n_before} vs {len(probes)} probes")

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
    if dcfg.get("reasoning_traces"):
        from . import reasoning
        pc_rows = reasoning.apply(pc_rows, parts["train"], seed=seed)
        print("[build] reasoning_traces: ON — prepended <think> traces to train_pc "
              "(do NOT also enable the platform reasoning_traces recipe; A/B them instead)")
    write_jsonl(os.path.join(out_dir, "train_pc.jsonl"), pc_rows)
    if novel_test:
        write_jsonl(os.path.join(out_dir, "test_novel.jsonl"), novel_test)

    # 7. Stats for the model card
    total_rows = sum(len(v) for v in parts.values())
    all_rows = [r for rows in parts.values() for r in rows]
    src_counts = Counter(r["meta"]["source"] for r in all_rows)
    n_no_tool = sum(1 for r in all_rows if r["meta"].get("hn_kind") == "no_tool")
    n_miss_param = sum(1 for r in all_rows if r["meta"].get("mt_kind") == "miss_param")
    realized = {k: round(v / max(total_rows, 1), 4) for k, v in src_counts.items()}
    no_tool_share = round(n_no_tool / max(total_rows, 1), 4)
    intended = {
        "positive": round(positive_share, 4),
        "hard_negative": hn_ratio, "multiturn": mt_ratio, "schema_drift": sd_ratio,
    }
    # The refuse/clarify moat rests on the no_tool slice; guard it (research optimum ~10% of total).
    mix_ok = 0.06 <= no_tool_share <= 0.14 and n_miss_param >= 10
    if not mix_ok:
        print(
            f"[build] WARNING mix off target: no_tool={no_tool_share:.1%} of total "
            f"(want ~10%), miss_param rows={n_miss_param} (want >=10). "
            "Check hard_negatives.kinds / multiturn.kinds weights."
        )
    mix = {
        "intended_shares": intended, "realized_shares": realized,
        "no_tool_rows": n_no_tool, "no_tool_share_of_total": no_tool_share,
        "miss_param_rows": n_miss_param, "mix_ok": mix_ok,
    }
    stats = {"total": total_rows, "novel_test": len(novel_test), "mix": mix}
    if contamination is not None:
        stats["contamination"] = contamination
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

    # Reproducibility manifest: SHA-256 of every artifact + seed + config/git/lib provenance.
    try:
        from . import manifest as _manifest
        m = _manifest.write(out_dir=out_dir, config_path="config.yaml")
        print(f"[build] manifest: {len(m['artifacts_sha256'])} artifacts hashed, commit {str(m['git_commit'])[:8]}")
    except Exception as e:
        print(f"[build] manifest skipped ({type(e).__name__}: {e})")

    print("[build] done:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

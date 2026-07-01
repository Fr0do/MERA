#!/usr/bin/env python3
"""Compute the official MMReD headline metric from an lm-eval results.json.

Why this is a post-hoc reporter and not an in-harness group aggregation:
lm-eval's group-of-groups aggregation flattens to leaf metrics before calling a
group's aggregation function, so a harmonic mean placed on the top-level `mmred`
group would receive the 15 raw leaf exact_match values and silently drop the 1/2/4
length weighting (verified end-to-end). The per-type subgroups DO apply the length
weighting correctly (utils.group_length_weighted_aggregate), so the headline is the
harmonic mean of the five per-type subgroup scores — computed here by NAME from the
subgroup rows the harness already emits, with no positional task-order constant.

Usage:
  python scripts/mmred_score.py path/to/results.json [--group mmred]
"""
import argparse
import importlib.util
import json
from pathlib import Path

# Reuse the exact harmonic used by the harness so the number matches the config.
_UTILS = Path(__file__).resolve().parent.parent / "benchmark_tasks" / "mmred" / "utils.py"
_spec = importlib.util.spec_from_file_location("mmred_utils", _UTILS)
_u = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_u)
harmonic = _u.group_harmonic_mean_aggregate


def _pick(d):
    """Pull the em.dc_aggregate (fallback exact_match) scalar out of a results entry."""
    for k in ("em.dc_aggregate,scoring", "exact_match,scoring",
              "em.dc_aggregate,none", "exact_match,none"):
        if k in d:
            return d[k], k
    for k, v in d.items():
        if isinstance(v, (int, float)) and "stderr" not in k and k != "alias":
            return v, k
    return None, None


def mmred_headline(obj, group="mmred"):
    """Return (headline, rows) where rows = [(subgroup, len_weighted_score, {leaf: em})]."""
    groups = obj.get("groups", {})
    results = obj.get("results", {})
    gsub = obj.get("group_subtasks", {})
    rows = []
    for sg in gsub.get(group, []):
        val, _ = _pick(groups.get(sg, results.get(sg, {})))
        leaves = {lf: _pick(results.get(lf, {}))[0] for lf in gsub.get(sg, [])}
        rows.append((sg, val, leaves))
    per_type = [v for _, v, _ in rows if v is not None]
    headline = harmonic(per_type, [1] * len(per_type)) if per_type else 0.0
    return headline, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", type=Path)
    ap.add_argument("--group", default="mmred")
    args = ap.parse_args()
    obj = json.loads(args.results_json.read_text())
    headline, rows = mmred_headline(obj, args.group)

    print("MMReD headline = harmonic mean of per-type length-weighted (1/2/4) scores\n")
    print(f"{'question type':<26}{'len-weighted':>12}   leaves 32/64/128")
    for sg, val, leaves in rows:
        lv = "  ".join(f"{k.rsplit('_', 1)[1]}={v:.3f}" for k, v in leaves.items()
                       if v is not None)
        vs = f"{val:.4f}" if val is not None else "N/A"
        print(f"{sg:<26}{vs:>12}   {lv}")
    print(f"\nMMReD ({args.group}) headline = {headline:.4f}")


if __name__ == "__main__":
    main()

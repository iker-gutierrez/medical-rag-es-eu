#!/usr/bin/env python
"""Generate 66 Qwen3-8B SF configs by cloning the no-SF configs (184-249) with self_feedback=true.

Run IDs 250-315.
  250-271: SNS1064     (11 no_think_sf + 11 think_sf)
  272-293: CasiMedicos (11 no_think_sf + 11 think_sf)
  294-315: mixed       (11 no_think_sf + 11 think_sf)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs/experiments"

NOSF_START = 184   # first no-SF config ID
SF_START   = 250   # first SF config ID

FEEDBACK_MAX_NEW_TOKENS = 512   # feedback pass; no <think> needed here


def main() -> None:
    written = []
    for offset in range(66):
        nosf_id = NOSF_START + offset
        sf_id   = SF_START   + offset

        # Find the source config
        matches = sorted(CFG_DIR.glob(f"{nosf_id}_*.json"))
        assert len(matches) == 1, f"Expected 1 config for ID {nosf_id}, found {matches}"
        src = json.loads(matches[0].read_text(encoding="utf-8"))

        # Derive new names and paths
        old_name = src["experiment_name"]          # e.g. qwen3_8b_no_rag_no_think_extractive_sns1064
        new_name = old_name + "_sf"
        new_output = f"experiments/runs/{sf_id}_{new_name}_dev/predictions.jsonl"

        cfg = dict(src)
        cfg["experiment_name"]        = new_name
        cfg["output"]                 = new_output
        cfg["self_feedback"]          = True
        cfg["feedback_max_new_tokens"] = FEEDBACK_MAX_NEW_TOKENS

        out_path = CFG_DIR / f"{sf_id}_{new_name}_dev.json"
        out_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(out_path.name)
        print(f"  {sf_id}: {new_name}")

    print(f"\nWrote {len(written)} configs (IDs {SF_START}–{SF_START + len(written) - 1})")


if __name__ == "__main__":
    main()

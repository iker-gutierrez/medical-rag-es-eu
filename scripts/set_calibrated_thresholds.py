#!/usr/bin/env python
"""Write the calibrated MA-RAG conflict thresholds into the pipeline configs.

The multiple-choice threshold needs no calibration: MA-RAG treats *any*
disagreement between candidates as conflict, so the bar is 0.

The open-answer threshold does. Semantic conflict (1 - mean pairwise cosine over
the candidates' short answers) never reaches 0 for correct paraphrases and its
scale is model-dependent -- on the calibration subsample Llama-3.1-8B's open-answer
conflict spread out to 0.183 while Qwen3.5-9B, being far more self-consistent,
stayed much tighter. A single shared threshold would therefore make the refinement
loop fire on nearly every Basque record and on almost no Spanish one, which says
more about the backbone than about the questions.

Each model's threshold is set to the median round-1 conflict it produced on the
24-record stratified calibration subsample, so the loop engages on the
more-disagreeing half of open-answer records -- the same rough engagement rate
that MA-RAG's own any-disagreement rule already yields on the multiple-choice half.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"
CONFIG_DIR = ROOT / "configs" / "experiments"

# calibration run -> the marag config it calibrates. The EU calibration is the
# _topk3 one: the threshold is read off the candidates' conflict distribution, and
# those candidates depend on the retrieved evidence, so a calibration run under the
# old rerank-top-5 retrieval does not transfer to the e5-top-3 config.
TARGETS = {
    "calib_marag_es": "1303_qwen35_9b_marag_e5_rerank5_extractive_mixed_dev",
    "calib_marag_eu_topk3": "1313_llama31_8b_marag_e5_topk3_extractive_mixed_eu_dev",
    "calib_marag_eu_latxa_topk1": "1323_latxa_llama31_8b_marag_e5_topk1_extractive_mixed_eu_dev",
}


def open_answer_conflicts(tag: str) -> list[float]:
    path = RUNS / tag / "predictions.jsonl"
    values = []
    for line in path.open():
        record = json.loads(line)
        log = record["reasoning_trace"]["round_log"]
        if log and log[0]["conflict_mode"] != "option_disagreement":
            values.append(float(log[0]["conflict"]))
    return values


def main() -> None:
    for calib_tag, config_stem in TARGETS.items():
        values = open_answer_conflicts(calib_tag)
        if not values:
            print(f"{calib_tag}: no calibration data, leaving config untouched")
            continue
        threshold = round(statistics.median(values), 3)
        iterate = sum(1 for v in values if v > threshold)

        path = CONFIG_DIR / f"{config_stem}.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config.pop("conflict_threshold", None)
        config["conflict_threshold_mc"] = 0.0
        config["conflict_threshold_open"] = threshold
        config["conflict_threshold_provenance"] = (
            f"median round-1 semantic conflict over {len(values)} open-answer records "
            f"of the 24-record stratified calibration subsample ({calib_tag})"
        )
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"{config_stem}\n"
            f"    n={len(values)} open-answer calibration records\n"
            f"    conflict spread: min={min(values):.3f} median={threshold:.3f} max={max(values):.3f}\n"
            f"    -> conflict_threshold_open = {threshold}  "
            f"({iterate}/{len(values)} = {100 * iterate / len(values):.0f}% of open records would iterate)\n"
            f"    -> conflict_threshold_mc   = 0.0 (any disagreement, MA-RAG's own criterion)"
        )


if __name__ == "__main__":
    main()

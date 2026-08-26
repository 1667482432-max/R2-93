from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "matched_phase7_pas_aggregate_graph_metric.json"
SCALE = 0.025


def action_key(record: dict) -> tuple:
    return (
        record["variant"],
        record["mode"],
        int(record["neighbors"]),
        record["kernel"],
    )


def stats(delta: np.ndarray) -> dict:
    return {
        "deltas": delta.tolist(),
        "mean_delta": float(delta.mean()),
        "min_delta": float(delta.min()),
        "lcb": float(delta.mean() - 0.75 * delta.std()),
    }


def run() -> None:
    source = json.loads(SOURCE.read_text())
    output: dict[str, object] = {"scale": SCALE, "groups": {}}
    for group in range(11):
        records = [
            record
            for record in source["groups"]
            if int(record["group"]) == group
            and abs(float(record["scale"]) - SCALE) < 1e-9
            and float(record["mean_delta"]) > 0.0
        ]
        # Keep one copy of identical action vectors and limit the combinatorial
        # search to candidates that have at least a plausible robust signal.
        unique: dict[tuple, dict] = {action_key(record): record for record in records}
        records = sorted(
            unique.values(), key=lambda record: float(record["lcb"]), reverse=True
        )[:96]
        vectors = np.asarray([record["deltas"] for record in records], dtype=np.float64)
        candidates: list[dict] = []
        for size in (1, 2, 3):
            best: dict | None = None
            if size == 1:
                combinations = ((index,) for index in range(len(records)))
            else:
                combinations = itertools.combinations(range(len(records)), size)
            for indices in combinations:
                delta = vectors[list(indices)].mean(axis=0)
                if np.min(delta) <= 0.0:
                    continue
                candidate = {
                    "size": size,
                    "actions": [action_key(records[index]) for index in indices],
                    **stats(delta),
                }
                if best is None or candidate["lcb"] > best["lcb"]:
                    best = candidate
            if best is not None:
                candidates.append(best)
        output["groups"][str(group)] = {
            "records_considered": len(records),
            "best_safe_by_size": candidates,
        }
        print(json.dumps({"group": group, "safe": candidates}))
    (ROOT / "matched_phase7_pas_aggregate_portfolio_screen.json").write_text(
        json.dumps(output, indent=2)
    )


if __name__ == "__main__":
    run()

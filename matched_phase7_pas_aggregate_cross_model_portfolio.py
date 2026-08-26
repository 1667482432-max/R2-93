from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCES = {
    "canonical": "matched_phase7_pas_aggregate_canonical.json",
    "harmonic": "matched_phase7_pas_aggregate_harmonic.json",
    "graph_metric": "matched_phase7_pas_aggregate_graph_metric.json",
    "local_gp": "matched_phase7_pas_aggregate_local_gp.json",
    "anchor": "matched_phase7_pas_aggregate_anchor_canonical.json",
}


def compact_action(source: str, row: dict) -> dict:
    ignored = {"group", "deltas", "mean_delta", "min_delta", "lcb"}
    return {"source": source, **{key: value for key, value in row.items() if key not in ignored}}


def stats(vector: np.ndarray) -> dict:
    return {
        "deltas_linearized": vector.tolist(),
        "mean_delta": float(vector.mean()),
        "min_delta": float(vector.min()),
        "lcb": float(vector.mean() - 0.75 * vector.std()),
    }


def shortlist(rows: list[dict], limit: int = 48) -> list[dict]:
    chosen: dict[str, dict] = {}
    rankings = [
        sorted(rows, key=lambda row: float(row["lcb"]), reverse=True),
        sorted(rows, key=lambda row: float(row["mean_delta"]), reverse=True),
    ]
    for fold in range(5):
        rankings.append(
            sorted(rows, key=lambda row: float(row["deltas"][fold]), reverse=True)
        )
    for ranking in rankings:
        for row in ranking[:limit]:
            key = json.dumps(row, sort_keys=True)
            chosen[key] = row
    return list(chosen.values())


def run() -> None:
    by_source = {
        source: json.loads((ROOT / filename).read_text())["groups"]
        for source, filename in SOURCES.items()
    }
    output = {}
    for group in range(11):
        candidates = {}
        for source, rows in by_source.items():
            usable = [
                row
                for row in rows
                if int(row["group"]) == group
                and float(row["mean_delta"]) > 0.0
                and float(row.get("scale", 0.0)) <= 0.30
            ]
            candidates[source] = shortlist(usable)
        best_pairs = []
        for left_source, right_source in itertools.combinations(SOURCES, 2):
            best = None
            for left in candidates[left_source]:
                left_vector = np.asarray(left["deltas"], dtype=np.float64)
                for right in candidates[right_source]:
                    vector = left_vector + np.asarray(right["deltas"], dtype=np.float64)
                    if vector.min() <= 0.0:
                        continue
                    result = {
                        "actions": [
                            compact_action(left_source, left),
                            compact_action(right_source, right),
                        ],
                        **stats(vector),
                    }
                    if best is None or result["lcb"] > best["lcb"]:
                        best = result
            if best is not None:
                best_pairs.append(best)
        best_pairs.sort(key=lambda row: row["lcb"], reverse=True)

        # Add one action from a third, independent model family to each of the
        # strongest pair portfolios. This is a bounded beam search rather than
        # an unconstrained hyperparameter combination.
        best_triples = []
        for pair in best_pairs[:20]:
            used = {action["source"] for action in pair["actions"]}
            pair_vector = np.asarray(pair["deltas_linearized"], dtype=np.float64)
            for source in SOURCES.keys() - used:
                for row in candidates[source]:
                    vector = pair_vector + np.asarray(row["deltas"], dtype=np.float64)
                    if vector.min() <= 0.0:
                        continue
                    result = {
                        "actions": pair["actions"] + [compact_action(source, row)],
                        **stats(vector),
                    }
                    best_triples.append(result)
        best_triples.sort(key=lambda row: row["lcb"], reverse=True)
        output[str(group)] = {
            "source_candidate_counts": {
                source: len(rows) for source, rows in candidates.items()
            },
            "best_pairs": best_pairs[:20],
            "best_triples": best_triples[:20],
        }
        print(
            json.dumps(
                {
                    "group": group,
                    "pair": best_pairs[:1],
                    "triple": best_triples[:1],
                }
            ),
            flush=True,
        )
    (ROOT / "matched_phase7_pas_aggregate_cross_model_portfolio.json").write_text(
        json.dumps(output, indent=2)
    )


if __name__ == "__main__":
    run()

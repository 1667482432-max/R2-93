from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import r2_pipeline as rp


ROOT = Path(__file__).resolve().parent


def objective(vector: np.ndarray, mode: str, penalty: float) -> float:
    if mode == "lcb":
        return float(vector.mean() - penalty * vector.std())
    if mode == "min_mean":
        return float(vector.min() + penalty * vector.mean())
    raise ValueError(mode)


def run() -> None:
    source = json.loads(
        (ROOT / "matched_phase7_pas_aggregate_portfolio_exact.json").read_text()
    )
    test_groups = rp.official_island_labels(np.load(ROOT / "Round2_Test_Pos.npy"))
    counts = {int(g): int(n) for g, n in zip(*np.unique(test_groups, return_counts=True))}
    total = sum(counts.values())
    options = {}
    for group in counts:
        rows = list(source["groups"][str(group)])
        rows.append(
            {
                "action_weights": [0.0] * len(source["portfolios"][str(group)]),
                "deltas": [0.0] * 5,
                "mean_delta": 0.0,
                "min_delta": 0.0,
                "lcb": 0.0,
            }
        )
        options[group] = rows

    results = []
    settings = [("lcb", value) for value in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)] + [
        ("min_mean", value) for value in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    ]
    for mode, penalty in settings:
        selected = {
            group: max(
                range(len(options[group])),
                key=lambda index: objective(
                    np.asarray(options[group][index]["deltas"]), mode, penalty
                ),
            )
            for group in counts
        }
        for _ in range(20):
            changed = False
            combined = sum(
                counts[group]
                * np.asarray(options[group][selected[group]]["deltas"])
                for group in counts
            ) / total
            for group in counts:
                old = counts[group] * np.asarray(
                    options[group][selected[group]]["deltas"]
                ) / total
                remainder = combined - old
                best = max(
                    range(len(options[group])),
                    key=lambda index: objective(
                        remainder
                        + counts[group]
                        * np.asarray(options[group][index]["deltas"])
                        / total,
                        mode,
                        penalty,
                    ),
                )
                if best != selected[group]:
                    selected[group] = best
                    changed = True
                combined = remainder + counts[group] * np.asarray(
                    options[group][selected[group]]["deltas"]
                ) / total
            if not changed:
                break
        combined = sum(
            counts[group] * np.asarray(options[group][selected[group]]["deltas"])
            for group in counts
        ) / total
        result = {
            "mode": mode,
            "penalty": penalty,
            "objective": objective(combined, mode, penalty),
            "c1_deltas": combined.tolist(),
            "score_deltas_approx": (0.4 * combined).tolist(),
            "mean_score_delta_approx": float(0.4 * combined.mean()),
            "min_score_delta_approx": float(0.4 * combined.min()),
            "selected": {
                str(group): options[group][selected[group]] for group in counts
            },
        }
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key != "selected"}))
    (ROOT / "matched_phase7_pas_aggregate_global_selection.json").write_text(
        json.dumps({"results": results}, indent=2)
    )


if __name__ == "__main__":
    run()

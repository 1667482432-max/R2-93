from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import r2_pipeline as rp
from los_map_features import load_mesh


ROOT = Path(__file__).resolve().parent
BASES = np.asarray([[-18.413, -65.881, 25.0], [52.0, 35.0, 22.0]])


def build_segments(vertices: np.ndarray, faces: list[list[int]]) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        for index in range(len(face)):
            left, right = sorted((face[index], face[(index + 1) % len(face)]))
            if np.linalg.norm(vertices[left, :2] - vertices[right, :2]) > 0.25:
                edges.add((left, right))
    return np.asarray(
        [
            [
                *vertices[left, :2],
                *vertices[right, :2],
                max(vertices[left, 2], vertices[right, 2]),
            ]
            for left, right in edges
        ],
        dtype=np.float32,
    )


def intersections(point: np.ndarray, base: np.ndarray, segments: np.ndarray) -> list[tuple[float, float]]:
    lo = np.minimum(base[:2], point[:2])
    hi = np.maximum(base[:2], point[:2])
    mask = (
        (segments[:, 0] <= hi[0])
        & (segments[:, 2] >= lo[0])
        & (segments[:, 1] <= hi[1])
        & (segments[:, 3] >= lo[1])
        & (segments[:, 4] >= 1.5)
    )
    selected = segments[mask]
    ray = point[:2] - base[:2]
    hits: list[tuple[float, float]] = []
    for segment in selected:
        offset = segment[:2] - base[:2]
        edge = segment[2:4] - segment[:2]
        denominator = ray[0] * edge[1] - ray[1] * edge[0]
        if abs(float(denominator)) < 1e-9:
            continue
        t = (offset[0] * edge[1] - offset[1] * edge[0]) / denominator
        u = (offset[0] * ray[1] - offset[1] * ray[0]) / denominator
        if 0 < t < 1 and 0 <= u <= 1:
            hits.append((float(t), float(segment[4])))
    hits.sort()
    unique: list[tuple[float, float]] = []
    for hit in hits:
        if not unique or abs(hit[0] - unique[-1][0]) > 1e-3:
            unique.append(hit)
        elif hit[1] > unique[-1][1]:
            unique[-1] = hit
    return unique


def line_features(point: np.ndarray, base: np.ndarray, segments: np.ndarray) -> list[float]:
    hits = intersections(point, base, segments)
    if hits:
        t = np.asarray([row[0] for row in hits])
        height = np.asarray([row[1] for row in hits])
        los_height = base[2] + t * (point[2] - base[2])
        clearance = height - los_height
        max_index = int(np.argmax(clearance))
        core = [
            len(hits),
            t[0],
            t[-1],
            height.max(),
            height.mean(),
            clearance.max(),
            np.maximum(clearance, 0).mean(),
            np.sum(clearance > 0),
            t[max_index],
        ]
    else:
        t = np.empty(0)
        clearance = np.empty(0)
        core = [0.0] * 9
    binned = []
    for left in np.linspace(0.0, 1.0, 9)[:-1]:
        right = left + 0.125
        selected = clearance[(t >= left) & (t < right)]
        binned.extend(
            [
                float(len(selected)),
                float(selected.max()) if len(selected) else -32.0,
            ]
        )
    return [float(value) for value in core + binned]


def local_angular_features(
    point: np.ndarray, base: np.ndarray, vertices: np.ndarray, tree: cKDTree
) -> list[float]:
    distance, index = tree.query(point[:2], k=256)
    nearby = vertices[index]
    offset = nearby[:, :2] - point[:2]
    reference = np.arctan2(base[1] - point[1], base[0] - point[0])
    angle = np.arctan2(offset[:, 1], offset[:, 0]) - reference
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    bins = np.floor((angle + np.pi) / (2 * np.pi) * 16).astype(int) % 16
    output = []
    for radius in (10.0, 20.0, 40.0, 80.0):
        active = (distance <= radius) & (nearby[:, 2] > 2.0)
        for bin_index in range(16):
            selected = nearby[active & (bins == bin_index), 2]
            output.append(float(selected.max()) if len(selected) else 0.0)
        for bin_index in range(16):
            output.append(float(np.sum(active & (bins == bin_index))) / 16.0)
    return output


def run() -> None:
    vertices, faces = load_mesh()
    segments = build_segments(vertices, faces)
    tree = cKDTree(vertices[:, :2])
    points = np.vstack(
        (np.load(ROOT / "Round2_Train_Pos.npy"), np.load(ROOT / "Round2_Test_Pos.npy"))
    )
    rows = []
    for index, point in enumerate(points):
        side = int(point[1] > 0)
        active_base = BASES[side]
        other_base = BASES[1 - side]
        feature = []
        feature.extend(line_features(point, active_base, segments))
        feature.extend(line_features(point, other_base, segments))
        feature.extend(local_angular_features(point, active_base, vertices, tree))
        rows.append(feature)
        if (index + 1) % 250 == 0 or index + 1 == len(points):
            print(json.dumps({"done": index + 1, "total": len(points)}), flush=True)
    result = np.asarray(rows, dtype=np.float32)
    result = np.column_stack(
        (
            result,
            np.load(ROOT / "los_map_features.npy"),
            np.load(ROOT / "map_features.npy"),
        )
    ).astype(np.float32)
    np.save(ROOT / "rich_map_features.npy", result)
    print(
        json.dumps(
            {
                "shape": list(result.shape),
                "finite": bool(np.isfinite(result).all()),
                "segments": int(len(segments)),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()

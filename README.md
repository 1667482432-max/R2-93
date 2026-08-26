# Phase93 raw-data end-to-end reconstruction

This repository rebuilds the historical Phase93 submission from the official
Round2 source data. The main entry point no longer downloads or requires P9,
P10, Phase40, Phase6, PAS, PDP, map-feature, or other prediction arrays.

## Required official files

Place these four files in the repository root:

```text
Round2_Train_Channel.npy
Round2_Train_Pos.npy
Round2_Test_Pos.npy
Round2_Map.ply
```

`train_energy.npy` is calculated from `Round2_Train_Channel.npy`. Rows whose
channel energy is zero are removed from model fitting as outliers. The setup
JSON is optional and is not used by the model.

## Install and run

```powershell
python -m pip install -r requirements.txt
python build_phase93_end_to_end.py
```

The entry point invokes `build_phase93_from_raw.py`. It reconstructs:

1. LOS/local/rich map features from `Round2_Map.ply`.
2. Five deterministic official-like rectangular OOF validation folds.
3. Phase4, Phase5 and Phase6 OOF models and 500-point test predictions.
4. The Phase7 selection features used by the frozen robust Phase10 target.
5. P9, Phase10, Phase40 and Phase93.

No historical test prediction is accepted as an input. To inspect the exact
57-stage execution order, run:

```powershell
python build_phase93_end_to_end.py --list-stages
```

An interrupted run can resume from a named stage while reusing earlier caches:

```powershell
python build_phase93_end_to_end.py --from-stage phase6-test
```

The final submission is written to:

```text
Round2_Test_Channel_phase93_g56_antip10_plus_symmetric_clamp.npy
```

Expected shape and dtype are `(500, 256, 4, 192)` and `complex64`. The final
validator checks shape, dtype, finite values, nonzero rows and SHA256, then
writes `phase93_raw_end_to_end_manifest.json`.

## Reproducibility boundary

The code and fixed hyperparameters are part of the model. Large arrays created
during fitting or inference are disposable caches, not model inputs. The old
Release downloader remains only for auditing historical frozen outputs; it is
not called by the raw-data end-to-end entry point.

## Algorithm documentation

- `docs/Round2_Phase93_端到端模型算法说明.docx`: detailed raw-data-to-output model description.
- `docs/Phase93_答辩汇报.pptx`: editable PowerPoint defense deck.

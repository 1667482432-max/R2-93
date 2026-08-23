# Phase93 end-to-end project

This repository contains the complete Phase93 model/validation code path. It is
the frozen `g5/g6 full-192 anti-P10 + symmetric clamp` pipeline and includes all
local Python modules imported by the two entry points.

## One-click data setup

The original competition data (training channel, train/test point clouds and
sampling positions) is intentionally not redistributed. Provide those files
from the authorized competition data package at the paths expected by the
original code. The generated Phase93/P9/P10/Phase40 arrays are release assets,
not Git objects. From the repository directory run:

```powershell
powershell -ExecutionPolicy Bypass -File .\download_phase93_release_assets.ps1
```

The script downloads the generated arrays and verifies every SHA256 in
`phase93_release_assets.json`. No source files need to be edited afterwards.
By default it downloads only rebuild prerequisites; add
`-IncludeReferenceOutputs` to place the frozen P9/P10/Phase40/Phase93 reference
arrays under `phase93_reference_assets` for comparison.

## End-to-end commands

After supplying the authorized original competition data and the frozen
derived prerequisites listed in `phase93_release_assets.json`, the complete
rebuild path is:

```powershell
python .\build_phase93_end_to_end.py
```

The orchestrator runs P9 → Phase10 → Phase40 → Phase93. Each stage writes a
500-row complex64 result and refuses to overwrite an existing output. The
standalone Phase93 builder remains available for auditing the final stage.

The project intentionally preserves the original Phase93 protocol. It does
not tune against test truth and does not silently substitute later phases.

## Frozen provenance

- Phase93 predeclare: `phase93_g56_antip10_plus_symmetric_clamp_predeclared.json`
- Validation result: `phase93_g56_antip10_plus_symmetric_clamp_validation.json`
- Submission manifest: `phase93_g56_antip10_plus_symmetric_clamp_submission_manifest.json`
- Release asset pointer: `phase93_release_assets.json`

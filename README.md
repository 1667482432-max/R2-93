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

## End-to-end commands

After the one-click setup, the frozen validation path is:

```powershell
python .\phase93_g56_antip10_plus_symmetric_clamp_validation.py
python .\build_phase93_g56_antip10_plus_symmetric_clamp_submission.py
```

The builder writes the 500-row complex64 output and a QA manifest. It refuses
to overwrite an existing output, detects zero-channel outliers, and checks the
frozen source hashes before producing anything.

The project intentionally preserves the original Phase93 protocol. It does
not tune against test truth and does not silently substitute later phases.

## Frozen provenance

- Phase93 predeclare: `phase93_g56_antip10_plus_symmetric_clamp_predeclared.json`
- Validation result: `phase93_g56_antip10_plus_symmetric_clamp_validation.json`
- Submission manifest: `phase93_g56_antip10_plus_symmetric_clamp_submission_manifest.json`
- Release asset pointer: `phase93_release_assets.json`

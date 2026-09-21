# Execution notes

- A scorer startup failed before scoring because a generic `import run` resolved an unrelated historical module after its path setup. The new scorer now uses its fully qualified diagnosis-module import. This occurred before the analysis code lock; it changed no model, input, protocol, inference output or evaluator. It is not a PoseFix accuracy/numerical failure.
- CPU regression suite: 6/6 passed. Source code compiles. The GPU inference protocol and canonical checkpoint hashes stayed fixed.
- A warning from the existing PyTorch installation reports its applied cuDNN/nvrtc workaround. No libraries/drivers were installed or changed. Actual layerwise repeat differences are measured in `NUMERICAL_REPLAY.json`, not inferred from the warning.
- GPU operation retains RustDesk and the display, refuses foreign compute and checks temperature below80°C. No restart, driver, power limit, or clock changes.
- The user explicitly approved pushing both pre-existing commits (including the 1,050-file approximately160MB prior research commit) together with this diagnosis. New staging still includes only this diagnosis's code/docs/figures, never unrelated untracked work.
- Overlapping CPU scoring finished just before the GPU worker published its final source-hash verification receipt. Aggregation stopped on the missing completion receipt; after inference completed, the exact same locked scorer reused every finished split and aggregated successfully. No inference, metric or threshold was changed. Use sequential stage commands to avoid this harmless dependency race.

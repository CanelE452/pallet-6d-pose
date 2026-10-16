# Observation intervention and direction correction (V25)

Research-only experiment; frozen operating detector and prior artifacts retained.

- Design and results: [`pallet_observation_v25.md`](../../../_docs/notes/point_line_fusion/04_failure_analysis/pallet_observation_v25.md).
- Outputs: `data/pallet/results/pallet_observation_v25/`.
- Interpreter: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`.
- `diagnose25.py`: existing-cache condition breakdown, no GPU needed.
- `prepare25.py --phase extract`, then `--phase parents`: actual opposite crops and GPU parent inference.
- `intervention25.py`: original/opposite direction distribution diagnosis.
- `run25.py`: intervention diagnosis → GPU training → inference → scoring → independent audit and provenance replay → Markdown/HTML and figures → desktop opening → completion receipt.
- `network25.py`: callable `ObservationCorrectionNetwork(seed, arm)` combines frozen parents and trained correction. Input is the canonical YOLO observation dictionary, with `edge` containing the actual grayscale/derivative ROI for pixel-evidence arms. Forward receives no GT.

Seven registered arms × three seeds × 2,000 final updates = 42,000. The original six observation slots remain in all batches. An extra mild-crop slot repeats the original view in the control; paired arms alternate original and opposite views from the same source group. Detector availability changes exposure slightly, disclosed in `GROUPS_seed*.json` and `AUDIT.json`.

Gates retain V20 movement-weighted continuous-alpha BCE; vectors predict a bounded 2D residual with point-distance loss. This comparison changes output and objective together. Within the vector family, point/line and pixel-evidence comparisons have identical parameter initialization, batches and loss. Local controls still share frozen line-trained ancestry.

Signed image derivatives are measured evidence, not renderer ground truth for visible edges or endpoints. Opposite crops do not change the camera pose. Existing real and synthetic sets are reused development data; synthetic check originals remain separate from correction training. No operating checkpoint is replaced.

Completed model folders are reused only with verified training-source hashes. Do not edit frozen experiment sources to rerun this study; use a new registered output namespace. The driver records failures in `PROGRESS.json` and requires completion artifacts for every phase.

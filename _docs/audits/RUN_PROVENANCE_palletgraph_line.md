# RUN PROVENANCE — paper_s2_palletgraph_line_screen

- created: 2026-07-31T17:16:16.653266+00:00
- git HEAD: a6bf187ca13a561be01bd84eadb15c4d302935ee
- checkpoint: /home/minjae/Documents/github/pallet-pose/weights/paper_s2_stageB/net_epoch_0057.pth (SHA-256 c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896, unchanged)
- python 3.10.20 / torch 2.1.1+cu118 / opencv 4.9.0
- mechanism-val membership SHA: 8d086cd7f8a20cf6fd76af4b26fd5b2ed01ed2a1173e441ae777909202260914
- final-test open count: 0
- close-range rule (fixed before metrics): bbox_area_ratio_top_25pct
- Canny settings reported: [(50, 150), (100, 200), (150, 250)] (used: (100, 200))
- DGP: lambda_point=1.0, iterations=6, trust=(rot 0.05 rad, trans 0.05 m)
- lambda_line calibrated to E_line/E_point fractions [0.25, 0.5, 1.0] (primary 0.5); values in line_lambda_calibration.json

## Arm frame counts
- P0: 87 frames, fallback 0.000
- P1: 87 frames, fallback 0.195
- P2: 87 frames, fallback 0.195
- P3: 87 frames, fallback 0.195
- P4: 87 frames, fallback 0.195
- P2_f025: 87 frames, fallback 0.195
- P2_f100: 87 frames, fallback 0.195
- P3_f025: 87 frames, fallback 0.195
- P3_f100: 87 frames, fallback 0.195
- P4_f025: 87 frames, fallback 0.195
- P4_f100: 87 frames, fallback 0.195

## Reused (not reimplemented)
- evaluator/decoder/geometry: `paper_s2_mechanism_diagnostic`, `paper_s2_frozen_diagnostic`
- canonical 3D corners: `annotate_pnp.make_pallet_keypoints_3d` via `pallet_graph_geometry`
- no dataset file was written or modified


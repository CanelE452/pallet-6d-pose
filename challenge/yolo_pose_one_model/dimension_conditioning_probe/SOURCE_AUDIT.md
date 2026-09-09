# Dimension-conditioning probe checkpoint source audit

Audit captured at 2026-08-28T14:13:47.061092351Z. This audit is artifact-backed: no checkpoint was modified, no training was run, and checkpoint choice was not made from real DEV results.

## Resolution

The current paper-facing YOLO baseline is the vanilla G38 epoch-60 last.pt artifact. It is the same artifact as the requested G38 60ep vanilla checkpoint, not a third checkpoint. The three requested logical roles therefore resolve to two unique checkpoint files.

| Logical role | Selected checkpoint | SHA-256 | Training membership | Epoch evidence | Embedded Ultralytics | Commit evidence | Decision |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| Y0 vanilla G38 30ep | challenge/yolo_pose_one_model/runs_posecls_g38/Y26_G38_Y0_VANILLA_30EP_SEED42/weights/last.pt | 37f904b975db3e95297af5acb51f6e99360f4b59245cef04d0511af3f5a189b1 | G38 generic-only: 38,002 train / 1,998 val | args=30; results rows=30 | 8.4.60 | checkpoint commit null; source-audit HEAD 96ddf1967ecee2759e5d36578a84f2e4eb021efe, dirty | comparator |
| G38 60ep vanilla | challenge/yolo_pose_one_model/runs_camera_facing_loss/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt | 1a806ca497fde5175334e908540ebae22a45deeb08862f014ac8460f8d43ac3c | G38 generic-only: 38,002 train / 1,998 val | args=60; results rows=60 | 8.4.60 | checkpoint commit null; HEAD 152012b6757d792f8e32179815523d14f10ca85c reconstructed from reflog only | primary artifact |
| Current paper baseline | same G38 60ep last.pt above | 1a806ca497fde5175334e908540ebae22a45deeb08862f014ac8460f8d43ac3c | same G38 membership | same 60/60 evidence | 8.4.60 | same caveat as G38 60ep | alias of G38 60ep; no third artifact |

The primary lock is PRIMARY_G38_GENERIC_ONLY_60EP.

## Why this is the paper baseline

- _docs/paper/baselines/BASELINE_READINESS.md:205-218 names the G38 epoch-60 last.pt as the eligible DEV_INFERENCE_READY checkpoint, records its full SHA-256 and size, and explicitly says best.pt and the later pose-classification run are not substitutes.
- challenge/evaluation_v2/dev_results/YOLO26_G38_DEV.json records the identical path and SHA. Its selection policy says this was the pre-existing REAL_G38 epoch-60 last.pt source of truth and was not selected from DEV results.
- challenge/evaluation_v2/selector_diagnostic/PLASTIC_SELECTOR_DIAGNOSTIC.json independently records the same checkpoint, run-args hash, and G38 data-contract hash.
- challenge/yolo_pose_one_model/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930/G38_ADAPT_INIT_LOCK.json binds the same path and hash and says it was chosen by artifact.

This evidence overrides the misleading historical runs_paper directory name.

## Accepted artifact detail

### Y0 vanilla G38 30ep

- Run: challenge/yolo_pose_one_model/runs_posecls_g38/Y26_G38_Y0_VANILLA_30EP_SEED42
- Selected last.pt: 6,546,471 bytes; SHA-256 37f904b975db3e95297af5acb51f6e99360f4b59245cef04d0511af3f5a189b1.
- Non-selected best.pt: SHA-256 1964ba22364677b4ef244e2be65cdc527e82d647c26b2053e1b0de547db10f39. Existing Y0 audits bind last.pt.
- args.yaml: SHA-256 6696634419ef7f7dbcb9fee34c6d4b690d574abec85a8e45acf70f84847f9a65; epochs 30, patience 0, batch 32, image size 640, seed 42, optimizer SGD.
- Checkpoint header: date 2026-08-25T16:07:28.361979, version 8.4.60, class ultralytics.nn.tasks.PoseModel, 3,043,704 parameters, top-level epoch -1, and no embedded branch or commit.
- The -1 epoch is expected for this optimizer-stripped final artifact. args.yaml plus 30 data rows in results.csv establish completion.
- RUNTIME_AUDIT.json reports PoseLoss26, 1,188 batches through epoch 30, and zero PC/ROLE/POSEALIGN/QPOSE calls, supporting the vanilla designation.
- challenge/yolo_pose_one_model/runs_posecls_g38/SOURCE_AUDIT.json records HEAD 96ddf1967ecee2759e5d36578a84f2e4eb021efe but also records a dirty worktree. This is direct source-audit evidence, not a clean commit-bound build.

### G38 vanilla 60ep / current paper alias

- Run: challenge/yolo_pose_one_model/runs_camera_facing_loss/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42
- Selected last.pt: 6,552,807 bytes; SHA-256 1a806ca497fde5175334e908540ebae22a45deeb08862f014ac8460f8d43ac3c.
- Rejected sibling best.pt: SHA-256 0e8f0ee0abfc46cd7a8d387e992126849c8378468d6f70fed15aa1e01c3bc414. The paper artifact lock explicitly requires last.pt.
- args.yaml: SHA-256 baad2df25b15bcbdfc34e6b85f6769de9c19eee52ceacdd9ab6aaf505539b5cd; epochs 60, patience 15, batch 32, image size 640, seed 42, optimizer SGD.
- Checkpoint header: date 2026-08-23T22:53:07.728346, version 8.4.60, class ultralytics.nn.tasks.PoseModel, 3,043,704 parameters, top-level epoch -1, and no embedded branch or commit.
- args.yaml plus 60 data rows in results.csv establish completion. RUNTIME_AUDIT.json reports PoseLoss26 and zero PC/ROLE calls, supporting the vanilla designation.
- Reflog places HEAD 152012b6757d792f8e32179815523d14f10ca85c at 2026-08-20T19:19:55+09:00 and the next HEAD, 96ddf1967ecee2759e5d36578a84f2e4eb021efe, at 2026-08-25T10:24:01+09:00. The Aug-23 checkpoint falls inside that interval, but this is REFLOG_RECONSTRUCTED_NOT_CHECKPOINT_BOUND evidence, not an embedded or clean training commit.

Both accepted runs initialize from challenge/weights/pretrained_yolo/yolo26n-pose.pt, SHA-256 eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9. That file is only the clean pretrained initializer; it is not the fine-tuned paper baseline.

## Frozen G38 training membership

- Data YAML: challenge/yolo_pose_one_model/datasets/g38_generic_only/data.yaml, SHA-256 8191c1111294b52b396175e2f38210e3484fd5ad3e6af70e103f8ce278432f88.
- Manifest: challenge/yolo_pose_one_model/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930/G38_GENERIC_ONLY_MANIFEST.json, SHA-256 a085075c86ad65ecdb6ff7a5961baa6424451ec7024ec330b9df10688b7aff48.
- Train: 38,002 unique stems; SHA-256 of sorted stems joined by newlines without a terminal newline is 74eb31741c34addb97c1d396f6fe4a620f1fa993e325083b7d3551c82560484b.
- Val: 1,998 unique stems; corresponding hash is 0a8c70e3db287f76e9a3f90f9b0ea916ca9373a41b2cee5051b41cb886a45d9d.
- Train/val intersection is zero. Current image and label symlink stems match the manifest exactly, with no missing or extra stems.
- The convention is camera_dynamic_0123_v4 with nine keypoints and zero real training examples. challenge/yolo_pose_one_model/runs_posecls_g38/DATA_CONTRACT.json is corroborating evidence.

## Rejected lookalikes

### Historical literal paper-named run

challenge/yolo_pose_one_model/runs_paper/yolo26n_paper_generic_v1_seed42/weights/last.pt has SHA-256 6a40a4d430fd205a427e38a1927aad2a0a0bef20b984484e0b77318e7bd355ea, size 6,552,679 bytes, checkpoint date 2026-08-21T01:10:30.513680, embedded Ultralytics 8.4.60, and 60 configured/completed epochs. Its args hash is 2d427905180dc200ff3871e89a4677f52280754c4bbdbae1d828dc2b2a349334.

It is rejected because its membership is datasets/broad40k/data.yaml with 39,500 train and 500 val, not frozen G38. The associated paper_generic_v1_manifest.json hash is 6d00ea81488f7db02cf9e4c467c85fff022db7a8cc3cc7c0f5659a02403c2829. Its checkpoint embeds no commit; 152012b6757d792f8e32179815523d14f10ca85c is only a reflog reconstruction.

### G38 exposure-matched control

challenge/yolo_pose_one_model/runs_camera_facing_loss/OLD_ROOT_G38_EXP73916_60EP_SEED42/weights/last.pt has SHA-256 cde730597b7f62f48d87b8e0bae8223ff92c95d36fd14bf47cb12e0ba2010093, size 6,552,807 bytes, checkpoint date 2026-08-24T07:01:14.919994, embedded Ultralytics 8.4.60, and 60 configured/completed epochs. Its args hash is 94d598e60e1a497e72a229001095cc637500122bc3769c00c3010fc4116458ca.

It is rejected because G38_EXP73916 is a 73,916-exposure control made from 38,002 unique items plus 35,914 duplicated aliases; it is not vanilla G38. Its manifest hash is 3579513eee8b70e0173d10c7f38fc8095b92e904895a9af61436eca1c0f4bbd8. Its checkpoint embeds no commit; 152012b6757d792f8e32179815523d14f10ca85c is only a reflog reconstruction.

## Git and source caveats

- Repository snapshot: main at f8472ac08cdbf748754dacbca4b54fb4c576d666.
- git status --short had 100 entries: 34 worktree modifications, one worktree type change, and 65 untracked entries. The SHA-256 of the LF status stream was 982dc776558ad559346da27734554cf074a9b05681b538bc355efbdcfa42e504. The probe scope was already reported as one untracked directory.
- .gitignore:52 ignores all *.pt files, so none of these checkpoints is Git tracked.
- Every inspected checkpoint stores root as the string None and branch/commit/message/origin as null.
- Candidate args/results files first appear in eb8ed3a5264ea8329e112144431c58602b047f19, which post-dates training and is not a training commit.
- Exact source reconstruction is therefore unavailable for the historical, G38 60ep, and EXP checkpoints. Y0 has a direct source audit, but that audit itself records dirty and ignored/untracked inputs.

## Software and GPU snapshot

The default shell interpreter was Python 3.13.9 with torch 2.5.1+cu121; importing Ultralytics there failed. Checkpoint inspection therefore used /home/minjae/anaconda3/envs/pallet-yolo26/bin/python:

- Python 3.10.20
- torch 2.1.1+cu118 and torchvision 0.16.1+cu118
- torch CUDA build 11.8 and cuDNN 8700
- Ultralytics 8.4.60
- opencv-python 4.9.0.80 / cv2 4.9.0

The GPU was an NVIDIA GeForce RTX 3080 with 10,331,226,112 bytes reported by torch (10,240 MiB by nvidia-smi), compute capability 8.6, UUID GPU-8e902e93-41c0-df35-240a-e53c17a21542, and driver 580.173.02. nvidia-smi's CUDA 13.0 is the driver's advertised capability; it is not the torch build toolkit.

Installed source hashes used during inspection:

- ultralytics/nn/modules/head.py: 150c4913bd7599eb252372d4ef7e6a2d2339b5840c083f1b6d9b1675a885e883
- ultralytics/utils/loss.py: 8b4efa8f0c9e734c62edd62bac6509acd718b599c690bb763a786855333e546b

## Reproduction commands

Run from the repository root:

~~~bash
git rev-parse HEAD
git branch --show-current
git status --short
git status --short | sha256sum

sha256sum \
  challenge/yolo_pose_one_model/runs_posecls_g38/Y26_G38_Y0_VANILLA_30EP_SEED42/weights/last.pt \
  challenge/yolo_pose_one_model/runs_camera_facing_loss/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt \
  challenge/yolo_pose_one_model/runs_paper/yolo26n_paper_generic_v1_seed42/weights/last.pt \
  challenge/yolo_pose_one_model/runs_camera_facing_loss/OLD_ROOT_G38_EXP73916_60EP_SEED42/weights/last.pt

sha256sum \
  challenge/yolo_pose_one_model/datasets/g38_generic_only/data.yaml \
  challenge/yolo_pose_one_model/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930/G38_GENERIC_ONLY_MANIFEST.json

for csv in \
  challenge/yolo_pose_one_model/runs_posecls_g38/Y26_G38_Y0_VANILLA_30EP_SEED42/results.csv \
  challenge/yolo_pose_one_model/runs_camera_facing_loss/OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/results.csv
do
  awk 'END { print FILENAME, NR - 1 }' "$csv"
done
~~~

Checkpoint metadata requires the training environment because the default interpreter lacks Ultralytics:

~~~bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python - checkpoint.pt <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
model = checkpoint["model"]
print({
    "date": checkpoint.get("date"),
    "version": checkpoint.get("version"),
    "epoch": checkpoint.get("epoch"),
    "git": checkpoint.get("git"),
    "model_class": f"{type(model).__module__}.{type(model).__name__}",
    "parameters": sum(p.numel() for p in model.parameters()),
})
PY
~~~

Machine-readable full paths, hashes, sizes, alias identity, membership hashes, and provenance statuses are in CHECKPOINT_INVENTORY.json.

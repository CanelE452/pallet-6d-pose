"""bridge 에 README/계약을 채우고, worker base 를 조립하고, 두 ZIP 을 봉인한다.

`mc_build_bridge_export.py` 가 만든 재료 위에서 이어 돈다.  학습 0, GPU 0.
Windows 는 Linux 값을 강제받지 않는다 — 다르면 ENVIRONMENT_DIFF 에 적게 한다.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_build_bridge_export as B                                  # noqa: E402

ROOT, TR, STAMP, EXPORT = B.ROOT, B.TR, B.STAMP, B.EXPORT
BRIDGE = os.path.join(EXPORT, f"FIXED_OBJECT_BRIDGE_{STAMP}")
WORKER = os.path.join(EXPORT, f"WINDOWS_YOLO_WORKER_BASE_{STAMP}")
DEST = r"E:\CODING\proj\sub\gpu_handoff"


BRIDGE_README = """# FIXED_OBJECT_BRIDGE — 먼저 읽을 것

`{stamp}` · camera-facing 라벨을 fixed-object 라벨로 바꾸는 **계약과 검증**이다.
데이터는 들어 있지 않다(40K 라벨은 별도 번들).

## 1. 입력

`camera_dynamic_0123_v4` 라벨. YOLO pose txt 한 줄:
`cls cx cy w h  (x y v) x 9`  — index 0..7 이 코너, 8 이 centroid.

## 2. 출력

fixed-object **물리 코너 인덱싱**. 같은 점들을 다시 번호 매긴 것이지 새로 계산한
좌표가 아니다.

## 3. perm 방향

```
perm_v4[new_camera_facing_index] = old_fixed_index
```
근거는 round-trip 이 아니라 `convert_to_camera_facing_v4.py` 의 `compute_perm_v4`
docstring 과 origin(`obj['cuboid'][:8]`) 이다. 방향을 뒤집어 쓰지 말 것.

## 4. 변환

```
fixed[perm_v4[i]] = camera_facing[i]      i = 0..7
```

## 5. centroid

index 8 은 **그대로 둔다**. 순열에 넣지 않는다.

## 6. visibility / in-frame

좌표와 함께 움직인다. 같은 inverse mapping 을 v 에도 적용한다(점 튜플 통째로 옮기면
자동으로 그렇게 된다).

## 7. 불변량

roundtrip · point set · bbox · centroid · visibility · membership 이 모두 불변이어야
한다. 40,000 전수에서 위반 0 으로 확인돼 있다(`audit/FIXED_OBJECT_DATA_AUDIT.json`).
bbox 는 점 집합이 같으므로 정의상 같고, 그것을 I4 로 실제 검증했다.

## 8. 3D 재투영기를 쓰지 않는다

`make_pallet_keypoints_3d_diagram` 은 **real 평가용**이고 synthetic 계약과 다르다.
이 bridge 는 좌표를 다시 만들지 않는다 — 순수 재인덱싱이다.

## 9. converter 를 고쳐 새 convention 을 만들지 않는다

여기 있는 코드는 이미 40,000 장을 만든 그 코드다. 수정하면 기존 라벨과 다른 것이
되고, 그 순간 학습된 seed42 결과와 비교 불가능해진다.

---

## 받은 PC 에서 가장 먼저 할 일

```
python reference_cases/verify_reference_cases.py
```
`REFERENCE_CASE_PASS  32/32 exact match` 가 나와야 한다. 하나라도 어긋나면 쓰지 말 것.

reference 는 perm class **8종 전부**(dominant 4 + rare 4)와 asset **4종 전부**를
덮는다. expected 는 이미 생성된 40K fixed label 원문이지 재계산값이 아니다.

## RARE36

odd permutation 36장은 **KEEP** 이다. parity 가 음수인 것은 손상이 아니라 손잡이가
뒤집힌 camera-facing 라벨을 바로잡는 순열이며, 실제 3D chirality 로 재니 inverse-perm
후 40,000 전부 +1 로 같아진다. 근거: `audit/RARE36_AUDIT.md`.

## 구성

```
converter/      make_fixed_labels.py (40K 를 실제로 만든 코드) · fixed_label_smoke.py
                convert_to_camera_facing_v4.py (perm 정의 원본)
audit/          FIXED_OBJECT_DATA_AUDIT.json (I1~I10 전수) · FIXED_LABEL_SMOKE.json
                PERM_V4_CENSUS.json · PERM_RARE_AUDIT.json
                RARE36_AUDIT.{{json,md}} · RARE36_FRAMES.csv
                FIXED_LABEL_PROVENANCE.json (40,000행 perm/asset)
                data.yaml · FIXED_60EP_DATA_CONTRACT.json · paper_generic_v1_manifest.json
reference_cases/ REFERENCE_CASES.json + .sha256 + verify_reference_cases.py
FIXED_BRIDGE_SOURCE_LOCK.{{md,json}}   VERSION.json   FILES.sha256
```
"""


CONTRACT = """# FIXED_OBJECT_DATA_CONTRACT

## 라벨 스키마 (YOLO pose txt, 한 줄 = 한 인스턴스)

```
cls  cx cy w h  x0 y0 v0  x1 y1 v1 ... x8 y8 v8
```
- 좌표는 이미지 정규화 [0,1]
- `v` 는 ultralytics 규약(0 = 프레임 밖/미표기, 2 = 프레임 안)
- index 0..7 = 코너, index 8 = centroid
- `kpt_shape: [9, 3]`, `nc: 1`, `names: {0: pallet}`

## 입력 convention — camera_dynamic_0123_v4

0-3 near face, {0,1,4,5} top, {2,3,6,7} bottom, 8 centroid.
near/far 는 **카메라 기준**이라 같은 물체라도 시점에 따라 번호가 바뀐다.

## 출력 convention — fixed object

물체 고정 코너 인덱싱. 시점이 바뀌어도 같은 물리 코너가 같은 번호를 갖는다.

## 변환 규칙 (유일)

```python
fixed = [None] * 8
for i in range(8):
    fixed[perm_v4[i]] = camera_facing[i]
fixed.append(camera_facing[8])          # centroid 고정
# cls, bbox 는 그대로
```

## perm_v4 출처

프레임별 라벨 JSON `objects[0]['perm_v4']`. 8원소 bijection 이어야 한다(I1).
40,000 프레임에서 관측된 class 는 8종:

```
dominant 4   (0,1,2,3,4,5,6,7) 10,308 · (5,4,7,6,1,0,3,2) 10,328
             (1,5,6,2,0,4,7,3)  9,802 · (4,0,3,7,5,1,2,6)  9,526
rare 4       (0,4,7,3,1,5,6,2) 12 · (4,5,6,7,0,1,2,3) 10
             (1,0,3,2,5,4,7,6)  9 · (5,1,2,6,4,0,3,7)  5      합 36 = RARE36
```

## 불변량 (I1~I10)

```
I1  perm 이 8원소 bijection            위반 0 / 40,000
I2  roundtrip (inverse 적용 시 원본)    위반 0
I3  점 집합 불변                        위반 0
I4  visible 점 집합 = bbox 불변         위반 0
I5  centroid 불변                       위반 0
I6  visibility 불변                     위반 0
I7  perm class 열거                     8종
I10 membership 불변 (39,500 + 500)      True
```

## 하지 않는 것

- 3D 재투영 금지 (`make_pallet_keypoints_3d_diagram` 은 real 평가용)
- centroid 를 순열에 넣지 않음
- bbox 재계산 금지 (점 집합이 같으므로 정의상 동일)
- converter 수정 금지
"""


WORKER_README = """# WINDOWS_YOLO_WORKER_BASE — 먼저 읽을 것

`{stamp}` · Windows RTX 4070 worker 가 fixed-object 60ep 학습을 **모델 PC 와 같은
recipe 로** 시작하기 위한 최소 자산이다.

## 지금 GPU 를 쓰지 말 것

이 PC 의 GPU 는 BROAD_FAMILY_V2 를 렌더 중이라고 전달받았다. 아래 1~6 은 전부
**CPU 만으로** 된다. GPU 는 7 단계(renderer pause 합의) 이후에만 쓴다.

## 실행 순서

```
1. conda create -n pallet-yolo26 python=3.10
2. env/PAPER_YOLO_ENV_LOCK.txt 의 버전대로 설치
     torch 2.1.1+cu118 계열 · ultralytics 8.4.60 · opencv 4.9.0 · numpy/scipy/pyyaml
     (Windows CUDA 빌드는 다를 수 있다 — 다르면 ENVIRONMENT_DIFF.txt 에 적는다)
3. import test        python -c "import torch, ultralytics, cv2, numpy, yaml; print('ok')"
4. cuda test          python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
5. weight load        python -c "from ultralytics import YOLO; YOLO(r'weights/yolo26n-pose.pt', task='pose'); print('weight ok')"
6. CPU data dryrun    code/check_dataset.py  (data.yaml 경로/개수만 검사, 학습 아님)
7. ★ renderer pause 합의 후에만 GPU 사용
8. smoke              64 sample forward/backward 1 step
9. 60ep clean start   code/train_fixed_60ep.py
```

## args 는 기억으로 쓰지 않는다

`args/PAPER_60EP_ARGS_LOCK.yaml` 이 seed42 run 의 **실제 args.yaml 에서 읽은 값**이다.
`locked` 블록이 브리프가 지정한 키, `full_args` 가 전체다. launcher 가 이 파일을
그대로 읽으므로 손으로 옮겨 적지 말 것.

## 경로

Linux 절대경로를 그대로 쓰지 않는다. `code/worker_paths.template.yaml` 의
`WORKER_ROOT` 만 Windows 경로로 채우면 나머지가 상대경로로 풀린다.

## workers 값

Linux 는 `workers=4` 다. Windows 는 multiprocessing 구조가 달라 hang/DataLoader 오류가
나면 **줄여도 된다**. 단 바꾼 값을 `ENVIRONMENT_DIFF.txt` 에 기록한다 — 그 외 args 는
바꾸지 않는다.

## 완료 판정

`results.csv` 의 마지막 epoch 마크가 60 인지로만 판단한다. exit code 나 프로세스 존재로
판단하지 않는다(둘 다 거짓말한 이력이 있다).

## 구성

```
env/       PAPER_YOLO_ENV_LOCK.txt · conda yml · pip freeze · nvidia_smi (모델 PC)
args/      PAPER_60EP_ARGS_LOCK.yaml + .sha256
weights/   yolo26n-pose.pt  (seed42 가 실제로 초기화에 쓴 그 파일)
code/      train_fixed_60ep.py · check_run_complete.py · check_dataset.py
           semantic_diagnostic.py · pack_results.py · worker_paths.template.yaml
reference/ seed42 args.yaml · results.csv · CONVERGENCE.csv · ADAPTIVE_60EP_VERDICT.json
FILES.sha256
```

## 참고 — 모델 PC seed42 실측 (비교 기준)

```
60 epoch 완주 · NaN 0 · pose mAP50 0.8085 · mAP50-95 0.6411
identity_best 0.8293 · yaw180_best 0.1585 · case H3 PARTIAL_FIXED_SEMANTIC_LEARNING
checkpoint 규칙 = last.pt (결과를 보고 바꾸지 않는다)
```
"""


PATHS_TEMPLATE = """# worker_paths.template.yaml — WORKER_ROOT 만 채운다
WORKER_ROOT: 'E:\\CODING\\proj\\sub\\gpu_handoff'

dataset_root: '{WORKER_ROOT}\\data\\broad40k_fixed'
images_train: '{WORKER_ROOT}\\data\\broad40k_fixed\\images\\train'
images_val:   '{WORKER_ROOT}\\data\\broad40k_fixed\\images\\val'
labels_train: '{WORKER_ROOT}\\data\\broad40k_fixed\\labels\\train'
labels_val:   '{WORKER_ROOT}\\data\\broad40k_fixed\\labels\\val'
weights:      '{WORKER_ROOT}\\WINDOWS_YOLO_WORKER_BASE\\weights\\yolo26n-pose.pt'
project:      '{WORKER_ROOT}\\runs_fixed'
run_name:     'FIXED_OBJECT_BROAD40K_60EP_SEED42_WIN'

# 모델 PC 원본(참고용, 그대로 쓰지 않는다)
linux_dataset_root: 'challenge/yolo_pose_one_model/datasets/broad40k_fixed'
"""

TRAIN_LAUNCHER = '''"""fixed 60ep 학습 launcher — args 를 LOCK 에서 읽는다. 손으로 옮기지 않는다."""
import os, subprocess, sys, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
LOCK = yaml.safe_load(open(os.path.join(BASE, "args", "PAPER_60EP_ARGS_LOCK.yaml")))
PATHS = yaml.safe_load(open(os.path.join(HERE, "worker_paths.yaml")))

SKIP = {"model", "data", "project", "name", "save_dir", "resume"}


def main():
    args = dict(LOCK["full_args"])
    cmd = ["yolo", "pose", "train",
           f"model={PATHS['weights']}",
           f"data={os.path.join(PATHS['dataset_root'], 'data.yaml')}",
           f"project={PATHS['project']}", f"name={PATHS['run_name']}",
           "resume=False"]
    for key, value in args.items():
        if key in SKIP or value is None:
            continue
        cmd.append(f"{key}={value}")
    print(" ".join(cmd), flush=True)
    log = os.path.join(BASE, "train_fixed_60ep.log")
    with open(log, "a", encoding="utf-8") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    print(f"-> {log}. 완주 판정은 check_run_complete.py 로만 한다.", flush=True)


if __name__ == "__main__":
    main()
'''

CHECK_RUN = '''"""완주 판정 — results.csv 의 마지막 epoch 마크로만. exit code/프로세스 안 본다."""
import csv, json, math, os, sys

TARGET_EPOCHS = 60


def bad(value):
    try:
        v = float(value)
        return math.isnan(v) or math.isinf(v)
    except Exception:
        return False


def main(run_dir):
    path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(path):
        print(json.dumps({"COMPLETE": False, "reason": "results.csv 없음"})); sys.exit(1)
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(json.dumps({"COMPLETE": False, "reason": "행 0"})); sys.exit(1)
    last = int(float(rows[-1]["epoch"]))
    nan = sum(1 for r in rows for v in r.values() if bad(v))
    report = {"COMPLETE": last >= TARGET_EPOCHS, "last_epoch": last,
              "rows": len(rows), "nan_inf": nan,
              "checkpoint_rule": "last.pt"}
    print(json.dumps(report, indent=1))
    sys.exit(0 if report["COMPLETE"] else 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
'''

CHECK_DATASET = '''"""CPU dryrun — 개수/스키마만 본다. 모델을 올리지 않는다."""
import os, sys, yaml

EXPECT_TRAIN, EXPECT_VAL, KPTS = 39500, 500, 9


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    paths = yaml.safe_load(open(os.path.join(here, "worker_paths.yaml")))
    problems = []
    counts = {}
    for split, expect in (("train", EXPECT_TRAIN), ("val", EXPECT_VAL)):
        images = paths[f"images_{split}"]
        labels = paths[f"labels_{split}"]
        for name, folder in (("images", images), ("labels", labels)):
            if not os.path.isdir(folder):
                problems.append(f"{name}_{split} 폴더 없음: {folder}"); continue
            n = len([f for f in os.listdir(folder)
                     if f.lower().endswith((".png", ".jpg", ".txt"))])
            counts[f"{name}_{split}"] = n
            if n != expect:
                problems.append(f"{name}_{split} {n} != {expect}")
    sample = None
    folder = paths["labels_train"]
    if os.path.isdir(folder):
        names = sorted(os.listdir(folder))[:1]
        if names:
            field = open(os.path.join(folder, names[0])).read().split()
            sample = {"file": names[0], "fields": len(field),
                      "expected_fields": 5 + 3 * KPTS}
            if len(field) != 5 + 3 * KPTS:
                problems.append(f"라벨 필드 {len(field)} != {5 + 3 * KPTS}")
    print({"counts": counts, "sample": sample, "problems": problems})
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
'''

PACK_RESULTS = '''"""학습 산출물을 모델 PC 로 되돌릴 최소 묶음으로 packing."""
import hashlib, os, sys, zipfile

WANT = ("args.yaml", "results.csv", "CONVERGENCE.csv")
WEIGHTS = ("last.pt", "best.pt")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(run_dir, out_zip):
    picked = []
    for name in WANT:
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            picked.append(p)
    for name in WEIGHTS:
        p = os.path.join(run_dir, "weights", name)
        if os.path.exists(p):
            picked.append(p)
    for name in os.listdir(run_dir):
        if name.endswith(".json"):
            picked.append(os.path.join(run_dir, name))
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in picked:
            zf.write(p, os.path.relpath(p, run_dir))
        zf.writestr("FILES.sha256",
                    "\\n".join(f"{sha256(p)}  {os.path.relpath(p, run_dir)}"
                               for p in picked))
    print(f"{out_zip}  {len(picked)} files  sha256 {sha256(out_zip)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
'''


def main():
    # ---------- bridge 마무리 ----------
    open(os.path.join(BRIDGE, "README_FIRST.md"), "w").write(
        BRIDGE_README.format(stamp=STAMP))
    open(os.path.join(BRIDGE, "FIXED_OBJECT_DATA_CONTRACT.md"), "w").write(CONTRACT)
    n_bridge = B.write_files_sha(BRIDGE)
    bridge_zip = B.make_zip(BRIDGE, os.path.join(EXPORT, f"{os.path.basename(BRIDGE)}.zip"))
    B.log(f"BRIDGE  files {n_bridge}  zip {bridge_zip['bytes']:,}B  "
          f"entries {bridge_zip['file_count']}  crc_bad {bridge_zip['crc_bad_entry']}")

    # ---------- worker base ----------
    B.log("PHASE D/E — env / args / weight lock")
    locked, full_args, weight = B.phase_de(os.path.join(WORKER, "args"))
    shutil.move(os.path.join(WORKER, "args", "PAPER_YOLO_ENV_LOCK.txt"),
                os.path.join(WORKER, "env_tmp.txt"))
    os.makedirs(os.path.join(WORKER, "env"), exist_ok=True)
    shutil.move(os.path.join(WORKER, "env_tmp.txt"),
                os.path.join(WORKER, "env", "PAPER_YOLO_ENV_LOCK.txt"))
    for name in ("env_pallet_yolo26.txt", "pip_freeze_pallet_yolo26.txt",
                 "conda_pallet_yolo26.yml", "nvidia_smi.txt"):
        src = os.path.join(WORKER, "args", name)
        if os.path.exists(src):
            shutil.move(src, os.path.join(WORKER, "env", name))
    B.log(f"  args locked {len(locked)} keys  "
          f"weight exists={weight['exists']} match_args={weight.get('match_args')}")

    if weight["exists"]:
        B.copy(B.PRETRAIN, os.path.join(WORKER, "weights"))
    json.dump(weight, open(os.path.join(WORKER, "weights", "WEIGHT_LOCK.json"), "w")
              if weight["exists"] else
              open(os.path.join(WORKER, "PRETRAIN_WEIGHT_SOURCE_MISSING.json"), "w"),
              indent=1, ensure_ascii=False)

    code = os.path.join(WORKER, "code")
    os.makedirs(code, exist_ok=True)
    open(os.path.join(code, "worker_paths.template.yaml"), "w").write(PATHS_TEMPLATE)
    open(os.path.join(code, "train_fixed_60ep.py"), "w").write(TRAIN_LAUNCHER)
    open(os.path.join(code, "check_run_complete.py"), "w").write(CHECK_RUN)
    open(os.path.join(code, "check_dataset.py"), "w").write(CHECK_DATASET)
    open(os.path.join(code, "pack_results.py"), "w").write(PACK_RESULTS)
    for src in (os.path.join(B.RUNS_FIXED, "overnight_60ep.py"),
                os.path.join(B.RUNS_FIXED, "seed43_driver.py")):
        if os.path.exists(src):
            B.copy(src, code, rename="semantic_diagnostic_reference_"
                                     + os.path.basename(src))

    ref = os.path.join(WORKER, "reference")
    for name in ("args.yaml", "results.csv", "CONVERGENCE.csv",
                 "ADAPTIVE_60EP_VERDICT.json"):
        src = os.path.join(B.SEED42, name)
        if os.path.exists(src):
            B.copy(src, ref, rename=f"seed42_{name}")
    for name in ("FIXED_60EP_CONFIG_LOCK.json", "FIXED_60EP_DATA_CONTRACT.json",
                 "PAPER_FIXED_SOURCE_LOCK.json"):
        src = os.path.join(B.RUNS_FIXED, name)
        if os.path.exists(src):
            B.copy(src, ref)
    src = os.path.join(B.DS_FX, "data.yaml")
    if os.path.exists(src):
        B.copy(src, ref, rename="broad40k_fixed_data.yaml")

    open(os.path.join(WORKER, "README_FIRST.md"), "w").write(
        WORKER_README.format(stamp=STAMP))
    open(os.path.join(WORKER, "ENVIRONMENT_DIFF.txt"), "w").write(
        "# Windows 에서 Linux 값과 다르게 설정한 것을 여기 적는다.\n"
        "# 예: workers=4 -> 2 (Windows DataLoader hang)\n"
        "# args 는 그 외 바꾸지 않는다.\n")
    n_worker = B.write_files_sha(WORKER)
    worker_zip = B.make_zip(WORKER, os.path.join(EXPORT, f"{os.path.basename(WORKER)}.zip"))
    B.log(f"WORKER  files {n_worker}  zip {worker_zip['bytes']:,}B  "
          f"entries {worker_zip['file_count']}  crc_bad {worker_zip['crc_bad_entry']}")

    # ---------- PHASE H ----------
    hits = []
    for base in ("/home/minjae", os.path.join(ROOT, "challenge"), TR):
        for depth_root, _, names in os.walk(base):
            if depth_root.count(os.sep) - base.count(os.sep) > 3:
                continue
            for name in names:
                low = name.lower()
                if "matched10k" in low or "v1_fixed" in low or "v1_matched" in low:
                    hits.append(os.path.join(depth_root, name))
    v1 = {"MANIFEST_RECEIVED": bool(hits), "hits": hits,
          "action": ("manifest 를 받아야 V1_FIXED_MATCHED10K 를 materialize 할 수 있다. "
                     "n=10,000 / unique 10,000 / target positive 0 / leakage 0 조건은 "
                     "manifest 없이 검증 불가.") if not hits else "hits 검토"}

    report = {"stamp": STAMP, "export_root": EXPORT,
              "bridge": bridge_zip, "worker": worker_zip,
              "args_locked": locked, "weight": weight, "v1": v1,
              "windows_dest": DEST}
    json.dump(report, open(os.path.join(EXPORT, "EXPORT_REPORT.json"), "w"),
              indent=1, ensure_ascii=False)
    B.log(f"V1 matched10k manifest RECEIVED={v1['MANIFEST_RECEIVED']}")
    B.log(f"-> {EXPORT}")


if __name__ == "__main__":
    main()

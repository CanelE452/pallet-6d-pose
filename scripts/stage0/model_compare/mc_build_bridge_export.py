"""Windows worker 를 풀기 위한 전달 번들 2종을 조립한다.  학습 0, 추론 0, GPU 0.

repository 는 읽기만 한다.  산출은 전부 repo 밖 `~/pallet_bridge_export_<STAMP>/` 다.

새로 만드는 것은 **reference_cases 하나뿐**이다.  나머지는 이미 검증된 자산
(`~/pallet_worker_transfer_20260821T105141Z/`, `runs_fixed/`)을 그대로 복사한다 —
40,000 fixed label 을 실제로 만든 코드와 감사 결과가 거기 있고, 다시 만들면 그것이
같은 것이라는 보장이 사라진다.

reference_cases 가 필요한 이유: bridge 를 받은 다른 PC 가 converter 를 돌렸을 때
**같은 결과가 나오는지 자동으로 확인할 수단**이 지금 없다.  expected 는 지어내지
않고 이미 생성돼 있는 fixed label 파일을 그대로 쓴다.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict

ROOT = "/home/minjae/Documents/github/pallet-pose"
TR = "/home/minjae/pallet_worker_transfer_20260821T105141Z"
STAMP = os.environ.get("BRIDGE_STAMP", "20260822T1255")
EXPORT = os.path.expanduser(f"~/pallet_bridge_export_{STAMP}")

DS_CF = os.path.join(ROOT, "challenge/yolo_pose_one_model/datasets/broad40k")
DS_FX = os.path.join(ROOT, "challenge/yolo_pose_one_model/datasets/broad40k_fixed")
SRC_JSON = os.path.join(ROOT, "data/pallet/training_data/paper_release/"
                              "v2_prod40k_clean_merged/labels")
STAGE_FX = os.path.join(TR, "staging", "fixed_labels")
MANIFEST = os.path.join(ROOT, "challenge/yolo_pose_one_model/datasets/"
                              "paper_generic_v1_manifest.json")
RUNS_FIXED = os.path.join(ROOT, "challenge/yolo_pose_one_model/runs_fixed")
SEED42 = os.path.join(RUNS_FIXED, "FIXED_OBJECT_BROAD40K_60EP_SEED42_ADAPTIVE_CONFIRM")
PRETRAIN = os.path.join(ROOT, "challenge/weights/pretrained_yolo/yolo26n-pose.pt")

# 브리프가 고정을 요구한 args 키.  값은 실제 args.yaml 에서만 읽는다.
ARG_KEYS = ["model", "task", "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf",
            "momentum", "weight_decay", "warmup_epochs", "cos_lr", "mosaic",
            "close_mosaic", "scale", "hsv_h", "hsv_s", "hsv_v", "single_cls",
            "fliplr", "patience", "seed", "workers", "amp", "deterministic",
            "save_period"]
PER_CLASS = 4          # perm class 당 목표 reference 수


def log(message):
    print(message, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src, dst_dir, rename=None):
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, rename or os.path.basename(src))
    shutil.copy2(src, dst)
    return dst


# ------------------------------------------------------- PHASE A source lock
def phase_a():
    """40,000 fixed label 을 실제로 만든 provenance 에서만 파일을 고른다."""
    entries = [
        (os.path.join(TR, "scripts/make_fixed_labels.py"),
         "converter — 40K fixed relabel 을 실제로 수행한 코드"),
        (os.path.join(TR, "scripts/fixed_label_smoke.py"),
         "I1~I8 표본 검증"),
        (os.path.join(ROOT, "scripts/annotate/convert_to_camera_facing_v4.py"),
         "perm_v4 정의 원본 (compute_perm_v4 docstring 'perm[new_idx] = old_idx')"),
        (os.path.join(TR, "manifests/FIXED_OBJECT_DATA_AUDIT.json"),
         "40K 전수 I1~I10 감사 결과"),
        (os.path.join(TR, "manifests/FIXED_LABEL_SMOKE.json"),
         "표본 감사 결과"),
        (os.path.join(TR, "manifests/PERM_V4_CENSUS.json"),
         "perm class 8종 + asset 교차 census"),
        (os.path.join(TR, "manifests/PERM_RARE_AUDIT.json"),
         "rare perm 감사"),
        (os.path.join(TR, "manifests/RARE36_AUDIT.json"), "RARE36 최종 판정 KEEP"),
        (os.path.join(TR, "manifests/RARE36_AUDIT.md"), "RARE36 판정 근거"),
        (os.path.join(TR, "manifests/RARE36_FRAMES.csv"), "RARE36 프레임 목록"),
        (os.path.join(TR, "manifests/FIXED_LABEL_PROVENANCE.json"),
         "프레임별 perm_v4 / asset / det (40,000행)"),
        (os.path.join(DS_FX, "data.yaml"), "fixed dataset 계약"),
        (os.path.join(RUNS_FIXED, "FIXED_60EP_DATA_CONTRACT.json"),
         "60ep 데이터 계약 12/12 PASS"),
        (MANIFEST, "train/val stem 정본 (39,500 + 500)"),
    ]
    lock, missing = [], []
    for path, role in entries:
        if not os.path.exists(path):
            missing.append(path); continue
        lock.append({"path": path, "sha256": sha256(path),
                     "bytes": os.path.getsize(path), "role": role})
    return lock, missing


# ---------------------------------------------------- PHASE B reference cases
def parse_label(line):
    field = line.split()
    return field[0], field[1:5], [tuple(field[5 + 3 * i:8 + 3 * i]) for i in range(9)]


def phase_b(out_dir):
    """이미 생성된 fixed label 을 expected 로 삼는다 — 새로 계산하지 않는다."""
    manifest = json.load(open(MANIFEST))
    provenance = json.load(open(os.path.join(TR, "manifests/FIXED_LABEL_PROVENANCE.json")))
    by_class = defaultdict(list)
    for row in provenance:
        by_class[tuple(row["perm_v4"])].append(row)

    picked = []
    for perm, rows in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        # asset 이 서로 다르게 퍼지도록 asset 별로 돌아가며 뽑는다
        by_asset = defaultdict(list)
        for row in rows:
            by_asset[row.get("source_asset")].append(row)
        order, index = list(by_asset), 0
        while len([p for p in picked if tuple(p["perm_v4"]) == perm]) < PER_CLASS:
            asset = order[index % len(order)]
            index += 1
            if not by_asset[asset]:
                if all(not by_asset[a] for a in order):
                    break
                continue
            picked.append(by_asset[asset].pop(0))
            if index > 4 * len(order) + PER_CLASS:
                break

    cases = []
    for row in picked:
        stem, split = row["stem"], row["split"]
        cf_path = os.path.join(DS_CF, "labels", split, f"{stem}.txt")
        fx_path = os.path.join(STAGE_FX, split, f"{stem}.txt")
        if not (os.path.exists(cf_path) and os.path.exists(fx_path)):
            continue
        cf_line = open(cf_path).read().strip()
        fx_line = open(fx_path).read().strip()
        cls, box, kp_cf = parse_label(cf_line)
        _, _, kp_fx = parse_label(fx_line)
        cases.append({
            "source_frame_id": stem, "split": split,
            "source_asset": row.get("source_asset"),
            "perm_v4": row["perm_v4"], "perm_det": row.get("perm_det"),
            "perm_class": str(tuple(row["perm_v4"])),
            "is_rare_odd": bool(row.get("mirrored")),
            "source_camera_facing_label": cf_line,
            "expected_fixed_label": fx_line,
            "cls": cls, "bbox_cxcywh": box,
            "centroid_kp8": list(kp_cf[8]),
            "visibility_camera_facing": [k[2] for k in kp_cf],
            "visibility_fixed": [k[2] for k in kp_fx],
        })

    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "stamp": STAMP, "n_cases": len(cases),
        "perm_classes_covered": sorted({c["perm_class"] for c in cases}),
        "assets_covered": sorted({str(c["source_asset"]) for c in cases}),
        "rule": "fixed[perm_v4[i]] = camera_facing[i], index 8(centroid) 고정",
        "expected_source": "이미 생성된 40K fixed label 파일 원문 "
                           "(재계산하지 않았다)",
        "how_to_verify": "verify_reference_cases.py 를 bridge 안에서 실행하면 "
                         "converter 규칙을 재현해 expected_fixed_label 과 문자열 "
                         "exact 일치를 검사한다.",
        "cases": cases}
    path = os.path.join(out_dir, "REFERENCE_CASES.json")
    json.dump(payload, open(path, "w"), indent=1, ensure_ascii=False)
    open(os.path.join(out_dir, "REFERENCE_CASES.sha256"), "w").write(
        f"{sha256(path)}  REFERENCE_CASES.json\n")
    return payload


VERIFIER = '''"""reference case 로 converter 규칙을 검증한다.  다른 PC 에서 그대로 실행.

    fixed[perm_v4[i]] = camera_facing[i]      (i = 0..7)
    fixed[8] = camera_facing[8]               centroid 고정
    cls / bbox 는 손대지 않는다

expected 와 **문자열 exact** 일치여야 한다.  하나라도 어긋나면 bridge 를 쓰지 말 것.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def convert(line, perm):
    field = line.split()
    cls, box = field[0], field[1:5]
    kp = [tuple(field[5 + 3 * i:8 + 3 * i]) for i in range(9)]
    fixed = [None] * 8
    for i in range(8):
        fixed[perm[i]] = kp[i]
    fixed.append(kp[8])
    return " ".join([cls] + box + [v for t in fixed for v in t])


def main():
    payload = json.load(open(os.path.join(HERE, "REFERENCE_CASES.json")))
    bad = []
    for case in payload["cases"]:
        got = convert(case["source_camera_facing_label"], case["perm_v4"])
        if got != case["expected_fixed_label"]:
            bad.append(case["source_frame_id"])
    total = len(payload["cases"])
    print(f"reference cases {total}  perm classes "
          f"{len(payload['perm_classes_covered'])}  assets "
          f"{len(payload['assets_covered'])}")
    if bad:
        print(f"FAIL {len(bad)}/{total}: {bad[:10]}")
        sys.exit(1)
    print(f"REFERENCE_CASE_PASS  {total}/{total} exact match")


if __name__ == "__main__":
    main()
'''


# ------------------------------------------------------------ PHASE D/E lock
def phase_de(out_dir):
    import yaml
    args_path = os.path.join(SEED42, "args.yaml")
    args = yaml.safe_load(open(args_path))
    locked = {k: args.get(k) for k in ARG_KEYS}
    os.makedirs(out_dir, exist_ok=True)

    args_out = os.path.join(out_dir, "PAPER_60EP_ARGS_LOCK.yaml")
    with open(args_out, "w") as fh:
        yaml.safe_dump({"source": os.path.relpath(args_path, ROOT),
                        "source_sha256": sha256(args_path),
                        "run": "FIXED_OBJECT_BROAD40K_60EP_SEED42_ADAPTIVE_CONFIRM",
                        "locked": locked, "full_args": args},
                       fh, sort_keys=False, allow_unicode=True)
    open(args_out + ".sha256", "w").write(
        f"{sha256(args_out)}  PAPER_60EP_ARGS_LOCK.yaml\n")

    env_lines = ["# PAPER_YOLO_ENV_LOCK  " + STAMP,
                 "# 모델 PC 실측값. Windows 는 강제 대상이 아니며, 다르면",
                 "# ENVIRONMENT_DIFF.txt 에 기록한다.", ""]
    for name in ("env_pallet_yolo26.txt", "pip_freeze_pallet_yolo26.txt",
                 "conda_pallet_yolo26.yml", "nvidia_smi.txt"):
        path = os.path.join(TR, "staging/env", name)
        if os.path.exists(path):
            env_lines += [f"===== {name} =====", open(path).read().rstrip(), ""]
            copy(path, out_dir)
    env_out = os.path.join(out_dir, "PAPER_YOLO_ENV_LOCK.txt")
    open(env_out, "w").write("\n".join(env_lines))

    weight = {"path": PRETRAIN, "exists": os.path.exists(PRETRAIN)}
    if weight["exists"]:
        weight["sha256"] = sha256(PRETRAIN)
        weight["bytes"] = os.path.getsize(PRETRAIN)
        weight["used_by_seed42_args"] = args.get("model")
        weight["match_args"] = (os.path.basename(str(args.get("model")))
                                == os.path.basename(PRETRAIN))
    return locked, args, weight


# -------------------------------------------------------------- zip helpers
def write_files_sha(root):
    lines = []
    for base, _, names in os.walk(root):
        for name in sorted(names):
            if name == "FILES.sha256":
                continue
            full = os.path.join(base, name)
            lines.append(f"{sha256(full)}  {os.path.relpath(full, root)}")
    open(os.path.join(root, "FILES.sha256"), "w").write("\n".join(sorted(lines)) + "\n")
    return len(lines)


def make_zip(src_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for base, _, names in os.walk(src_dir):
            for name in sorted(names):
                full = os.path.join(base, name)
                zf.write(full, os.path.join(os.path.basename(src_dir),
                                            os.path.relpath(full, src_dir)))
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        count = len(zf.namelist())
    open(zip_path + ".sha256", "w").write(
        f"{sha256(zip_path)}  {os.path.basename(zip_path)}\n")
    return {"zip": zip_path, "bytes": os.path.getsize(zip_path),
            "sha256": sha256(zip_path), "file_count": count,
            "crc_bad_entry": bad}


def main():
    os.makedirs(EXPORT, exist_ok=True)
    report = {"stamp": STAMP, "export_root": EXPORT}

    # ---------------- PHASE A ----------------
    log("PHASE A — source lock")
    lock, missing = phase_a()
    log(f"  파일 {len(lock)}  누락 {len(missing)}")
    if missing:
        for path in missing:
            log(f"  MISSING {path}")

    bridge_dir = os.path.join(EXPORT, f"FIXED_OBJECT_BRIDGE_{STAMP}")
    os.makedirs(bridge_dir, exist_ok=True)
    json.dump({"stamp": STAMP, "files": lock, "missing": missing},
              open(os.path.join(bridge_dir, "FIXED_BRIDGE_SOURCE_LOCK.json"), "w"),
              indent=1, ensure_ascii=False)
    with open(os.path.join(bridge_dir, "FIXED_BRIDGE_SOURCE_LOCK.md"), "w") as fh:
        fh.write(f"# FIXED_BRIDGE_SOURCE_LOCK — {STAMP}\n\n"
                 "40,000 fixed label 을 실제로 만든 provenance 에서만 골랐다.\n\n"
                 "```\n")
        for row in lock:
            fh.write(f"{row['sha256'][:16]}  {row['bytes']:>9}  "
                     f"{os.path.relpath(row['path'], os.path.dirname(ROOT))}\n"
                     f"{'':>28}{row['role']}\n")
        fh.write("```\n")

    # ---------------- PHASE B ----------------
    log("PHASE B — reference cases")
    ref_dir = os.path.join(bridge_dir, "reference_cases")
    payload = phase_b(ref_dir)
    open(os.path.join(ref_dir, "verify_reference_cases.py"), "w").write(VERIFIER)
    log(f"  cases {payload['n_cases']}  perm class "
        f"{len(payload['perm_classes_covered'])}  asset "
        f"{len(payload['assets_covered'])}")

    result = subprocess.run([sys.executable,
                             os.path.join(ref_dir, "verify_reference_cases.py")],
                            capture_output=True, text=True)
    log("  " + result.stdout.strip().replace("\n", "\n  "))
    reference_pass = (result.returncode == 0)
    report["reference_case_pass"] = reference_pass
    report["reference_cases"] = {"n": payload["n_cases"],
                                 "perm_classes": payload["perm_classes_covered"],
                                 "assets": payload["assets_covered"]}

    # ---------------- PHASE C 재료 ----------------
    log("PHASE C — bridge 재료 복사")
    conv_dir = os.path.join(bridge_dir, "converter")
    for row in lock:
        target = (conv_dir if row["path"].endswith(".py")
                  else os.path.join(bridge_dir, "audit"))
        copy(row["path"], target)
    json.dump({"stamp": STAMP, "bundle": "FIXED_OBJECT_BRIDGE",
               "rule": "fixed[perm_v4[i]] = camera_facing[i]",
               "perm_direction": "perm[new_camera_facing_index] = old_fixed_index",
               "centroid": "index 8 unchanged",
               "no_3d_reprojector": True,
               "rare36": "KEEP (RARE36_AUDIT.json STATUS)",
               "source_git_HEAD": subprocess.run(
                   ["git", "-C", ROOT, "rev-parse", "HEAD"],
                   capture_output=True, text=True).stdout.strip()},
              open(os.path.join(bridge_dir, "VERSION.json"), "w"),
              indent=1, ensure_ascii=False)
    return bridge_dir, report, lock, payload, reference_pass


if __name__ == "__main__":
    main()

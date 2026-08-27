"""V1_FIXED_MATCHED10K — 렌더 PC 의 manifest 가 가리키는 10,000 장을 materialize 한다.

새로 렌더하거나 라벨을 다시 만들지 않는다.  RGB 는 hardlink, label 은 **이미 검증된
fixed 40K** 에서 hardlink 로 가져온다.  camera-facing 라벨은 쓰지 않는다.

stem 은 추측하지 않는다 — manifest 가 준 `rgb` 파일명에서 그대로 뽑는다.
(sample_id 의 숫자만 보고 zero-pad 를 벗기면 `f0016` 이 `f16` 이 되어 어긋난다.)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_build_bridge_export as B                                  # noqa: E402

ROOT, TR, STAMP, EXPORT = B.ROOT, B.TR, B.STAMP, B.EXPORT
INCOMING = os.path.join(EXPORT, "_v1_incoming")
VIEW = os.path.join(EXPORT, f"V1_FIXED_MATCHED10K_{STAMP}")
SRC_RGB = os.path.join(ROOT, "data/pallet/training_data/paper_release/"
                             "v2_prod40k_clean_merged/rgb")
FX = os.path.join(TR, "staging", "fixed_labels")
MANIFEST_40K = os.path.join(ROOT, "challenge/yolo_pose_one_model/datasets/"
                                  "paper_generic_v1_manifest.json")
SHARD_BYTES = 3_500_000_000


def link(src, dst):
    if os.path.exists(dst):
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main():
    handoff = json.load(open(os.path.join(INCOMING, "v1_matched10k_manifest.json")))
    frames = handoff["frames"]
    stems = [os.path.basename(r["rgb"]).replace("_rgb.png", "") for r in frames]
    manifest40k = json.load(open(MANIFEST_40K))
    train40k, val40k = set(manifest40k["train_stems"]), set(manifest40k["val_stems"])

    checks = {"n": len(stems), "unique": len(set(stems)),
              "in_40k": 0, "missing_40k": [], "from_train": 0, "from_val": 0,
              "rgb_found": 0, "label_found": 0}
    rows, modes = [], {"hardlink": 0, "copy": 0, "exists": 0}
    for split in ("train", "val"):
        os.makedirs(os.path.join(VIEW, "images", split), exist_ok=True)
        os.makedirs(os.path.join(VIEW, "labels", split), exist_ok=True)

    for stem, entry in zip(stems, frames):
        if stem in train40k:
            split = "train"; checks["from_train"] += 1; checks["in_40k"] += 1
        elif stem in val40k:
            split = "val"; checks["from_val"] += 1; checks["in_40k"] += 1
        else:
            checks["missing_40k"].append(stem); continue
        rgb_src = os.path.join(SRC_RGB, f"{stem}_rgb.png")
        lbl_src = os.path.join(FX, split, f"{stem}.txt")
        if not os.path.exists(rgb_src) or not os.path.exists(lbl_src):
            checks["missing_40k"].append(stem); continue
        checks["rgb_found"] += 1; checks["label_found"] += 1
        modes[link(rgb_src, os.path.join(VIEW, "images", split, f"{stem}.png"))] += 1
        modes[link(lbl_src, os.path.join(VIEW, "labels", split, f"{stem}.txt"))] += 1
        rows.append({"stem": stem, "split_in_broad40k": split,
                     "sample_id": entry["sample_id"],
                     "elevation_stratum": entry["elevation_stratum"],
                     "appearance_regime": entry["appearance_regime"],
                     "size_bin": entry.get("size_bin"),
                     "pallet_type": entry.get("pallet_type")})

    counted = {s: len(os.listdir(os.path.join(VIEW, "images", s))) for s in ("train", "val")}
    label_counted = {s: len(os.listdir(os.path.join(VIEW, "labels", s))) for s in ("train", "val")}

    fixed_yaml = os.path.join(ROOT, "challenge/yolo_pose_one_model/datasets/"
                                    "broad40k_fixed/data.yaml")
    yaml_text = open(fixed_yaml).read() if os.path.exists(fixed_yaml) else ""
    open(os.path.join(VIEW, "data.yaml"), "w").write(
        "# V1_FIXED_MATCHED10K — path 는 Windows 에서 채운다\n"
        "path: .\ntrain: images/train\nval: images/val\n"
        "kpt_shape: [9, 3]\nnc: 1\nnames:\n  0: pallet\n"
        "flip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n\n"
        "# 참고 — broad40k_fixed/data.yaml 원문\n"
        + "".join(f"# {line}\n" for line in yaml_text.splitlines()))

    audit = {
        "name": "V1_FIXED_MATCHED10K", "stamp": STAMP,
        "source_manifest": {
            "file": "v1_matched10k_manifest.json",
            "sha256": B.sha256(os.path.join(INCOMING, "v1_matched10k_manifest.json")),
            "declared_n": handoff["n"],
            "declared_target_positive": handoff["target_positive"],
            "selection": handoff["selection"]},
        "stem_rule": "manifest 의 rgb basename 에서 추출 (sample_id 숫자 파싱 금지 — "
                     "zero-pad 를 벗기면 f0016 이 f16 이 되어 어긋난다)",
        "label_source": "이미 검증된 fixed 40K (staging/fixed_labels). "
                        "camera-facing 라벨 미사용.",
        "checks": {k: (v if not isinstance(v, list) else len(v))
                   for k, v in checks.items()},
        "materialized": {"images": counted, "labels": label_counted,
                         "total_images": sum(counted.values()),
                         "total_labels": sum(label_counted.values())},
        "link_modes": modes,
        "GATE": {
            "n_10000": sum(counted.values()) == 10000,
            "unique_10000": checks["unique"] == 10000,
            "labels_10000": sum(label_counted.values()) == 10000,
            "missing_0": len(checks["missing_40k"]) == 0,
            "target_positive_0": handoff["target_positive"] == 0},
        "★val_overlap": {
            "n_from_broad40k_val": checks["from_val"],
            "note": "manifest 가 40,000 전체(train 39,500 + val 500)를 모집단으로 "
                    "뽑았기 때문에 BROAD40K 의 val 이 섞여 있다. 원 소속을 그대로 "
                    "보존했고 여기서 임의로 재분할하지 않았다. V1/V2 비교 설계에서 "
                    "어떤 split 을 쓸지는 사용자가 정한다."},
        "★exporter_parity": {
            "claim_from_render_pc": "V2 는 paper_generic_v1 contract"
                                    "(PAD=100 / BORDER_REFLECT_101 / in-canvas visibility)를 구현",
            "our_side": "V1 label 은 broad40k(prepare_yolo_pose.py 산출)의 perm 재인덱싱본",
            "status": "[추정] 두 익스포터의 바이트 수준 대조는 하지 않았다. "
                      "비교 전에 양쪽에서 같은 프레임 한 장을 직렬화해 맞춰볼 것."},
        "frames": rows,
    }
    json.dump(audit, open(os.path.join(VIEW, "V1_FIXED_MATCHED10K_AUDIT.json"), "w"),
              indent=1, ensure_ascii=False)

    for name in ("README.md", "SHA256SUMS.txt", "V1_MATCHED10K_MATCHING_REPORT.md",
                 "build_v1_matched10k.py", "v1_matched10k_manifest.json"):
        src = os.path.join(INCOMING, name)
        if os.path.exists(src):
            B.copy(src, os.path.join(VIEW, "source_manifest"),
                   rename=f"handoff_{name}")

    open(os.path.join(VIEW, "README.md"), "w").write(f"""# V1_FIXED_MATCHED10K — {STAMP}

렌더 PC 의 `V1_MATCHED10K_MANIFEST_HANDOFF` 가 지정한 10,000 장을 모델 PC 의
**이미 검증된 fixed 40K** 에서 그대로 꺼낸 것이다. 새로 렌더하지도, 라벨을 다시
만들지도 않았다.

```
images/train {counted['train']:>6}   labels/train {label_counted['train']:>6}
images/val   {counted['val']:>6}   labels/val   {label_counted['val']:>6}
합계         {sum(counted.values()):>6}                {sum(label_counted.values()):>6}
```

## 라벨

fixed-object indexing (`fixed[perm_v4[i]] = camera_facing[i]`, centroid 고정).
camera-facing 라벨은 들어 있지 않다. 계약은 FIXED_OBJECT_BRIDGE 번들 참조.

## ★ BROAD40K val 이 {checks['from_val']} 장 섞여 있다

manifest 가 40,000 전체를 모집단으로 뽑았기 때문이다. 원 소속을 그대로 보존했고
임의로 재분할하지 않았다. V1/V2 비교에서 어떤 split 을 쓸지는 사용자가 정한다.

## ★ exporter parity 는 아직 미확인

렌더 PC 는 V2 를 paper_generic_v1 contract 로 만들었다고 밝혔고, 이쪽 V1 라벨은
`prepare_yolo_pose.py` 산출의 perm 재인덱싱본이다. **두 익스포터를 바이트 수준으로
대조하지는 않았다.** 비교 실험 전에 같은 프레임 한 장을 양쪽에서 직렬화해 맞춰볼 것.

## 구성

```
data.yaml                      path 만 Windows 에서 채운다
images/{{train,val}}             PNG (원본 hardlink)
labels/{{train,val}}             fixed-object YOLO txt
V1_FIXED_MATCHED10K_AUDIT.json 게이트 결과 + 10,000행 stratum
source_manifest/               렌더 PC 원본 handoff 5 파일 (무수정)
```
""")
    return VIEW, audit, counted


def bundle(view, audit):
    """RGB 는 shard, 나머지는 한 zip.  PNG 는 압축이 안 되므로 STORE 로 담는다."""
    meta_zip = os.path.join(EXPORT, f"V1_FIXED_MATCHED10K_BUNDLE_{STAMP}.zip")
    with zipfile.ZipFile(meta_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for base, _, names in os.walk(view):
            if os.sep + "images" in base:
                continue
            for name in sorted(names):
                full = os.path.join(base, name)
                zf.write(full, os.path.join(os.path.basename(view),
                                            os.path.relpath(full, view)))
    with zipfile.ZipFile(meta_zip) as zf:
        meta = {"bytes": os.path.getsize(meta_zip), "entries": len(zf.namelist()),
                "crc_bad": zf.testzip()}
    meta["sha256"] = B.sha256(meta_zip)
    open(meta_zip + ".sha256", "w").write(
        f"{meta['sha256']}  {os.path.basename(meta_zip)}\n")

    images = []
    for split in ("train", "val"):
        folder = os.path.join(view, "images", split)
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            images.append((path, os.path.join("images", split, name),
                           os.path.getsize(path)))
    shards, current, size = [], [], 0
    for item in images:
        if size + item[2] > SHARD_BYTES and current:
            shards.append(current); current, size = [], 0
        current.append(item); size += item[2]
    if current:
        shards.append(current)

    shard_info = []
    for index, group in enumerate(shards):
        path = os.path.join(EXPORT,
                            f"V1_FIXED_MATCHED10K_RGB_{STAMP}_part{index:03d}.zip")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            for full, arc, _ in group:
                zf.write(full, os.path.join(f"V1_FIXED_MATCHED10K_{STAMP}", arc))
        with zipfile.ZipFile(path) as zf:
            entries, bad = len(zf.namelist()), zf.testzip()
        digest = B.sha256(path)
        open(path + ".sha256", "w").write(f"{digest}  {os.path.basename(path)}\n")
        shard_info.append({"file": os.path.basename(path),
                           "bytes": os.path.getsize(path), "images": entries,
                           "crc_bad": bad, "sha256": digest})
        B.log(f"  shard {index}: {entries} imgs  {os.path.getsize(path):,}B")
    return meta_zip, meta, shard_info


if __name__ == "__main__":
    B.log("V1_FIXED_MATCHED10K — materialize")
    view, audit, counted = main()
    B.log(f"  images {sum(counted.values())}  labels "
          f"{audit['materialized']['total_labels']}  link {audit['link_modes']}")
    B.log(f"  GATE {audit['GATE']}")
    B.log("bundling")
    meta_zip, meta, shards = bundle(view, audit)
    report = {"stamp": STAMP, "view": view, "gate": audit["GATE"],
              "meta_zip": {"file": os.path.basename(meta_zip), **meta},
              "rgb_shards": shards,
              "val_overlap": audit["★val_overlap"]["n_from_broad40k_val"]}
    json.dump(report, open(os.path.join(EXPORT, "V1_BUNDLE_REPORT.json"), "w"),
              indent=1, ensure_ascii=False)
    B.log(f"  meta zip {meta['bytes']:,}B entries {meta['entries']} "
          f"crc_bad {meta['crc_bad']}")
    B.log(f"  rgb shards {len(shards)}")

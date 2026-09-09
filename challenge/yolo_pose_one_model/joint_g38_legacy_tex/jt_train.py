"""G38 + legacy v1/v2 + legacy-tex 를 한 번에 학습한다 (순차 finetune 아님).

기존에 legacy 를 넣는 방법은 "G38 로 먼저 학습 -> 그 체크포인트에서 finetune" 뿐이었고,
그 결과가 검출은 크게 올리고 keypoint 정밀도는 떨어뜨렸다.  여기서는 세 소스를 **하나의
학습에 섞어** pretrained 에서 바로 돌린다.

레시피는 `Y26_G38_Y0_VANILLA_30EP_SEED42` 와 글자 그대로 같다.  데이터 구성만 다른
대조가 되도록 한 것이고, 그래야 차이를 데이터 탓으로 읽을 수 있다.

치수 조건화는 넣지 않는다 — 별도 트랙에서 STOP 판정이 났고, 섞으면 두 변수가 한 번에
바뀐다(PURPOSE.md 참조).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = "/home/minjae/Documents/github/pallet-pose"
HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(ROOT, "challenge/yolo_pose_one_model/datasets/"
                        "g38_legacy_v1v2_p0_tex20k")
INIT = os.path.join(ROOT, "challenge/weights/pretrained_yolo/yolo26n-pose.pt")
RUN = "JOINT_G38_LEGACY_TEX_30EP_SEED42"

# Y0 VANILLA 와 동일.  하나라도 바꾸면 데이터-only 대조가 깨진다.
RECIPE = dict(epochs=30, batch=32, imgsz=640, seed=42, optimizer="SGD",
              lr0=0.01, lrf=0.01, cos_lr=True, warmup_epochs=3.0, patience=0,
              single_cls=True, fliplr=0.0, flipud=0.0, mosaic=0.3, scale=0.25,
              erasing=0.4, close_mosaic=10, deterministic=True, resume=False,
              val=True, plots=False, save_period=-1, workers=8, verbose=True)


def audit():
    """학습 전에 데이터 구성을 실측해 남긴다 — 나중에 '무엇으로 학습했나' 를 못 물게."""
    out = {"dataset": os.path.relpath(DS, ROOT), "splits": {}}
    for split in ("train", "val"):
        img = os.path.join(DS, "images", split)
        lab = os.path.join(DS, "labels", split)
        names = sorted(os.listdir(img))
        src = {}
        for n in names:
            src[n.split("__")[0]] = src.get(n.split("__")[0], 0) + 1
        empty = sum(1 for f in os.listdir(lab)
                    if os.path.getsize(os.path.join(lab, f)) == 0)
        cols = set()
        for f in sorted(os.listdir(lab))[:200]:
            p = os.path.join(lab, f)
            if os.path.getsize(p):
                cols.add(len(open(p).readline().split()))
        out["splits"][split] = {"n_images": len(names),
                                "n_labels": len(os.listdir(lab)),
                                "by_source": src, "empty_labels": empty,
                                "label_columns_seen": sorted(cols)}
    out["init"] = os.path.relpath(INIT, ROOT)
    out["init_sha256"] = hashlib.sha256(open(INIT, "rb").read()).hexdigest()
    out["recipe"] = RECIPE
    out["mode"] = "one-shot joint training (NOT sequential finetune)"
    out["dimension_conditioning"] = False
    return out


def main():
    info = audit()
    os.makedirs(HERE, exist_ok=True)
    json.dump(info, open(os.path.join(HERE, "DATASET_AUDIT.json"), "w"),
              indent=1, ensure_ascii=False)
    tr = info["splits"]["train"]
    assert tr["n_images"] == tr["n_labels"] == 55980, tr["n_images"]
    assert tr["empty_labels"] == 0, tr["empty_labels"]
    assert tr["label_columns_seen"] == [32], tr["label_columns_seen"]
    print(f"  train {tr['n_images']}  {tr['by_source']}", flush=True)
    print(f"  val   {info['splits']['val']['n_images']}  "
          f"{info['splits']['val']['by_source']}", flush=True)
    print(f"  init  {info['init_sha256'][:16]}…", flush=True)

    from ultralytics import YOLO
    model = YOLO(INIT, task="pose")
    model.train(data=os.path.join(DS, "data.yaml"),
                project=os.path.join(HERE, "runs"), name=RUN, exist_ok=True,
                **RECIPE)
    print(f"DONE -> {os.path.join(HERE, 'runs', RUN)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

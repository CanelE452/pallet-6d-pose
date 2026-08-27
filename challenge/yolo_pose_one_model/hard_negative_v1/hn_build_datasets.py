"""HC / HM·HF 학습셋 구축 — G38 심볼릭 링크 재사용, 물리 복사 없음.

두 폴더만 만든다.  HM 과 HF 는 **membership 이 완전히 같고 loss 만 다르므로**
같은 폴더를 쓴다.  따로 만들면 두 arm 의 데이터가 갈라졌는지 매번 증명해야 한다.

    hn_hc     G38 38,002 + positive repeat 1,900   = 39,902
    hn_hard   G38 38,002 + HARD_NEG1900   1,900    = 39,902

positive repeat membership 은 spec 8 대로 sha256(stem) 오름차순 앞 1,900.
negative 는 **빈 라벨**(0 바이트)로 넣는다 — YOLO 의 background 표현이다.

val 은 두 폴더 모두 G38 val 1,998 그대로.  선택 신호를 val 에서 만들지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil

ROOT = "/home/minjae/Documents/github/pallet-pose"
DS = os.path.join(ROOT, "challenge/yolo_pose_one_model/datasets")
G38 = os.path.join(DS, "g38_generic_only")
HN = os.path.join(ROOT, "challenge/yolo_pose_one_model/hard_negative_v1")
NEG_RGB = os.path.join(ROOT, "data/pallet/training_data/paper_release/negative/"
                             "extracted/negative_synth_v1_train/rgb")
MEMBERSHIP = os.path.join(HN, "PHASE_A/HARD_NEG1900_MEMBERSHIP__Y0.txt")
N = 1900

YAML = """path: {path}
train: images/train
val: images/val
nc: 1
kpt_shape: [9, 3]
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]
names:
  0: pallet
"""


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


def base(name):
    """G38 원본을 심볼릭으로 그대로 옮긴다."""
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            out = os.path.join(name, kind, split)
            os.makedirs(out, exist_ok=True)
            src_dir = os.path.join(G38, kind, split)
            for f in sorted(os.listdir(src_dir)):
                link(os.path.join(src_dir, f), os.path.join(out, f))


def build_hc(out):
    base(out)
    stems = sorted(os.path.splitext(f)[0]
                   for f in os.listdir(os.path.join(G38, "images/train")))
    stems.sort(key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    picked = stems[:N]
    for s in picked:
        link(os.path.join(G38, "images/train", s + ".png"),
             os.path.join(out, "images/train", s + "__rep1.png"))
        link(os.path.join(G38, "labels/train", s + ".txt"),
             os.path.join(out, "labels/train", s + "__rep1.txt"))
    return picked


def build_hard(out):
    base(out)
    members = [ln.strip() for ln in open(MEMBERSHIP) if ln.strip()]
    assert len(members) == N, f"membership {len(members)} != {N}"
    for m in members:
        src = os.path.join(NEG_RGB, m + ".png")
        assert os.path.exists(src), src
        link(src, os.path.join(out, "images/train", "NEG__" + m + ".png"))
        # 빈 라벨 = background.  0 바이트로 만든다.
        open(os.path.join(out, "labels/train", "NEG__" + m + ".txt"), "w").close()
    return members


def audit(path, expect_train, expect_neg):
    tr = os.path.join(path, "images/train")
    lb = os.path.join(path, "labels/train")
    n_img = len(os.listdir(tr))
    n_lab = len(os.listdir(lb))
    empty = sum(1 for f in os.listdir(lb)
                if os.path.getsize(os.path.join(lb, f)) == 0)
    ok = (n_img == expect_train and n_lab == expect_train and empty == expect_neg)
    return {"path": os.path.relpath(path, ROOT), "n_images": n_img,
            "n_labels": n_lab, "n_empty_labels": empty,
            "expect_train": expect_train, "expect_empty": expect_neg, "ok": ok}


def main():
    report = {}
    for name, fn, n_neg in (("hn_hc", build_hc, 0), ("hn_hard", build_hard, N)):
        out = os.path.join(DS, name)
        if os.path.exists(out):
            shutil.rmtree(out)
        picked = fn(out)
        open(os.path.join(out, "data.yaml"), "w").write(YAML.format(path=out))
        report[name] = audit(out, 38002 + N, n_neg)
        report[name]["membership_sha256"] = hashlib.sha256(
            ("\n".join(picked) + "\n").encode()).hexdigest()
        print(f"  {name:10} images {report[name]['n_images']}  "
              f"empty-label {report[name]['n_empty_labels']}  "
              f"ok={report[name]['ok']}", flush=True)
    json.dump(report, open(os.path.join(HN, "preflight/DATASET_AUDIT.json"), "w"),
              indent=1)
    assert all(v["ok"] for v in report.values()), "DATASET AUDIT FAILED"
    print("DATASET_AUDIT OK")


if __name__ == "__main__":
    os.makedirs(os.path.join(HN, "preflight"), exist_ok=True)
    main()

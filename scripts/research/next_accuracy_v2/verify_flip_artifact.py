"""빌드된 YOLO 라벨에서 flip 파생본이 부모의 미러+순열인지 직접 검증한다 (§6).

계약 테스트는 코드를 묶지 산출물을 묶지 않는다.  생성기를 고쳐도 디스크에 남은
낡은 증강본은 여전히 초록불로 통과한다 — 그 구멍을 여기서 막는다.

불변식:  flip 라벨의 index dst 는 부모 index FLIP_PERM_8[dst] 의 좌우 미러여야 한다.
        x_flip[dst] + x_base[src] == W - 1      (padded 캔버스 폭 기준으로 환산)
        y_flip[dst] == y_base[src]

읽기 전용.  새 학습·추론 없음.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATASETS = REPO / "challenge/yolo_pose_one_model/datasets"
FLIP_PERM_8 = (1, 0, 3, 2, 5, 4, 7, 6)
FLIP_PERM = FLIP_PERM_8 + (8,)


def png_size(p: Path):
    with open(p, "rb") as f:
        head = f.read(26)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def read_label(p: Path):
    f = p.read_text(encoding="utf-8").strip().split()
    return [(float(f[5 + 3 * i]), float(f[6 + 3 * i]), int(f[7 + 3 * i]))
            for i in range(9)]


def check(ds: str):
    root = DATASETS / ds
    img_dir, lbl_dir = root / "images/train", root / "labels/train"
    if not lbl_dir.exists():
        return None
    stems = {p.stem for p in lbl_dir.glob("*.txt")}
    pairs, res = [], {"dataset": ds, "n_flip": 0, "n_paired": 0,
                      "n_mirror_ok": 0, "n_identical_to_parent": 0,
                      "n_other": 0, "max_err_px": 0.0, "examples": []}
    for s in sorted(stems):
        if not s.endswith("_f"):
            continue
        res["n_flip"] += 1
        # 부모 stem: aug__<session>_<frame>_f  ->  <session>_manual_gt__<frame>
        body = s[:-2]
        if body.startswith("aug__"):
            body = body[len("aug__"):]
        cand = [t for t in stems if t.endswith("__" + body.rsplit("_", 1)[-1])
                and body.rsplit("_", 1)[-1].isdigit()
                and body.rsplit("_", 1)[0] in t]
        if len(cand) != 1:
            continue
        pairs.append((s, cand[0]))

    for fs, bs in pairs:
        fw, fh = png_size(img_dir / f"{fs}.png")
        bw, bh = png_size(img_dir / f"{bs}.png")
        if (fw, fh) != (bw, bh):
            continue
        fl, bl = read_label(lbl_dir / f"{fs}.txt"), read_label(lbl_dir / f"{bs}.txt")
        res["n_paired"] += 1
        mirror_err = ident_err = 0.0
        for dst in range(9):
            src = FLIP_PERM[dst]
            if fl[dst][2] == 0 or bl[src][2] == 0:
                continue
            mirror_err = max(mirror_err,
                             abs(fl[dst][0] * fw + bl[src][0] * bw - (fw - 1)),
                             abs(fl[dst][1] * fh - bl[src][1] * bh))
        for i in range(9):
            if fl[i][2] == 0 or bl[i][2] == 0:
                continue
            ident_err = max(ident_err, abs(fl[i][0] - bl[i][0]) * fw,
                            abs(fl[i][1] - bl[i][1]) * fh)
        if mirror_err <= 2.0:
            res["n_mirror_ok"] += 1
        elif ident_err <= 2.0:
            res["n_identical_to_parent"] += 1
            if len(res["examples"]) < 5:
                res["examples"].append(
                    {"flip": fs, "parent": bs, "mirror_err_px": round(mirror_err, 1),
                     "identity_err_px": round(ident_err, 1)})
        else:
            res["n_other"] += 1
            if len(res["examples"]) < 5:
                res["examples"].append(
                    {"flip": fs, "parent": bs, "mirror_err_px": round(mirror_err, 1),
                     "identity_err_px": round(ident_err, 1)})
        res["max_err_px"] = max(res["max_err_px"], round(mirror_err, 1))
    return res


def main() -> int:
    out = []
    for ds in sorted(p.name for p in DATASETS.iterdir() if p.is_dir()):
        r = check(ds)
        if r and r["n_flip"]:
            out.append(r)
    dst = REPO / "data/pallet/results/next_accuracy_v2/FLIP_ARTIFACT_VERIFY.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'dataset':<26}{'flip':>6}{'paired':>8}{'mirror_ok':>11}"
          f"{'= parent':>10}{'other':>7}{'max_err_px':>12}")
    for r in out:
        print(f"{r['dataset']:<26}{r['n_flip']:>6}{r['n_paired']:>8}"
              f"{r['n_mirror_ok']:>11}{r['n_identical_to_parent']:>10}"
              f"{r['n_other']:>7}{r['max_err_px']:>12.1f}")
    print(f"\nwrote {dst.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

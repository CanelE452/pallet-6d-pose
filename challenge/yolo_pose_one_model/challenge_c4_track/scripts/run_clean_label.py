#!/usr/bin/env python3
"""정본 라벨 + flip 제외 C4 재학습 — 학습부터 판정·알림까지 한 파일에서.

    STEP 1  yolo26n-ft -> [indexed FT 120ep] -> stage1_indexed
    STEP 2  stage1     -> [C4 FT 120ep]      -> stage2_c4
    STEP 3  stage1/stage2/F2/v4 를 **같은 새 val** 로 재채점
    STEP 4  사전등록 gate 로 판정하고 Discord 로 판정과 수치를 보낸다

완료 판정은 exit code 가 아니라 산출물(weights/best.pt + results.csv 의 최종 epoch)로
한다.  gate 는 `_docs/notes/c4-rotation-symmetry.md` §1 에 사전등록한 값을 여기
하드코딩한다 — 결과를 보고 고치지 못하게.

데이터셋 경로를 커맨드라인에 노출하지 않는다(`purpose_gate` 가 `*_v숫자` 를 실험
폴더로 오인한다).  실행은 `clean_label/` 에서 하고 PURPOSE.md 는 거기 있다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TRACK = HERE.parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
ONE = REPO / "challenge/yolo_pose_one_model"

from run_c4_arms import RECIPE, sha256  # noqa: E402

OUT = TRACK / "clean_label"
DS = ONE / "datasets/live_gt_v6_clean"
BASE = ONE / "release/pallet-pose-yolo26n-ft/pallet_yolo26n_pose_ft.pt"
# `--variant nopad` 로 무패딩 데이터셋/출력으로 통째로 갈아끼운다.  Jetson 처럼 추론
# 예산이 빠듯할 때 쓴다 — padding 은 입력 텐서를 640x480 -> 640x544 로 키워 실측
# +12.8% 를 먹는다.  ★학습과 추론의 padding 은 반드시 같아야 한다.
VARIANTS = {
    "pad100": ("datasets/live_gt_v6_clean", "clean_label"),
    "nopad":  ("datasets/live_gt_v7_nopad", "nopad_label"),
}
NOTIFY = Path.home() / ".claude/hooks/discord-notify.sh"

# 합의된 증강 — 확대축소·색감·밝기를 올리고 flip 은 끈 채로 둔다.
AUG_OVERRIDE = dict(epochs=120, scale=0.40, hsv_v=0.50, hsv_s=0.60)

# 사전등록 gate (notes §1).  결과를 보고 고치지 않는다.
GATE_MARGIN_PX = 0.15


def notify(msg: str) -> None:
    if not NOTIFY.is_file():
        print(f"[notify 없음] {msg}")
        return
    try:
        subprocess.run([str(NOTIFY), msg], timeout=30, check=False)
    except Exception as exc:                      # 알림 실패가 학습을 죽이면 안 된다
        print(f"[notify 실패] {exc}")


def finished(run_dir: Path, epochs: int) -> bool:
    """완료는 산출물로만 판정한다 — 프로세스 생존이나 exit code 를 믿지 않는다."""
    w = run_dir / "weights/best.pt"
    csv = run_dir / "results.csv"
    if not (w.is_file() and csv.is_file()):
        return False
    lines = csv.read_text(encoding="utf-8").strip().splitlines()
    return len(lines) - 1 >= epochs


def train(name: str, base: Path, c4: bool, epochs: int) -> Path:
    from ultralytics import YOLO
    from pallet_yolo_loss.trainer import ChallengeC4Trainer

    run_dir = OUT / name
    if finished(run_dir, epochs):
        print(f"[skip] {name} 이미 완료 — {run_dir/'weights/best.pt'}")
        return run_dir / "weights/best.pt"

    if c4:
        os.environ["C4_CONFIG"] = str(TRACK / "c4_config_enabled.json")
    else:
        os.environ.pop("C4_CONFIG", None)

    for c in (DS / "labels").glob("*.cache"):     # stale 캐시가 pose mAP 를 뒤집은 이력
        c.unlink()
        print(f"   캐시 제거: {c.name}")

    print(f"\n=== {name} === base {base.name}  loss {'C4' if c4 else 'indexed'}")
    started = time.time()
    YOLO(str(base)).train(data=str(DS / "data.yaml"), project=str(OUT), name=name,
                          trainer=ChallengeC4Trainer,
                          **{**RECIPE, **AUG_OVERRIDE, "epochs": epochs})
    mins = (time.time() - started) / 60

    if not finished(run_dir, epochs):
        notify(f"❌ {name} 실패 — 산출물이 없거나 epoch 미달 ({run_dir})")
        raise SystemExit(f"[FAIL] {name} 산출물 없음: {run_dir}")
    w = run_dir / "weights/best.pt"
    print(f"완료 {mins:.1f}분  {w}  sha {sha256(w)[:16]}…")
    return w


def evaluate(arms: dict) -> dict:
    """evaluate_c4_arms 의 채점 함수를 새 val 로 재사용한다."""
    import evaluate_c4_arms as ev
    ev.DS = DS                                    # 모듈 전역을 새 데이터셋으로 돌린다
    out = {}
    for arm, w in arms.items():
        if not Path(w).is_file():
            print(f"[skip] {arm} 가중치 없음: {w}")
            continue
        print(f"\n--- 채점 {arm} ---")
        rows = ev.per_frame(Path(w))
        s = ev.summarize(rows)
        s["detection"] = ev.detection_metrics(Path(w))
        out[arm] = s
        print(f"  검출 {s['n_detected']}/{s['n_frames']}  "
              f"fixed med {s['fixed_index']['median']:.2f}  "
              f"C4 med {s['c4_equivalent']['median']:.2f}  "
              f"collapse {s['true_collapse']}  perm {s['selected_permutation_hist']}")
    return out


def verdict(res: dict) -> tuple[str, str]:
    """사전등록 gate.  기준은 notes §1 에 먼저 쓰고 여기 하드코딩했다."""
    if "stage2_c4" not in res or "F2" not in res:
        return "INCOMPLETE", "stage2 또는 F2 채점이 없다"
    new, ref = res["stage2_c4"], res["F2"]
    dm = new["c4_equivalent"]["median"] - ref["c4_equivalent"]["median"]
    dc = new["true_collapse"] - ref["true_collapse"]
    dd = new["n_detected"] - ref["n_detected"]
    detail = (f"C4 median {new['c4_equivalent']['median']:.2f} vs F2 "
              f"{ref['c4_equivalent']['median']:.2f} ({dm:+.2f}px) | "
              f"collapse {new['true_collapse']} vs {ref['true_collapse']} ({dc:+d}) | "
              f"검출 {new['n_detected']} vs {ref['n_detected']} ({dd:+d}) | "
              f"fixed median {new['fixed_index']['median']:.2f} vs "
              f"{ref['fixed_index']['median']:.2f}")
    if dm <= -GATE_MARGIN_PX and dc <= 0 and dd >= 0:
        return "PASS", detail
    if abs(dm) < GATE_MARGIN_PX and dc <= 0 and dd >= 0:
        return "NEUTRAL", detail
    return "FAIL", detail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=AUG_OVERRIDE["epochs"])
    ap.add_argument("--smoke", action="store_true", help="2 epoch 으로 배선만 확인")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="pad100")
    args = ap.parse_args(argv)
    epochs = 2 if args.smoke else args.epochs

    global DS, OUT
    ds_rel, out_rel = VARIANTS[args.variant]
    DS, OUT = ONE / ds_rel, TRACK / out_rel
    print(f"variant {args.variant}  데이터 {ds_rel}  출력 {out_rel}")

    if not DS.is_file() and not (DS / "data.yaml").is_file():
        print(f"[STOP] 데이터셋 없음: {DS}")
        return 1
    if not BASE.is_file():
        print(f"[STOP] base 없음: {BASE}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    w1 = train("stage1_indexed", BASE, c4=False, epochs=epochs)
    w2 = train("stage2_c4", w1, c4=True, epochs=epochs)

    arms = {"stage1_indexed": w1, "stage2_c4": w2}
    if args.variant == "pad100":
        # 기존 arm 들도 패딩 학습본이라 같은 val 로 재는 것이 공정하다.
        arms["F2"] = TRACK / "F2/weights/best.pt"
        arms["v4"] = ONE / "runs_live_gt/ft_live_gt_v4/weights/best.pt"
    res = evaluate(arms)
    if args.variant == "pad100":
        v, detail = verdict(res)
    else:
        # 패딩본을 무패딩 val 로 재면 parity 가 깨져 부당하게 나쁘다.  두 variant 비교는
        # 원본 좌표계로 맞춰 주는 별도 스크립트(compare_padding_variants.py)가 한다.
        s = res["stage2_c4"]
        v, detail = "INFO", (
            f"C4 median {s['c4_equivalent']['median']:.2f} | "
            f"fixed median {s['fixed_index']['median']:.2f} | "
            f"collapse {s['true_collapse']} | 검출 {s['n_detected']}/{s['n_frames']} "
            "(패딩본과의 비교는 compare_padding_variants.py 로)")

    payload = {
        "verdict": v, "detail": detail, "epochs": epochs,
        "dataset": str(DS.relative_to(REPO)),
        "gate_margin_px": GATE_MARGIN_PX,
        "aug_override": AUG_OVERRIDE,
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "arms": res,
    }
    (OUT / "RESULT.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"\n판정 {v}\n{detail}\n  -> {OUT/'RESULT.json'}")
    icon = {"PASS": "✅", "NEUTRAL": "➖", "FAIL": "❌"}.get(v, "⚠️")
    notify(f"{icon} 정본라벨+flip제외 C4 재학습 — {v}\n{detail}\n"
           f"{payload['elapsed_min']}분, {epochs}ep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

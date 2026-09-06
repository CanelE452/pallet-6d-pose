"""self-training 에 DiffPnP 를 얹는 스크린 — 학습에서 끝내지 않는다.

    STEP 0 Q0  이 설정에서 DiffPnP 항의 기울기 대역을 재서 lambda 를 고른다
    STEP 1     대조(lambda=0) · 처치(lambda=lambda*) 두 arm 학습
               R0 체크포인트에서 이어가고, exposure lock 예산·증강을 그대로 쓴다
    STEP 2     PAPER_EVAL 319 추론 (정본 recipe)
    STEP 3     pose 평가 (정본 selector·GT·metric)
    STEP 4     사전등록 게이트로 판정
    STEP 5     판정과 핵심 수치를 남긴다

완료 판정은 exit code 가 아니라 산출물로 한다.

    conda run -n pallet-yolo26 python -u \
        scripts/research/diffpnp_yolo_v1/run_selftrain_screen.py --probe-only
    conda run -n pallet-yolo26 python -u \
        scripts/research/diffpnp_yolo_v1/run_selftrain_screen.py --lambda-dp 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/paper/pose_metric_closure_v1"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_screen as rs  # noqa: E402  (infer/evaluate/verdict 재사용)

OUT = REPO / "data/pallet/results/diffpnp_yolo_v1"
RUNS = OUT / "selftrain_runs"
PMC = REPO / "data/pallet/results/paper_pose_metric_closure_v1"
LOCK = OUT / "SELFTRAIN_DIFFPNP_METHOD_LOCK.json"

# --st-arm 으로 pseudo-label 표집 draw 를 바꾼다.  ultralytics 의 args.seed 는
# dataloader 에 도달하지 않아 seed 복제가 비트 동일해지므로, 유효한 replicate 는
# 표집을 바꾼 데이터셋(P43/P44)이다 — repo 의 기존 방식과 같다.
ARM = "R5_PROPOSED"
INDEX = OUT / f"selftrain_{ARM}"
DATA = (REPO / "challenge/yolo_pose_one_model/datasets/paper_selftrain_v1"
        / ARM / "data.yaml")


def select_arm(name: str) -> None:
    global ARM, INDEX, DATA
    ARM = name
    INDEX = OUT / f"selftrain_{ARM}"
    DATA = (REPO / "challenge/yolo_pose_one_model/datasets/paper_selftrain_v1"
            / ARM / "data.yaml")

# run_paper_selftraining.py 의 TRAIN_ARGS 를 그대로 옮긴다 — 이 실험의 유일한 변수는
# lambda_dp 이므로 self-training 레시피는 한 글자도 바꾸지 않는다.
TRAIN_ARGS = dict(
    imgsz=640, batch=32, optimizer="SGD", lr0=0.002, lrf=0.01, cos_lr=True,
    momentum=0.937, weight_decay=0.0005, warmup_epochs=1.0, patience=0,
    box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.35, degrees=0.0, translate=0.1, scale=0.25,
    shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0,
    mosaic=0.15, close_mosaic=3, mixup=0.0, copy_paste=0.0, erasing=0.4,
    seed=42, deterministic=True, val=False, plots=False, save_period=-1,
    workers=2, cache=False,
)


def base_checkpoint() -> Path:
    entry = json.loads((PMC / "POSE_ARM_CHECKPOINT_LOCK.json").read_text())["arms"]["R0"]
    return REPO / entry["checkpoint"]


def write_config(name: str, lam: float) -> Path:
    path = OUT / f"diffpnp_config_ST_{name}.json"
    path.write_text(json.dumps({
        "enabled": lam != 0.0, "lambda_dp": lam, "gn_steps": 5,
        "huber_delta_norm": 0.10, "damping": 1e-3, "delta_clip": 0.5,
        "min_visible": 6, "affine_residual_max_px": 0.5,
        "index_dir": str(INDEX)}, indent=1))
    return path


def train_code(name: str, epochs: int) -> str:
    args = dict(TRAIN_ARGS, model=str(base_checkpoint()), data=str(DATA),
                epochs=epochs, device="0", save=True, project=str(RUNS),
                name=name, exist_ok=True, verbose=False)
    return ("import sys; sys.path.insert(0, %r)\n"
            "from pallet_yolo_loss.trainer import DiffPnPTrainer\n"
            "DiffPnPTrainer(overrides=%r).train()\n") % (str(REPO), args)


def train_arm(name: str, lam: float, epochs: int) -> Path:
    weights = RUNS / name / "weights" / "last.pt"
    if weights.exists() and weights.stat().st_size > 0:
        print(f"[{name}] 이미 있음 — 건너뜀", flush=True)
        return weights
    env = dict(os.environ, DIFFPNP_CONFIG=str(write_config(name, lam)))
    env.pop("DIFFPNP_PROBE", None)
    print(f"[{name}] 학습 시작 lambda_dp={lam}", flush=True)
    started = time.time()
    subprocess.run([sys.executable, "-u", "-c", train_code(name, epochs)],
                   env=env, cwd=str(REPO))
    print(f"[{name}] 학습 종료 {time.time() - started:.0f}s", flush=True)
    if not weights.exists() or weights.stat().st_size == 0:
        raise SystemExit(f"[{name}] 산출물 없음 — 학습이 실제로 끝나지 않았다")
    return weights


def probe(epochs: int = 1) -> dict:
    """STEP 0 — 이 설정에서 lambda 대역을 잰다 (판정 아님)."""
    env = dict(os.environ, DIFFPNP_CONFIG=str(write_config("PROBE", 1.0)),
               DIFFPNP_PROBE="1")
    code = train_code("PROBE", epochs).replace(
        "DiffPnPTrainer(overrides=", "t = DiffPnPTrainer(overrides=").replace(
        ").train()\n", ")\n"
        "rows = []\n"
        "def cb(tr):\n"
        "    c = getattr(tr.model, 'criterion', None)\n"
        "    inner = [x for x in (getattr(c,'one2many',None), getattr(c,'one2one',None))\n"
        "             if x is not None and hasattr(x,'dp_stats')]\n"
        "    if not inner and hasattr(c,'dp_stats'): inner=[c]\n"
        "    for x in inner:\n"
        "        s = x.dp_stats\n"
        "        if 'grad_base' in s and s.get('n_valid',0)>0:\n"
        "            rows.append([s['grad_base'], s['grad_dp_at_lambda1'],\n"
        "                         s['lambda_for_5pct'], s['n_valid'],\n"
        "                         s['mean_corner_norm']])\n"
        "t.add_callback('on_train_batch_end', cb)\n"
        "t.train()\n"
        "import json, numpy as np\n"
        "a = np.array(rows) if rows else np.zeros((0,5))\n"
        "json.dump({'batches': len(a),\n"
        "           'grad_ratio_at_lambda1_median': float(np.median(a[:,1]/np.maximum(a[:,0],1e-12))) if len(a) else 0.0,\n"
        "           'lambda_for_5pct_median': float(np.median(a[:,2])) if len(a) else 0.0,\n"
        "           'lambda_for_5pct_p25': float(np.percentile(a[:,2],25)) if len(a) else 0.0,\n"
        "           'lambda_for_5pct_p75': float(np.percentile(a[:,2],75)) if len(a) else 0.0,\n"
        "           'median_supervised_per_head': float(np.median(a[:,3])) if len(a) else 0.0,\n"
        "           'median_corner_norm': float(np.median(a[:,4])) if len(a) else 0.0},\n"
        "          open(%r,'w'), indent=2)\n" % str(OUT / "ST_Q0_GRAD_BAND.json"))
    subprocess.run([sys.executable, "-u", "-c", code], env=env, cwd=str(REPO))
    return json.loads((OUT / "ST_Q0_GRAD_BAND.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lambda-dp", type=float, default=None)
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--st-arm", default="R5_PROPOSED")
    args = ap.parse_args()
    select_arm(args.st_arm)
    suffix = "" if args.st_arm == "R5_PROPOSED" else "_" + args.st_arm.split("_")[-1]

    RUNS.mkdir(parents=True, exist_ok=True)
    if args.probe_only or args.lambda_dp is None:
        q0 = probe()
        print("\nQ0: " + json.dumps(q0, indent=2), flush=True)
        if args.probe_only:
            return 0
        args.lambda_dp = float(q0["lambda_for_5pct_median"])

    lock = json.loads(LOCK.read_text())
    started = time.time()
    arms = {f"ST_C_LAMBDA0{suffix}": 0.0, f"ST_T_DIFFPNP{suffix}": args.lambda_dp}
    preds = {}
    for name, lam in arms.items():
        w = train_arm(name, lam, args.epochs)
        preds[name] = rs.infer_arm(name, w)
    summaries = {name: rs.evaluate_arm(p) for name, p in preds.items()}
    for name, s in summaries.items():
        print(f"[{name}] {json.dumps(s)}", flush=True)

    names = list(arms)
    v = rs.verdict(summaries[names[0]], summaries[names[1]], lock)
    report = {"schema_version": "selftrain_diffpnp_result_v1",
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_sec": round(time.time() - started, 1),
              "self_training_arm": ARM, "epochs": args.epochs,
              "lambda_dp": args.lambda_dp,
              "index_meta": json.loads((INDEX / "SELFTRAIN_INDEX_META.json").read_text()),
              "arms": summaries, "verdict": v,
              "leakage_note": lock["leakage_note"]}
    (OUT / f"SELFTRAIN_DIFFPNP_RESULT{suffix}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    lines = [f"# self-training + DiffPnP ({ARM} 데이터셋, lambda_dp 만 다르다)", "",
             f"{args.epochs} epochs · 900 optimizer update · lambda={args.lambda_dp:.4g} · "
             f"{report['elapsed_sec'] / 60:.0f}분", "",
             "| 지표 | 대조 λ=0 | 처치 λ>0 | 상대변화 |", "|---|---:|---:|---:|"]
    for m, d in v["by_metric"].items():
        lines.append(f"| {m} | {d['control']:.4f} | {d['treatment']:.4f} | "
                     f"{d['relative_change']:+.2%} |")
    lines += ["", f"판정 **{v['verdict']}** · 실질 {v['effective_reading']} · "
                  f"개선 {v['improved_metric_count']}/4", "",
              f"> {lock['leakage_note']}", ""]
    (OUT / f"SELFTRAIN_DIFFPNP_REPORT{suffix}.md").write_text("\n".join(lines))

    print("\n" + "=" * 60)
    print(f"판정: {v['verdict']}  (실질 {v['effective_reading']}, "
          f"개선 {v['improved_metric_count']}/4)")
    for m, d in v["by_metric"].items():
        print(f"  {m:26s} {d['control']:.4f} -> {d['treatment']:.4f}  "
              f"{d['relative_change']:+.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

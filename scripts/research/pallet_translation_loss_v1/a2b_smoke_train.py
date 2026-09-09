"""학습 경로 스모크 — 1 epoch, 데이터 1%.  배선만 본다."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "challenge/yolo_pose_one_model/pallet_translation_loss_v1"
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    from a2b_preflight import screen_args, R0_CKPT, DATA_YAML
    from a2b_model import A2BTrainer
    cal = json.loads((REPO/"data/pallet/results/pallet_translation_loss_v1/A2B_LAMBDA_CALIBRATION.json").read_text())
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else cal["lambda_lc"]
    name = sys.argv[2] if len(sys.argv) > 2 else "SMOKE_LC"
    cfg = {"enabled": True, "lambda_geo": lam, "calibration": False,
           "train_stem_list": str(PKG/"TRAIN_STEMS.txt")}
    cp = PKG/f"_a2b_config_{name}.json"; cp.write_text(json.dumps(cfg))
    os.environ["A2B_CONFIG"] = str(cp)
    a = screen_args()
    ov = {k: v for k, v in vars(a).items() if not k.startswith("_")}
    ov.update(dict(model=str(R0_CKPT), data=str(DATA_YAML), epochs=1, fraction=0.01,
                   project=str(PKG/"runs_smoke"), name=name, exist_ok=True,
                   patience=0, val=False, plots=False, seed=42, deterministic=True,
                   workers=2, batch=32, imgsz=640, save_period=-1))
    A2BTrainer(overrides=ov).train()
    st = PKG/"runs_smoke"/name/"A2B_EPOCH_STATS.jsonl"
    print("\n--- A2B_EPOCH_STATS ---")
    print(st.read_text() if st.is_file() else "MISSING")
    bo = PKG/"runs_smoke"/name/"A2B_BATCH_ORDER.json"
    print("--- BATCH ORDER ---"); print(bo.read_text() if bo.is_file() else "MISSING")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

"""DOPE(backbone_dope_final_v1)를 PAPER_EVAL 319 에 추론한다.

YOLO arm 과 **같은 정본 recipe·같은 predictions 스키마·같은 evaluator** 를 쓴다.
새 학습 0.  lock 파일은 읽기만 한다.

DOPE 는 belief map 이라 YOLO 의 box conf 가 없다 — box 는 검출된 keypoint 의
bounding box 로, conf 는 belief peak 의 최솟값으로 채운다(스키마 호환용이며
YOLO 의 box_conf 와 같은 양이 아니다).
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, torch, cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# decoder_paths 는 filters 폴더를 path 에 넣지 않는다 — 여기서 보충한다
for _p in (ROOT / "Deep_Object_Pose/common", ROOT / "scripts/stage0",
           ROOT / "challenge/scripts", ROOT / "scripts/data_prep/filters",
           ROOT / "scripts/data_prep/eval"):
    sys.path.insert(0, str(_p))
C = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
OUT = ROOT / "data/pallet/results/backbone_compare_v1"
CKPT = ROOT / "weights/backbone_dope_final_v1/run/final_net_epoch_0060.pth"
ARM = "BB_DOPE_G38"
INPUT = 400
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def main():
    from models import DopeNetwork
    # decoder_paths 는 배포 게이트 스택 전체를 끌어온다.  필요한 건 추출기 하나뿐이라
    # 정본 함수만 직접 가져온다 (같은 구현, 같은 threshold).
    from filter_pr_camfacing import extract_keypoints_from_belief

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(str(k).startswith("module.") for k in state):
        state = {str(k).removeprefix("module."): v for k, v in state.items()}
    # 체크포인트 텐서 접두어가 vgg + m1_1..m6_2 뿐이고 m_seg* 가 없다 ->
    # seg head 없이 만든 모델이다.  strict=True 로 그 사실을 강제 확인한다.
    n_belief = state["m6_2.12.weight"].shape[0] if "m6_2.12.weight" in state else 9
    has_seg = any(str(k).startswith("m_seg") for k in state)
    print(f"  belief channels {n_belief}  seg head {'있음' if has_seg else '없음'}")
    model = DopeNetwork(numVec=0, numSeg=1 if has_seg else 0)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).to(dev).eval()
    print(f"loaded {CKPT.name}  tensors {len(state)}  device {dev}")

    lock = json.loads((C / "INFERENCE_REPLAY_LOCK.json").read_text())
    assert lock["status"] == "FROZEN"
    spec = lock["recipe"]
    pad = int(spec["pad_px"])          # memory: DOPE 추론은 reflect-padding 필수
    frames = json.loads((C / "AXIS_REVIEW_MANIFEST.json").read_text())["frames_list"]

    preds, nod = {}, 0
    for i, fr in enumerate(frames):
        if i % 80 == 0:
            print(f"  {i}/{len(frames)}", flush=True)
        img = cv2.imread(str(ROOT / fr["image"]))
        if img is None:
            preds[fr["frame_id"]] = {"status": "IMAGE_MISSING"}; continue
        p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        H, W = p.shape[:2]
        rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        t = cv2.resize(rgb, (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)
        t = ((t.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        with torch.inference_mode():
            out = model(torch.from_numpy(t)[None].to(dev))
            belief = out[0][-1][0].detach().float().cpu().numpy()
        # belief 격자 -> padded 픽셀 -> 원본 픽셀(pad 빼기)
        sx, sy = W / belief.shape[2], H / belief.shape[1]
        raw = extract_keypoints_from_belief(belief)
        kp, peaks = [], []
        for x, y, pk in raw:
            peaks.append(float(pk))
            kp.append([float("nan"), float("nan")] if (x < 0 and y < 0)
                      else [float(x) * sx - pad, float(y) * sy - pad])
        ok = [k for k in kp[:8] if np.isfinite(k[0])]
        if len(ok) < 6:
            preds[fr["frame_id"]] = {"status": "NO_DETECTION"}; nod += 1; continue
        a = np.asarray(ok)
        preds[fr["frame_id"]] = {
            "status": "OK",
            "box_xyxy": [float(a[:, 0].min()), float(a[:, 1].min()),
                         float(a[:, 0].max()), float(a[:, 1].max())],
            "box_conf": float(min(peaks[:8])),
            "keypoints_xy": kp, "keypoints_conf": [float(x) for x in peaks],
            "detections": 1}

    payload = {"schema_version": "frozen_arm_prediction_v1", "arm": ARM,
               "checkpoint": str(CKPT.relative_to(ROOT)),
               "checkpoint_sha256": hashlib.sha256(CKPT.read_bytes()).hexdigest(),
               "recipe": spec,
               "recipe_lock_sha256": hashlib.sha256(
                   (C / "INFERENCE_REPLAY_LOCK.json").read_bytes()).hexdigest(),
               "population_frame_order_sha256": lock["population"]["frame_order_sha256"],
               "n_frames": len(frames), "no_detection": nod,
               "new_training": 0, "checkpoint_reselection": 0,
               "note": ("DOPE belief-map arm. box/conf are derived from detected "
                        "keypoints and belief peaks; they are not the YOLO box_conf."),
               "frames": preds}
    (C / "predictions" / f"{ARM}.json").write_text(json.dumps(payload, indent=2) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"predictions_{ARM}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n{ARM}: {len(frames)} frames, no_detection {nod}")


if __name__ == "__main__":
    raise SystemExit(main())

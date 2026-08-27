"""conf calibration — PHASE 0 audit + conf=0.001 전수 후보 덤프.

positive 와 negative 에 **동일한 전처리 계약**을 건다 (PAD=100 REFLECT_101,
imgsz=640, top-1 by conf).  다르면 FP/image 가 비교 불가능해진다.

conf=0.001 로 한 번만 추론하고 threshold 는 전부 오프라인에서 쓸어본다 —
threshold 마다 재추론하면 같은 모델이 다른 NMS 를 타서 비교가 흐려진다.
"""
from __future__ import annotations
import json, os, sys
import cv2, numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
A2 = os.path.join(ROOT, "challenge/yolo_pose_one_model/analysis_pre_v2")
PIPE = os.path.join(ROOT, "challenge/yolo_pose_one_model/paper_generic_pipeline")
CK = os.path.join(ROOT, "challenge/yolo_pose_one_model/runs_paper/"
                        "yolo26n_paper_generic_v1_seed42/weights/last.pt")
NEG_SEQ = os.path.join(ROOT, "data/pallet/raw_data/outside/"
                             "forklift_raw_20260528_163408/rgb")
NEG_JSON = os.path.join(ROOT, "challenge/yolo_pose_one_model/runs_ft/"
                              "forklift_raw_conf.json")
FT_LABELS = os.path.join(ROOT, "challenge/yolo_pose_one_model/datasets/ft_a/"
                               "labels/train")
PAD, IMGSZ, DUMP_CONF = 100, 640, 0.001


def phase0_audit():
    """negative 259 장의 per-frame inventory + 역할 계약."""
    rows = json.load(open(NEG_JSON))
    conf_by = {r["frame"]: r for r in rows}
    # 파일명은 neg__forklift_raw__000575[__repN].txt — repeat 사본이 있으므로
    # 프레임 번호로 dedupe 한다.  259 는 고유 프레임 수다.
    import re as _re
    frames = sorted({int(m.group(1)) for f in os.listdir(FT_LABELS)
                     if (m := _re.match(r"neg__forklift_raw__(\d{6})", f))
                     and f.endswith(".txt")
                     and os.path.getsize(os.path.join(FT_LABELS, f)) == 0})
    # positive DEV 와의 session overlap
    pos = json.load(open(os.path.join(PIPE, "eval_manifest.json")))["items"]
    pos_sessions = sorted({i["set"] for i in pos})
    inv = []
    for fr in frames:
        c = conf_by.get(fr, {})
        img = os.path.join(NEG_SEQ, f"{fr:06d}.png")
        inv.append({
            "frame": fr, "image": os.path.relpath(img, ROOT),
            "exists": os.path.exists(img),
            "pallet_present": False,
            "presence_evidence": "prepare_real_ft 선정 + 전수 육안 검수 (빈 라벨)",
            "session": "forklift_raw_20260528_163408",
            "source": "data/pallet/raw_data/outside",
            "used_in_paper_generic_training": False,
            "used_in_yolo26n_ft_training": True,
            "prior_model_max_conf": c.get("max_conf"),
            "prior_model_n_boxes": c.get("n"),
            "overlaps_positive_dev_session": False})
    audit = {
        "role": "REAL_NEG_DEV_V1",
        "n_frames": len(inv),
        "sequence": "data/pallet/raw_data/outside/forklift_raw_20260528_163408/rgb",
        "sequence_total_frames": len(rows),
        "all_pallet_absent": all(not r["pallet_present"] for r in inv),
        "used_in_paper_generic_training": False,
        "paper_generic_train_set": "BROAD 합성 40,000 (real positive 0, real negative 0)",
        "positive_dev_sessions": pos_sessions,
        "positive_dev_session_dates": {
            "eval_noapril": "2026-04-03", "eval_cad": "2026-05-13",
            "eval_outside": "2026-05-13", "eval_pallet07": "2026-05-13",
            "eval_pallet09": "2026-05-13", "eval_night08": "2026-05-22",
            "eval_night09": "2026-05-22"},
        "negative_session_date": "2026-05-28",
        "session_overlap_with_positive_dev": False,
        "overlap_evidence": "촬영일·시퀀스가 모두 다르다. positive DEV 는 "
                            "capture* 세션(04-03~05-22), negative 는 "
                            "forklift_raw 05-28. 파일명 체계도 다르다"
                            "(ns 타임스탬프 vs 000000 연번).",
        "★SELECTION_BIAS": {
            "mechanism": "prepare_real_ft.py 가 `max_conf < 0.20` 인 프레임만 "
                         "negative 로 채택했다 (911 중 259).",
            "prior_model": "challenge stage_a/ft 계열 (paper_generic 아님)",
            "consequence": "팔레트가 없는데 이전 모델이 conf>=0.2 로 오검출한 "
                           "프레임은 이 셋에서 **체계적으로 배제**됐다. "
                           "따라서 여기서 잰 FP/image 는 **하한(lower bound)**이다.",
            "not_a_leak": "paper_generic 은 real 을 한 장도 안 봤으므로 학습 "
                          "누수는 아니다. 표본 편향이지 누수가 아니다.",
            "unadjudicated_frames": len(rows) - len(inv),
            "note": "나머지 652 장은 팔레트 유무가 판정되지 않았다 "
                    "(시퀀스 median max_conf 0.81 — 대부분 팔레트가 실제로 있다)."},
        "sealed": "이 259 장은 이 순간부터 threshold calibration DEV 다. "
                  "final test 로 재사용 금지.",
        "caveat_ft_models": "yolo26n_ft / yolo26m_ft 는 이 259 장을 학습했다. "
                            "그 두 모델의 FP 와 섞어 비교 금지.",
        "frames": inv}
    json.dump(audit, open(os.path.join(A2, "REAL_NEG_DEV_AUDIT.json"), "w"),
              indent=1, ensure_ascii=False)
    return audit


def boxes_of(result):
    if result.boxes is None or not len(result.boxes):
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    kps = result.keypoints.xy.cpu().numpy()
    kpc = (result.keypoints.conf.cpu().numpy()
           if result.keypoints.conf is not None else None)
    out = []
    for i in range(len(conf)):
        out.append({"conf": float(conf[i]),
                    "xyxy": (xyxy[i] - PAD).tolist(),
                    "kps": (kps[i] - PAD).tolist(),
                    "kp_conf": None if kpc is None else kpc[i].tolist()})
    return sorted(out, key=lambda b: -b["conf"])


def main():
    audit = phase0_audit()
    print(f"PHASE 0  REAL_NEG_DEV_V1  n={audit['n_frames']}  "
          f"paper_generic 학습사용={audit['used_in_paper_generic_training']}  "
          f"session overlap={audit['session_overlap_with_positive_dev']}")
    from ultralytics import YOLO
    model = YOLO(CK, task="pose")

    def run(image_path):
        im = cv2.imread(image_path)
        if im is None:
            return None
        p = cv2.copyMakeBorder(im, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
        return boxes_of(model.predict(p, imgsz=IMGSZ, conf=DUMP_CONF,
                                      verbose=False)[0])

    pos = json.load(open(os.path.join(PIPE, "eval_manifest.json")))["items"]
    dump = {"weights": os.path.relpath(CK, ROOT),
            "recipe": {"pad": PAD, "imgsz": IMGSZ, "dump_conf": DUMP_CONF,
                       "border": "BORDER_REFLECT_101",
                       "selection": "top-1 by box conf among survivors"},
            "positive": [], "negative": []}
    for it in pos:
        b = run(os.path.join(ROOT, it["image"]))
        dump["positive"].append({"fid": it["frame_id"], "set": it["set"],
                                 "population": it["population"],
                                 "boxes": b or []})
    print(f"  positive {len(dump['positive'])} 장 덤프 완료")
    for r in audit["frames"]:
        b = run(os.path.join(ROOT, r["image"]))
        dump["negative"].append({"frame": r["frame"], "boxes": b or []})
    print(f"  negative {len(dump['negative'])} 장 덤프 완료")
    json.dump(dump, open(os.path.join(A2, "_cc_raw_dump.json"), "w"), indent=1)
    npos = sum(1 for e in dump["positive"] if e["boxes"])
    nneg = sum(1 for e in dump["negative"] if e["boxes"])
    print(f"  conf>={DUMP_CONF} 후보 보유: positive {npos}/{len(dump['positive'])}  "
          f"negative {nneg}/{len(dump['negative'])}")


if __name__ == "__main__":
    main()

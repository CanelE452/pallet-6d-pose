"""YOLO 덤프에 FINAL40K 와 **같은** pose 경로를 태우고 같은 규약으로 그린다.

`ft_vdo_infer` 에서 `draw` · 색 · 엣지 · 치수 가정을 그대로 가져온다.  따로 그리면
두 그림이 다른 것을 말하게 된다.  K 도 같은 `internet_pallet_infer.fit_K`(HFOV
스윕) 를 쓴다 -- 이 영상에는 cam_K 가 없어서 어느 쪽이든 지어내야 하는데, 한쪽만
다른 방식을 쓰면 비교가 아니라 두 실험이 된다.

한 가지는 같게 만들 수 없고, 그래서 헤더에 적는다.

    FINAL40K   corner + line -> `solve_arms` 의 F3 (line 이 회전을 다시 맞춘다)
    YOLO       점 9 개뿐 -> `APNP.solve_pose` (점 PnP).  F0 에 대응한다.

검출 수의 정의도 두 모델이 다르다.  DOPE 는 belief threshold 0.3 을 넘은 채널을
세지만, YOLO 는 박스가 잡히면 가려진 코너까지 9 개를 **항상** 출력한다.  그래서
`det 9/9` 가 공짜로 나온다 -- 여기서는 keypoint confidence 0.5 를 넘은 것만 세고
그 기준을 헤더에 박는다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = "/home/minjae/Documents/github/pallet-pose"
for sub in ("scripts/stage0", "scripts/stage0/paper_s2", "scripts/stage0/multihead",
            "scripts/stage0/line", "scripts/stage0/real_eval", "challenge",
            "scripts/annotate", "scripts/stage0/final_train"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import cv2                                        # noqa: E402
import annotate_pnp as APNP                       # noqa: E402
import internet_pallet_infer as IPI               # noqa: E402
import ft_vdo_infer as FV                         # noqa: E402

OUT = os.path.join(ROOT, "data/pallet/results/model_compare/vdo_infer")
SRC = FV.SRC
DIMS = FV.DIMS
KP_CONF = 0.5


def pose_from_points(pred8, pred_c, shape):
    """FINAL40K 와 같은 HFOV 스윕으로 K 를 잡고, 그 K 로 점 PnP 를 푼다."""
    fit = IPI.fit_K(pred8, pred_c, DIMS, shape)
    if fit is None:
        return None, None, "PnP skipped (det<6)"
    K, hfov, reproj = fit
    kps9 = IPI._kps9_from_pred(pred8, pred_c)
    pose = APNP.solve_pose(kps9, K, dims=DIMS, img_shape=shape)
    if pose is None:
        return None, None, "PnP FAIL"
    model_points = APNP.make_pallet_keypoints_3d_diagram(
        width=DIMS[0], depth=DIMS[1], height=DIMS[2])[:8]
    R = np.asarray(pose["R"], float)
    t = np.asarray(pose["t"], float).reshape(3)
    camera = (R @ model_points.T).T + t
    depth = np.clip(camera[:, 2], 1e-6, None)
    projection = (K @ (camera / depth[:, None]).T).T[:, :2]
    note = (f"point-PnP  hfov {hfov:.0f}deg  reproj {reproj:.1f}px  "
            f"dist {np.linalg.norm(t):.2f}m")
    return projection, pose, note


def run(name):
    source = os.path.join(OUT, f"vdo_kps_{name}.json")
    if not os.path.exists(source):
        print(f"  {name}: 덤프 없음 -> 건너뜀", flush=True)
        return None
    payload = json.load(open(source))
    panels, records = [], []

    for entry in payload["frames"]:
        stem = entry["frame"]
        image = cv2.imread(os.path.join(SRC, f"{stem}.png"))
        height, width = image.shape[:2]
        pred8 = np.full((8, 2), np.nan)
        pred_c = None
        n_det = 0
        if entry["kps"] is not None:
            kps = np.asarray(entry["kps"], float)
            confidence = (np.asarray(entry["kp_conf"], float)
                          if entry["kp_conf"] is not None
                          else np.ones(len(kps)))
            for i in range(min(8, len(kps))):
                if confidence[i] >= KP_CONF:
                    pred8[i] = kps[i]
                    n_det += 1
            if len(kps) > 8 and confidence[8] >= KP_CONF:
                pred_c = kps[8].tolist()

        projection, _, note = pose_from_points(pred8, pred_c, image.shape)
        colour = (0, 255, 0) if n_det >= 6 else (0, 200, 255)
        # cv2.putText 는 한글을 못 그린다(물음표로 나온다). 헤더는 ASCII 로만.
        header = (f"{stem}   det {n_det}/8 @kp_conf{KP_CONF}   "
                  f"box_conf {entry['box_conf']:.2f}" if entry["box_conf"]
                  is not None else f"{stem}   NO DETECTION (0 instances)")
        sub = (note + "   | dims ASSUMED 1.1x1.3x0.11m  K from HFOV fit  "
               f"| {name}")
        panels.append(FV.draw(image, pred8, pred_c, projection,
                              (header, colour), (sub, (200, 200, 200))))
        records.append({"frame": stem, "n_det": n_det,
                        "box_conf": entry["box_conf"], "note": note})
        print(f"  {name:26} {stem}  det {n_det}/8  {note}", flush=True)
        cv2.imwrite(os.path.join(OUT, f"{stem}_{name}.jpg"), panels[-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    path = os.path.join(OUT, f"vdo_grid_{name}.jpg")
    cv2.imwrite(path, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"model": name, "weights": payload["weights"],
            "train_data": payload["train_data"], "frames": records,
            "pose_route": "point PnP (APNP.solve_pose) -- YOLO 에는 line 이 없다. "
                          "FINAL40K 의 F3 는 line 으로 회전을 다시 맞추므로 같은 팔이 아니다.",
            "det_rule": f"keypoint confidence >= {KP_CONF} "
                        "(YOLO 는 박스가 잡히면 9 점을 항상 출력한다)",
            "grid": os.path.relpath(path, ROOT)}


def main():
    reports = [r for r in (run("yolo26n_paper_generic_v1"),
                           run("yolo26n_broad40k_5ep")) if r is not None]
    target = os.path.join(OUT, "VDO_YOLO_PAPER.json")
    json.dump({"models": reports,
               "scope_warning": "논문 트랙 = BROAD 합성 40,000 뿐, real 0장. "
                                "이 배포영상은 적용범위 밖이다. 못 잡아도 결함이 아니라 "
                                "범위 밖이라는 뜻이다.",
               "K": "unknown -> HFOV sweep (internet_pallet_infer.fit_K), "
                    "FINAL40K 와 동일",
               "dims": "ASSUMED PALLET_DIMS 1.1 x 1.3 x 0.11 m, FINAL40K 와 동일",
               "compare_with": "data/pallet/results/paper_s2_multihead/final_train/"
                               "vdo_infer/vdo_infer_seed1.json"},
              open(target, "w"), indent=1, ensure_ascii=False)
    on_disk = len([f for f in os.listdir(OUT) if f.endswith(".jpg")])
    print(f"완료: 모델 {len(reports)}종, 디스크 jpg {on_disk}개 -> "
          f"{os.path.relpath(OUT, ROOT)}", flush=True)


if __name__ == "__main__":
    main()

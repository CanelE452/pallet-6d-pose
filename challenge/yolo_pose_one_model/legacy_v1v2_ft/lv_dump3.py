"""목표/이전/개선 3종 키포인트 덤프 — 기존 mc_dump_yolo 를 그대로 태운다.

추론 계약(PAD=100 reflect, imgsz 640, conf 0.4, 최고신뢰 인스턴스)을 새로 쓰지
않는다. MODELS 만 갈아끼우고 main() 을 부른다 — 계약이 갈라지면 그림이 표와
다른 말을 한다.
"""
import os
import sys

ROOT = "/home/minjae/Documents/github/pallet-pose"
sys.path.insert(0, os.path.join(ROOT, "scripts/stage0/model_compare"))
import mc_dump_yolo as MD  # noqa: E402

MD.MODELS = {
    # 목표 — real GT 로 학습한 배포본 (upper bound)
    "LV3_TARGET": "/home/minjae/Documents/github/25y_automatic_lifter-master/"
                  "pallet_yolo26n_pose_ft.pt",
    # 이전 — 현 challenge BEST, target 을 한 장도 안 본 모델
    "LV3_BEFORE": "challenge/yolo_pose_one_model/runs_camera_facing_loss/"
                  "OLD_ROOT_G38_GENERIC_ONLY_60EP_SEED42/weights/last.pt",
    # 개선 — 위에 LEGACY_V1V2_P0_10K 를 15ep FT
    "LV3_AFTER": "challenge/yolo_pose_one_model/legacy_v1v2_ft/runs/"
                 "LV1V2_FT_15EP_SEED42/weights/last.pt",
}

if __name__ == "__main__":
    MD.main()

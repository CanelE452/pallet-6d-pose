# calib/perception.py
# YOLO 추론으로 front class 마스크/박스 선택 (GPU/FP16 안전 버전, PERF 로깅 제거)
# + CUDA/GPU 사용 상태를 명확히 보여주는 print 로그 추가

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import Optional, Tuple, List
from .config import MODEL_PATH, FRONT_CLASS_NAME, CONF_THR, USE_GPU, CUDA_DEVICE, USE_HALF
from .geometry import clamp_bbox


class Perception:
    def __init__(self,
                 model_path: str = MODEL_PATH,
                 front_class_name: str = FRONT_CLASS_NAME,
                 conf_thr: float = CONF_THR):
        # 1) 디바이스 선택
        cuda_available = torch.cuda.is_available()
        self.device = f"cuda:{CUDA_DEVICE}" if (USE_GPU and cuda_available) else "cpu"

        # ---- 상태 출력 (초기화) ----
        print("==========[Perception / CUDA Status]==========")
        print(f"[Perception] torch.__version__           : {torch.__version__}")
        # torch.version.cuda는 빌드된 CUDA 버전 정보(없을 수도 있음)
        try:
            print(f"[Perception] torch.version.cuda         : {torch.version.cuda}")
        except Exception:
            print(f"[Perception] torch.version.cuda         : (unavailable)")
        print(f"[Perception] USE_GPU (config)           : {USE_GPU}")
        print(f"[Perception] CUDA available             : {cuda_available}")
        print(f"[Perception] Requested CUDA_DEVICE      : {CUDA_DEVICE}")
        print(f"[Perception] Selected device            : {self.device}")

        if str(self.device).startswith("cuda") and cuda_available:
            # 안전하게 device index 추출
            try:
                dev_idx = int(str(self.device).split(":")[1])
            except Exception:
                try:
                    dev_idx = torch.cuda.current_device()
                except Exception:
                    dev_idx = 0
            try:
                print(f"[Perception] CUDA device name          : {torch.cuda.get_device_name(dev_idx)}")
            except Exception:
                print(f"[Perception] CUDA device name          : (unavailable)")
            try:
                print(f"[Perception] torch.cuda.current_device : {torch.cuda.current_device()}")
            except Exception:
                print(f"[Perception] torch.cuda.current_device : (unavailable)")
        else:
            print("[Perception] Running on CPU (no CUDA).")
        print("==============================================")

        # 2) 모델 로드 (여기서는 half() 하지 않음! → fuse 전에 dtype 섞임 방지)
        self.model = YOLO(model_path)

        moved_ok = False
        try:
            # 가능하면 디바이스만 이동
            self.model.model.to(self.device)
            moved_ok = True
        except Exception:
            # 일부 버전/백엔드에서 .model 접근이 없을 수도 있으므로 무시
            moved_ok = False

        print(f"[Perception] Model.to({self.device}) attempted; success={moved_ok}")

        self.front_class_name = front_class_name
        self.conf_thr = conf_thr
        self.front_idx: Optional[int] = None

    def _resolve_front_idx(self) -> bool:
        """모델 names에서 FRONT_CLASS_NAME에 해당하는 클래스 인덱스를 찾는다."""
        if self.front_idx is not None:
            return True
        names = getattr(self.model, "names", None)
        if isinstance(names, dict) and names:
            for k, v in names.items():
                try:
                    if v == self.front_class_name:
                        self.front_idx = int(k)
                        return True
                except Exception:
                    continue
        return False

    def infer_front(self, color_img) -> Tuple[bool, Optional[np.ndarray], Optional[tuple]]:
        """
        color_img: (H, W, 3) BGR uint8
        return: (det_ok, mask_bin(H,W,0/1), bbox_xyxy or None)
        """
        H, W = color_img.shape[:2]

        # half는 GPU에서만 유효. (여기서만 half를 켠다 → fuse 이후 안전)
        use_half = bool(USE_HALF and str(self.device).startswith("cuda"))

        # ---- 상태 출력 (추론 직전) ----
        print("-----[Perception / Inference]-----")
        print(f"[Perception] Inference device : {self.device}")
        print(f"[Perception] Use FP16 (half)  : {use_half}")
        print("----------------------------------")

        # YOLO 추론
        res = self.model.predict(
            source=color_img,
            device=self.device,
            verbose=False,
            conf=self.conf_thr,
            half=use_half
        )[0]

        det_ok = False
        mask_bin = None
        bbox_now = None

        # 유효성 검사
        if res.masks is None or res.boxes is None:
            print("[Perception] No masks/boxes in result.")
            return det_ok, mask_bin, bbox_now
        if (not hasattr(res.masks, "data")) or (res.masks.data is None) or len(res.masks.data) == 0:
            print("[Perception] Empty mask data.")
            return det_ok, mask_bin, bbox_now
        if (not hasattr(res.boxes, "xyxy")) or (res.boxes.xyxy is None) or len(res.boxes) == 0:
            print("[Perception] Empty boxes.")
            return det_ok, mask_bin, bbox_now

        # 클래스 인덱스 식별 (모델에서 실패했으면 결과에서 재시도)
        if not self._resolve_front_idx():
            names_from_res = getattr(res, "names", {})
            if isinstance(names_from_res, dict) and names_from_res:
                for k, v in names_from_res.items():
                    try:
                        if v == self.front_class_name:
                            self.front_idx = int(k)
                            break
                    except Exception:
                        continue
        if self.front_idx is None:
            print(f"[Perception] Could not resolve front class index for '{self.front_class_name}'.")
            return det_ok, mask_bin, bbox_now

        # 결과 텐서 → numpy
        masks = res.masks.data.detach().cpu().numpy()
        classes = res.boxes.cls.detach().cpu().numpy().astype(int) if hasattr(res.boxes, "cls") else None
        confs   = res.boxes.conf.detach().cpu().numpy().astype(float) if hasattr(res.boxes, "conf") else None
        xyxy    = res.boxes.xyxy.detach().cpu().numpy().astype(float) if hasattr(res.boxes, "xyxy") else None

        if classes is None or confs is None or xyxy is None:
            print("[Perception] Missing classes/confs/xyxy tensors.")
            return det_ok, mask_bin, bbox_now

        # 1) 클래스=front & 2) confidence>=thr 필터링
        idxs: List[int] = [i for i, (c, p) in enumerate(zip(classes, confs))
                           if (c == self.front_idx and p >= self.conf_thr)]
        if not idxs:
            print(f"[Perception] No detection for class '{self.front_class_name}' over conf {self.conf_thr}.")
            return det_ok, mask_bin, bbox_now

        # === 선택 기준 ===
        TARGET_AR = 1100.0 / 125.0  # ≈ 8.8 (표준 파렛트 front 가로/세로 비)
        def ar_distance(i: int) -> tuple:
            x1, y1, x2, y2 = xyxy[i]
            w = max(1.0, (x2 - x1))
            h = max(1.0, (y2 - y1))
            ar = w / h
            # 1순위: |ar - 8.8| 최소, 2순위: conf 큰 것(-confs[i])
            return (abs(ar - TARGET_AR), -float(confs[i]))

        best_idx = min(idxs, key=ar_distance)

        # 선택된 인덱스의 마스크/바운딩 박스 반환
        x1f, y1f, x2f, y2f = xyxy[best_idx]
        m_full = cv2.resize(masks[best_idx], (W, H), interpolation=cv2.INTER_NEAREST)
        mask_bin = (m_full > 0.5).astype(np.uint8)
        bbox_now = clamp_bbox(int(x1f), int(y1f), int(x2f), int(y2f), W, H)

        det_ok = True
        print(f"[Perception] Selected detection idx={best_idx}, conf={float(confs[best_idx]):.4f}")
        return det_ok, mask_bin, bbox_now

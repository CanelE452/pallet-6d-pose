#!/usr/bin/env python3
"""배포용 checkpoint 에서 커스텀 모델 클래스 참조를 떼어낸다.

`pallet_yolo_loss` 의 커스텀 loss 로 학습하면 checkpoint 안의 `model` / `ema` 가
`pallet_yolo_loss.model.*PoseModel` 인스턴스로 pickle 된다.  그 모듈이 없는 환경
(= 모델을 내려받은 사람)에서는 `ModuleNotFoundError` 로 **아예 로드되지 않는다.**
2026-09-06 확인: 이미 공개된 `pallet-pose-yolo26n-c4` 도 이 상태였다.

커스텀 클래스는 `PoseModel` 을 상속해 `init_criterion` 만 바꾼 것이고 그 메서드는
학습에만 쓰인다.  그래서 `__class__` 를 표준 `PoseModel` 로 되돌려도 추론 결과는
바뀌지 않는다 — 이 스크립트는 바꾼 뒤 **같은 입력의 출력이 동일한지 검증한다.**

    python strip_custom_model_class.py <in.pt> <out.pt> [--keep-optimizer]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))          # 커스텀 클래스를 불러올 수 있어야 로드된다

import torch  # noqa: E402
from ultralytics.nn.tasks import PoseModel  # noqa: E402


def rebase(obj) -> str | None:
    """커스텀 클래스면 PoseModel 로 되돌리고 원래 이름을 돌려준다."""
    if obj is None:
        return None
    name = f"{type(obj).__module__}.{type(obj).__name__}"
    if type(obj) is PoseModel:
        return None
    if not isinstance(obj, PoseModel):
        raise SystemExit(f"[STOP] PoseModel 파생이 아니다: {name}")
    obj.__class__ = PoseModel
    return name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--keep-optimizer", action="store_true",
                    help="배포본은 보통 optimizer/scaler 를 뺀다 (기본 제거)")
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    ck = torch.load(src, map_location="cpu", weights_only=False)

    # 변환 전 출력을 기록해 둔다 — 변환이 모델을 바꾸지 않았음을 보이기 위해.
    # ★ `.float()` 은 in-place 다.  원본에 직접 부르면 fp16 저장본이 fp32 가 되어
    #   배포 파일이 두 배로 불어난다(실제로 6.55 -> 12.80 MB 로 겪었다).  사본에서 잰다.
    import copy
    x = torch.zeros(1, 3, 640, 640)

    def probe(net):
        with torch.no_grad():
            out = copy.deepcopy(net).float().eval()(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    before = probe(ck.get("ema") or ck["model"])

    changed = [n for n in (rebase(ck.get("model")), rebase(ck.get("ema"))) if n]
    if not changed:
        print("커스텀 클래스 없음 — 그대로 복사할 필요도 없다")
        return 0
    print("되돌린 클래스:", ", ".join(sorted(set(changed))), "-> ultralytics PoseModel")

    after = probe(ck.get("ema") or ck["model"])
    delta = (before - after).abs().max().item()
    if delta != 0.0:
        raise SystemExit(f"[STOP] 변환이 출력을 바꿨다 (max |diff| = {delta})")
    print(f"출력 불변 확인: max |diff| = {delta}")

    if not args.keep_optimizer:
        for k in ("optimizer", "scaler", "updates"):
            ck.pop(k, None)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)
    print(f"저장 {dst}  ({dst.stat().st_size/1e6:.2f} MB, 원본 {src.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

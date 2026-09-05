"""기존 pose closure evaluator 를 **한 줄도 고치지 않고** 이 트랙의 예측에 돌린다.

모듈을 import 한 뒤 두 전역만 이 트랙 폴더로 바꾼다.

    PREDICTIONS -> 이 트랙의 predictions/
    OUT_DIR     -> 이 트랙의 결과 폴더

GT_PATH · MANIFEST 는 import 시점에 이미 읽기 전용 원본으로 확정돼 있어 건드리지 않는다.
따라서 6D 지표 코드·물체 계약·selector 는 논문 트랙과 동일하다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M

CLOSURE_SCRIPTS = M.REPO_ROOT / "scripts/paper/pose_metric_closure_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out-dir", default=str(M.GATE_D))
    args = parser.parse_args()

    sys.path.insert(0, str(CLOSURE_SCRIPTS))
    sys.path.insert(0, str(M.REPO_ROOT))
    import run_pose_evaluation as rpe

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rpe.PREDICTIONS = M.PREDICTIONS
    rpe.OUT_DIR = out_dir

    sys.argv = ["run_pose_evaluation",
                "--pose-object-contract", str(M.POSE_CONTRACT_PATH),
                "--arm", args.arm]
    return rpe.main()


if __name__ == "__main__":
    raise SystemExit(main())

"""PHASE 10 — paper negative FT 데이터셋 조립 (준비만, 기본 HARD BLOCK).

ratio 를 지금 확정하지 않는다.  `--negative-ratio` 를 반드시 명시해야 돌아간다.
target positive synthetic 과 target real positive 는 어떤 경로로도 들어오지 않는다.
"""
from __future__ import annotations
import argparse, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negative-ratio", type=float, default=None,
                    help="필수. 미지정이면 HARD BLOCK — 기본값을 발명하지 않는다")
    ap.add_argument("--negative-src", required=True,
                    help="target-free negative 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-have-user-approval", action="store_true")
    a = ap.parse_args()
    if a.negative_ratio is None:
        sys.exit("HARD BLOCK: --negative-ratio 를 명시하라. 기본값 없음.")
    if not a.i_have_user_approval:
        sys.exit("HARD BLOCK: 사용자 승인 플래그가 없다.")
    verdict = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "../../evaluation/PAPER_YOLO_VERDICT.json")
    if not os.path.exists(verdict) or \
            json.load(open(verdict))["verdict"] != "STRONG_PASS":
        sys.exit("HARD BLOCK: positive 가 STRONG_PASS 여야 한다.")
    sys.exit("여기서부터는 REAL_NEG_DEV 확보 후 구현한다 — 지금은 스텁이다.")


if __name__ == "__main__":
    main()

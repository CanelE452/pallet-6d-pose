"""seed43/44 의 args 가 seed42 와 seed 말고 전부 같은지 검사한다.

"같은 레시피" 는 선언이 아니라 검사다.  허용 차이는 seed / name / save_dir 뿐이고,
나머지가 하나라도 다르면 재현 실험이 아니라 다른 실험이다.
"""
from __future__ import annotations
import json, sys, yaml, pathlib

ALLOWED = {"seed", "name", "save_dir", "project"}
R = pathlib.Path("/home/minjae/Documents/github/pallet-pose")
BASE = R/"challenge/yolo_pose_one_model/runs_paper/yolo26n_paper_generic_v1_seed42/args.yaml"


def compare(other):
    a = yaml.safe_load(open(BASE))
    b = yaml.safe_load(open(other))
    diff = {k: {"seed42": a.get(k), "other": b.get(k)}
            for k in set(a) | set(b)
            if k not in ALLOWED and a.get(k) != b.get(k)}
    return diff


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: compare_args.py <other args.yaml>")
    d = compare(sys.argv[1])
    print(json.dumps(d, indent=1, default=str) if d
          else "IDENTICAL (seed/name/path 제외)")
    sys.exit(1 if d else 0)

"""§17 LINE_FUSION_IMPLEMENTATION_GATE — 기존 T1~T7 을 그대로 다시 돌린다.

    python3 scripts/paper/fast6d_screen_v1b/run_fusion_unit_tests.py \
        --output-dir data/pallet/results/paper_fast6d_screen_v1b

새 test 를 만들지 않는다.  `mh_fusion.run_tests` 를 **손대지 않고** 실행하되,
그 모듈의 OUT 만 V1B 네임스페이스로 돌려 read-only artifact
(`paper_s2_multihead/fusion_unit_tests.json`) 를 덮어쓰지 않는다.  읽어야 하는
두 파일은 symlink 로 걸어준다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for sub in ("scripts/stage0", "scripts/stage0/paper_s2", "scripts/stage0/multihead",
            "scripts/stage0/line", "scripts/stage0/real_eval", "challenge",
            "scripts/annotate"):
    sys.path.insert(0, str(ROOT / sub))

SOURCE = ROOT / "data/pallet/results/paper_s2_multihead"
NEEDED = ["theta_posealigned_d0.json",
          "mh_predcache_e3confirm25k_seed1_D2_MH_DEV512.npz"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    sandbox = out_dir / "audit" / "fusion_test_sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    for name in NEEDED:
        link = sandbox / name
        if not link.exists():
            os.symlink(SOURCE / name, link)

    import mh_fusion as FU
    FU.OUT = sandbox
    before = (SOURCE / "fusion_unit_tests.json").read_bytes()
    FU.run_tests(None)                          # 실패하면 SystemExit 을 던진다
    after = (SOURCE / "fusion_unit_tests.json").read_bytes()
    assert before == after, "read-only artifact was modified — abort"

    report = json.loads((sandbox / "fusion_unit_tests.json").read_text())
    historical = json.loads(before)
    verdict = {
        "LINE_FUSION_IMPLEMENTATION_GATE": "PASS" if report.get("PASS") else "FAIL",
        "rerun": report,
        "historical_record": historical,
        "agrees_with_historical_pass": bool(report.get("PASS") and historical.get("PASS")),
        "read_only_artifact_unmodified": True,
        "note": "mh_fusion.run_tests executed unchanged; only its OUT was redirected",
    }
    (out_dir / "audit" / "LINE_FUSION_UNIT_TESTS.json").write_text(
        json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))
    return 0 if report.get("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())

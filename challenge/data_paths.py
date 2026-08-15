"""challenge/data 경로 레지스트리 — 하드코딩을 한 곳으로 모으기 위한 단일 진실원천.

지금까지 `challenge/data/...` 문자열이 92개 파일 272 라인에 흩어져 있어서 폴더를
옮기는 것 자체가 불가능했다. 이 모듈이 그 경로를 전부 이름으로 들고 있고, 재편
(01_real / 02_synthetic / 03_derived / 04_results)은 여기 값만 바꾸면 반영된다.

2026-08-14 재편으로 아래 값은 **실제 위치**다. 호환용 symlink 71개는 참조를 모두
전환한 뒤 제거했으므로, 옛 경로(`challenge/data/capturepallet07_manual_gt` 등)는
더 이상 존재하지 않는다.

파이썬에서:

    import sys, pathlib
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    from challenge.data_paths import EVAL_CANONICAL, get

    for name, path in EVAL_CANONICAL.items():
        ...

셸/yaml 에서 (import 이 안 되므로 CLI 로):

    python challenge/data_paths.py --get eval_cad          # ROOT 상대 경로
    python challenge/data_paths.py --get eval_cad --abs    # 절대 경로
    python challenge/data_paths.py --list
    python challenge/data_paths.py --check                 # 전 경로 존재 검사
"""
from __future__ import annotations

import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = "challenge/data"

# ── 정본 평가셋 ───────────────────────────────────────────────────────────
# objects[0].split == "eval" 인 manual GT. 다른 셋으로 대체 금지.
# 상세: _docs/EVAL_SET_CANONICAL.md / 강제: challenge/tests/test_eval_set_canonical.py
#
# ⚠ 프레임 수는 161 이지 56 이 아니다. CLAUDE.md 와 memory 의 "56장" 은 07-3x 기준
# 이라 한 세대 뒤처졌다 — 2026-08-07/08-08 에 metric_split_lock.md §1.6 의
# final-test 세션 4개를 봉인 해제해 정본에 편입했기 때문. 근거는 테스트의
# EVAL_FOLDERS/EXPECTED_TOTAL 이고, 이 모듈은 그쪽을 따른다.
EVAL_CANONICAL = {
    "eval_outside": f"{DATA}/01_real/eval_canonical/_outside_eval_manual_gt",       # 22
    "eval_noapril": f"{DATA}/01_real/eval_canonical/capture0403noapril_manual_gt",  # 12
    "eval_cad": f"{DATA}/01_real/eval_canonical/capturepalletcad_manual_gt",        # 22
    # 아래 넷은 lock §1.6 의 final-test 세션. 정본에 편입됐지만 PL 풀·threshold
    # 캘리브에 들어가면 안 된다(transductive 차단). FINAL_TEST 로도 노출한다.
    "eval_pallet07": f"{DATA}/01_real/manual_gt/capturepallet07_manual_gt",         # 27
    "eval_pallet09": f"{DATA}/01_real/manual_gt/capturepallet09_manual_gt",         # 36
    "eval_night08": f"{DATA}/01_real/manual_gt/capturenight08_manual_gt",           # 17
    "eval_night09": f"{DATA}/01_real/manual_gt/capturenight09_manual_gt",           # 25
}
EVAL_CANONICAL_TOTAL = 161

# lock §1.6 final-test — 위 정본의 부분집합. PL 풀에 넣으면 안 된다.
FINAL_TEST = ("eval_pallet07", "eval_pallet09", "eval_night08", "eval_night09")

# lock §1.6 filter-val 세션(threshold 캘리브에 쓰임). 정본에 남지만
# final-test 로 보고하면 안 된다.
FILTER_VAL_SESSIONS = ("capturepallet02", "capturepallet03", "capturepallet04",
                       "capturepallet05", "capturepallet08")

# 폐기된 구본. 평가·development·probe 어디에도 쓰지 않는다.
FORBIDDEN_EVAL_SOURCES = ("_eval_sets/outside_combined", "_eval_sets/night_combined")

# ── 촬영본 + 손 어노테이션 GT ─────────────────────────────────────────────
REAL_MANUAL_GT = {
    "pallet02": f"{DATA}/01_real/manual_gt/capturepallet02_manual_gt",
    "pallet03": f"{DATA}/01_real/manual_gt/capturepallet03_manual_gt",
    "pallet04": f"{DATA}/01_real/manual_gt/capturepallet04_manual_gt",
    "pallet05": f"{DATA}/01_real/manual_gt/capturepallet05_manual_gt",
    "pallet07": f"{DATA}/01_real/manual_gt/capturepallet07_manual_gt",  # = EVAL_CANONICAL
    "pallet08": f"{DATA}/01_real/manual_gt/capturepallet08_manual_gt",
    "pallet09": f"{DATA}/01_real/manual_gt/capturepallet09_manual_gt",  # = EVAL_CANONICAL
    "night04": f"{DATA}/01_real/manual_gt/capturenight04_manual_gt",
    "night05": f"{DATA}/01_real/manual_gt/capturenight05_manual_gt",
    "night06": f"{DATA}/01_real/manual_gt/capturenight06_manual_gt",
    "night07": f"{DATA}/01_real/manual_gt/capturenight07_manual_gt",
    "night08": f"{DATA}/01_real/manual_gt/capturenight08_manual_gt",  # = EVAL_CANONICAL
    "night09": f"{DATA}/01_real/manual_gt/capturenight09_manual_gt",  # = EVAL_CANONICAL
    "night_eval": f"{DATA}/01_real/manual_gt/_night_eval_manual_gt",
    "wood_183705": f"{DATA}/01_real/manual_gt/wood_pallet_20260618_183705_manual_gt",
    "wood_184309": f"{DATA}/01_real/manual_gt/wood_pallet_20260618_184309_manual_gt",
    "pallet11": f"{DATA}/01_real/manual_gt/pallet11_gt",
    # ★리프터에 실제로 장착하고 찍은 유일한 세션 (2026-08-15 어노테이션 시작).
    # 원본 영상 data/pallet/raw_data/outside/forklift_raw_20260528_163408.mp4 (911f, 640x480),
    # 프레임은 같은 폴더의 forklift_raw_20260528/rgb/ 에 추출해 두었다.
    # 카메라 높이·각도·포크가 화면에 들어오는 조건이 배포와 같아서 real finetune 의 핵심 소스다.
    "forklift_20260528": f"{DATA}/01_real/manual_gt/forklift_20260528_manual_gt",
    # 아래 셋은 현재 비어 있다(0 files). 사용자 지시로 보존.
    "night01_EMPTY": f"{DATA}/01_real/manual_gt/capturenight01_manual_gt",
    "night03_EMPTY": f"{DATA}/01_real/manual_gt/capturenight03_manual_gt",
    "pallet01_EMPTY": f"{DATA}/01_real/manual_gt/capturepallet01_manual_gt",
}

# ── 모델이 만든 라벨(pseudo) — 이미지는 촬영본 ────────────────────────────
REAL_PSEUDO_GT = {
    "pallet01": f"{DATA}/01_real/pseudo_gt/capturepallet01_pseudo_gt",
    "pallet02": f"{DATA}/01_real/pseudo_gt/capturepallet02_pseudo_gt",
    "pallet03": f"{DATA}/01_real/pseudo_gt/capturepallet03_pseudo_gt",
    "pallet04": f"{DATA}/01_real/pseudo_gt/capturepallet04_pseudo_gt",
    "pallet05": f"{DATA}/01_real/pseudo_gt/capturepallet05_pseudo_gt",
    "pallet06": f"{DATA}/01_real/pseudo_gt/capturepallet06_pseudo_gt",
    "pallet08": f"{DATA}/01_real/pseudo_gt/capturepallet08_pseudo_gt",
    "pallet09": f"{DATA}/01_real/pseudo_gt/capturepallet09_pseudo_gt",
    "pallet10": f"{DATA}/01_real/pseudo_gt/capturepallet10_pseudo_gt",
}

REAL_MISC = {
    "pallet07_augmented": f"{DATA}/01_real/augmented/capturepallet07_augmented",
    "live_captures_EMPTY": f"{DATA}/01_real/_live_captures",
}

# ── 합성 (photogrammetry 파렛트 기반, challenge 전용 — 논문 사용 금지) ────
# v1/v2 는 논문 트랙에서 쓰지 않는다. memory: v1v2-challenge-only-not-paper
SYNTHETIC = {
    "training_root": f"{DATA}/02_synthetic/training",
    "v1": f"{DATA}/02_synthetic/training/v1",
    "v2": f"{DATA}/02_synthetic/training/v2",
    "v3": f"{DATA}/02_synthetic/training/v3",  # mask 는 JSON mask_rle 사용, PNG 금지
    "addon_v1": f"{DATA}/02_synthetic/training/addon_v1",
    "addon_v1_train": f"{DATA}/02_synthetic/training/addon_v1_train",
    "addon_v1_val": f"{DATA}/02_synthetic/training/addon_v1_val",
    "truncation_addon_v1": f"{DATA}/02_synthetic/training/truncation_addon_v1",
}

# ── 위 데이터를 가공해 만든 학습 입력 ─────────────────────────────────────
DERIVED = {
    "truncation_crops": f"{DATA}/03_derived/truncation_crops",
    "truncation_crops_dope": f"{DATA}/03_derived/truncation_crops_dope",
    "truncation_crops_palletobj": f"{DATA}/03_derived/truncation_crops_palletobj",
    "truncation_crops_synth": f"{DATA}/03_derived/truncation_crops_synth",
    "yolo_pose": f"{DATA}/03_derived/yolo_pose",
    "yolo_pose_padded": f"{DATA}/03_derived/yolo_pose_padded",
    "yolo_pose_manual": f"{DATA}/03_derived/yolo_pose_manual",
    "yolo_pose_manual_padded": f"{DATA}/03_derived/yolo_pose_manual_padded",
    "yolo_pose_cropaug_padded": f"{DATA}/03_derived/yolo_pose_cropaug_padded",
    "yolo_pose_cropaug_v2_padded": f"{DATA}/03_derived/yolo_pose_cropaug_v2_padded",
    "train_manual_pseudo": f"{DATA}/03_derived/_train_manual_pseudo",
    "train_pallet07_aug": f"{DATA}/03_derived/_train_pallet07_aug",
    "train_capturepallet07": f"{DATA}/03_derived/_train_capturepallet07",
    "eval_real_gt_merged": f"{DATA}/03_derived/_eval_real_gt_merged",
    "holdout_stems": f"{DATA}/03_derived/holdout_stems.txt",
}

# ── 추론 산출물 (데이터 아님) ─────────────────────────────────────────────
RESULTS = {
    "ab_crop_eval": f"{DATA}/04_results/ab_crop_eval",
    "ab_crop_eval_cropaug": f"{DATA}/04_results/ab_crop_eval_cropaug",
    "ab_crop_eval_cropaug_v2": f"{DATA}/04_results/ab_crop_eval_cropaug_v2",
    "cropaug_truncation_eval": f"{DATA}/04_results/cropaug_truncation_eval",
    "challenge_ft_forklift_eval": f"{DATA}/04_results/challenge_ft_forklift_eval",
    "forklift_cropaug_infer_frames": f"{DATA}/04_results/forklift_cropaug_infer_frames",
    "forklift_cropaug_v2_frames": f"{DATA}/04_results/forklift_cropaug_v2_frames",
    "forklift_cropaug_v2_NEW_infer_frames": f"{DATA}/04_results/forklift_cropaug_v2_NEW_infer_frames",
    "sanity": f"{DATA}/04_results/_sanity",
    "verify_truncation_aug": f"{DATA}/04_results/_verify_truncation_aug",
    # forklift 추론 영상 9개 (171M)
    "mp4_cropaug": f"{DATA}/04_results/forklift_cropaug_infer.mp4",
    "mp4_cropaug_v2": f"{DATA}/04_results/forklift_cropaug_v2_infer.mp4",
    "mp4_cropaug_v2_NEW": f"{DATA}/04_results/forklift_cropaug_v2_NEW_infer.mp4",
    "mp4_ft_s2": f"{DATA}/04_results/forklift_ft_s2_infer.mp4",
    "mp4_ft_s2_PAD": f"{DATA}/04_results/forklift_ft_s2_PAD_infer.mp4",
    "mp4_ft_s2_PAD160": f"{DATA}/04_results/forklift_ft_s2_PAD160_infer.mp4",
    "mp4_otftrunc": f"{DATA}/04_results/forklift_otftrunc_infer.mp4",
    "mp4_otftrunc_PAD": f"{DATA}/04_results/forklift_otftrunc_PAD_infer.mp4",
    "mp4_otftrunc_PAD160": f"{DATA}/04_results/forklift_otftrunc_PAD160_infer.mp4",
    "yolo_convert_log": f"{DATA}/04_results/yolo_pose_padded_convert.log",
}

_GROUPS = {
    "EVAL_CANONICAL": EVAL_CANONICAL,
    "REAL_MANUAL_GT": REAL_MANUAL_GT,
    "REAL_PSEUDO_GT": REAL_PSEUDO_GT,
    "REAL_MISC": REAL_MISC,
    "SYNTHETIC": SYNTHETIC,
    "DERIVED": DERIVED,
    "RESULTS": RESULTS,
}

# 그룹 접두어를 붙여 평탄화한 조회용 테이블.
# 같은 짧은 이름이 그룹마다 있으므로(pallet02 등) 접두어가 필요하다.
_PREFIX = {
    "EVAL_CANONICAL": "",
    "REAL_MANUAL_GT": "manual.",
    "REAL_PSEUDO_GT": "pseudo.",
    "REAL_MISC": "real.",
    "SYNTHETIC": "synth.",
    "DERIVED": "derived.",
    "RESULTS": "results.",
}

ALL = {}
for _g, _d in _GROUPS.items():
    for _k, _v in _d.items():
        ALL[f"{_PREFIX[_g]}{_k}"] = _v


def get(name: str, absolute: bool = False) -> str:
    """이름으로 경로를 얻는다. absolute=True 면 절대 경로."""
    if name not in ALL:
        raise KeyError(f"unknown dataset name: {name!r}. --list 로 확인.")
    rel = ALL[name]
    return str(ROOT / rel) if absolute else rel


def missing() -> list[tuple[str, str]]:
    """존재하지 않는 경로를 (name, path) 로 돌려준다."""
    return [(n, p) for n, p in ALL.items() if not (ROOT / p).exists()]


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--get", metavar="NAME")
    ap.add_argument("--abs", action="store_true", help="--get 을 절대 경로로")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true", help="전 경로 존재 검사")
    a = ap.parse_args()

    if a.get:
        print(get(a.get, absolute=a.abs))
        return 0
    if a.list:
        for g, d in _GROUPS.items():
            print(f"[{g}]")
            prefix = _PREFIX[g]
            for k, v in d.items():
                print(f"  {prefix + k:42s} {v}")
        return 0
    if a.check:
        bad = missing()
        for n, p in bad:
            print(f"MISSING  {n:42s} {p}")
        print(f"{len(ALL) - len(bad)}/{len(ALL)} present")
        return 1 if bad else 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

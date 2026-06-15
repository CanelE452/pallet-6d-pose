#!/usr/bin/env python3
"""
split_lock.py — 사용자 확정 split 을 frame_id 리스트로 락 (P2 누수 차단의 CPU 파트).

원칙(어기지 않음): final-test 세션은 PL/threshold/checkpoint 어디에도 안 들어간다.
  pool ∩ val = ∅, pool ∩ test = ∅, val ∩ test = ∅ (세션 통째 단위 — 인접 프레임 누수 방지).
split 기준 = GT양 + 세션수(도메인당 2 test) + 대표성. 현재 모델 good/det 점수로 selection 안 함.

세션→frame 매핑은 raw rgb 폴더에서. GT 는 평탄화된 _eval_sets 를 frame_id 로 세션 역추적.
출력만 생성(읽기전용 audit). 데이터 수정 없음.
"""
import os, re, glob, json
from collections import defaultdict

OUTDIR = "data/pallet/eval_results/split_lock"

DOMAINS = {
    "outside": {"raw_root": "data/outside", "session_glob": "capturepallet[0-9]*",
                "gt": "data/_eval_sets/outside_combined"},
    "night":   {"raw_root": "data/night",   "session_glob": "capturenight[0-9]*",
                "gt": "data/_eval_sets/night_combined"},
}
# 사용자 확정 split (세션 단위)
SPLIT = {
    "outside": {
        "final_test": ["capturepallet09", "capturepallet07"],
        "filter_val": ["capturepallet08", "capturepallet02", "capturepallet03",
                       "capturepallet04", "capturepallet05"],
        "pl_pool":    ["capturepallet01", "capturepallet10", "capturepallet11"],
    },
    "night": {
        "final_test": ["capturenight09", "capturenight08"],
        "filter_val": ["capturenight06", "capturenight07", "capturenight05"],
        "pl_pool":    ["capturenight01", "capturenight02", "capturenight03",
                       "capturenight04", "capturenight10"],
    },
}
RGB_SUBDIR = "rgb"
EXCLUDE_FILE = "data/_eval_sets/_exclude.txt"


def frame_id(stem):
    nums = re.findall(r"\d{6,}", stem)
    return max(nums, key=len) if nums else stem


def list_session_frames(raw_root, session_glob):
    out = defaultdict(set)
    for sd in sorted(glob.glob(os.path.join(raw_root, session_glob))):
        if not os.path.isdir(sd):
            continue
        sess = os.path.basename(sd)
        rgb = os.path.join(sd, RGB_SUBDIR)
        search = rgb if os.path.isdir(rgb) else sd
        for p in glob.glob(os.path.join(search, "*")):
            if p.lower().endswith((".png", ".jpg", ".jpeg")):
                out[sess].add(frame_id(os.path.splitext(os.path.basename(p))[0]))
    return out


def load_exclude():
    if not os.path.exists(EXCLUDE_FILE):
        return set()
    return {l.strip() for l in open(EXCLUDE_FILE) if l.strip()}


def gt_frames(gt_dir, exclude):
    out = set()
    for p in glob.glob(os.path.join(gt_dir, "**", "*"), recursive=True):
        if p.lower().endswith(".png"):
            fid = frame_id(os.path.splitext(os.path.basename(p))[0])
            if fid not in exclude:
                out.add(fid)
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    exclude = load_exclude()
    assignment = {}
    final_test_ids, pl_pool_ids = [], []
    ok = True

    print(f"# split lock  (exclude={len(exclude)})\n")
    for dom, cfg in DOMAINS.items():
        sess_frames = list_session_frames(cfg["raw_root"], cfg["session_glob"])
        raw_map = {fid: s for s, fids in sess_frames.items() for fid in fids}
        gt = gt_frames(cfg["gt"], exclude)
        gt_by_sess = defaultdict(set)
        unmatched = 0
        for fid in gt:
            s = raw_map.get(fid)
            if s is None:
                unmatched += 1
            else:
                gt_by_sess[s].add(fid)

        spl = SPLIT[dom]
        # 검증: 배정 세션이 실재하나 + 세션 disjoint
        assigned = spl["final_test"] + spl["filter_val"] + spl["pl_pool"]
        missing = [s for s in assigned if s not in sess_frames]
        dup = len(assigned) != len(set(assigned))
        all_sess = set(sess_frames)
        unassigned = sorted(all_sess - set(assigned))

        def frames_of(sess_list):
            return set().union(*[sess_frames[s] for s in sess_list]) if sess_list else set()

        sets = {k: frames_of(spl[k]) for k in ("final_test", "filter_val", "pl_pool")}
        # 교집합(세션 단위는 자명히 disjoint, frame 단위까지 검증)
        inter = {
            "pool∩val": sets["pl_pool"] & sets["filter_val"],
            "pool∩test": sets["pl_pool"] & sets["final_test"],
            "val∩test": sets["filter_val"] & sets["final_test"],
        }
        disjoint = all(len(v) == 0 for v in inter.values())
        ok = ok and disjoint and not missing and not dup

        # GT 수 per split
        def gt_count(sess_list):
            return sum(len(gt_by_sess[s]) for s in sess_list)

        print(f"=== {dom} ===")
        for k in ("final_test", "filter_val", "pl_pool"):
            print(f"  {k:11} sessions={len(spl[k])}  frames={len(sets[k]):5}  GT={gt_count(spl[k])}"
                  f"   {spl[k]}")
        print(f"  disjoint(frame): {disjoint}  {dict((k, len(v)) for k,v in inter.items())}")
        if missing: print(f"  ⚠ 배정에 없는 세션: {missing}")
        if dup:     print(f"  ⚠ 세션 중복 배정")
        if unassigned: print(f"  ⚠ 미배정 세션(=풀/평가 어디에도 안 들어감): {unassigned}")
        if unmatched:  print(f"  · GT {unmatched}장 = raw 세션 매칭 안 됨(capturepalletcad 등 → split 제외)")
        print()

        final_test_ids += sorted(sets["final_test"])
        pl_pool_ids += sorted(sets["pl_pool"])
        assignment[dom] = {
            "final_test": {"sessions": spl["final_test"], "n_frames": len(sets["final_test"]), "n_gt": gt_count(spl["final_test"])},
            "filter_val": {"sessions": spl["filter_val"], "n_frames": len(sets["filter_val"]), "n_gt": gt_count(spl["filter_val"])},
            "pl_pool":    {"sessions": spl["pl_pool"],    "n_frames": len(sets["pl_pool"]),    "n_gt": gt_count(spl["pl_pool"])},
            "unassigned_sessions": unassigned,
            "gt_unmatched": unmatched,
        }

    # 출력 파일
    with open(os.path.join(OUTDIR, "final_test_exclude.txt"), "w") as f:
        f.write("\n".join(final_test_ids) + "\n")
    with open(os.path.join(OUTDIR, "pl_pool_frames.txt"), "w") as f:
        f.write("\n".join(pl_pool_ids) + "\n")
    with open(os.path.join(OUTDIR, "split_assignment.json"), "w") as f:
        json.dump(assignment, f, indent=2, ensure_ascii=False)

    print("생성:")
    print(f"  {OUTDIR}/final_test_exclude.txt   ({len(final_test_ids)} frame_id — PL/튜닝/ckpt 금지)")
    print(f"  {OUTDIR}/pl_pool_frames.txt        ({len(pl_pool_ids)} frame_id — PL 추출 허용)")
    print(f"  {OUTDIR}/split_assignment.json")
    print(f"\n전체 disjoint 검증: {'PASS ✅' if ok else 'FAIL ❌'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
session_inventory_v2.py — split 분기(A/B/C) 판정 + unlabeled∩test 누수 + 세션별 GT good 분포.

v1 대비 수정 (대화 합의 스펙):
  1) numeric-only filename 을 prefix/session 으로 안 씀 (H2 폭증 버그 제거).
  2) viability 를 전체 frame 수가 아니라 metrics 기반 good-count 로 판정. metrics 없으면 'A?'.
  3) session signal 보수적 선택 + singleton 과반이면 폐기.
  4) frame overlap 과 session overlap 분리.
  5) decision rule 도메인별 명시 출력.

v2 → v2.1 패치 (2026-06-10, 실제 데이터 구조 확인 후):
  6) [root] unlabeled = data/outside · data/night 의 capture 폴더. 세션 = 폴더명(capturepalletNN).
     rgb/ 만 카운트 (depth 더블카운트 제거 → master 9894/9134 와 일치).
  7) [GT 역매칭] GT(_eval_sets/*_combined)는 평탄화돼 세션정보 상실 → frame_id 로 raw 세션 역추적.
     (대화 '주의 1' 적중. 이게 없으면 GT 가 거짓 SINGLE.)
  8) [세션별 GT good 분포] final-test 세션 선택 = R1 유효성 판정의 입력. 세션 수만으론 부족 →
     세션별 detectable/good(<10px)/gross 분포 출력.
  - capturepalletcad 는 session_glob 에서 제외 (실제 사진이나 'CAD 보유 내 파렛트' 의심 → 논문
    unseen 풀에서 분리). 별도로 카운트만 리포트.

재구성본 — 원본 Codex v2 복구 시 diff 대조 권장. 순수 표준 라이브러리.
"""

import os, re, glob, json
from collections import defaultdict

# ============================== CONFIG ==============================
DOMAINS = {
    "outside": {"raw_root": "data/outside", "session_glob": "capturepallet[0-9]*",
                "gt": "data/_eval_sets/outside_combined"},
    "night":   {"raw_root": "data/night",   "session_glob": "capturenight[0-9]*",
                "gt": "data/_eval_sets/night_combined"},
}
RGB_SUBDIR = "rgb"
EXCLUDE_FILE = "data/_eval_sets/_exclude.txt"
METRICS_FILE = "data/pallet/eval_results/filter_pr_camfacing/per_frame_heldout_pretrain.json"
META_ID, META_ERR, META_DS, META_NDET = "frame", "mean_match_px", "dataset", "n_detected"
GOOD_PX, GROSS_PX, MIN_DET = 10.0, 20.0, 6
VIABLE_MIN_GOOD = 8          # final-test 후보 세션이 이 이상 good 가져야 쓸만
# capturepalletcad: 별도 리포트용 (논문 풀 제외 의심)
EXTRA_REPORT = {"outside": ("data/outside", "capturepalletcad")}
# ===================================================================


def load_exclude(path):
    if not path or not os.path.exists(path):
        return set()
    with open(path) as f:
        return {ln.strip() for ln in f if ln.strip()}


def frame_id(stem):
    nums = re.findall(r"\d{6,}", stem)
    return max(nums, key=len) if nums else stem


def list_session_frames(raw_root, session_glob, rgb_subdir):
    """{session_name: set(frame_id)} — capture 폴더의 rgb/ 만. [patch 6]"""
    out = defaultdict(set)
    for sess_dir in sorted(glob.glob(os.path.join(raw_root, session_glob))):
        if not os.path.isdir(sess_dir):
            continue
        sess = os.path.basename(sess_dir)
        rgb_dir = os.path.join(sess_dir, rgb_subdir)
        search = rgb_dir if os.path.isdir(rgb_dir) else sess_dir
        for p in glob.glob(os.path.join(search, "*")):
            if p.lower().endswith((".png", ".jpg", ".jpeg")):
                out[sess].add(frame_id(os.path.splitext(os.path.basename(p))[0]))
    return out


def list_gt_frames(gt_dir, exclude):
    out = set()
    for p in glob.glob(os.path.join(gt_dir, "**", "*"), recursive=True):
        if p.lower().endswith(".png"):
            fid = frame_id(os.path.splitext(os.path.basename(p))[0])
            if fid not in exclude:
                out.add(fid)
    return out


def load_metrics(path):
    """frame_id -> (err_px, n_detected). list-of-records 스키마. [v2 fix2 + patch]"""
    if not path or not os.path.exists(path):
        return None
    try:
        data = json.load(open(path))
    except Exception:
        return None
    recs = data if isinstance(data, list) else list(data.values())
    out = {}
    for r in recs:
        if not isinstance(r, dict) or META_ID not in r:
            continue
        try:
            err = float(r[META_ERR])
        except (KeyError, TypeError, ValueError):
            err = None
        out[frame_id(str(r[META_ID]))] = (err, r.get(META_NDET))
    return out or None


def classify(fid, metrics):
    """('good'|'gross'|'mid'|'undet'|'nometric')"""
    if metrics is None:
        return "nometric"
    rec = metrics.get(fid)
    if rec is None:
        return "nometric"
    err, ndet = rec
    if ndet is not None and ndet < MIN_DET:
        return "undet"
    if err is None:
        return "undet"
    if err < GOOD_PX:
        return "good"
    if err > GROSS_PX:
        return "gross"
    return "mid"


def decide(n_sess, total_good, n_good_sessions):
    if n_sess <= 1:
        return "B) temporal block + embargo (단일 세션 → frame random 금지)"
    if total_good is None:
        return "A?) 다세션 — metrics 붙여 good 분포 확인 후 확정"
    if n_good_sessions >= 2 and total_good >= 2 * VIABLE_MIN_GOOD:
        return ("A) session-level split 가능 — good 보유 세션 중 일부를 final-test 로. "
                "★단 기존 R1 이 전체 풀 학습이면 그 세션 제외하고 R1 재학습 필요")
    return "C) good 표본 부족 → domain-heldout 또는 final-test 격 하향"


def main():
    exclude = load_exclude(EXCLUDE_FILE)
    metrics = load_metrics(METRICS_FILE)
    print(f"# exclude={len(exclude)}  metrics={'loaded:'+str(len(metrics)) if metrics else 'NONE'}  "
          f"good<{GOOD_PX}px gross>{GROSS_PX}px det>={MIN_DET}\n")

    for dom, cfg in DOMAINS.items():
        sess_frames = list_session_frames(cfg["raw_root"], cfg["session_glob"], RGB_SUBDIR)
        raw_map = {fid: s for s, fids in sess_frames.items() for fid in fids}
        unl_total = sum(len(v) for v in sess_frames.values())
        gt = list_gt_frames(cfg["gt"], exclude)

        # 세션별 GT 분포 [patch 7+8]
        per = defaultdict(lambda: defaultdict(int))
        unmatched = 0
        for fid in gt:
            s = raw_map.get(fid)
            if s is None:
                unmatched += 1
                continue
            per[s]["gt"] += 1
            per[s][classify(fid, metrics)] += 1

        frame_int = gt & set(raw_map)                 # [fix4] frame overlap
        sess_int = set(per) & set(sess_frames)        # [fix4] session overlap (GT 세션이 풀에 있나)
        total_good = sum(d["good"] for d in per.values()) if metrics else None
        n_good_sess = sum(1 for d in per.values() if d["good"] > 0)

        print(f"=== {dom} ===")
        print(f"UNLABELED: sessions={len(sess_frames)}  frames(rgb)={unl_total}")
        print(f"GT: total={len(gt)}  matched={len(gt)-unmatched}  unmatched={unmatched}"
              + (f"  total_good={total_good}" if metrics else "  (no metrics)"))
        print(f"OVERLAP: frame={len(frame_int)}  session={len(sess_int)}  "
              f"(GT 세션 {len(sess_int)}/{len(per)} 가 unlabeled 풀에 존재)")
        print(f"BRANCH: {decide(len(sess_frames), total_good, n_good_sess)}")
        # 세션별 good 분포 (final-test 선택용)
        print("  per-session  [gt / good<10 / gross>20 / undet]:")
        for s in sorted(sess_frames):
            d = per.get(s)
            if not d:
                print(f"    {s:<18} gt=0  (GT 없음 = R1 누수 무관 → final-test 후보 아님)")
            else:
                print(f"    {s:<18} gt={d['gt']:<3} good={d['good']:<3} gross={d['gross']:<3} undet={d['undet']}")
        if unmatched:
            print(f"  ⚠ unmatched GT {unmatched}개 = raw 세션에서 frame_id 못 찾음 (다른 캡처/exclude 확인)")
        # 별도 리포트 (cad 등)
        if dom in EXTRA_REPORT:
            root, name = EXTRA_REPORT[dom]
            extra = list_session_frames(root, name, RGB_SUBDIR)
            n = sum(len(v) for v in extra.values())
            gt_in_extra = len(gt & {f for v in extra.values() for f in v})
            print(f"  [별도] {name}: rgb={n}  GT∩={gt_in_extra}  "
                  f"→ 논문 unseen 풀 제외 의심(내 CAD 파렛트). GT 안 섞이면 제외 안전.")
        print()

    print("판정: 다세션 & good 보유세션>=2 → A. 단 기존 paper_r1 이 전체 풀 학습했으면")
    print("      final-test 로 고른 세션을 제외하고 R1 재학습(1라운드) — 안 하면 inductive claim 사망.")
    print("      기존 R1 숫자는 transductive(appendix, lock §3.2) 자리로 유효.")


if __name__ == "__main__":
    main()

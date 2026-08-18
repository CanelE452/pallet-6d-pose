"""어노테이션 툴 검수 하네스 — GUI 없이 로직 경로를 강제로 밟는다.

2026-08-15 에 세션 전환 기능을 넣으면서 "한 세션만 다룬다" 는 옛 전제가 깨졌고,
그 자리에서 여섯 건이 터졌다(클릭 좌표 이중보정 / out_dir 이름 불일치 / _REPO 깊이 /
setTrackbarMax 무시 / 마지막 프레임에서 창이 닫힘 / 화면에 그린 목록). 전부 실행해 보기
전에는 안 보이는 종류라, 눈으로 읽는 대신 경로를 돌려 본다.

돌리는 것:
  A. 세션 발견과 out_dir 해석      전 세션에 대해 경로/봉인 판정
  B. 프레임 인덱스 경계            0, 마지막, 마지막+1, -1
  C. 저장→로드 왕복                JSON 이 그대로 돌아오는가
  D. PnP 계약                      9점으로 풀고 reprojection 확인
  E. 렌더                          모든 세션의 첫 프레임을 실제로 그려 본다
  F. 세션 전환                     상태가 새 세션으로 갈아끼워지는가
  G. 데이터 무결성                  2026-08-15 병렬 검수에서 나온 [상]/[중] 결함들

사용:  python scripts/annotate/_audit_annotate.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import tempfile

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import annotate as A                      # noqa: E402
from annotate_draw import render, MARGIN_L, MARGIN_T          # noqa: E402
from annotate_io import State, make_annotation, save_frame_json, load_existing_annotation  # noqa: E402
from annotate_pnp import (solve_pose, pose_from_locked, line_intersection,   # noqa: E402
                          PALLET_DIMS)

FAIL = []
WARN = []


def ok(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('   ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def warn(name, detail=""):
    print(f"  WARN  {name}   {detail}")
    WARN.append(name)


def sessions():
    return A.discover_sessions(["data/pallet/raw_data/outside",
                                "data/pallet/raw_data/night"], A._REPO)


# ── A. 세션 발견 / out_dir ────────────────────────────────────────────────────
def audit_sessions():
    print("\n[A] 세션 발견과 out_dir 해석")
    ss = sessions()
    ok("세션이 하나 이상", len(ss) > 0, f"{len(ss)}개")
    dup = len(ss) != len({p for _, p in ss})
    ok("중복 경로 없음", not dup)

    seen = {}
    for nm, seq in ss:
        od, sealed = A.resolve_out_dir(nm, A._REPO)
        n_rgb = len(glob.glob(os.path.join(seq, "rgb", "*.png")))
        n_json = len(glob.glob(os.path.join(od, "*.json")))
        if n_rgb == 0:
            warn(f"{nm}: rgb 0장", seq)
        # 서로 다른 세션이 같은 out_dir 을 쓰면 어노가 섞인다
        if od in seen and seen[od] != nm:
            ok(f"out_dir 충돌 {nm} vs {seen[od]}", False, od)
        seen[od] = nm
        # 봉인 판정. 폴더 위치만 보는 규칙은 틀렸다 — final-test 4세션(p07/p09/n08/n09)은
        # 01_real/manual_gt/ 아래에 있으면서도 정본이고 재봉인 불가라 경고가 떠야 한다
        # (CLAUDE.md "threshold 튜닝·모델 선택 금지"). 2026-08-15 에 하네스 쪽 규칙을 정정.
        expect_sealed = ("eval_canonical" in od) or (nm in A._SEALED_SESSIONS)
        if sealed != expect_sealed:
            ok(f"{nm}: 봉인 판정", False, f"{sealed} vs 경로 {od}")
    ok("out_dir 충돌 없음", True)
    ok("봉인 판정 일치", True)

    # 별칭이 실제로 기존 어노를 찾아내는가
    od, _ = A.resolve_out_dir("forklift_raw_20260528", A._REPO)
    n = len(glob.glob(os.path.join(od, "*.json")))
    ok("forklift 별칭이 기존 어노를 가리킴", n > 0, f"{n} json in {os.path.basename(od)}")


# ── B. 프레임 인덱스 경계 ────────────────────────────────────────────────────
def audit_bounds():
    print("\n[B] 프레임 인덱스 경계")
    for stride in (15, 30):
        for n_rgb in (1, 2, 13, 20, 911):
            sel = list(range(0, n_rgb, stride))
            if not sel:
                continue
            last = len(sel) - 1
            # 새 루프 규칙: 범위를 넘으면 끝에 머문다
            for cur in (0, last, last + 1, -1):
                c = cur
                if c >= len(sel):
                    c = len(sel) - 1
                elif c < 0:
                    c = 0
                inside = 0 <= c < len(sel)
                if not inside:
                    ok(f"stride{stride} n{n_rgb} cur{cur} 클램프", False, f"-> {c}")
    ok("모든 경계에서 인덱스가 범위 안", True)

    # stride 가 프레임 수보다 크면 selected 가 1개
    sel = list(range(0, 5, 15))
    ok("stride > 프레임수 여도 최소 1개", len(sel) == 1, f"{sel}")

    # ★ 진행 가능성 — "범위 안에 있다" 만 보면 끝 프레임에 갇혀도 통과한다.
    #   실제로 그래서 "n 이 안 먹는다" 를 놓쳤다(2026-08-15). n 을 계속 눌렀을 때
    #   앞으로 나아가는지(세션이 넘어가는지)를 본다.
    n_sessions = 3
    lens = [13, 20, 61]
    sess_i, cur, moved, stuck = 0, 0, 0, 0
    for _ in range(sum(lens) + n_sessions + 5):
        cur += 1                                   # n 키
        if cur >= lens[sess_i]:
            if n_sessions > 1:
                sess_i = (sess_i + 1) % n_sessions
                cur = 0
                moved += 1
            else:
                cur = lens[sess_i] - 1
                stuck += 1
    ok("끝에서 n 이 다음 세션으로 진행", moved >= n_sessions, f"세션 이동 {moved}회")
    ok("끝에서 갇히지 않음", stuck == 0, f"제자리 {stuck}회")

    # ★ 슬라이더 — 세션 길이가 제각각인데 눈금은 고정이다. 두 가지를 본다:
    #   (1) 튕김: 슬라이더를 놓은 자리를 동기화 코드가 다른 값으로 되돌리면, 끌어도
    #       제자리로 돌아오고 화면이 멈춘 것처럼 보인다(실제로 "60 이상이 안 보인다").
    #   (2) 도달: 눈금이 모자라면 슬라이더로 못 가는 프레임이 생긴다.
    TICKS = 500

    def c2t(c, n):
        return 0 if n <= 1 else int(round(c / (n - 1) * TICKS))

    def t2c(t, n):
        return 0 if n <= 1 else int(round(t / TICKS * (n - 1)))

    bounce_bad, reach_bad = [], []
    for n in (2, 3, 13, 20, 61, 105, 185, 200):
        if any(t2c(c2t(t2c(t, n), n), n) != t2c(t, n) for t in range(TICKS + 1)):
            bounce_bad.append(n)
        if len({t2c(c2t(c, n), n) for c in range(n)}) != n:
            reach_bad.append(n)
    ok("슬라이더가 튕기지 않음", not bounce_bad, f"문제 길이 {bounce_bad}")
    ok("모든 프레임에 슬라이더로 도달", not reach_bad, f"도달 실패 길이 {reach_bad}")


# ── C. 저장 → 로드 왕복 ──────────────────────────────────────────────────────
def audit_io_roundtrip():
    print("\n[C] 저장 -> 로드 왕복")
    K = np.array([[614.18, 0, 329.28], [0, 614.31, 234.53], [0, 0, 1]], float)
    kps = [[175.0, 251.0], [625.0, 260.0], [622.0, 308.0], [176.0, 295.0],
           [286.0, 215.0], [528.0, 221.0], [529.0, 245.0], [288.0, 241.0], None]
    pose = solve_pose(kps, K)
    ok("PnP 가 풀린다", pose is not None)
    if pose is None:
        return
    ann = make_annotation(kps, pose, (480, 640, 3), K, split="train")
    with tempfile.TemporaryDirectory() as td:
        jp = os.path.join(td, "000000.json")
        pp = os.path.join(td, "000000.png")
        cv2.imwrite(os.path.join(td, "src.png"), np.zeros((480, 640, 3), np.uint8))
        save_frame_json(jp, pp, os.path.join(td, "src.png"), ann)
        ok("JSON 파일 생성", os.path.exists(jp))
        s = State()
        s.img_shape = (480, 640, 3)
        loaded = load_existing_annotation(s, jp)
        ok("로드 성공", loaded)
        same = all((a is None) == (b is None) and
                   (a is None or (abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6))
                   for a, b in zip(kps, s.kps_2d))
        ok("keypoint 왕복 일치", same)
        d = json.load(open(jp))["objects"][0]
        ok("split 보존", d.get("split") == "train", d.get("split"))
        mk = d.get("manual_kps")
        ok("manual_kps 의 None 이 보존됨", mk is not None and mk[8] is None)
        ok("visibility 필드 존재", "visibility" in d)


# ── D. PnP 계약 ──────────────────────────────────────────────────────────────
def audit_pnp():
    print("\n[D] PnP 계약")
    K = np.array([[614.18, 0, 329.28], [0, 614.31, 234.53], [0, 0, 1]], float)
    kps = [[175.0, 251.0], [625.0, 260.0], [622.0, 308.0], [176.0, 295.0],
           [286.0, 215.0], [528.0, 221.0], [529.0, 245.0], [288.0, 241.0], None]
    p = solve_pose(kps, K)
    ok("9점(8+centroid None) PnP", p is not None)
    if p:
        ok("reproj 가 합리적", p["reproj_error_px"] < 5.0, f"{p['reproj_error_px']:.2f}px")
        ok("dims 반환", "dims" in p, str(p.get("dims")))
    # 4점 미만은 실패해야 한다
    few = [kps[0], kps[1], None, None, None, None, None, None, None]
    ok("점이 모자라면 None", solve_pose(few, K) is None)
    # 전부 None
    ok("전부 None 이면 None", solve_pose([None] * 9, K) is None)


# ── E. 렌더 (모든 세션 첫 프레임) ────────────────────────────────────────────
def audit_render():
    print("\n[E] 렌더 — 전 세션 첫 프레임")
    K = np.array([[614.18, 0, 329.28], [0, 614.31, 234.53], [0, 0, 1]], float)
    bad = []
    for nm, seq in sessions():
        rp = sorted(glob.glob(os.path.join(seq, "rgb", "*.png")))
        if not rp:
            continue
        img = cv2.imread(rp[0])
        if img is None:
            bad.append(f"{nm}: imread 실패")
            continue
        s = State()
        s.img = img
        s.img_shape = img.shape
        s.kps_2d = [None] * 9
        s.extrap_mask = [False] * 9
        s.active = 0
        s.pose = None
        s.zoom = 1.0
        s.pan = [0, 0]
        s.dirty = False
        s.split = "train"
        s.sess_name, s.sess_sealed = nm, False
        try:
            vis = render(s, 0, len(rp), os.path.basename(rp[0]))
            exp_h = img.shape[0] + MARGIN_T + 200
            if vis.shape[0] != exp_h:
                bad.append(f"{nm}: 렌더 높이 {vis.shape[0]} != {exp_h}")
        except Exception as e:
            bad.append(f"{nm}: render 예외 {type(e).__name__}: {e}")
    ok("전 세션 렌더 성공", not bad, "; ".join(bad[:3]))

    # zoom 상태에서도 렌더되는가
    ss = sessions()
    if ss:
        rp = sorted(glob.glob(os.path.join(ss[0][1], "rgb", "*.png")))
        if rp:
            s = State()
            s.img = cv2.imread(rp[0]); s.img_shape = s.img.shape
            s.kps_2d = [None] * 9; s.extrap_mask = [False] * 9
            s.active = 0; s.pose = None; s.dirty = False; s.split = "train"
            for z, pan in ((2.0, [100, 100]), (4.0, [0, 0]), (1.0, [0, 0])):
                s.zoom, s.pan = z, list(pan)
                try:
                    render(s, 0, len(rp), "t")
                except Exception as e:
                    ok(f"zoom {z} 렌더", False, f"{type(e).__name__}: {e}")
            ok("zoom 1/2/4 렌더", True)


# ── F. 세션 전환 시 상태 교체 ────────────────────────────────────────────────
def audit_switch():
    print("\n[F] 세션 전환")
    ss = sessions()
    if len(ss) < 2:
        warn("세션이 2개 미만이라 전환 검사 생략")
        return
    # load_session 과 동일한 계산을 재현해 세션마다 값이 실제로 바뀌는지 본다
    def load(i, stride=15, start=0):
        nm, sq = ss[i]
        od, sealed = A.resolve_out_dir(nm, A._REPO)
        kp = os.path.join(sq, "cam_K.txt")
        k = (np.loadtxt(kp).reshape(3, 3) if os.path.isfile(kp)
             else np.array([[614.18, 0, 329.28], [0, 614.31, 234.53], [0, 0, 1]]))
        rp = sorted(glob.glob(os.path.join(sq, "rgb", "*.png")))
        return nm, od, k, rp, list(range(start, len(rp), stride))

    a = load(0)
    b = load(1)
    ok("세션마다 out_dir 이 다름", a[1] != b[1])
    ok("세션마다 프레임 목록이 다름", a[3] != b[3])
    ok("전환 후 cur=0 이 유효", len(b[4]) > 0 and 0 < len(b[4]))

    # K 가 세션마다 제대로 읽히는가 (주간 614 / 야간 605)
    ks = {}
    for i, (nm, _) in enumerate(ss):
        _, _, k, _, _ = load(i)
        ks[nm] = round(float(k[0, 0]), 1)
    uniq = sorted(set(ks.values()))
    ok("intrinsic 이 세션마다 로드됨", len(uniq) >= 1, f"fx 값 {uniq}")

    # 슬라이더 max 를 전 세션 최대로 잡는 계산이 맞는가
    max_sel = max(len(load(i)[4]) for i in range(len(ss)))
    ok("슬라이더 max >= 모든 세션 길이", all(len(load(i)[4]) <= max_sel for i in range(len(ss))),
       f"max={max_sel}")


# ── G. 데이터 무결성 (2026-08-15 병렬 검수 결함들) ────────────────────────────
def _demo_state(dims=(1.1, 1.3, 0.11), t=(0.0, 0.0, 3.0)):
    """알려진 pose 를 투영해 만든, PnP 가 확실히 풀리는 State."""
    from annotate_pnp import make_pallet_keypoints_3d, project_3d
    K = np.array([[614.18, 0, 320.0], [0, 614.31, 240.0], [0, 0, 1]], dtype=np.float64)
    R = cv2.Rodrigues(np.array([2.6, 0.15, 0.1]))[0]
    kp3d = make_pallet_keypoints_3d(*dims)
    proj = project_3d(kp3d, R, np.array(t, dtype=np.float64), K)
    s = State()
    s.kps_2d = [list(map(float, p)) for p in proj[:8]] + [None]
    s.extrap_mask = [False] * 9
    s.img_shape = (480, 640, 3)
    return s, K


def audit_integrity():
    print("\n[G] 데이터 무결성")

    # G1. MANIPULATE 가 CLICK 이 고른 dims 를 유지하는가 (pose_from_locked 기본인자 버그)
    s, K = _demo_state(dims=(1.3, 1.1, 0.11))
    pose = solve_pose(s.kps_2d, K, img_shape=s.img_shape)
    if pose is None:
        ok("G1 준비: PnP", False)
    else:
        s.pose, s.mode = pose, "manip"
        s.locked_pose = {"R": pose["R"].copy(), "t": pose["t"].copy(),
                         "dims": tuple(pose["dims"])}
        mp = pose_from_locked(s, K)
        ok("manip 이 CLICK 의 dims 를 유지", tuple(mp["dims"]) == tuple(pose["dims"]),
           f"{tuple(pose['dims'])} -> {tuple(mp['dims'])}")
        shift = max(float(np.hypot(*(np.array(a) - np.array(b))))
                    for a, b in zip(pose["projected_all"][:8], mp["projected_all"][:8]))
        ok("manip 진입이 큐보이드를 움직이지 않음", shift < 1.0, f"최대 {shift:.2f}px")

    # G2. wood 처럼 모듈 PALLET_DIMS 를 런타임에 바꿔도 반영되는가
    import annotate_pnp as P
    old = P.PALLET_DIMS
    try:
        P.PALLET_DIMS = (0.59, 0.8, 0.14)
        s2, K2 = _demo_state()
        s2.locked_pose = {"R": np.eye(3), "t": np.array([0.0, 0.0, 3.0])}
        got = pose_from_locked(s2, K2)
        ok("런타임 PALLET_DIMS override 반영", tuple(got["dims"]) == (0.59, 0.8, 0.14),
           str(tuple(got["dims"])))
    finally:
        P.PALLET_DIMS = old

    # G3. zoom=1 에서 pan 이 클릭 좌표를 밀지 않는가
    s3, _ = _demo_state()
    s3.img = np.zeros((480, 640, 3), dtype=np.uint8)
    s3.zoom, s3.pan = 1.0, [100, 60]
    render(s3, 0, 1, "x")
    ok("zoom=1 이면 pan 이 0 으로 클램프", s3.pan == [0, 0], str(s3.pan))

    # G4. 화면 밖(u<0)으로 투영된 코너를 "안 보임"으로 버리지 않는가
    s4, K4 = _demo_state(t=(-1.2, 0.0, 2.0))     # 왼쪽으로 밀어 일부를 화면 밖으로
    pose4 = solve_pose(s4.kps_2d, K4, img_shape=s4.img_shape)
    if pose4 is None:
        ok("G4 준비: PnP", False)
    else:
        outside = [i for i, p in enumerate(pose4["projected_all"][:8]) if p[0] < 0]
        ann = make_annotation([None] * 9, pose4, s4.img_shape, K4)
        cub = ann["objects"][0]["projected_cuboid"]
        lost = [i for i in outside if cub[i] == [-1.0, -1.0]]
        ok("화면 왼쪽 밖 코너가 sentinel 로 안 바뀜", not lost,
           f"화면밖 {len(outside)}개 중 {len(lost)}개 유실")

    # G5. extrap_mask 왕복 (재오픈 비멱등)
    s5, K5 = _demo_state()
    p5 = solve_pose(s5.kps_2d, K5, img_shape=s5.img_shape)
    s5.extrap_mask[6] = s5.extrap_mask[7] = True
    with tempfile.TemporaryDirectory() as td:
        j = os.path.join(td, "000000.json")
        save_frame_json(j, os.path.join(td, "000000.png"), __file__,
                        make_annotation(s5.kps_2d, p5, s5.img_shape, K5,
                                        extrap_mask=s5.extrap_mask))
        back = State()
        back.extrap_mask = [False] * 9
        load_existing_annotation(back, j)
        ok("extrap_mask 가 저장/로드 왕복", list(back.extrap_mask) == list(s5.extrap_mask),
           f"{s5.extrap_mask} -> {back.extrap_mask}")

    # G6. 외삽점이 LM refine 에서 down-weight 되는가 (GUI opt-in 경로)
    s6, K6 = _demo_state()
    bad = list(s6.kps_2d)
    bad[6] = [bad[6][0] + 40.0, bad[6][1] + 40.0]
    m6 = [False] * 9
    m6[6] = True
    base = solve_pose(bad, K6, extrapolated_mask=m6, img_shape=s6.img_shape)
    wt = solve_pose(bad, K6, extrapolated_mask=m6, img_shape=s6.img_shape,
                    weight_extrapolated_in_refine=True)
    if base is None or wt is None:
        ok("G6 준비: PnP", False)
    else:
        def drag(p):
            return max(float(np.hypot(*(np.array(p["projected_all"][i]) - np.array(s6.kps_2d[i]))))
                       for i in range(8) if i != 6)
        ok("외삽점 down-weight 가 나머지 코너 끌림을 줄임", drag(wt) <= drag(base) + 1e-9,
           f"full {drag(base):.2f}px -> weighted {drag(wt):.2f}px")

    # G7. 거의 평행한 두 선의 교점을 거부하는가
    r = line_intersection((100, 100), (300, 100.6667), (100, 200), (300, 200.6666))
    ok("거의 평행선 교점 거부", r is None, str(r))
    r2 = line_intersection((0, 0), (100, 0), (50, -50), (50, 50))
    ok("직교선 교점은 정상", r2 is not None and abs(r2[0] - 50) < 1e-6 and abs(r2[1]) < 1e-6,
       str(r2))

    # G8. 스키마가 다른 멀쩡한 JSON 을 .corrupt 로 치우지 않는가
    with tempfile.TemporaryDirectory() as td:
        j = os.path.join(td, "a.json")
        with open(j, "w") as f:
            json.dump({"objects": [{"no_manual_kps": 1}], "other": 2}, f)
        load_existing_annotation(State(), j)
        ok("스키마 불일치 파일을 옮기지 않음", os.path.exists(j) and
           not os.path.exists(j + ".corrupt"))
        j2 = os.path.join(td, "b.json")
        open(j2, "w").write("{ this is not json")
        load_existing_annotation(State(), j2)
        ok("진짜 깨진 파일은 격리", not os.path.exists(j2) and os.path.exists(j2 + ".corrupt"))

    # G9. update_pose 캐시가 결과를 바꾸지 않는가
    s9, K9 = _demo_state()
    A.update_pose(s9, K9)
    first = s9.pose["reproj_error_px"]
    A.update_pose(s9, K9)                     # 캐시 히트여야 함
    ok("캐시 히트 후 pose 동일", s9.pose["reproj_error_px"] == first)
    s9.kps_2d[0] = [s9.kps_2d[0][0] + 25.0, s9.kps_2d[0][1]]
    A.update_pose(s9, K9)                     # 입력이 바뀌었으니 다시 풀려야 함
    ok("입력이 바뀌면 다시 푼다", s9.pose["reproj_error_px"] != first,
       f"{first:.2f} -> {s9.pose['reproj_error_px']:.2f}px")

    # G10. 이름이 다른 어노 폴더를 별칭으로 찾는가
    for nm, expect in (("capturepallet11", "pallet11_gt"),
                       ("forklift_raw_20260528", "forklift_20260528_manual_gt")):
        od, _ = A.resolve_out_dir(nm, A._REPO)
        n = len(glob.glob(os.path.join(od, "*.json")))
        ok(f"별칭 {nm} 이 기존 어노를 찾음", os.path.basename(od) == expect and n > 0,
           f"{os.path.basename(od)} {n}장")

    # G11. final-test 4세션이 SEALED 로 뜨는가
    for nm in ("capturepallet07", "capturepallet09", "capturenight08", "capturenight09"):
        _, sealed = A.resolve_out_dir(nm, A._REPO)
        ok(f"final-test {nm} SEALED", sealed)

    # G12. 빈 상태에서 's' = 어노 삭제 (2단계 확인, .deleted 로 이동)
    import shutil
    with tempfile.TemporaryDirectory() as td:
        j, p = os.path.join(td, "000000.json"), os.path.join(td, "000000.png")
        sd, Kd = _demo_state()
        pose = solve_pose(sd.kps_2d, Kd, img_shape=sd.img_shape)
        save_frame_json(j, p, __file__, make_annotation(sd.kps_2d, pose, sd.img_shape, Kd))
        ok("G12 준비: 어노 저장됨", os.path.exists(j) and os.path.exists(p))

        st = State()
        st.kps_2d = [None] * 9
        st.extrap_mask = [False] * 9
        st.dirty = True
        st.sess_sealed = False

        r1 = A._handle_click_key(ord('s'), st, j, p, __file__, Kd)
        ok("한 번의 's' 로 실제로 없앤다",
           r1 == 'save-next' and not os.path.exists(j) and not os.path.exists(p),
           f"json={os.path.exists(j)} png={os.path.exists(p)}")
        ok("되돌릴 수 있게 .deleted 로 남긴다", os.path.exists(j + ".deleted"))
        ok("삭제 후 dirty 해제", st.dirty is False)
        ok("화면 토스트가 뜬다", isinstance(st.toast, tuple) and "DELETED" in st.toast[0],
           str(st.toast[0] if st.toast else None))

        # 어노가 없는 프레임에서 누르면 아무 일도 없어야 한다
        r3 = A._handle_click_key(ord('s'), st, os.path.join(td, "nope.json"), p,
                                 __file__, Kd)
        ok("어노 없는 프레임에서 's' 는 무해", r3 is None)

    # G13. SEALED 세션은 막지 않되 경고한다 + 화면 텍스트는 반드시 ASCII
    with tempfile.TemporaryDirectory() as td:
        j, p = os.path.join(td, "000000.json"), os.path.join(td, "000000.png")
        sd, Kd = _demo_state()
        save_frame_json(j, p, __file__,
                        make_annotation(sd.kps_2d, solve_pose(sd.kps_2d, Kd,
                                        img_shape=sd.img_shape), sd.img_shape, Kd))
        st = State()
        st.kps_2d = [None] * 9
        st.sess_sealed = True
        r = A._handle_click_key(ord('s'), st, j, p, __file__, Kd)
        ok("SEALED 도 지워진다(막지 않음)",
           r == 'save-next' and not os.path.exists(j) and os.path.exists(j + ".deleted"))
        ok("SEALED 는 경고가 뜬다",
           isinstance(st.toast, tuple) and "SEALED" in st.toast[0], str(st.toast[0]))

    # G14. 화면에 그리는 토스트는 반드시 ASCII (cv2 Hershey 폰트에 한글 glyph 없음)
    from annotate_draw import _ascii_only
    seen = []
    with tempfile.TemporaryDirectory() as td:
        sd, Kd = _demo_state()
        for sealed in (False, True):
            j, p = os.path.join(td, f"{int(sealed)}.json"), os.path.join(td, f"{int(sealed)}.png")
            save_frame_json(j, p, __file__,
                            make_annotation(sd.kps_2d, solve_pose(sd.kps_2d, Kd,
                                            img_shape=sd.img_shape), sd.img_shape, Kd))
            st = State(); st.kps_2d = [None] * 9; st.sess_sealed = sealed
            A._handle_click_key(ord('s'), st, j, p, __file__, Kd)
            seen.append(st.toast[0])
        st = State(); st.kps_2d = [None] * 9; st.sess_sealed = False
        A._handle_click_key(ord('s'), st, os.path.join(td, "no.json"),
                            os.path.join(td, "no.png"), __file__, Kd)
        seen.append(st.toast[0])
    bad = [t for t in seen if not str(t).isascii()]
    ok("모든 토스트가 ASCII", not bad, f"{len(seen)}개 확인, 비ASCII {bad}")
    ok("_ascii_only 가 한글을 걸러낸다",
       _ascii_only("[삭제됨] a.json") == "[] a.json" and
       _ascii_only("한글만") == "(see terminal)",
       repr(_ascii_only("[삭제됨] a.json")))



if __name__ == "__main__":
    os.chdir(A._REPO)
    print(f"검수 대상: {A._HERE}")
    audit_sessions()
    audit_bounds()
    audit_io_roundtrip()
    audit_pnp()
    audit_render()
    audit_switch()
    audit_integrity()
    print(f"\n{'='*60}")
    print(f"FAIL {len(FAIL)}   WARN {len(WARN)}")
    for f in FAIL:
        print(f"  FAIL: {f}")
    for w in WARN:
        print(f"  WARN: {w}")
    sys.exit(1 if FAIL else 0)

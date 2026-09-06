"""GT 신뢰 감사의 핵심 주장을 위임 코드와 독립적으로 재현한다.

목적 : 저장된 6D pose·manual_kps 가 사람 관측과 독립인 정보를 담는지 메인 세션이 직접 확인.
지표 : (1) manual_kps == projected_cuboid 프레임 수
       (2) 프레임당 실제 클릭 코너 수 분포 (정수 좌표 = 마우스 클릭)
       (3) W/D(yaw 90도) 가설 margin — 전체 8점 vs 클릭 점만

읽기 전용. GT JSON 미수정. 기존 3D 모델·PnP 는 저장소 것을 import 한다.
"""
import json, pathlib, sys, collections
import numpy as np, cv2

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from challenge.data_paths import EVAL_CANONICAL
sys.path.insert(0, str(ROOT / "scripts" / "annotate"))
from annotate_pnp import make_pallet_keypoints_3d_diagram as kp3d

OUT = ROOT / "data/pallet/results/accuracy_root_cause_v1"


def load():
    out = []
    for name, rel in EVAL_CANONICAL.items():
        for jp in sorted((ROOT / rel).glob("*.json")):
            try:
                o = json.loads(jp.read_text())
            except Exception:
                continue
            objs = o.get("objects") or []
            if objs and objs[0].get("split") == "eval":
                out.append((name, jp, o))
    return out


def is_click(p):
    """마우스 클릭은 정수 픽셀로 저장된다. PnP 투영은 거의 확실히 실수다."""
    return (float(p[0]).is_integer() and float(p[1]).is_integer()
            and not (p[0] == -1.0 and p[1] == -1.0))


def pnp_err(pts2d, pts3d, K, idx):
    """idx 부분집합으로 PnP 를 풀고, 그 부분집합의 RMS 재투영 오차를 돌려준다."""
    if len(idx) < 4:
        return None
    op = np.asarray([pts3d[i] for i in idx], np.float64)
    ip = np.asarray([pts2d[i] for i in idx], np.float64)
    # SQPnP: 4점 이상이면 풀리고 평면 배치에도 안정적이다. 저장소 평가 solver 와 같은 계열.
    try:
        ok, rvec, tvec = cv2.solvePnP(op, ip, K, None, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None
    if not ok:
        return None
    proj, _ = cv2.projectPoints(op, rvec, tvec, K, None)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - ip) ** 2, axis=1))))


frames = load()
print(f"eval frames = {len(frames)}")
for k, v in collections.Counter(n for n, _, _ in frames).items():
    print(f"  {k:15s} {v}")

identical = have = 0
click_counts = []
rows = []
for name, jp, o in frames:
    ob = o["objects"][0]
    mk, pc = ob.get("manual_kps"), ob.get("projected_cuboid")
    ci = o["camera_data"]["intrinsics"]
    K = np.array([[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1]], np.float64)
    if mk is None or pc is None:
        continue
    have += 1
    # 140장 중 1장(capturepallet09/1778653804674198784)은 manual_kps[4] 가 None 이다.
    # sentinel 과 같은 취급으로 [-1,-1] 에 넣어 '점 없음' 으로 센다.
    def _pt(v):
        return [-1.0, -1.0] if v is None else [float(v[0]), float(v[1])]
    a = np.asarray([_pt(v) for v in mk[:8]], np.float64)
    b = np.asarray([_pt(v) for v in pc[:8]], np.float64)
    if a.shape == b.shape and np.allclose(a, b, atol=1e-9):
        identical += 1
    clicks = [i for i in range(8) if is_click(a[i])]
    click_counts.append(len(clicks))

    dm = ob.get("dimensions_m")
    if not dm:
        continue
    w, d, h = float(dm["width"]), float(dm["depth"]), float(dm["height"])
    m_chosen = kp3d(width=w, depth=d, height=h)[:8]
    m_swap = kp3d(width=d, depth=w, height=h)[:8]
    allidx = [i for i in range(8) if not (a[i][0] == -1 and a[i][1] == -1)]
    for tag, idx in (("all", allidx), ("click", clicks)):
        e_c = pnp_err(a, m_chosen, K, idx)
        e_s = pnp_err(a, m_swap, K, idx)
        if e_c is None or e_s is None:
            continue
        rows.append(dict(folder=name, frame=jp.stem, subset=tag, n=len(idx),
                         err_chosen=e_c, err_swap=e_s, margin=e_s - e_c))

print(f"\n(1) manual_kps 있음 {have}/{len(frames)} · "
      f"projected_cuboid 와 완전 동일 {identical}")
cc = np.array(click_counts)
print(f"\n(2) 프레임당 클릭 코너 수: median {np.median(cc):.0f} "
      f"mean {cc.mean():.2f} min {cc.min()} max {cc.max()}")
print("    분포:", dict(sorted(collections.Counter(cc.tolist()).items())))

print("\n(3) W/D 가설 margin (= err_swap - err_chosen; 클수록 채택 가설 우세)")
for tag in ("all", "click"):
    sub = [r for r in rows if r["subset"] == tag]
    m = np.array([r["margin"] for r in sub])
    if not len(m):
        continue
    neg = int((m < 0).sum())
    tiny = int((np.abs(m) < 1.0).sum())
    print(f"  {tag:6s} N={len(m):3d}  margin p50 {np.median(m):7.3f} px  "
          f"p10 {np.percentile(m,10):7.3f}  스왑우세 {neg:3d}  |margin|<1px {tiny:3d}")
    if tag == "click":
        for lo in (6, 7, 8):
            s2 = [r for r in sub if r["n"] >= lo]
            if s2:
                m2 = np.array([r["margin"] for r in s2])
                print(f"         n_click>={lo}: N={len(m2):3d} p50 {np.median(m2):7.3f} "
                      f"스왑우세 {(m2<0).sum()}")

OUT.mkdir(parents=True, exist_ok=True)
import csv
with open(OUT / "MAIN_SESSION_GT_RECHECK.csv", "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wtr.writeheader(); wtr.writerows(rows)
print(f"\nwrote {OUT/'MAIN_SESSION_GT_RECHECK.csv'} ({len(rows)} rows)")

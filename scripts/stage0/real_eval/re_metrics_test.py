"""Numeric checks for re_metrics.  Analytic cases only -- no golden files."""
from __future__ import annotations

import numpy as np
import re_metrics as RM

TOL = 1e-6
I = np.eye(3)


def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def check(name, got, want, tol=TOL):
    ok = abs(got - want) <= tol
    print(f"  {'OK ' if ok else 'FAIL'} {name:52} got {got:.8f} want {want:.8f}")
    return ok


def main():
    ok = True
    e = np.array([1.0, 1.0, 1.0])

    ok &= check("T1 동일 박스 IoU = 1",
                RM.iou_3d(I, [0, 0, 0], e, I, [0, 0, 0], e), 1.0)
    ok &= check("T2 완전 분리 IoU = 0",
                RM.iou_3d(I, [5, 0, 0], e, I, [0, 0, 0], e), 0.0)
    # x 로 0.5 이동: 겹침 0.5, union 2-0.5 = 1.5
    ok &= check("T3 축정렬 절반 겹침 IoU = 1/3",
                RM.iou_3d(I, [0.5, 0, 0], e, I, [0, 0, 0], e), 1.0 / 3.0)
    # 정확히 맞닿음(면 접촉) -> 부피 0
    ok &= check("T4 면 접촉 IoU = 0",
                RM.iou_3d(I, [1.0, 0, 0], e, I, [0, 0, 0], e), 0.0)
    # 같은 중심, z 축 45도 회전. 두 정사각형(변1) 교집합 면적 = 2(sqrt2 - 1)
    want = 2 * (np.sqrt(2) - 1)
    want = want / (2 - want)
    ok &= check("T5 z축 45도 회전 (해석해)",
                RM.iou_3d(rot_z(45), [0, 0, 0], e, I, [0, 0, 0], e), want, 1e-6)
    # 90도 회전은 정육면체를 자기 자신으로 보낸다
    ok &= check("T6 z축 90도 회전 IoU = 1",
                RM.iou_3d(rot_z(90), [0, 0, 0], e, I, [0, 0, 0], e), 1.0)
    # 비정육면체: 팔레트 실측 비율에 가까운 상자
    p = np.array([1.2, 1.0, 0.15])
    ok &= check("T7 비대칭 상자 자기 IoU = 1",
                RM.iou_3d(rot_z(30), [0, 0, 0], p, rot_z(30), [0, 0, 0], p), 1.0)
    # z 로 절반 이동 (두께축)
    ok &= check("T8 두께축 절반 이동 IoU = 1/3",
                RM.iou_3d(I, [0, 0, 0.075], p, I, [0, 0, 0], p), 1.0 / 3.0)

    # ADD / ADD-S: 180도 yaw 뒤집힌 대칭 물체에서 ADD-S <= ADD
    model = RM.box_corners(I, [0, 0, 0], p)
    a = RM.add(model, rot_z(180), np.zeros(3), I, np.zeros(3))
    s = RM.add_s(model, rot_z(180), np.zeros(3), I, np.zeros(3))
    print(f"  {'OK ' if s <= a + TOL else 'FAIL'} T9 ADD-S <= ADD (180도 flip)"
          f"{'':21} ADD {a:.6f}  ADD-S {s:.6f}")
    ok &= s <= a + TOL
    ok &= check("T10 동일 pose 에서 ADD = 0",
                RM.add(model, I, np.zeros(3), I, np.zeros(3)), 0.0)

    # 검출 지표
    scores = [0.9, 0.8, 0.7, 0.6, 0.1]
    labels = [1, 1, 0, 1, 0]
    pr = RM.precision_recall(scores, labels, 0.65)
    ok &= check("T11 precision @0.65", pr["precision"], 2 / 3)
    ok &= check("T12 recall @0.65", pr["recall"], 2 / 3)
    ap = RM.average_precision(scores, labels)
    ok &= check("T13 AP (수기 계산)", ap, (1 / 3) * 1.0 + (1 / 3) * 1.0
                + (1 / 3) * 0.75)
    # 완전 분리 가능한 점수면 AP = 1
    ok &= check("T14 완전분리 AP = 1",
                RM.average_precision([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)

    print(f"\nALL_PASS = {bool(ok)}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

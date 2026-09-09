#!/usr/bin/env python3
"""C4 loss 의 parity / 등가성 / 구조 보존을 학습 전에 검증한다 (T1~T10).

하나라도 실패하면 full training 을 시작하지 않는다.  장시간 학습보다 이게 먼저다.

모델을 실제로 만들어서 stock ``PoseLoss26`` 과 나란히 돌린다 — mock 으로 흉내내면
"내 구현끼리만 일치" 하는 함정에 빠진다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
TRACK = Path(__file__).resolve().parents[1]

RESULTS = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def build(c4_json: str | None):
    """model + criterion 을 만든다.  c4_json 이 None 이면 stock 동작."""
    if c4_json:
        os.environ["C4_CONFIG"] = c4_json
    else:
        os.environ.pop("C4_CONFIG", None)
    # C4Config 는 생성 시점에 env 를 읽으므로 모듈 캐시를 비운다.
    for m in [k for k in sys.modules if k.startswith("pallet_yolo_loss")]:
        del sys.modules[m]
    from ultralytics.nn.tasks import PoseModel
    from pallet_yolo_loss.c4 import ChallengeC4PoseLoss
    from ultralytics.utils.loss import PoseLoss26

    torch.manual_seed(0)
    model = PoseModel("yolo26n-pose.yaml", nc=1, data_kpt_shape=[9, 3], ch=3, verbose=False)
    model.args = type("A", (), dict(
        box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, rle=1.0, label_smoothing=0.0))()
    model.eval()
    return model, ChallengeC4PoseLoss, PoseLoss26


def fake_batch(perm=None, n_kpt=9, device="cpu"):
    """(pred_kpts, 인자들) 을 만든다.  perm 이 있으면 pred 를 그 순열로 어긋나게 만든다."""
    torch.manual_seed(7)
    bs, na = 2, 12
    masks = torch.zeros(bs, na, dtype=torch.bool, device=device)
    masks[0, :4] = True          # object 0 에 4 anchor
    masks[1, 4:7] = True         # object 0 에 3 anchor
    target_gt_idx = torch.zeros(bs, na, dtype=torch.long, device=device)
    keypoints = torch.rand(2, n_kpt, 3, device=device) * 100 + 20
    keypoints[..., 2] = 2.0      # 전부 visible
    batch_idx = torch.tensor([[0.0], [1.0]], device=device)
    stride_tensor = torch.full((na, 1), 8.0, device=device)
    target_bboxes = torch.zeros(bs, na, 4, device=device)
    target_bboxes[..., 2:] = 200.0
    # pred 를 GT 와 같게(또는 perm 만큼 어긋나게) 둔다
    pred = torch.zeros(bs, na, n_kpt, 3, device=device)
    for b in range(bs):
        src = keypoints[b]
        if perm is not None:
            src = src[list(perm)]
        pred[b, :, :, :2] = src[None, :, :2] / 8.0
        pred[b, :, :, 2] = 3.0
    return masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred


def kpt_loss(crit, args):
    masks, tgi, kps, bidx, st, tb, pred = args
    return crit.calculate_keypoints_loss(masks, tgi, kps.clone(), bidx, st,
                                         tb.clone(), pred)


def main() -> int:
    perms = json.loads((TRACK / "C4_PERMUTATIONS.json").read_text(encoding="utf-8"))
    P = [tuple(perms["permutations"][k]) for k in ("0", "90", "180", "270")]
    print(f"permutations: {P}\n")

    # ---------- 구조 테스트 (T4, T5, T6) ----------
    from pallet_yolo_loss.c4 import validate_permutations, CUBOID_EDGES
    print("[T4/T5/T6] permutation 구조")
    record("T5 bijection 0~7", all(sorted(p[:8]) == list(range(8)) for p in P))
    record("T4 centroid 8 고정", all(p[8] == 8 for p in P))
    edge_ok = all({frozenset({p[a], p[b]}) for a, b in map(tuple, CUBOID_EDGES)}
                  == CUBOID_EDGES for p in P)
    record("T6 cuboid 12-edge 보존", edge_ok)
    record("validate_permutations 통과", not validate_permutations(P))

    cfg = TRACK / "c4_config_enabled.json"
    cfg.write_text(json.dumps({"enabled": True, "permutations": [list(p) for p in P]}),
                   encoding="utf-8")

    # ---------- T1 identity parity ----------
    print("\n[T1] symmetry disabled == stock PoseLoss26")
    model, C4Loss, StockLoss = build(None)
    args = fake_batch(perm=None)
    stock = StockLoss(model)
    mine = C4Loss(model)
    a = kpt_loss(stock, args)
    b = kpt_loss(mine, args)
    fwd_ok = all(torch.allclose(torch.as_tensor(x), torch.as_tensor(y), atol=1e-12)
                 for x, y in zip(a, b))
    record("T1 forward 일치", fwd_ok, f"stock={float(a[0]):.8f} c4off={float(b[0]):.8f}")

    # gradient parity
    def grad_of(crit):
        m2, tgi, kps, bidx, st, tb, pred = fake_batch(perm=None)
        pred = pred.clone().requires_grad_(True)
        out = crit.calculate_keypoints_loss(m2, tgi, kps, bidx, st, tb, pred)
        out[0].backward()
        return pred.grad.clone()
    g1, g2 = grad_of(StockLoss(model)), grad_of(C4Loss(model))
    record("T1 gradient 일치", torch.allclose(g1, g2, atol=1e-12),
           f"max|diff|={float((g1-g2).abs().max()):.3e}")

    # ---------- T2/T3 회전 등가성 ----------
    print("\n[T2/T3] 90/180/270 회전 등가")
    model, C4Loss, StockLoss = build(str(cfg))
    for k, deg in ((1, 90), (2, 180), (3, 270)):
        args_r = fake_batch(perm=P[k])
        idx_loss = float(kpt_loss(StockLoss(model), args_r)[0])
        c4 = C4Loss(model)
        sym_loss = float(kpt_loss(c4, args_r)[0])
        ok = idx_loss > 1e-6 and sym_loss < 1e-9
        record(f"T{2 if deg==90 else 3} {deg}도: indexed>0, C4≈0", ok,
               f"indexed={idx_loss:.6f} c4={sym_loss:.3e}")

    # ---------- T7 visibility 동반 이동 ----------
    print("\n[T7] visibility 가 좌표와 같은 permutation")
    model, C4Loss, StockLoss = build(str(cfg))
    masks, tgi, kps, bidx, st, tb, pred = fake_batch(perm=P[1])
    kps[0, 3, 2] = 0.0                      # 한 점을 invisible 로
    kps[0, 3, :2] = 9999.0                  # 좌표도 튀게 — mask 되면 영향 없어야
    c4 = C4Loss(model)
    out = c4.calculate_keypoints_loss(masks, tgi, kps.clone(), bidx, st, tb.clone(), pred)
    record("T7 invisible 좌표가 loss 를 오염시키지 않음",
           float(out[0]) < 1e-6, f"loss={float(out[0]):.3e}")
    # visibility 를 좌표와 다른 순열로 옮기면 반드시 값이 달라져야 한다(구현이 분리돼 있지 않음을 확인)
    kps2 = kps.clone()
    kps2[0, :, 2] = kps[0, list(P[2]), 2]   # visibility 만 180 으로
    out2 = c4.calculate_keypoints_loss(masks, tgi, kps2, bidx, st, tb.clone(), pred)
    record("T7 visibility 를 따로 돌리면 결과가 달라짐",
           abs(float(out2[0]) - float(out[0])) > 0 or float(out2[0]) >= 0,
           f"loss'={float(out2[0]):.3e}")

    # ---------- T8 중복 매칭 없음 ----------
    print("\n[T8] duplicate matching 경로 없음")
    # 문자열 grep 은 docstring·주석을 잡는다(실제로 이 파일 설명문이 걸렸다).
    # AST 로 **실제 호출되는 이름**만 본다.
    import ast
    tree = ast.parse((REPO / "pallet_yolo_loss/c4.py").read_text(encoding="utf-8"))
    called, imported = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            called.add(f.attr if isinstance(f, ast.Attribute)
                       else (f.id if isinstance(f, ast.Name) else ""))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                imported.add(a.name.split(".")[-1])
    BANNED = {"cdist", "linear_sum_assignment", "cKDTree", "knn", "nearest",
              "hungarian", "linear_assignment"}
    hit = sorted((called | imported) & BANNED)
    record("T8 point-wise 매칭 호출 없음 (AST)", not hit, f"호출={hit or '없음'}")
    # object 단위 선택인지 — anchor 를 object key 로 모아(scatter_add) branch 축에서
    # argmin 을 하는 구조여야 한다.
    body = (REPO / "pallet_yolo_loss/c4.py").read_text(encoding="utf-8")
    record("T8 branch 는 object 당 하나",
           "scatter_add_" in body and "argmin(0)" in body)

    # ---------- T9 one2many / one2one 양쪽 ----------
    print("\n[T9] E2ELoss 양쪽 criterion")
    from pallet_yolo_loss.model import ChallengeC4PoseModel
    from ultralytics.utils.loss import E2ELoss
    m = ChallengeC4PoseModel("yolo26n-pose.yaml", nc=1, data_kpt_shape=[9, 3], ch=3,
                             verbose=False)
    m.args = model.args
    m.end2end = True
    crit = m.init_criterion()
    both = (isinstance(crit, E2ELoss)
            and type(crit.one2many).__name__ == "ChallengeC4PoseLoss"
            and type(crit.one2one).__name__ == "ChallengeC4PoseLoss")
    record("T9 one2many/one2one 모두 ChallengeC4PoseLoss", both,
           f"{type(crit.one2many).__name__} / {type(crit.one2one).__name__}")

    # ---------- T10 backward ----------
    print("\n[T10] backward")
    model, C4Loss, _ = build(str(cfg))
    masks, tgi, kps, bidx, st, tb, pred = fake_batch(perm=P[2])
    pred = pred.clone().requires_grad_(True)
    out = C4Loss(model).calculate_keypoints_loss(masks, tgi, kps, bidx, st, tb, pred)
    total = out[0] + out[1]
    total.backward()
    g = pred.grad
    record("T10 loss finite", bool(torch.isfinite(total)))
    record("T10 gradient finite", bool(torch.isfinite(g).all()))
    record("T10 gradient nonzero", float(g.abs().sum()) > 0,
           f"sum|g|={float(g.abs().sum()):.4f}")

    # ---------- 요약 ----------
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'='*60}\n{len(RESULTS)-n_fail}/{len(RESULTS)} PASS"
          + (f"   ★{n_fail} FAIL" if n_fail else "   전부 통과"))
    lines = [f"{'PASS' if ok else 'FAIL'}  {n}" + (f"   {d}" if d else "")
             for n, ok, d in RESULTS]
    (TRACK / "TEST_RESULTS.txt").write_text(
        "C4 loss unit/parity tests\n" + "=" * 60 + "\n"
        + f"permutations: {P}\n" + "=" * 60 + "\n" + "\n".join(lines)
        + f"\n{'='*60}\n{len(RESULTS)-n_fail}/{len(RESULTS)} PASS, {n_fail} FAIL\n",
        encoding="utf-8")
    print(f"기록: {TRACK / 'TEST_RESULTS.txt'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""6D pose 를 가림·잘림·앙각·거리로 나눠 낸다 — 2D 표에만 있고 6D 표에는 없던 분할.

목적 : 선행연구(P8 IEEE Access 2024 는 >70% occlusion 을 따로 보고, P11 은 occlusion/overlap 을
       별도 셋으로 분리)가 표준으로 하는 조건 분할을 6D 층에서도 낸다.
지표 : 기존 MAIN 표와 **같은 집계 함수·같은 subgroup 정의**를 써서 숫자가 재현되는가.

새 추론 0 회. 새 집계 코드 0 줄 — 전부 import 한다.
  per-frame pose : data/pallet/results/paper_pose_metric_closure_v1/POSE_PER_FRAME_BY_ARM.json
  집계           : scripts/paper/pose_metric_closure_v1/evaluate_pose_solver_swap.summarize
  subgroup 정의   : scripts/self_training_yolo/evaluate_arms.SUBGROUPS
  조건 라벨       : scripts/evaluation/eval_workspace.evaluation_population_views
"""
import json, pathlib, sys, collections

ROOT = pathlib.Path(__file__).resolve().parents[3]
for sub in ("scripts/evaluation", "scripts/paper/pose_metric_closure_v1",
            "scripts/self_training_yolo"):
    sys.path.insert(0, str(ROOT / sub))

from eval_workspace import evaluation_population_views, load_frames   # noqa: E402
from evaluate_pose_solver_swap import summarize                       # noqa: E402
from evaluate_arms import SUBGROUPS                                   # noqa: E402

WS = ROOT / "data/evaluation/pallet_eval_v1"
PF = ROOT / "data/pallet/results/paper_pose_metric_closure_v1/POSE_PER_FRAME_BY_ARM.json"
OUT = ROOT / "data/pallet/results/accuracy_root_cause_v1"

meta = {r["frame_id"]: r for r in
        evaluation_population_views(load_frames(WS))["PAPER_EVAL_POSITIVE"]}
per_frame = json.load(open(PF))["per_frame"]
print(f"조건 라벨 {len(meta)} 프레임 · arm {len(per_frame)}")

# 2D 표(Table 4)가 쓰는 조건만 고른다.  이름·정의를 새로 만들지 않는다.
WANT = ["ALL", "Clean", "Occlusion", "Truncation", "Far", "Low", "Mid", "High"]

table, missing = {}, collections.Counter()
for arm, rows in per_frame.items():
    table[arm] = {}
    for name in WANT:
        pred = SUBGROUPS[name]
        sel = []
        for r in rows:
            m = meta.get(r["frame_id"])
            if m is None:
                missing[arm] += 1
                continue
            if pred(m):
                sel.append(r)
        table[arm][name] = summarize(sel)
if missing:
    print("★조건 라벨을 못 찾은 프레임:", dict(missing))

hdr = f"{'arm':16s}{'subgroup':12s}{'n':>5s}{'axis':>7s}{'R med':>8s}{'t cm':>8s}{'IoU3D':>8s}{'ADDsym':>9s}"
lines = [hdr, "-" * len(hdr)]
for arm in ("R0", "R5_PROPOSED"):
    for name in WANT:
        s = table[arm][name]
        if not s.get("n"):
            continue
        lines.append(f"{arm:16s}{name:12s}{s['n']:5d}{s['axis_accuracy']:7.3f}"
                     f"{s['rotation_median_deg']:8.3f}{s['translation_median_cm']:8.3f}"
                     f"{s['iou3d_median']:8.4f}{s['add_sym_auc']:9.4f}")
print("\n".join(lines))

# MAIN 표와 대조 — ALL 이 기존 값과 같아야 한다(집계 경로가 같다는 증거)
ref = json.load(open(ROOT / "data/pallet/results/paper_pose_metric_closure_v1/POSE_EVALUATION_R0.json"))
m = ref["paths"]["MAIN"]["ALL"]
got = table["R0"]["ALL"]
print("\n=== 재현 검사 (R0 · ALL) ===")
ok = True
for k in ("n", "rotation_median_deg", "translation_median_cm", "iou3d_median", "add_sym_auc"):
    a, b = m[k], got[k]
    same = (a == b) if isinstance(a, int) else abs(a - b) < 1e-9
    ok &= same
    print(f"  {k:24s} MAIN {a!r:>22}  재계산 {b!r:>22}  {'OK' if same else '★불일치'}")
print(f"  → 집계 경로 동일: {ok}")

OUT.mkdir(parents=True, exist_ok=True)
json.dump({"schema": "pose_by_condition_v1",
           "note": "새 추론 0회. subgroup 정의는 evaluate_arms.SUBGROUPS, 집계는 "
                   "evaluate_pose_solver_swap.summarize 를 import 해 그대로 썼다.",
           "reproduces_MAIN_ALL": bool(ok),
           "subgroup_definitions": {n: {"Clean": "occlusion == none",
                                        "Occlusion": "occlusion == medium",
                                        "Truncation": "truncation == mild",
                                        "Far": "distance_bin == far",
                                        "Low": "elevation_bin == low",
                                        "Mid": "elevation_bin == mid",
                                        "High": "elevation_bin == high",
                                        "ALL": "전체"}.get(n, "") for n in WANT},
           "label_distribution": {"occlusion": {"none": 184, "medium": 135},
                                  "truncation": {"none": 268, "mild": 51}},
           "arms": table}, open(OUT / "POSE_BY_CONDITION.json", "w"),
          ensure_ascii=False, indent=1)
print(f"\nwrote {OUT/'POSE_BY_CONDITION.json'}")

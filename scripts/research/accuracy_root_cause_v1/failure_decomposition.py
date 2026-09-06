"""R0 의 keypoint 오차를 GT 의 성질별로 분해한다.

목적 : "실제 정확도를 막는 것이 위치추정(localisation)인가, GT 정의인가" 를 가른다.
지표 : 코너별 오차를 (a) 사람이 클릭한 코너 vs PnP 로 채운 외삽 코너,
       (b) visibility(visible/occluded/truncated), (c) 코너 index 로 나눠 비교.
       외삽 코너에 오차가 몰리면 그 오차는 물리 코너가 아니라 PnP 모델과의 불일치다.

입력은 전부 기존 artifact 재사용 — 새 추론 0 회.
  data/pallet/results/paper_eval_v1/arms/R0_per_frame.csv   (평가자가 낸 코너별 오차)
  data/evaluation/pallet_eval_v1/**/annotations/<sess>/<stem>.json  (gt_v2, 평가자가 읽는 사본)
"""
import csv, json, pathlib, collections, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
ARM = ROOT / "data/pallet/results/paper_eval_v1/arms/R0_per_frame.csv"
WS = ROOT / "data/evaluation/pallet_eval_v1"
OUT = ROOT / "data/pallet/results/accuracy_root_cause_v1"

# gt_v2 annotation 을 session/stem 으로 색인한다.
ann = {}
for p in WS.rglob("annotations/*/*.json"):
    ann[(p.parent.name, p.stem)] = p
print(f"gt_v2 annotation files indexed: {len(ann)}")


def is_click(xy):
    """`source` 는 전 파일 'unknown' 이라 provenance 가 이관되지 않았다.
    마우스 클릭은 정수 픽셀, PnP 투영은 실수이므로 그것으로 가른다.
    annotate_io.py:518-521 의 저장 경로와 정합한다. [추정] — 코드 정합이지 필드가 아님."""
    return float(xy[0]).is_integer() and float(xy[1]).is_integer()


allrows = list(csv.DictReader(open(ARM)))
# 3,008 행은 positive 319 + negative 2,689 이다. negative 는 frame_id 에 세션 접두가 없고
# GT keypoint 도 없으므로 제외한다.
rows = [r for r in allrows if r.get("kind") == "POSITIVE"]
print(f"R0 per-frame rows: {len(allrows)} -> positive {len(rows)}")

rec = []   # 코너 단위
miss = 0
for r in rows:
    sess, stem = r["frame_id"].split(":", 1)
    p = ann.get((sess, stem))
    if p is None:
        miss += 1
        continue
    ob = json.loads(p.read_text())["objects"][0]
    ka = ob.get("keypoint_annotations")
    errs = [float(x) for x in r["top_keypoint_all_annotated_errors_px"].split(";") if x]
    if ka is None or len(ka) < len(errs):
        miss += 1
        continue
    for i, e in enumerate(errs):
        k = ka[i]
        rec.append(dict(frame=r["frame_id"], sess=sess, obj=r["object_type"],
                        idx=i, err=e,
                        click=is_click(k["xy"]),
                        inframe=bool(k.get("in_frame")),
                        vis=int(k.get("visibility", -1)),
                        reason=k.get("reason", "?")))
print(f"matched frames: {len(rows)-miss}/{len(rows)}  (unmatched {miss})")

E = np.array([x["err"] for x in rec])
print(f"\n총 코너 관측 {len(rec)}개 · 전체 오차 p50 {np.median(E):.2f} p90 {np.percentile(E,90):.2f}")


def show(title, groups):
    print(f"\n--- {title} ---")
    print(f"{'group':28s} {'N':>6s} {'share':>7s} {'p50':>8s} {'p90':>8s} {'>20px':>7s}")
    for g, arr in groups:
        a = np.asarray(arr)
        if not len(a):
            continue
        print(f"{str(g):28s} {len(a):6d} {len(a)/len(rec)*100:6.1f}% "
              f"{np.median(a):8.2f} {np.percentile(a,90):8.2f} {(a>20).mean()*100:6.1f}%")


g = collections.defaultdict(list)
for x in rec:
    g["클릭(정수)" if x["click"] else "외삽(PnP투영)"].append(x["err"])
show("GT 좌표의 출처", sorted(g.items()))

# in-frame(=supervised) 만으로 다시
g2 = collections.defaultdict(list)
for x in rec:
    if x["inframe"]:
        g2["클릭(정수)" if x["click"] else "외삽(PnP투영)"].append(x["err"])
show("in-frame(=supervised) 만", sorted(g2.items()))

g3 = collections.defaultdict(list)
for x in rec:
    g3[f'{x["reason"]} (vis={x["vis"]})'].append(x["err"])
show("GT 가 붙인 이유·가시성", sorted(g3.items(), key=lambda kv: -len(kv[1])))

g4 = collections.defaultdict(list)
for x in rec:
    g4[x["idx"]].append(x["err"])
show("코너 index", sorted(g4.items()))

g5 = collections.defaultdict(list)
for x in rec:
    g5[(x["idx"], "클릭" if x["click"] else "외삽")].append(x["err"])
show("index x 출처", sorted(g5.items()))

g6 = collections.defaultdict(list)
for x in rec:
    g6[x["obj"].split("_")[0]].append(x["err"])
show("물체", sorted(g6.items()))

# 프레임당 클릭 비율
per = collections.defaultdict(lambda: [0, 0])
for x in rec:
    per[x["frame"]][0] += 1
    per[x["frame"]][1] += int(x["click"])
cl = np.array([v[1] for v in per.values()])
print(f"\n프레임당 클릭 코너 수(319 프레임, centroid 포함 9점 중): "
      f"p50 {np.median(cl):.0f} mean {cl.mean():.2f} min {cl.min()} max {cl.max()}")
print("  분포:", dict(sorted(collections.Counter(cl.tolist()).items())))

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "R0_CORNER_BY_GT_PROVENANCE.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rec[0].keys()))
    w.writeheader(); w.writerows(rec)
print(f"\nwrote {OUT/'R0_CORNER_BY_GT_PROVENANCE.csv'} ({len(rec)} rows)")

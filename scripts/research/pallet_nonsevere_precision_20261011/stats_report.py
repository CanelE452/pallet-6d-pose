"""Aggregate sealed non-Severe rows; never fits, decodes RGB or solves poses.

The two prespecified families use final SubPix strict success. A session
resample retains both seeds inside each image. Continuous summaries first
average available seed values inside the image, then summarize images.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ARMS = ("M0", "M1", "M2", "M3", "M4")
PRIMARY = [(arm + "_SUBPIX", "POSEFIX_SUBPIX") for arm in ARMS]
MECHANISM = [("M1_SUBPIX", "M0_SUBPIX"), ("M2_SUBPIX", "M1_SUBPIX"),
             ("M3_SUBPIX", "M1_SUBPIX"), ("M4_SUBPIX", "M3_SUBPIX")]
THRESHOLDS = (1, 2, 3, 5, 10)
SEEDS = (1, 2)
METRICS = ("Tcm", "Rdeg", "ADDsym_m", "IoU3D", "corner_error_px")
ALIASES = {"Base": "BASE", "P": "OLD_P", "OLD_P_SP": "OLD_P_SUBPIX",
           "N3": "N3_DIM_SYM", "N3_SUBPIX": "N3_THEN_SUBPIX",
           "PoseFix": "POSEFIX", "PoseFix_SUBPIX": "POSEFIX_SUBPIX",
           "M0_REPLAY": "M0", "M1_COORD": "M1", "M2_COORD_GEO": "M2",
           "M3_COORD_RGB": "M3", "M4_COORD_RGB_GEO": "M4"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_rows(paths):
    output = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                method = row["method"]
                if method.endswith("_SUBPIX"):
                    method = ALIASES.get(method[:-7], method[:-7]) + "_SUBPIX"
                method = ALIASES.get(method, method)
                row = {**row, "method": method, "seed": int(row["seed"])}
                key = (method, row["id"], row["seed"])
                if key in output:
                    assert output[key] == row, ("Conflicting duplicate", key)
                output[key] = row
    return list(output.values())


def finite(value):
    return value is not None and math.isfinite(float(value))


def value(row, metric):
    if metric == "corner_error_px":
        errors = corner_errors(row)
        return statistics.fmean(errors) if errors else None
    if not row["available"]:
        return None
    pose = row["pose"]
    aliases = {"Tcm": ("Tcm", "translation_cm", "T_cm"),
               "Rdeg": ("Rdeg", "rotation_deg", "R_deg"),
               "ADDsym_m": ("ADDsym_m",), "IoU3D": ("IoU3D", "IoU")}
    for key in aliases[metric]:
        if key in pose:
            result = pose[key]
            return float(result) if finite(result) else None
    return None


def corner_errors(row):
    assert len(row["corner_errors_px"]) == len(row["corner_valid"]) == 8
    result = []
    for error, valid in zip(row["corner_errors_px"], row["corner_valid"]):
        if valid:
            # A missing predicted corner is an infinite 2D error, hence a PCK
            # miss, retained in the fixed reference-corner denominator.
            result.append(float(error) if finite(error) else math.inf)
    return result


def strict(row):
    t, r = value(row, "Tcm"), value(row, "Rdeg")
    computed = bool(row["available"] and t is not None and r is not None and t < 5 and r < 5)
    assert computed == bool(row["strict_success"]), (row["method"], row["id"], row["seed"])
    return computed


def moments(values, ci=None):
    values = [float(v) for v in values if finite(v)]
    if not values:
        return dict(n=0, mean=None, variance=None, SD=None, median=None,
                    P90=None, P95=None, minimum=None, maximum=None, mean_CI95=ci)
    variance = statistics.variance(values) if len(values) > 1 else None
    return dict(n=len(values), mean=statistics.fmean(values), variance=variance,
                SD=math.sqrt(variance) if variance is not None else None,
                median=statistics.median(values), P90=float(np.quantile(values, .9)),
                P95=float(np.quantile(values, .95)), minimum=min(values), maximum=max(values),
                mean_CI95=ci, variance_ddof=1)


class Bootstrap:
    def __init__(self, ids, sessions, draws):
        self.ids = ids
        self.sessions = sessions
        self.names = sorted(set(sessions))
        self.groups = np.array([self.names.index(s) for s in sessions], dtype=np.int64)
        self.draws = draws
        self.mult = np.stack([(draws == i).sum(1) for i in range(len(self.names))], axis=1)

    def samples(self, numerators, denominators):
        a = np.asarray(numerators, dtype=float)
        b = np.asarray(denominators, dtype=float)
        ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
        numer = np.bincount(self.groups, weights=np.where(ok, a, 0), minlength=len(self.names))
        denom = np.bincount(self.groups, weights=np.where(ok, b, 0), minlength=len(self.names))
        n, d = self.mult @ numer, self.mult @ denom
        return np.divide(n, d, out=np.full(len(d), np.nan), where=d > 0)

    def ci(self, numerators, denominators=None):
        if denominators is None:
            denominators = [1 if finite(x) else 0 for x in numerators]
        sample = self.samples(numerators, denominators)
        sample = sample[np.isfinite(sample)]
        return np.quantile(sample, [.025, .975]).tolist() if len(sample) else None

    def paired_binary(self, before, after):
        before, after = np.asarray(before, dtype=np.int64), np.asarray(after, dtype=np.int64)
        assert before.shape == after.shape and before.shape == (len(self.ids), 2)
        totals = np.bincount(self.groups, weights=(after - before).sum(1), minlength=len(self.names)).astype(np.int64)
        counts = np.bincount(self.groups, minlength=len(self.names)).astype(np.int64) * 2
        full_n, full_d = int(totals.sum()), int(counts.sum())
        draw_n, draw_d = self.mult @ totals, self.mult @ counts
        assert (draw_d > 0).all()
        sample = draw_n / draw_d * 100
        if full_n > 0:
            extreme = (draw_n <= 0) | (draw_n * full_d >= 2 * full_n * draw_d)
        elif full_n < 0:
            extreme = (draw_n >= 0) | (draw_n * full_d <= 2 * full_n * draw_d)
        else:
            extreme = np.ones(len(draw_n), dtype=bool)
        return dict(status="CALCULATED", effect_pp=full_n / full_d * 100,
                    CI95=np.quantile(sample, [.025, .975]).tolist(),
                    p_raw=(1 + int(extreme.sum())) / (1 + len(sample)), extreme_draws=int(extreme.sum()),
                    numerator=full_n, denominator=full_d, frames=len(self.ids), seeds=list(SEEDS),
                    per_seed_effect_pp=((after - before).sum(0) / len(self.ids) * 100).tolist(),
                    estimand="frame-weighted, both seeds within image; paired session clusters",
                    p_definition="centered paired bootstrap, exact integer tie rule, plus-one",
                    reference="same manual-corner geometric reference; DEV exploratory")


def holm(records, declared_size):
    # Missing/blocked hypotheses keep the originally declared family size.
    valid = sorted((row["p_raw"], i) for i, row in enumerate(records) if row.get("p_raw") is not None)
    running = 0.
    for rank, (p, index) in enumerate(valid):
        running = max(running, min(1., (declared_size - rank) * p))
        records[index]["p_Holm"] = running
    for row in records:
        row.setdefault("p_Holm", None)
        row["declared_family_size"] = declared_size
        row["estimable_family_tests"] = len(valid)
    return records


def summarize(index, method, ids, seeds, boot):
    rr = [[index[method, fid, seed] for seed in seeds] for fid in ids]
    vectors = {}
    for metric in METRICS:
        vectors[metric] = [statistics.fmean(v) if (v := [value(r, metric) for r in records if finite(value(r, metric))]) else None for records in rr]
    success = [sum(strict(r) for r in records) / len(seeds) for records in rr]
    availability = [sum(bool(r["available"]) for r in records) / len(seeds) for records in rr]
    result = dict(frames=len(ids), seeds=list(seeds), prediction_rows=len(ids) * len(seeds),
                  strict_success= dict(rate=statistics.fmean(success), successes=sum(sum(strict(r) for r in records) for records in rr),
                                       denominator=len(ids)*len(seeds), CI95=boot.ci(success)),
                  pose_availability=dict(rate=statistics.fmean(availability), available=sum(sum(bool(r["available"]) for r in records) for records in rr),
                                         denominator=len(ids)*len(seeds), CI95=boot.ci(availability)),
                  metrics={metric:moments(values, boot.ci(values)) for metric, values in vectors.items()},
                  continuous_policy="available values mean within image, then moments across images; finite sample size explicit",
                  binary_policy="each seed decision first; both seeds remain in image in bootstrap",
                  PCK={}, per_seed={})
    for threshold in THRESHOLDS:
        n = [sum(sum(e <= threshold for e in corner_errors(r)) for r in records) / len(seeds) for records in rr]
        d = [sum(len(corner_errors(r)) for r in records) / len(seeds) for records in rr]
        assert all(x > 0 for x in d)
        result["PCK"][str(threshold)] = dict(rate=sum(n)/sum(d), hits_seed_mean=sum(n), reference_corners=sum(d),
                                           CI95=boot.ci(n,d), threshold_px=threshold, operator="<=",
                                           policy="raw 2D; all fixed valid reference corners; missing prediction is a miss; independent of PnP availability")
    axes = [r["axis_wrong"] for records in rr for r in records if r.get("axis_wrong") is not None]
    result["axis_wrong"] = dict(errors=sum(bool(v) for v in axes), denominator=len(axes), rate=sum(bool(v) for v in axes)/len(axes) if axes else None)
    confusion = [sum(bool(r["wd_confusion"]) for r in records)/len(seeds) for records in rr]
    result["legacy_WD_confusion"] = dict(rate=statistics.fmean(confusion), CI95=boot.ci(confusion),
                                         definition="available R>45deg and abs(yaw)>=60deg; distinct from direct axis error")
    for seed in seeds:
        r = [index[method,fid,seed] for fid in ids]
        result["per_seed"][str(seed)] = dict(strict_successes=sum(strict(x) for x in r), denominator=len(ids),
                                              unavailable=sum(not x["available"] for x in r),
                                              metrics={metric:moments([value(x,metric) for x in r]) for metric in METRICS},
                                              PCK={str(t):sum(sum(e <= t for e in corner_errors(x)) for x in r)/sum(len(corner_errors(x)) for x in r) for t in THRESHOLDS})
    return result


def family(pairs, index, ids, boot, statuses):
    records = []
    for after, before in pairs:
        complete = all((m,fid,s) in index for m in (after,before) for fid in ids for s in SEEDS)
        if not complete:
            records.append(dict(after=after,before=before,status="BLOCKED_OR_INCOMPLETE",p_raw=None,CI95=None,effect_pp=None,
                                reason=statuses.get(after.replace("_SUBPIX",""), "Required paired rows missing")))
            continue
        a = [[strict(index[after,fid,s]) for s in SEEDS] for fid in ids]
        b = [[strict(index[before,fid,s]) for s in SEEDS] for fid in ids]
        records.append(dict(after=after,before=before,**boot.paired_binary(b,a)))
    return holm(records, len(pairs))


def all_transitions(index, ids):
    output=[]
    for after,before in PRIMARY+MECHANISM:
        for fid in ids:
            for seed in SEEDS:
                if (after,fid,seed) not in index or (before,fid,seed) not in index:continue
                a,b=index[after,fid,seed],index[before,fid,seed]
                a_success,b_success=strict(a),strict(b)
                ea,eb=corner_errors(a),corner_errors(b)
                assert len(ea)==len(eb)
                output.append(dict(after=after,before=before,id=fid,seed=seed,session=a["session"],grade=a["grade"],
                    event="RECOVERY" if a_success and not b_success else "DAMAGE" if b_success and not a_success else "UNCHANGED",
                    before_success=b_success,after_success=a_success,
                    before_available=b["available"],after_available=a["available"],
                    before_Tcm=value(b,"Tcm"),after_Tcm=value(a,"Tcm"),before_Rdeg=value(b,"Rdeg"),after_Rdeg=value(a,"Rdeg"),
                    before_corner_mean_px=statistics.fmean(eb) if all(map(math.isfinite,eb)) else None,
                    after_corner_mean_px=statistics.fmean(ea) if all(map(math.isfinite,ea)) else None,
                    corners_damaged=sum(x>y+1e-8 for x,y in zip(ea,eb)),corners_improved=sum(x<y-1e-8 for x,y in zip(ea,eb)),
                    before_hyp=b.get("selected_hyp"),after_hyp=a.get("selected_hyp")))
    return output


def csv_table(path, headers, records):
    with Path(path).open("x",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=headers);writer.writeheader();writer.writerows(records)


def number(value, digits=3):
    return "NA" if value is None else f"{value:.{digits}f}"


def report(doc, metrics, paired, selection, benchmark, statuses):
    selected=selection.get("selected_arm") if selection else None
    selected=ALIASES.get(selected,selected)
    row=next((x for x in paired["primary"] if x["after"]==f"{selected}_SUBPIX"),None)
    if row is None or row.get("effect_pp") is None:
        first="현재 PoseFix-style+SubPix 대비 정확도 개선 여부는 미확정이며, 실행·자료 상태를 그대로 공개한다."
    elif row["effect_pp"]>0 and row["CI95"][0]>0 and row["p_Holm"]<.05:
        first=f"Synthetic VAL로 선택한 {selected}+SubPix가 동일 non-Severe 개발 영상에서 PoseFix-style+SubPix보다 strict 5cm·5° 성공률을 {row['effect_pp']:.3f}%p 높였다(탐색적 개발 평가)."
    elif row["effect_pp"]>0:
        first=f"Synthetic VAL로 선택한 {selected}+SubPix의 성공률 점추정은 PoseFix-style+SubPix보다 {row['effect_pp']:.3f}%p 높지만, Holm 보정 후 정확도 우월을 확인하지 못했다."
    else:
        first=f"Synthetic VAL로 선택한 {selected}+SubPix는 PoseFix-style+SubPix를 이기지 못했다(성공률 차이 {row['effect_pp']:.3f}%p)."
    lines=[first,"",f"[확인] 주평가는 Severe 및 물리 보호 구간을 제외한 **{metrics['frames']}장, {metrics['sessions']}개 세션**, 공통 seed 1·2다([DATA_CONTRACT.json](DATA_CONTRACT.json), [ROW_METRICS.jsonl.gz](ROW_METRICS.jsonl.gz)). Severe 제외는 사용자 지정 범위이며 자동 가림 판별의 성능 주장이 아니다.","",
           "[확인] 참조 자세는 수동 코너 기반 기하 재구성이다. 아래 수치는 같은 참조에 대한 차이이며 직접 계측한 물리 정확도의 독립 검증으로 해석하지 않는다. 기존 RECT245와 분모·보호/노출 계약이 달라 수치를 합치지 않는다.","",
           "[확인] 5cm·5°는 **T<5cm AND R<5°**다. 자세 미산출은 성공률 분모에 남는다. 연속 오차는 산출된 값의 seed 평균을 먼저 영상 안에서 계산한 뒤 영상별 평균·표본 SD·중앙값·P90·P95를 구한다. PCK는 영상 안 참조 코너의 고정 분모로 계산하며 PnP 실패와 별개인 원래 2D 지표다.","",
           "동일 non-Severe 최종 정확도 (공통 seed 1·2)","",
           "| 방법 | 성공/220 | 성공률 % | T cm 평균±SD | T 중앙/P90/P95 | R ° 평균±SD | R 중앙/P90/P95 | 미산출 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    if (doc/"TRAIN_RECEIPTS.json").exists():
        training=read(doc/"TRAIN_RECEIPTS.json");aux=read(doc/"AUX_WEIGHT_LOCK.json")
        introduction=["",f"[확인] 모든 완료 arm은 seed 1·2에서 동일 마지막 **{training['common_steps']} updates/arm/seed**, batch 16으로 학습했다. 총 추가 optimizer updates는 {training['total_updates']}이며 상태는 {training['status']}다. 원래 OLD_P의 같은-seed checkpoint에서 출발하고 기반 모델은 고정했다([TRAIN_RECEIPTS.json](TRAIN_RECEIPTS.json)).", "",
            f"[확인] 새 보정기의 동일 용량은 {aux['all_arm_capacity']} parameters(기존 {aux['original_corrector_parameters']}, 추가 {aux['additional_parameters']})다. PoseFix-style 정본의 보정기 68,661,641 parameters 및 6,000 updates/seed와 학습 자원이 같다고 주장하지 않는다. M0/M1/M2에서는 RGB CNN의 2,768 parameters 실행을 건너뛰지만, RGB 입력 열 512 parameters를 포함한 81차원 scorer matmul은 그대로 실행하며 해당 입력은 0이다([SOURCE_BINDINGS.json](SOURCE_BINDINGS.json), [LATENCY.json](LATENCY.json)).", "",
            f"[확인] native coordinate SmoothL1 및 covariance 감독 가중치는 고정 TRAIN 32 minibatch의 actual-logit gradient norm으로 한 번 정했다: {json.dumps(aux['weights'],ensure_ascii=False)}. 실사 결과로 바꾸지 않았다. 후보 hull의 outer radius는 box diagonal×0.08이며 context sampling의 stencil_fraction과 구분한다([AUX_WEIGHT_LOCK.json](AUX_WEIGHT_LOCK.json), [DECODER_AUDIT.json](DECODER_AUDIT.json)).", ""]
        lines[7:7]=introduction
    if (doc/"SELECTION_INVALID_BOX_ADAPTER_AMENDMENT.json").exists():
        repair=read(doc/"SELECTION_INVALID_BOX_ADAPTER_AMENDMENT.json")
        lines[7:7]=["",f"[확인] 첫 synthetic calibration/selection 프로세스는 미검출 row의 NaN box를 native RGB tile 정수 반경으로 바꾸다 실패했다({repair['failed_stage_wall_seconds']:.3f}초). 이후 source Generic의 [0,0,1,1] fallback을 미사용 RGB tile allocation에만 적용했다. 원래 model boxes/support/points/masks와 782장 분모를 보존했으며 재학습·조건 탐색·실패 row 제거는 없다. 실패/수선 프로세스 시간은 모두 예산에 포함한다([SELECTION_INVALID_BOX_ADAPTER_AMENDMENT.json](SELECTION_INVALID_BOX_ADAPTER_AMENDMENT.json), [STAGE_selection.json](STAGE_selection.json)).", ""]
    order=["BASE","SUBPIX","OLD_P","OLD_P_SUBPIX","N3_DIM_SYM","N3_THEN_SUBPIX","POSEFIX","POSEFIX_SUBPIX"]
    if selected and selected+"_SUBPIX" not in order:order.append(selected+"_SUBPIX")
    groups=metrics["groups"]
    status_display={k:statuses[k] for k in ("status","executed_arms","blocked_arms","common_steps","total_updates","seeds") if k in statuses} or statuses
    def table_row(method):
        packet=groups.get(method,{}).get("common_seed12")
        if not packet:return f"| {method} | NA | NA | NA | NA | NA | NA | BLOCKED |"
        t,r=packet["metrics"]["Tcm"],packet["metrics"]["Rdeg"]
        return f"| {method} | {packet['strict_success']['successes']}/{packet['strict_success']['denominator']} | {100*packet['strict_success']['rate']:.2f} | {number(t['mean'])}±{number(t['SD'])} | {number(t['median'])}/{number(t['P90'])}/{number(t['P95'])} | {number(r['mean'])}±{number(r['SD'])} | {number(r['median'])}/{number(r['P90'])}/{number(r['P95'])} | {packet['pose_availability']['denominator']-packet['pose_availability']['available']} |"
    lines += [table_row(m) for m in order]+["","[확인] 위 분모 220은 동일 110장×2seed의 예측 수다. 독립 영상 수는 110이며 bootstrap은 실제 세션 단위를 재표집하고 영상 내 두 seed를 묶는다. 기존 seed3 및 seed별 수치는 [METRICS.json](METRICS.json)에 별도로 보존한다.","",
        "[확인] 직접 축 오류는 사전 축 참조와 선택 W/D 가설의 불일치다. 기존 W/D confusion은 available R>45° AND |yaw|≥60°인 별도 기술 지표이며 두 값을 섞지 않는다. 직접 축 참조가 불명확한 1장은 축 오류 분모에서만 제외하고 최종 정확도 110장 분모에는 그대로 둔다.", "",
        "| 방법 | 직접 축 오류/평가 가능 예측 | 기존 W/D confusion % | ADDsym m 평균±SD | IoU3D 평균±SD |", "|---|---:|---:|---:|---:|"]
    for method in order:
        packet=groups.get(method,{}).get("common_seed12")
        if not packet:
            lines.append(f"| {method} | NA | NA | NA | NA |")
            continue
        axis=packet["axis_wrong"];add=packet["metrics"]["ADDsym_m"];iou=packet["metrics"]["IoU3D"]
        lines.append(f"| {method} | {axis['errors']}/{axis['denominator']} | {number(100*packet['legacy_WD_confusion']['rate'],2)} | {number(add['mean'])}±{number(add['SD'])} | {number(iou['mean'])}±{number(iou['SD'])} |")
    if (doc/"DECODER_AUDIT.json").exists():
        decoder=read(doc/"DECODER_AUDIT.json")
        examples=", ".join(f"{x['desired_dx_native_px']:g}px → {x['expected_dx_native_px']:.3f}px" for x in decoder["source_target_small_displacement_examples"])
        counts=decoder["corner_counts"]
        lines += ["",f"[확인] 실제 source target을 그대로 expected-decode한 box diagonal 340px, T=1/λ=1/cap 없음의 수식 예시는 {examples}다. 이는 정확한 target 분포도 작은 이동에서 축소될 수 있음을 보인 것이며 실제 checkpoint 모두에 같은 비율이 있다는 뜻은 아니다. 60,000장 진단의 support 코너 {counts['supported']:,}개 중 auxiliary eligible은 {counts['auxiliary_eligible']:,}개이고 hull 밖 {counts['outside_hull']:,}개, native cap 밖 {counts['beyond_native_cap']:,}개를 각각 기록했다. 두 제외 조건은 중첩될 수 있다([DECODER_AUDIT.json](DECODER_AUDIT.json)).", "",
            "[확인] 유한 후보의 확률평균은 연속 좌표다. 후보 간격을 최소 가능한 좌표 오차라고 하지 않으며 임의 1.33 배율 보정은 적용하지 않았다. source feature-center와 native pixel-center 검사, 실제 TRAIN logits의 기대좌표 gradient/finite difference를 통과한 뒤 학습했다.", "", "![실제 source decoder와 전체 synthetic 잔차 진단](FIGURES/decoder_evidence.png)", ""]
    lines += [
        "학습 메커니즘 절제 (모든 완료 arm, 최종 SubPix 포함)","",
        "| 방법 | 성공/220 | 성공률 % | T cm 평균±SD | T 중앙/P90/P95 | R ° 평균±SD | R 중앙/P90/P95 | 미산출 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]+[table_row(a+"_SUBPIX") for a in ARMS]
    lines += ["","[확인] M0는 기존 후보 CE 추가 학습 대조, M1은 기대좌표 직접 감독, M2는 고정 가중 covariance 기하 감독, M3는 native RGB 분기, M4는 RGB+기하 감독이다. 치수는 M2/M4의 TRAIN 감독과 모든 방법의 inference PnP에 사용되며, 치수를 새 추론 scorer 입력으로 넣은 효과와 구분한다.","",
              "[확인] 최종 후보는 synthetic VAL의 마지막 공통 checkpoint로만 정한다. REAL 최고 행으로 후보를 재선택하지 않는다. 선택 기준 및 완료·차단 상태는 선택/학습 receipt에 기록한다.","",
              "[확인] 추가 학습과 auxiliary 가중치 산출은 TRAIN만 쓴다. 최종 후보 선택은 고정 selection 1,031장 중 사전 C2 물리회전 조건을 만족한 782장으로만 수행한다. 반면 decoder/geometry의 수학적 검사는 synthetic source 60,000장 전체(heldout 1,985장 포함)를 열람했고 기존 frozen OLD_P 확률의 calibration+selection+heldout 진단 cohort는 seed별 4,020장(지원 4,006장)이다. 이 analytical exposure를 숨기지 않으며 synthetic heldout을 미열람 독립 TEST라고 부르지 않는다. 새 M0–M4의 heldout 성능 추론·선택은 수행하지 않았다([DECODER_AUDIT.json](DECODER_AUDIT.json), [SYNTHETIC_SELECTION_LOCK.json](SYNTHETIC_SELECTION_LOCK.json)).", "",
              "고정된 짝 검정", "", "| Family | 대비 | 효과 %p | 95% CI | raw p | Holm p | 상태 |", "|---|---|---:|---:|---:|---:|---|"]
    for fam in ("primary","mechanism"):
        for r in paired[fam]:
            ci=r.get("CI95");interval="NA" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            lines.append(f"| {fam} | {r['after']} − {r['before']} | {number(r.get('effect_pp'))} | {interval} | {number(r.get('p_raw'),6)} | {number(r.get('p_Holm'),6)} | {r['status']} |")
    lines += ["", "[확인] 구성의 증분은 최종 성공률뿐 아니라 같은 영상의 T/R 꼬리와 실제 비용을 함께 읽는다. 아래 기술적 차이는 새 가설검정 family를 만들지 않는다."]
    for contrast in paired["mechanism"]:
        after,before=contrast["after"],contrast["before"]
        a=groups.get(after,{}).get("common_seed12");b=groups.get(before,{}).get("common_seed12")
        if not a or not b:
            lines += ["",f"[확인] {after} − {before}: 필요한 arm이 완료되지 않아 증분은 NA다."]
            continue
        improved=contrast["effect_pp"]>0 and contrast["CI95"][0]>0 and contrast["p_Holm"]<.05
        conclusion="Holm 보정 후 성공률 증분이 확인됐다" if improved else "이 설정에서 Holm 보정 후 성공률 증분을 확인하지 못했다"
        diff=lambda metric,stat:number(a["metrics"][metric][stat]-b["metrics"][metric][stat]) if a["metrics"][metric][stat] is not None and b["metrics"][metric][stat] is not None else "NA"
        ba=(benchmark or {}).get("methods",{}).get(after,{}).get("latency_ms",{}).get("median")
        bb=(benchmark or {}).get("methods",{}).get(before,{}).get("latency_ms",{}).get("median")
        cost=number(ba-bb) if ba is not None and bb is not None else "NA"
        lines += ["",f"[확인] {after} − {before}: {conclusion}. 성공률 {number(contrast['effect_pp'])}%p, T 중앙/P95 차이 {diff('Tcm','median')}/{diff('Tcm','P95')}cm, R 중앙/P95 차이 {diff('Rdeg','median')}/{diff('Rdeg','P95')}°, 전체 latency 중앙값 차이 {cost}ms다."]
    if selected in ARMS:
        retained={"M0":"기존 CE 추가 학습", "M1":"기대좌표 직접 감독", "M2":"기대좌표+기하 감독",
                  "M3":"기대좌표+RGB 분기", "M4":"기대좌표+RGB+기하 감독"}[selected]
        lines += ["",f"[확인] Synthetic VAL에서 고정한 실험 후보는 **{selected} ({retained})**다. 이 선택은 정확도 개선 제품으로 채택했다는 뜻이 아니다. 완료된 모든 arm과 구성은 표·원행·실패 사례에 남기고 실사 최고 행으로 후보를 갈아타거나 다시 학습하지 않는다([SYNTHETIC_SELECTION.json](SYNTHETIC_SELECTION.json))."]
        if row and row.get("effect_pp") is not None and not (row["effect_pp"]>0 and row["CI95"][0]>0 and row["p_Holm"]<.05):
            lines += ["", "[확인] 이번 screen은 PoseFix+SubPix 대비 정확도 개선을 확보하지 못했으므로 새 후보의 운영 채택을 권고하지 않는다. 기하 항은 M2−M1 및 M4−M3, RGB 분기는 M3−M1의 실제 증분·꼬리·추가 비용을 기준으로 읽는다. 아래 대비에서 증분을 확인하지 못한 구성은 개선된 핵심 기여 또는 운영 추가 기능에서 제외한다. 무차이는 동등성 증명이 아니며, 기하·RGB 일반의 불가능성을 뜻하지 않는다."]
        changes=read(doc/"CHANGED_SUCCESS_CASES.json")["rows"]
        own=[c for c in changes if c["after"]==selected+"_SUBPIX" and c["before"]=="POSEFIX_SUBPIX"]
        events=Counter(c["event"] for c in own)
        lines += ["",f"[확인] 선택 후보와 PoseFix+SubPix가 성공 여부에서 달라진 동일 영상×seed는 복구 {events['RECOVERY']}건, 손상 {events['DAMAGE']}건이다. 이 건수는 성공률 순증가와 구별하며 양쪽 전체 ID·T/R·2D 오류를 남긴다."]
        candidate=groups[selected+"_SUBPIX"]["common_seed12"];comparator=groups["POSEFIX_SUBPIX"]["common_seed12"]
        for seed in SEEDS:
            n=candidate["per_seed"][str(seed)]["strict_successes"];b=comparator["per_seed"][str(seed)]["strict_successes"]
            counts=Counter(c["event"] for c in own if c["seed"]==seed)
            lines += ["",f"[확인] seed {seed}: {selected}+SubPix {n}/{metrics['frames']}, PoseFix+SubPix {b}/{metrics['frames']}; 복구 {counts['RECOVERY']}건, 손상 {counts['DAMAGE']}건이다. 동일 참조에서의 관측 결과이며 seed별 변동의 원인이나 새로운 감독의 일반적 실패 메커니즘을 인과적으로 입증한 것은 아니다."]
    if (doc/"TRAIN_RECEIPTS.json").exists():
        train=read(doc/"TRAIN_RECEIPTS.json")
        lines += ["", "학습 중 감독 gradient 비율 (actual logits 기준)","", "| seed | arm | step | 좌표/CE | 기하/CE |", "|---:|---|---:|---:|---:|"]
        for receipt_name in train["receipts"]:
            receipt=read(doc/receipt_name)
            for h in receipt["history"]:
                if h["step"] not in {100,1000,train["common_steps"]}:continue
                for arm,p in h["arms"].items():
                    ratio=p["gradient_ratios"]
                    coordinate=0. if arm=="M0" else ratio.get("coord_to_CE")
                    geometry=ratio.get("geo_to_CE") if arm in ("M2","M4") else 0.
                    lines.append(f"| {receipt['seed']} | {arm} | {h['step']} | {number(coordinate,4)} | {number(geometry,4)} |")
        lines += ["", "[확인] 위 표는 해당 arm에 실제 적용되는 항의 비율이다. M0 좌표/기하 항과 M1/M3 기하 항은 0이다. 마지막 step의 비율이 기록되지 않았으면 NA로 남긴다. covariance 항은 고정 가중 local surrogate이며 LC 전체 Eq17 재현 또는 실제 T/R 손실이라고 하지 않는다."]
    lines += ["","[확인] primary는 다섯 새 arm+SubPix−PoseFix+SubPix의 strict 성공률 5검정, mechanism은 M1−M0/M2−M1/M3−M1/M4−M3 최종 SubPix의 strict 성공률 4검정이다. 두 family의 Holm 보정은 별개다. 세션 bootstrap 10,000회(seed 20260917), 중심화한 양측 검정과 exact 정수 동률을 사용한다([PAIRED_RESULTS.json](PAIRED_RESULTS.json)). 다른 지표는 기술통계이며 같은 family에 사후 추가하지 않는다.","",
        "2D 코너 정확도 (고정 참조 코너 분모, %)","", "| 방법 | PCK@1px | @2px | @3px | @5px | @10px |", "|---|---:|---:|---:|---:|---:|"]
    for method in list(dict.fromkeys(order+[a+"_SUBPIX" for a in ARMS])):
        p=groups.get(method,{}).get("common_seed12")
        cells=[number(100*p["PCK"][str(t)]["rate"],2) if p else "NA" for t in THRESHOLDS]
        lines.append("| "+method+" | "+" | ".join(cells)+" |")
    lines += ["", "[확인] 평균·표본 분산·SD·중앙값·P90·P95 및 세션 bootstrap 평균 95% CI, ADDsym/IoU와 직접 축 오류·기존 W/D 지표의 분리 수치는 [METRICS.json](METRICS.json)과 [ACCURACY_TABLE.csv](ACCURACY_TABLE.csv)에 있다. 아래 T/R 누적 분포는 220개 영상×seed 원행의 전체 유한 오차 범위를 표시하며 자세 실패도 분모에 남긴다. 표의 연속 요약은 110개 영상 안 seed 평균이므로 CDF와 요약 단위를 구별한다.","",
        "![고정 기반과 학습 분기](FIGURES/01_pipeline.png)","", "![PCK와 위치·회전 누적 분포](FIGURES/02_PCK_pose_CDF.png)","", "![동일 영상의 복구와 손상](FIGURES/03_recovery_damage.png)","", "![고정된 primary 및 mechanism 효과](FIGURES/04_fixed_paired_effects.png)", "",
        "[확인] 복구·손상과 좌표 오류 증가를 모두 남긴다. 전체 영상×seed 전이는 [ALL_PAIRED_TRANSITIONS.csv](ALL_PAIRED_TRANSITIONS.csv), 성공이 달라진 모든 행은 [CHANGED_SUCCESS_CASES.json](CHANGED_SUCCESS_CASES.json), 자세 미산출 ID는 [FAILURES.json](FAILURES.json)에 있다. 공개 사례 그림은 원본 RGB를 포함하지 않는 수치/코너 도식이다.","",
        "동일 정밀도 설정의 비용", "", "| 방법 | 전체/추가 파라미터 | 전체 latency ms 중앙/P90 | peak memory MiB | 측정 상태 |", "|---|---:|---:|---:|---|"]
    bench=benchmark.get("methods",{}) if benchmark else {}
    for method in ["BASE","SUBPIX","OLD_P","OLD_P_SUBPIX","N3_DIM_SYM","N3_THEN_SUBPIX","POSEFIX","POSEFIX_SUBPIX"]+[a+s for a in ARMS for s in ("","_SUBPIX")]:
        b=bench.get(method,{})
        latency=b.get("latency_ms",{})
        lines.append(f"| {method} | {b.get('total_params','NA')}/{b.get('additional_params','NA')} | {number(latency.get('median'))}/{number(latency.get('P90'))} | {number(b.get('peak_memory_MiB'))} | {b.get('status','미측정')} |")
    lines += ["","[확인] 비용은 실제 완료한 독립 프로세스 측정만 쓴다. 미측정 항목은 NA이며 일부 accuracy-speed 점으로 전역 최적성을 주장하지 않는다. RGB를 쓰지 않는 대조 arm의 유지된 슬롯과 실제 실행 분기 비용은 runtime receipt로 구분한다.","",
              "[확인] model 연산은 FP32이고 AMP 및 TF32는 끈다. 기존 frozen feature capture는 p3/p4를 FP16으로 반올림해 저장한 뒤 GenericPointRefiner가 features.float()로 사용한다. 이 중간 반올림은 모든 feature 기반 비교 경로에 공통이며 모든 중간 tensor가 FP32라는 뜻은 아니다. PoseFix native RGB 입력·연산은 FP32다([ENVIRONMENT.json](ENVIRONMENT.json), [NEW_EVALUATION_RECEIPT.json](NEW_EVALUATION_RECEIPT.json)).", "",
              "[확인] M1의 후보 확률 기대좌표는 미분 가능한 weighted sum으로 계산한다. 이는 [Integral Human Pose Regression](https://arxiv.org/abs/1711.08229)의 expectation 개념을 이산 후보에 적용한 것이다. M2/M4는 reference 위치에서 캐시한 pose Jacobian A를 detach하고 residual만 미분하여 24차원 box-corner covariance trace의 제곱근 평균을 감독한다. [Linear-Covariance Loss](https://arxiv.org/html/2303.11516v2)의 E_cov를 고정 가중 local surrogate로 쓰며 E_prior/E_linear를 포함한 전체 Eq17 구현이 아니다. RGB 후보 분기는 [Deep ChArUco](https://arxiv.org/abs/1812.03247)의 이미지 기반 subpixel refinement와 목적상 관련되지만 marker 전용 네트워크를 재현하거나 팔레트에서 같은 효과를 보장하지 않는다.", "",
              "[추정] 짧은 2seed screen에서 실패하거나 개선되지 않은 arm은 이 설정의 증분 가치가 확인되지 않은 것이다. 방법의 영구 기각 또는 무차이의 동등성 증명으로 해석하지 않는다. geometry/RGB 제거 판단은 synthetic VAL 선택과 실제 accuracy·꼬리·비용을 연결해 적는다.","",
              "[확인] 실행 상태: "+json.dumps(status_display,ensure_ascii=False)+". 실제 완료한 학습량·gradient·calibration·자료 노출 및 실패 원인은 TRAIN_RECEIPTS/DECODER_AUDIT/COMPUTE_RECEIPT에 남긴다. 새 실험은 개발 자료를 재사용하므로 독립 최종 검증이 아니다.","",
              "재현 및 검산: [REPRODUCE.md](REPRODUCE.md), [STATISTICS_RECEIPT.json](STATISTICS_RECEIPT.json), [INDEPENDENT_VERIFICATION.json](INDEPENDENT_VERIFICATION.json). 기존 checkpoint/원고/과거 결과는 덮어쓰지 않는다.",""]
    (doc/"RESULTS_KO.md").write_text("\n".join(lines),encoding="utf-8")


def run(args):
    doc=Path(args.public_dir);doc.mkdir(parents=True,exist_ok=True)
    assert not (doc/"METRICS.json").exists(), "Scientific outputs already exist"
    contract=read(args.contract)
    ids=sorted(contract["allowed_ids"])
    assert len(ids)==110, "Reconfirm changed population before aggregation"
    rr=load_rows(args.rows);index={(r["method"],r["id"],r["seed"]):r for r in rr}
    assert len(index)==len(rr)
    assert all(r["id"] in ids and r["grade"].lower() in ("clean","moderate") for r in rr)
    metadata={}
    for r in rr:
        item=(r["session"],r["grade"].lower(),r["corner_valid"])
        if r["id"] in metadata:assert metadata[r["id"]]==item,("Metadata changed",r["id"])
        metadata[r["id"]]=item
        strict(r)
    assert set(metadata)==set(ids)
    sessions=[metadata[fid][0] for fid in ids];names=sorted(set(sessions))
    draw=np.random.default_rng(20260917).integers(0,len(names),size=(10000,len(names)))
    np.save(doc/"BOOTSTRAP_DRAWS.npy",draw,allow_pickle=False)
    boot=Bootstrap(ids,sessions,draw)
    lock=dict(seed=20260917,repetitions=10000,sessions=names,frames=len(ids),seeds=list(SEEDS),
              sha256=sha(doc/"BOOTSTRAP_DRAWS.npy"),unit="actual session; both seeds and paired methods inside image",
              population="non-Severe110 physical-protection intersection; no RECT245 merging")
    write(doc/"BOOTSTRAP_LOCK.json",lock)
    groups={}
    for method in sorted({r["method"] for r in rr}):
        seeds=sorted({r["seed"] for r in rr if r["method"]==method})
        assert all((method,fid,s) in index for fid in ids for s in seeds),("Incomplete method",method)
        groups[method]=dict(all_existing_seeds=summarize(index,method,ids,seeds,boot),
                            common_seed12=summarize(index,method,ids,SEEDS,boot) if set(SEEDS)<=set(seeds) else None)
    status=read(args.status) if args.status else {}
    statuses=status.get("arms",status)
    paired=dict(status="COMPLETE",metric="strict T<5cm AND R<5deg",primary=family(PRIMARY,index,ids,boot,statuses),
                mechanism=family(MECHANISM,index,ids,boot,statuses),bootstrap=lock,
                multiplicity="Holm separately in fixed5 primary and fixed4 mechanism; blocked tests remain planned")
    metrics=dict(status="COMPLETE",frames=len(ids),sessions=len(names),grades=dict(Counter(metadata[fid][1] for fid in ids)),
                 common_seeds=list(SEEDS),groups=groups,bootstrap=lock,
                 reference_kinds=sorted({r.get("reference_kind","UNKNOWN") for r in rr}),
                 existing_seed3_separate_from_new_seed12=True,additional_inference_or_PnP_or_training=0)
    transitions=all_transitions(index,ids)
    changed=[r for r in transitions if r["event"]!="UNCHANGED"]
    failures={method:{str(seed):[fid for fid in ids if not index[method,fid,seed]["available"]] for seed in sorted({r["seed"] for r in rr if r["method"]==method})} for method in groups}
    write(doc/"METRICS.json",metrics);write(doc/"PAIRED_RESULTS.json",paired)
    write(doc/"CHANGED_SUCCESS_CASES.json",dict(status="COMPLETE",rows=changed,all_recovery_and_damage_retained=True))
    write(doc/"FAILURES.json",dict(status="COMPLETE",unavailable=failures,blocked=statuses))
    if transitions:csv_table(doc/"ALL_PAIRED_TRANSITIONS.csv",list(transitions[0]),transitions)
    table=[]
    for method,g in groups.items():
        for scope,packet in g.items():
            if not packet:continue
            table.append(dict(method=method,seed_scope=scope,frames=packet["frames"],seeds="/".join(map(str,packet["seeds"])),successes=packet["strict_success"]["successes"],denominator=packet["strict_success"]["denominator"],success_percent=100*packet["strict_success"]["rate"],**{metric+"_"+k:v for metric,m in packet["metrics"].items() for k,v in m.items() if k in ("mean","SD","variance","median","P90","P95","n")}))
    csv_table(doc/"ACCURACY_TABLE.csv",list(table[0]),table)
    selection=read(args.selection) if args.selection else None
    benchmark=read(args.benchmark) if args.benchmark else None
    if args.report:report(doc,metrics,paired,selection,benchmark,statuses)
    write(doc/"STATISTICS_RECEIPT.json",dict(status="COMPLETE",source_sha256=sha(Path(__file__)),
          inputs=[dict(name=Path(p).name,sha256=sha(p)) for p in args.rows],contract_sha256=sha(args.contract),
          rows=len(rr),groups=len(groups),all_pair_rows=len(transitions),changed_success_rows=len(changed),
          additional_PnP=0,additional_training=0,additional_model_forwards=0,
          selection_sha256=sha(args.selection) if args.selection else None,benchmark_sha256=sha(args.benchmark) if args.benchmark else None))
    print(json.dumps(dict(status="COMPLETE",rows=len(rr),groups=len(groups),primary=paired["primary"],mechanism=paired["mechanism"]),ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--public-dir",required=True);parser.add_argument("--rows",nargs="+",required=True)
    parser.add_argument("--contract",required=True);parser.add_argument("--status");parser.add_argument("--selection")
    parser.add_argument("--benchmark");parser.add_argument("--report",action="store_true")
    run(parser.parse_args())

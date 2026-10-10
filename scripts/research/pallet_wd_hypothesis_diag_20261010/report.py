"""Stage-1 scientific table/figure and Korean report from frozen public numbers."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import common as C

POP_LABELS = {"REAL_DEV":"REAL_DEV", "SYNTH_HELDOUT":"SYNTH_HELDOUT", "AUX":"REAL_DEV AUX"}
METHOD_ORDER = {"BASE":0, "N3_DIM_SYM":1, "SUBPIX":2, "N3_THEN_SUBPIX":3}
PREFIX = "scripts.research.pallet_wd_hypothesis_diag_20261010"
SOURCE_LINK = "../../../scripts/research/pallet_wd_hypothesis_diag_20261010/"
OFFICIAL_COMMIT = "eb5b6fac0dc155e4e52f576e304fbf11655ff339"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, text):
    Path(path).write_text(text.rstrip()+"\n", encoding="utf-8")


def number(value, digits=3):
    return "NA" if value is None or not math.isfinite(float(value)) else f"{value:.{digits}f}"


def interval(value, scale=1.):
    return "NA" if value is None else f"[{number(value[0]*scale)}, {number(value[1]*scale)}]"


def rate(summary, name):
    r=summary["rates"][name]
    return f'{number(100*r["rate"])}%<br>CI {interval(r["CI95"],100)}'


def distribution(stats):
    return (f'{number(stats["mean"])} ± {number(stats["std"])}<br>'
            f'분산 {number(stats["variance"])}<br>'
            f'중앙값 {number(stats["median"])} / P90 {number(stats["P90"])}<br>'
            f'평균 CI {interval(stats["CI95"])}')


def ordered_groups(metrics):
    for population in ("REAL_DEV", "SYNTH_HELDOUT", "AUX"):
        if population not in metrics:
            continue
        groups=metrics[population]["groups"]
        for name in sorted(groups,key=lambda x:(x.split("::")[0],METHOD_ORDER.get(x.split("::")[1],99))):
            yield population,name,groups[name]


def stage1_table(metrics):
    lines=["[확인] STAGE1_DIAGNOSTIC_ONLY: 같은 예측 코너에 GT의 W/D parity를 공급한 진단이며, 새 선택 규칙의 검증 결과가 아닙니다.",
        "", "T 단위는 cm, R 단위는 °, 분산은 각각 cm²/°²입니다. ±는 표본 표준편차(ddof=1)입니다. CI는 평균 또는 비율의 95% bootstrap 구간입니다. seed mean의 오차 분포는 영상별 세 seed의 스칼라 오차 평균이며, 비율은 각 seed에서 먼저 계산한 이진 지표의 평균입니다. 각 seed와 나쁜 결과를 모두 표시합니다.",
        "", "| 모집단 / backbone | 방법 | 집계 | 선택 | 자세 산출/전체 | 혼동률 | 5 cm·5° 성공률 | T: 평균±SD, 분산, 중앙값/P90, CI | R: 평균±SD, 분산, 중앙값/P90, CI | S0−oracle R 평균 차이·CI |",
        "|---|---|---|---|---:|---|---|---|---|---|"]
    for population,name,group in ordered_groups(metrics):
        backbone,method=name.split("::")
        scopes=[("seed mean",group["seed_mean"])] + [(f"seed {s}",group["per_seed"][str(s)]) for s in group["seeds"]]
        for scope,arms in scopes:
            for arm in ("S0","ORACLE"):
                summary=arms[arm]
                gap=group["R_mean_difference_F_minus_oracle"]
                gapcell=f'{number(gap["mean"])}°<br>{interval(gap["CI95"])}<br>paired n={gap["n"]}' if scope=="seed mean" and arm=="S0" else "—"
                lines.append(f'| {POP_LABELS[population]} / {backbone} | {method} | {scope} | {arm} | '
                    f'{summary["available_all_seeds"]}/{summary["frames"]} | {rate(summary,"confusion_rate")} | '
                    f'{rate(summary,"success_rate")} | {distribution(summary["metrics"]["T_cm"])} | '
                    f'{distribution(summary["metrics"]["R_deg"])} | {gapcell} |')
    lines += ["", "[확인] 자세 미산출 영상은 오차 평균에서 제외하며, 산출 수를 바로 옆에 표시합니다. 성공률·혼동률은 전체 영상 분모로 계산하고 미산출을 false로 처리하므로, 특히 AUX의 낮은 혼동률은 미산출 수와 함께 해석해야 합니다. 전체 ADDsym·IoU3D, 최댓값, subgroup와 경고 ROC 수치는 [STAGE1_METRICS.json](STAGE1_METRICS.json)에 보존됩니다."]
    return "\n".join(lines)


def figure(metrics, doc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for path in (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")):
        if path.exists():
            plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams.update({"axes.unicode_minus":False,"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(13.6,11.4),gridspec_kw={"height_ratios":[1,1.22]})
    colors={"S0":"#2369a0","ORACLE":"#e37d24"}
    for ax,pop in zip(axes[0],("REAL_DEV","SYNTH_HELDOUT")):
        group=metrics[pop]["groups"]["YOLO::N3_THEN_SUBPIX"]
        bins=group["bins"]["elevation_deg"]
        labels=["<5","5–10","10–20","20–30",">=30"]
        if bins.get("UNKNOWN",{}).get("frames",0):labels.append("UNKNOWN")
        x=np.arange(len(labels))
        for arm,offset in (("S0",-.13),("ORACLE",.13)):
            for j,label in enumerate(labels):
                b=bins[label]
                if not b["frames"]:continue
                r=b["seed_mean"][arm]["rates"]["confusion_rate"]
                mean=100*r["rate"];ci=r["CI95"]
                error=None if ci is None else [[max(0,mean-100*ci[0])],[max(0,100*ci[1]-mean)]]
                ax.errorbar(j+offset,mean,yerr=error,fmt="o",color=colors[arm],capsize=3)
            # Use proxy legend handles even if the first bin is empty.
        ax.plot([],[],"o",color=colors["S0"],label="S0")
        ax.plot([],[],"o",color=colors["ORACLE"],label="GT-parity oracle")
        handles,legends=ax.get_legend_handles_labels();unique=dict(zip(legends,handles))
        ax.legend(unique.values(),unique.keys(),loc="upper right",fontsize=9)
        ax.set_xticks(x,[f'{label}°\nn={bins[label]["frames"]}' for label in labels]);ax.set_ylim(-2,102)
        ax.set_ylabel("혼동률 (%)");ax.set_title(f'{pop}: 고도별 혼동 — N3_THEN_SUBPIX',fontsize=11)
        ax.grid(axis="y",alpha=.22)
    panels=[[(p,n,g) for p,n,g in ordered_groups(metrics) if p in ("REAL_DEV","AUX")],
            [(p,n,g) for p,n,g in ordered_groups(metrics) if p=="SYNTH_HELDOUT"]]
    for ax,groups in zip(axes[1],panels):
        y=np.arange(len(groups))
        for j,(population,name,g) in enumerate(groups):
            for arm,offset in (("S0",-.13),("ORACLE",.13)):
                s=g["seed_mean"][arm]["metrics"]["R_deg"]
                if s["mean"] is None:continue
                ci=s["CI95"];error=None if ci is None else [[max(0,s["mean"]-ci[0])],[max(0,ci[1]-s["mean"])]]
                ax.errorbar(s["mean"],j+offset,xerr=error,fmt="o",color=colors[arm],capsize=3)
            gap=g["R_mean_difference_F_minus_oracle"]["mean"]
            common_n=g["R_mean_difference_F_minus_oracle"]["n"]
            ax.text(.985,j+.28,f'공통 {common_n} paired Δ {number(gap,2)}°',transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=7.7)
        ax.set_yticks(y,[n.replace("::"," / ") for p,n,g in groups],fontsize=9)
        ax.invert_yaxis();ax.set_xlabel("R 평균 (°): 점=평균, 선=평균 95% CI");ax.grid(axis="x",alpha=.22)
        ax.set_xlim(left=0);ax.margins(x=.18)
        ax.set_title("REAL_DEV + 기존 AUX" if groups and groups[0][0]!="SYNTH_HELDOUT" else "SYNTH_HELDOUT",fontsize=11)
    fig.suptitle("STAGE1_DIAGNOSTIC_ONLY — 고도별 뒤바뀜과 GT-parity oracle 진단",fontsize=16,y=.985)
    fig.text(.04,.016,"파란색 S0 / 주황색 GT-parity oracle. REAL: 13 session cluster, SYNTH: frame bootstrap 10,000회.\n"
             "코너는 동일하며 oracle만 GT W/D parity를 공급한다. DOPE 산출 수 S0 210/oracle 267: R 평균의 분모가 다르다.\n"
             "paired Δ는 두 결과 모두 산출된 영상의 R 차이 평균이다. 오차 최소 후보 탐색·인과 분해·새 규칙 효과 검증이 아니다.",fontsize=9)
    fig.subplots_adjust(left=.195,right=.97,bottom=.11,top=.92,hspace=.37,wspace=.44)
    path=doc/"STAGE1_figure.png";fig.savefig(path,dpi=145,pil_kwargs={"compress_level":9});plt.close(fig)
    sources={name:C.sha(doc/name) for name in ("STAGE1_METRICS.json","STAGE0_PARITY.json","STAGE1_SELECTION_SEAL.json","STAGE1_SOURCE_LOCK.json","STAGE1_FINAL_SOURCE_LOCK.json") if (doc/name).exists()}
    evidence={"status":"GENERATED_FROM_FIXED_PUBLIC_NUMBERS","figures":[{"path":path.name,"sha256":C.sha(path),"bytes":path.stat().st_size}],
              "evidence_source_sha256":sources,"raw_RGB_published":False,"rights_basis":"Own Matplotlib chart of numerical research outputs",
              "visual_inspection":{"status":"PENDING"}}
    write(doc/"STAGE1_FIGURE_INDEX.json",json.dumps(evidence,ensure_ascii=False,indent=2))


def headline(metrics):
    group=metrics["REAL_DEV"]["groups"]["YOLO::N3_THEN_SUBPIX"]
    s0,oracle=group["seed_mean"]["S0"],group["seed_mean"]["ORACLE"]
    return (f'[확인] 주 경로 N3_THEN_SUBPIX의 REAL_DEV R 평균은 S0 {number(s0["metrics"]["R_deg"]["mean"])}° / '
            f'GT-parity oracle {number(oracle["metrics"]["R_deg"]["mean"])}°이며, 혼동률은 '
            f'{number(100*s0["rates"]["confusion_rate"]["rate"])}% / {number(100*oracle["rates"]["confusion_rate"]["rate"])}%입니다.')


def subgroup_table(metrics,field):
    lines=["| 모집단 / 방법 | 구간 | 영상 수 | S0 혼동률 | oracle 혼동률 | S0 성공률 | oracle 성공률 | S0 R 평균 | oracle R 평균 |",
           "|---|---|---:|---|---|---|---|---:|---:|"]
    for population,name,g in ordered_groups(metrics):
        if population=="AUX":continue
        for label,b in g["bins"][field].items():
            if not b["frames"]:
                lines.append(f'| {population} / {name.replace("::"," / ")} | {label} | 0 | NA | NA | NA | NA | NA | NA |')
                continue
            a,o=b["seed_mean"]["S0"],b["seed_mean"]["ORACLE"]
            lines.append(f'| {population} / {name.replace("::"," / ")} | {label} | {b["frames"]} | {rate(a,"confusion_rate")} | {rate(o,"confusion_rate")} | {rate(a,"success_rate")} | {rate(o,"success_rate")} | {number(a["metrics"]["R_deg"]["mean"])} | {number(o["metrics"]["R_deg"]["mean"])} |')
    return "\n".join(lines)


def warning_table(metrics):
    lines=["| 모집단 / backbone / 방법 | AUC: seed 평균 | pooled AUC (기술) | 경고 기준 px | 경고율 | 혼동 포착률 | 경고 정밀도 | 유효 gap/예측 수 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for pop,name,g in ordered_groups(metrics):
        auc=g["score_gap_auc"]
        for threshold,w in g["warnings"].items():
            percent=lambda v: "NA" if v is None else number(100*v)+"%"
            lines.append(f'| {POP_LABELS[pop]} / {name.replace("::"," / ")} | {number(auc["macro_seed_mean"])} | {number(auc["pooled_descriptive"]["AUC"])} | <{threshold} | {percent(w["warning_rate"])} | {percent(w["caught_confusion_rate"])} | {percent(w["warning_precision"])} | {w["valid_score_predictions"]}/{w["predictions"]} |')
    return "\n".join(lines)


def coverage_table(metrics):
    lines=["| 모집단 / backbone | 방법 | S0 자세 산출/전체 | oracle 자세 산출/전체 | paired R 차이 공통 영상 수 |",
           "|---|---|---:|---:|---:|"]
    for pop,name,g in ordered_groups(metrics):
        backbone,method=name.split("::")
        s0,oracle=g["seed_mean"]["S0"],g["seed_mean"]["ORACLE"]
        lines.append(f'| {POP_LABELS[pop]} / {backbone} | {method} | {s0["available_all_seeds"]}/{g["frames"]} | '
            f'{oracle["available_all_seeds"]}/{g["frames"]} | {g["R_mean_difference_F_minus_oracle"]["n"]} |')
    return "\n".join(lines)


def documents(metrics,doc):
    parity=load(doc/"STAGE0_PARITY.json");audit=load(doc/"INPUT_AUDIT.json");execution=load(doc/"STAGE1_EXECUTION.json")
    assert parity["status"]=="PASS" and execution["stage2_rules_executed"]==0
    source_lock_name=execution.get("source_lock_path","STAGE1_SOURCE_LOCK.json")
    amendment=(load(doc/"STAGE1_POSTSEAL_AMENDMENT.json") if (doc/"STAGE1_POSTSEAL_AMENDMENT.json").exists() else None)
    amendment_method=("실제 최초 실행에서는 SYNTH GT 준비의 반복 NPZ 압축 해제로 메모리 문제가 발생해 중단했습니다. 봉인된 선택과 원래 core를 보존하고 별도 stage1_resume의 배열 1회 읽기로 scoring만 복구했습니다. 완료한 REAL/SYNTH F 선택 재실행은 0입니다. 중단 전 참조 margin fit 횟수는 NOT_MEASURED이고, 완료된 복구 참조 fit은 3,970회입니다. 이 provenance 제한을 숨기지 않습니다. [STAGE1_POSTSEAL_AMENDMENT.json](STAGE1_POSTSEAL_AMENDMENT.json)" if amendment else "이번 fresh 실행은 원래 frozen selector와 동일한 scoring 산술을 사용하며 SYNTH NPZ 배열을 한 번 읽는 wrapper를 이용합니다.")
    write(doc/"STAGE1_table.md",stage1_table(metrics))
    group=metrics["REAL_DEV"]["groups"]["YOLO::N3_THEN_SUBPIX"];s0,o=group["seed_mean"]["S0"],group["seed_mean"]["ORACLE"]
    synthetic=metrics["SYNTH_HELDOUT"]["groups"]["YOLO::N3_THEN_SUBPIX"];ss,so=synthetic["seed_mean"]["S0"],synthetic["seed_mean"]["ORACLE"]
    snippet=(f'With identical predicted corners, supplying the ground-truth W/D parity changed the seed-averaged mean rotation error from '
        f'{number(s0["metrics"]["R_deg"]["mean"])}° to {number(o["metrics"]["R_deg"]["mean"])}° on REAL_DEV (319 frames), and from '
        f'{number(ss["metrics"]["R_deg"]["mean"])}° to {number(so["metrics"]["R_deg"]["mean"])}° on SYNTH_HELDOUT (1,985 frames). '
        f'The corresponding confusion rates were {number(100*s0["rates"]["confusion_rate"]["rate"])}% versus '
        f'{number(100*o["rates"]["confusion_rate"]["rate"])}% on REAL_DEV. '
        'This is a GT-parity diagnostic, not a minimum-error oracle or a causal decomposition; the real reference is geometry reconstructed from manual 2D annotations rather than independent physical metrology.')
    write(doc/"PAPER_SNIPPET_EN.md",snippet)
    unclear=audit["human_axis_review"]["recheck"]["unclear_ids"]
    unavailable=[]
    for pop,name,g in ordered_groups(metrics):
        for arm in ("S0","ORACLE"):
            a=g["seed_mean"][arm]
            unavailable.append(f'{POP_LABELS[pop]} / {name.replace("::"," / ")} / {arm}: {a["available_all_seeds"]}/{g["frames"]}')
    report=["[확인] 판정 = STAGE1_DIAGNOSTIC_ONLY. 현재 산출물은 W/D 가설 진단이며 S1–S4의 효과 판정은 하지 않았습니다.",
        headline(metrics),
        f'[확인] 첨부 기획의 oracle 혼동률 0%·R 약 4.17°·성공률 약 49.8%는 이번 고정 절차에서 재현되지 않았습니다. 실제 주 경로 oracle 혼동률은 {number(100*o["rates"]["confusion_rate"]["rate"])}%, R 평균은 {number(o["metrics"]["R_deg"]["mean"])}°, 성공률은 {number(100*o["rates"]["success_rate"]["rate"])}%입니다. S0의 원래 가설과 T/R은 최대 차이 0으로 재현되므로 OpenCV 버전 차이를 불일치 원인으로 주장할 근거가 없습니다.',
        "[확인] GT-parity oracle은 기존 GT의 긴 축 parity를 공급합니다. R 오차가 최소인 후보를 고르는 oracle이 아니며, S0−oracle 평균 차이를 원인별 오차 기여량으로 해석하지 않습니다.",
        "[확인] REAL GT는 수동 2D 코너와 알려진 치수로 복원한 기하 참조입니다. 독립적인 실제 6D 계측값보다 신뢰도가 낮으며, 원래 319 CONFIRMED와 후속 40장 재검토의 36 CONFIRMED·4 UNCLEAR를 구분합니다.",
        "", "# W/D 가설 선택의 1단계 진단", "", "## 실행 범위와 실제 parity", "",
        f'[확인] Stage0에서 기존 REAL_DEV {parity["rows"]}개 예측 경로를 실제 원래 F로 다시 계산했습니다. 가설 일치 {parity["hypothesis_exact"]}/{parity["rows"]}, 자세 산출 여부 일치 {parity["availability_exact"]}/{parity["rows"]}이며, 최대 T 차이 {number(parity["max_translation_cm_delta"],9)} cm, 최대 R 차이 {number(parity["max_rotation_deg_delta"],9)}°입니다. 허용 오차는 0.01 cm/0.01°입니다. [STAGE0_PARITY.json](STAGE0_PARITY.json)',
        "", f'[확인] penalty 없이 최종 두 가설의 전체 9점 RMSE만 비교한 단순 선택은 원래 선택과 {parity["simple_same_as_official"]}/{parity["simple_comparison_rows"]}회 일치합니다. 이 비율은 실제 공식 점수 계약과 단순 RMSE 선택의 차이를 확인하는 값입니다. 가설 진단의 formal score gap은 **대안−선택의 공식 가중 점수 차이**이고 penalty를 포함하므로 순수 영상 RMSE 차이와 같다고 가정하지 않습니다. [solver.py]({SOURCE_LINK}solver.py)',
        "", f'[확인] 실제 Stage1 F 호출 수는 {execution["actual_original_F_calls_total"]}회이며 모집단별 {execution["actual_original_F_calls"]}입니다. SYNTH 참조 margin 계산의 추가 8점 참조 solver 호출 {execution["reference_only_SYNTH_margin_solver_calls"]}회는 진단용입니다. 새 신경망 추론·학습·합성 이미지 생성은 없으며 Stage2 규칙 실행은 0입니다. [STAGE1_EXECUTION.json](STAGE1_EXECUTION.json)',
        "", f"[확인] 고정 좌표·support·K·치수만으로 S0와 두 후보를 계산하고, 각 모집단의 선택을 봉인한 뒤 해당 참조 자세와 옛 오류 값을 읽었습니다. REAL parity PASS 후 SYNTH와 AUX를 진행했습니다. 원래 core와 새 선택 코드는 실제 계산 전에 SHA를 고정하고 실행 후 확인했습니다. [{source_lock_name}]({source_lock_name}), [STAGE1_SELECTION_SEAL.json](STAGE1_SELECTION_SEAL.json), [INPUT_AUDIT.json](INPUT_AUDIT.json)",
        "", "## 모든 방법의 정확도와 oracle 차이", "",
        "[확인] YOLO 네 경로 BASE, N3_DIM_SYM, SUBPIX, N3_THEN_SUBPIX를 같은 REAL_DEV 319장·SYNTH_HELDOUT 1,985장으로 비교합니다. 존재하는 DOPE/ResNet-18의 기존 BASE·N3 예측도 같은 319장으로 진단합니다. 각 영상의 반복 seed를 별도 독립 영상으로 세지 않습니다.",
        "", "전체 seed별/seed mean 평균·표본 분산·표준편차·중앙값·P90·95% CI·산출 수는 **[단일 Stage1 표](STAGE1_table.md)**에 있으며, [기계 판독 수치](STAGE1_METRICS.json)와 원행 [REAL](STAGE1_ROWS_REAL.jsonl.gz), [SYNTH](STAGE1_ROWS_SYNTH.jsonl.gz), [AUX](STAGE1_ROWS_AUX.jsonl.gz)를 함께 제공합니다.",
        "", "![고도별 혼동과 GT-parity oracle](STAGE1_figure.png)",
        "", "[확인] seed mean R 차이(S0−oracle)는 두 결과의 영상별 paired 차이로 집계합니다. 이 값이 작거나 음수인 방법도 그대로 남기며, oracle은 예측 코너의 오류를 제거하지 않습니다. 가설 parity가 맞아도 코너 오류·PnP 오차·참조 불확실성이 남습니다.",
        "", "[확인] YOLO의 네 경로는 REAL에서 두 선택 모두 319/319, SYNTH에서 모두 1,985/1,985 자세를 산출했습니다. DOPE BASE/N3는 S0 210/319, oracle 267/319로 산출 수가 다르며, ResNet-18은 모두 319/319입니다. DOPE의 R 평균은 서로 다른 available-only 분포이므로, 두 평균의 단순 차이와 paired Δ를 혼동하지 않아야 합니다. 표·그림의 paired Δ는 두 결과 모두 산출된 영상의 차이 평균입니다.",
        "",coverage_table(metrics),
        "", "[확인] 미산출·oracle 선택 전후 가설·성공/혼동 회복 및 손상의 모든 영상 ID는 [STAGE1_FAILURES.json](STAGE1_FAILURES.json)에 보존했습니다.",
        "", f'[확인] oracle의 진단 이점도 균일하지 않습니다. 주 경로의 REAL 고도 ≥30° 85장에서는 혼동률이 S0 {number(100*group["bins"]["elevation_deg"][">=30"]["seed_mean"]["S0"]["rates"]["confusion_rate"]["rate"])}%에서 oracle {number(100*group["bins"]["elevation_deg"][">=30"]["seed_mean"]["ORACLE"]["rates"]["confusion_rate"]["rate"])}%로 높아졌습니다. 이 구간과 잔여 실패를 그대로 남깁니다.',
        "", "## 고도·가설 구분 margin·거리·재질·가림", "",
        "[확인] 고도 구간은 <5 / 5–10 / 10–20 / 20–30 / ≥30°, 참조 W/D margin은 <2 / 2–4 / 4–8 / 8–16 / ≥16 px, 카메라에서의 유클리드 거리는 <2 / 2–3 / 3–4 / 4–6 / ≥6 m로 미리 고정했습니다. margin은 참조 기하의 가설 구분 정도이며 예측 formal gap과 구분합니다. 구간별 장수가 0인 경우 NA로 표시합니다. 재질과 가림 등급은 기존 레이블이며 새 어노테이션이 아닙니다.",
    ]
    if amendment:
        report.insert(15,"[확인] SYNTH 참조 준비에서 NPZ 배열을 ID마다 다시 압축 해제하고 큰 backing array를 보존하는 메모리 문제가 발생해 중단했습니다. 이미 봉인한 REAL/SYNTH 선택과 원래 source는 보존했고, 배열을 한 번 읽는 별도 scoring recovery로 이어갔습니다. 완료한 F 선택의 재실행은 0회입니다. 복구 참조 margin fit은 3,970회이며, 중단 전 참조 fit 횟수는 NOT_MEASURED로 남겨 총 호출 수를 꾸미지 않습니다. [STAGE1_POSTSEAL_AMENDMENT.json](STAGE1_POSTSEAL_AMENDMENT.json), [stage1_resume.py]("+SOURCE_LINK+"stage1_resume.py)")
        report.insert(15,"")
    for field,title in (("elevation_deg","고도"),("reference_margin_px","참조 margin"),("distance_m","카메라 거리"),("material","재질"),("grade","가림 등급")):
        report += ["",f"### {title}","",subgroup_table(metrics,field)]
    report += ["", "## 공식 점수 gap의 경고 진단", "",
        "[확인] 작은 gap이 혼동을 예측하는 방향으로 ROC score를 −gap으로 둡니다. AUC는 seed별 결과의 평균과 반복 예측을 합친 기술용 pooled 값을 구분합니다. threshold는 1/2/3/5 px이며 strict gap<threshold입니다. 경고율은 모든 경로 예측 분모, 혼동 포착률은 실제 혼동 예측 분모, 정밀도는 경고 예측 분모입니다. 이것은 S4 threshold 선택이나 성능 판정이 아닙니다.","",warning_table(metrics),
        "", "## GT와 해석의 제한", "",
        f'[확인] 원래 축 검토는 319 CONFIRMED입니다. 후속 독립 재검토는 40장 중 36 CONFIRMED·4 UNCLEAR이며, UNCLEAR ID는 {", ".join(f"`{x}`" for x in unclear)}입니다. 이번 진단은 고정된 원래 참조를 유지하며 4장을 다른 GT로 바꾸거나 숨기지 않습니다. [INPUT_AUDIT.json](INPUT_AUDIT.json)',
        "", "[확인] 공식 혼동은 proper symmetry 최소 R>45° 및 |yaw|≥60°이고, 5 cm·5° 성공은 strict T<5 cm 및 proper symmetry 최소 R<5°입니다. REAL cluster CI는 13 session, SYNTH 기본 CI는 frame bootstrap이며 재표본 10,000회·seed 20260917입니다. 다중 비교 보정은 하지 않았습니다. 가림·고도 subgroup와 AUX는 기술 진단으로 읽어야 합니다. [statistics.py]("+SOURCE_LINK+"statistics.py), [verdict.py]("+SOURCE_LINK+"verdict.py)",
        "", "[추정] 낮은 고도나 작은 참조 margin에서 S0/oracle 차이가 나타나는 것은 W/D 식별 난이도와 양립합니다. 동일 영상의 관측 기하·코너 오류·참조 생성 절차가 함께 바뀌므로 인과관계를 입증하지 않습니다. 원래 GT의 기하 복원 불확실성은 oracle 진단에도 적용됩니다.",
        "", "[확인] S1·S2·S3는 이후 고정 절차로 평가하며 사전 판정 규칙을 유지합니다. S3의 카메라 외부 자세 고정 기록은 아직 확인되지 않았고, GT 높이 SD≤5 cm·normal RMS≤2°로 고른 session은 사후 선택이므로 최대 FEASIBILITY_ONLY입니다. S4의 Metric3D-v2는 코드·가중치 private 준비와 import/전처리 검산만 했으며 모델 생성·forward는 0입니다. Stage2 보고 뒤 깊이 정확도 gate를 확인하기 전 성능 결과를 주장하지 않습니다. [DEPTH_PREPARATION.json](DEPTH_PREPARATION.json)",
        "", "## 코드와 재현", "", "[방법](METHOD_KO.md), [실제 재현 명령](REPRODUCE.md), [영어 3문장](PAPER_SNIPPET_EN.md), [그림 SHA와 근거](STAGE1_FIGURE_INDEX.json). 공개 산출물은 숫자와 직접 생성한 그래프만 포함하며 원본 RGB·개인 경로·검토자 정보는 포함하지 않습니다."]
    write(doc/"REPORT_KO.md","\n".join(report))
    method = f'''[확인] 이 문서는 1단계 진단 및 2·3단계 사전 계약입니다. 기존 예측 코너를 고정하고 W/D 가설 선택만 진단하며, 단계별 실제 결과는 [REPORT_KO.md](REPORT_KO.md)에 기록합니다.

# 방법과 사전 계약

N3 seed 1·2·3, cornerSubPix, 1% diagonal cap, 예측 코너·support·K·치수, SQPnP/RefineLM, proper symmetry metric은 기존 결과와 같습니다. native 차원 순서는 `[W,D,H]`, PnP 고정 차원은 `[W,H,D]`입니다. 원래 F를 실제 호출하여 S0를 복원하고 두 가설의 실제 최종 8점 fit도 저장합니다. [solver.py]({SOURCE_LINK}solver.py), [{source_lock_name}]({source_lock_name})

S0는 원래 공식 selector입니다. 공식 weighted score의 대안−선택 차이는 penalty를 포함한 nominal px 단위이며 순수 9점 RMSE와 동일하지 않습니다. 단순 선택은 두 최종 fit의 전체 9점 RMSE 최소값만 사용하고 동률이면 고정된 긴 앞면/짧은 앞면 순서를 따릅니다. GT-parity oracle은 고정 참조가 제공하는 긴 축 parity를 선택합니다. 동일 qFinal의 후보 중 metric R이 최소인 후보를 탐색하지 않습니다.

각 모집단의 선택은 해당 참조값을 읽기 전에 qFinal·K·고정 치수·support·source flag만으로 계산하고 봉인합니다. REAL의 봉인·parity PASS가 SYNTH/AUX 계산보다 먼저입니다. 참조는 그 모집단의 봉인 이후 오차·고도·참조 margin·거리와 명시적인 oracle 진단에만 읽습니다. SYNTH 참조 margin은 렌더러 GT 코너의 두 8점 fit 잔차 차이입니다. REAL margin은 기존 GT-builder 값을 유지합니다. [inputs.py]({SOURCE_LINK}inputs.py), [STAGE1_SELECTION_SEAL.json](STAGE1_SELECTION_SEAL.json)

{amendment_method}

혼동: proper symmetry 최소 R>45°와 |yaw|≥60°. 성공: strict T<5 cm와 proper symmetry 최소 R<5°. 오차는 영상별 seed 평균의 분포, 비율은 seed별 이진 판정 뒤 seed 평균입니다. REAL 319장 CI는 13 session cluster; SYNTH 1,985장 CI는 frame bootstrap입니다. 재표본 수 10,000, RNG seed 20260917, 표본 분산 ddof=1, P90은 NumPy linear quantile입니다. 전체 반복 seed를 957/5,955개 독립 영상으로 세지 않습니다. missing pose의 수치 오차는 available-only이고 비율의 전체 영상 분모에는 false로 남깁니다. [statistics.py]({SOURCE_LINK}statistics.py), [verdict.py]({SOURCE_LINK}verdict.py)

S1은 VIS_RULE_V1의 visible∩supported 코너가 6개 이상일 때 각 가설을 그 부분집합으로 fit하고 부분집합+center RMSE 최소값을 택합니다. 6개 미만은 S0 fallback입니다. S2는 full8와 8개의 leave-one-out7, 두 가설의 18 fit 중 각 부분집합+center RMSE 최소값을 택합니다. S3는 target frame을 제외한 session GT의 median normal·plane offset을 쓰는 사후 camera-fixed feasibility 진단입니다. SYNTH S3도 oracle 상한으로만 읽습니다. 새 코너 이동이나 score coefficient tuning은 하지 않습니다. [rules.py]({SOURCE_LINK}rules.py)

주 경로는 N3_THEN_SUBPIX입니다. 이후 SUPPORTED는 REAL 혼동 변화 CI 상한<0, SYNTH 혼동 평균 변화<0·2개 이상 seed 개선, 두 모집단 모두 성공 변화 CI 상한<0이 아님을 함께 만족해야 합니다. 어느 모집단이든 혼동 변화 CI 하한>0 또는 성공 변화 CI 상한<0이면 WORSENED입니다. 나머지는 UNRESOLVED이고 S3는 FEASIBILITY_ONLY입니다. 3규칙×2지표의 다중 비교 보정은 하지 않습니다. 1단계에는 이 판정을 적용하지 않습니다.

Metric3D-v2는 공식 기본 예제의 ViT-Small/RAFT-4만 미리 고정했습니다. [공식 hubconf]({"https://github.com/YvanYin/Metric3D/blob/"+OFFICIAL_COMMIT+"/hubconf.py#L18-L21"})와 [Small 입력 설정]({"https://github.com/YvanYin/Metric3D/blob/"+OFFICIAL_COMMIT+"/mono/configs/HourglassDecoder/vit.raft5.small.py#L18-L31"})의 616×1064 입력·canonical focal 1000을 사용합니다. 원본 RGB는 private에서 비율 유지 resize, RGB 평균값 padding, 공식 mean/std 정규화 후 추론하며, unpad·원래 해상도 보간·scaled fx/1000 곱으로 metric depth를 복원합니다. K는 fx/fy/cx/cy 전처리 계약으로 지원되며 모델에 K 전체 행렬을 직접 넣는 API라고 주장하지 않습니다. [depth.py]({SOURCE_LINK}depth.py)

공식 코드 commit은 `{OFFICIAL_COMMIT}`, HF revision은 `80d2d1410afb4b23cd9d18c6be9144483d4b70b6`, Small 가중치는 150,120,967 bytes/SHA256 `b34b2a2be9148054991cef7e417930e1320602ba7bc503b0ee4e7888543728f6`입니다. 공식 [BSD-2-Clause LICENSE]({"https://github.com/YvanYin/Metric3D/blob/"+OFFICIAL_COMMIT+"/LICENSE"})와 [저자의 비상업 제한 제거 안내](https://github.com/YvanYin/Metric3D/issues/115#issuecomment-2245665482)를 근거로 연구 사용합니다. 공식 HF 모델카드/API에는 checkpoint별 별도 license 표시가 확인되지 않았다는 제한을 남깁니다. 연구 전용 라이선스라는 표현은 쓰지 않습니다.

[확인] Metric3D v2는 [arXiv 2404.15506](https://arxiv.org/abs/2404.15506)에 TPAMI 2024 게재 정보가 있고 DOI는 `10.1109/TPAMI.2024.3444912`입니다. 첨부의 “publication venue 미확인” 전제를 이 공식 정보로 정정합니다. 코드·checkpoint·K 지원·실제 준비 상태의 근거는 [DEPTH_PREPARATION.json](DEPTH_PREPARATION.json)에 함께 보존했습니다. 공식 최소 GPU VRAM은 확인되지 않았습니다. **준비 시점**의 GPU는 RTX 3080 10 GiB이며 source/hash/import·전처리 검산만 했습니다. 이 준비 근거만으로 실제 추론 가능 여부나 깊이 정확도를 검증했다고 표현하지 않습니다.

준비 시점에는 기존 env를 바꾸지 않고 private target에 mmcv1.7.2/addict2.4.0/yapf0.40.1만 보충했습니다. 기존 torch2.1.1/torchvision0.16.1/numpy1.26.4는 공식 requirements_v2의 torch2.0.1/torchvision0.15.2/numpy1.23.1과 다릅니다. xformers가 없으면 공식 소스가 제공하는 Torch Attention fallback을 사용하며 소스를 수정하지 않습니다. source/hash/import와 전처리 검산만 통과했고 모델 생성·forward는 0이었습니다. 실제 추론과 깊이 정확도는 이후 Stage3 실행 결과로 별도 보고합니다.

깊이는 예측 앞면 quad 0–3을 centroid 기준 0.85배로 축소한 영역의 positive finite depth 중앙값입니다. 모델·원본 RGB·adapter의 SHA는 모델 생성과 추론 전에 봉인합니다. SYNTH 1,985장과 final-test 4 session을 제외한 REAL 231장에서 **네 경로·세 seed별로** 앞면 평균 Z 상대오차의 coverage·절대오차 median/P90·signed bias를 모두 보고합니다.

S4 정확도 gate의 주 경로는 N3_THEN_SUBPIX입니다. REAL 231의 각 영상에서 세 seed의 **절대 상대오차를 먼저 구한 뒤 산술 평균**하고, 이 231개 영상 값의 median을 gate에 사용합니다. signed error를 먼저 평균하여 상쇄하지 않습니다. 주 경로의 어느 영상·seed라도 깊이 추정이 누락되면 ID를 공개하고 gate FAIL로 S4를 실행하지 않습니다. 유효한 subset만으로 통과시키지 않습니다. 완전 coverage에서 이 median>5%면 S4를 실행하지 않습니다.

통과하면 margin m={{1,2,3}} px 중 **SYNTH 주 경로 혼동률의 seed 평균** 최소값을 택하고 동률은 작은 m으로 고정하여 REAL에 한 번 적용합니다. final-test 4 session은 모델·threshold 선택에서 제외하며 과거 개발 용도를 새 test로 재봉인하지 않습니다. 이 기준은 실제 깊이 추론 전에 고정했습니다.
'''
    write(doc/"METHOD_KO.md",method)
    reproduce = f'''[확인] 아래 명령은 기존 학습·코너 보정 결과를 입력으로 1단계 계산만 재현합니다. 새 학습·데이터 생성은 없습니다.

# 재현

게시 저장소 checkout에서 기존 pose 환경을 사용합니다. 원래 private 입력은 `PALLET_SOURCE_ROOT`로 지정하며 공개 보고서에는 개인 경로를 쓰지 않습니다. `PALLET_WD_OUTPUT`은 존재하지 않는 새 디렉터리로 지정합니다. 완료한 산출물 위에 preflight/stage1을 다시 실행하면 보존 검사로 중단합니다.

```bash
export PALLET_PYTHON=/path/to/existing/pallet-pose/bin/python
export PALLET_SOURCE_ROOT=/path/to/original/private/source
export PALLET_WD_OUTPUT=/path/to/new/nonexistent/output
export PYTHONDONTWRITEBYTECODE=1
"$PALLET_PYTHON" -B -m {PREFIX}.fresh_stage1
"$PALLET_PYTHON" -B -m {PREFIX}.summarize
"$PALLET_PYTHON" -B -m {PREFIX}.report --doc "$PALLET_WD_OUTPUT"
```

공개 수치만으로 표·그림을 다시 만들 때는 다음을 사용합니다. private RGB·checkpoint가 필요하지 않습니다. `--documents-only`는 그림/index를 보존하며 Markdown만 갱신합니다.

```bash
"$PALLET_PYTHON" -B -m {PREFIX}.report --doc _docs/experiments/pallet_wd_hypothesis_diag_20261010
"$PALLET_PYTHON" -B -m {PREFIX}.report --doc _docs/experiments/pallet_wd_hypothesis_diag_20261010 --documents-only
```

Depth 준비는 다음처럼 새 private 디렉터리에 pinned 공식 source와 Small checkpoint만 받습니다. 기존 Python 환경은 바꾸지 않고 private target에 부족한 세 패키지만 설치합니다. 설치 도중 build isolation에서 pkg_resources가 없는 문제를 피하기 위해 기존 setuptools를 사용하는 --no-build-isolation을 고정합니다. fetch와 prepare는 모델 생성·forward가 0입니다.

```bash
export PALLET_DEPTH_PRIVATE=/path/to/new/private/metric3d
"$PALLET_PYTHON" -B -m {PREFIX}.depth fetch --private-root "$PALLET_DEPTH_PRIVATE"
"$PALLET_PYTHON" -B -m pip install --no-deps --no-compile --no-cache-dir --no-build-isolation \\
  --target "$PALLET_DEPTH_PRIVATE/deps" mmcv==1.7.2 addict==2.4.0 yapf==0.40.1
"$PALLET_PYTHON" -B -m {PREFIX}.depth prepare --private-root "$PALLET_DEPTH_PRIVATE"
```

실제 depth cache는 Stage2 보고 완료 뒤의 Stage3 entry point입니다. private JSONL에는 id/population/image_path/K/raw_hw/crop_lrtb만 두고 GT·오류·final-test label을 입력하지 않습니다. K는 crop 후 native 좌표에 맞아야 합니다. 원본 RGB와 전체 깊이 map은 공개하지 않습니다. 아직 Stage3 정확도 gate를 평가하지 않은 1단계에서 cache 명령을 실행하거나 S4 결과를 주장하지 않습니다.
'''
    write(doc/"REPRODUCE.md",reproduce)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc",type=Path,default=C.DOC)
    parser.add_argument("--documents-only",action="store_true")
    args=parser.parse_args();doc=args.doc.resolve()
    metrics=load(doc/"STAGE1_METRICS.json")
    if not args.documents_only:figure(metrics,doc)
    documents(metrics,doc)
    print(json.dumps({"status":"STAGE1_DOCUMENTS_GENERATED","figure_generated":not args.documents_only,"phase":"STAGE1_DIAGNOSTIC_ONLY"}))


if __name__=="__main__":main()

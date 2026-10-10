"""Accumulate Stage-2 fixed-rule outcomes without changing Stage-1 evidence."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import common as C
from . import report as R
from . import verdict as V


def read_results(doc):
    results={}
    for path in sorted(doc.glob("RESULTS_S[123]_*.json")):
        value=R.load(path)
        rule=value["rule"];pop=value["population"]
        assert rule in ("S1","S2","S3") and (rule,pop) not in results
        results[rule,pop]=(path,value)
    return results


def freeze_stage1(doc,commit):
    """Keep the exact first-published texts/figure and their original SHA."""
    receipt=doc/"STAGE1_PUBLISHED_SNAPSHOT.json"
    if receipt.exists():
        return R.load(receipt)
    assert len(commit)==40 and all(c in "0123456789abcdef" for c in commit)
    sources=("REPORT_KO.md","METHOD_KO.md","REPRODUCE.md","STAGE1_table.md","PAPER_SNIPPET_EN.md","STAGE1_figure.png","STAGE1_FIGURE_INDEX.json")
    bindings=[]
    for name in sources:
        src=doc/name
        copyname="STAGE1_"+name if not name.startswith("STAGE1_") else "FIRST_PUBLISHED_"+name
        target=doc/copyname
        assert not target.exists()
        shutil.copyfile(src,target)
        bindings.append(dict(original=name,preserved=copyname,sha256=C.sha(src),bytes=src.stat().st_size))
    value=dict(status="FIRST_STAGE1_PUBLICATION_PRESERVED",publication_commit=commit,files=bindings,
        original_phase="STAGE1_DIAGNOSTIC_ONLY",description="Exact first-publication text/image snapshots; later reports are cumulative")
    R.write(receipt,json.dumps(value,ensure_ascii=False,indent=2))
    return value


def table(results):
    lines=["단위: T cm, R °, ADDsym m. ±는 표본 SD(ddof=1), 분산·P90·CI는 저장된 수치를 그대로 표시합니다. CI의 target은 평균/비율입니다.","",
        "| 규칙 / 모집단 | 경로 | 집계 | 선택 | 자세 산출/전체 | 혼동률 | 5 cm·5° 성공률 | T 평균±SD·분산·중앙값/P90·CI | R 평균±SD·분산·중앙값/P90·CI | ADDsym 평균±SD·분산·중앙값/P90·CI | IoU3D 평균±SD·분산·중앙값/P90·CI |",
        "|---|---|---|---|---:|---|---|---|---|---|---|"]
    for (rule,pop),(path,result) in sorted(results.items()):
        for method in C.METHODS:
            if method not in result["metrics"]:continue
            m=result["metrics"][method]
            scopes=[("seed mean",m["seed_mean"])]+[(f"seed {s}",m["per_seed"][str(s)]) for s in C.SEEDS]
            for scope,arms in scopes:
                for arm in ("S0",rule):
                    summary=arms[arm]
                    cells=[R.distribution(summary["metrics"][k]) for k in ("T_cm","R_deg","ADDsym_m","IoU3D")]
                    lines.append(f'| {rule} / {pop} | {method} | {scope} | {arm} | {summary["available_all_seeds"]}/{summary["frames"]} | '
                        f'{R.rate(summary,"confusion_rate")} | {R.rate(summary,"success_rate")} | '+" | ".join(cells)+" |")
    return "\n".join(lines)


def paired_table(results):
    lines=["| 규칙 / 모집단 | 경로 | 지표 | seed mean 변화 (규칙−S0) | 95% CI | seed1 / 2 / 3 변화 | paired 영상 수 | SYNTH scenario cluster 보조 CI |",
           "|---|---|---|---:|---|---|---:|---|"]
    for (rule,pop),(path,result) in sorted(results.items()):
        for method in C.METHODS:
            if method not in result["paired"]:continue
            for key,value in result["paired"][method]["seed_mean"].items():
                scale=100 if key.endswith("_rate") else 1
                unit=" pp" if scale==100 else ""
                lines.append(f'| {rule} / {pop} | {method} | {key} | {R.number(value["delta"]*scale)}{unit} | {R.interval(value["CI95"],scale)} | '
                    f'{" / ".join(R.number(x*scale) if x is not None else "NA" for x in value["per_seed_delta"])} | {value["paired_frames"]} | '
                    f'{R.interval(value.get("scenario_cluster_secondary_CI95"),scale)} |')
    return "\n".join(lines)


def failures_table(results):
    fields=("success_to_failure","failure_to_success","confusion_recovery","confusion_damage","hypothesis_change","fallback","unavailable")
    lines=["| 규칙 / 모집단 | 경로 | seed | 성공→실패 | 실패→성공 | 혼동 회복 | 혼동 손상 | 가설 변경 | fallback | 미산출 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (rule,pop),(path,result) in sorted(results.items()):
        for method in C.METHODS:
            if method not in result["failures"]:continue
            for s in C.SEEDS:
                f=result["failures"][method][str(s)]
                lines.append(f'| {rule} / {pop} | {method} | {s} | '+" | ".join(str(f[k+"_count"]) for k in fields)+" |")
    lines += ["", "모든 ID와 가설 변경 전후는 각 RESULTS JSON의 failures에 보존됩니다. 성공→실패와 혼동 손상 사례도 제외하지 않습니다."]
    return "\n".join(lines)


def performance_figure(results,doc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus":False,"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(12,8.5))
    for column,pop in enumerate(("SYNTH_HELDOUT","REAL_DEV")):
        for row,key in enumerate(("confusion_rate","success_rate")):
            ax=axes[row,column];ax.axvline(0,color="#777",linewidth=1)
            for j,rule in enumerate(("S1","S2","S3")):
                found=results.get((rule,pop))
                if found is None:
                    ax.text(.5,j,"미실행 / gate 상태 확인",transform=ax.get_yaxis_transform(),ha="center",color="#777",fontsize=9)
                    continue
                path,result=found
                if not result.get("primary") or result["primary"][key].get("delta") is None:
                    ax.text(.5,j,"산출 불가 / 근거 확인",transform=ax.get_yaxis_transform(),ha="center",color="#777",fontsize=9)
                    continue
                v=result["primary"][key]
                mean=100*v["delta"];ci=v["CI95"]
                color="#8b63a8" if rule=="S3" else "#2369a0"
                error=None if ci is None else [[max(0,mean-100*ci[0])],[max(0,100*ci[1]-mean)]]
                ax.errorbar(mean,j,xerr=error,fmt="o",color=color,capsize=4)
                ax.annotate(f'{R.number(mean,3)} pp',xy=(mean,j),xytext=(0,13),textcoords="offset points",ha="center",fontsize=9)
            ax.set_yticks(range(3),["S1 VIS","S2 full+LOO","S3 사후 feasibility"])
            ax.set_ylim(2.5,-.6);ax.grid(axis="x",alpha=.2)
            ax.set_xlabel("규칙−S0 변화 (percentage points)")
            ax.set_title(pop+": "+("혼동률 (음수 개선)" if key=="confusion_rate" else "성공률 (양수 개선)"))
    fig.suptitle("Stage2 — 고정 주 경로 N3_THEN_SUBPIX의 paired 변화",fontsize=15,y=.985)
    fig.text(.04,.02,"점=seed 평균, 선=95% CI. S3는 GT 사후 session 선택·target 제외 plane이며 FEASIBILITY_ONLY.\n"
        "REAL full319: session cluster; SYNTH: frame bootstrap. SYNTH S3는 oracle 상한, REAL S3는 사후 subset의 기술 CI다.",fontsize=9)
    fig.subplots_adjust(left=.19,right=.97,top=.91,bottom=.14,hspace=.43,wspace=.5)
    path=doc/"STAGE2_figure.png";fig.savefig(path,dpi=145,pil_kwargs={"compress_level":9});plt.close(fig)
    R.write(doc/"STAGE2_FIGURE_INDEX.json",json.dumps(dict(figures=[dict(path=path.name,sha256=C.sha(path),bytes=path.stat().st_size)],
        evidence_source_sha256={p.name:C.sha(p) for p,result in results.values()},raw_RGB_published=False,
        rights_basis="Own Matplotlib chart of saved fixed-rule statistics",visual_inspection=dict(status="PENDING")),ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc",type=Path,default=C.DOC)
    parser.add_argument("--stage1-commit",required=True)
    parser.add_argument("--documents-only",action="store_true")
    args=parser.parse_args()
    doc=args.doc.resolve()
    verdict=R.load(doc/"VERDICT_STAGE2.json")
    assert set(verdict["rules"])=={"S1","S2","S3"} and verdict["no_combined_method_verdict"]
    results=read_results(doc)
    for rule in ("S1","S2"):
        assert (rule,"SYNTH_HELDOUT") in results
        if (rule,"REAL_DEV") not in results:
            assert (doc/f"SKIPPED_{rule}_REAL.json").exists(), "Missing REAL results require an explicit closed gate receipt"
    snapshot=freeze_stage1(doc,args.stage1_commit)
    if not args.documents_only:performance_figure(results,doc)
    R.write(doc/"STAGE2_table.md",table(results))
    labels={rule:verdict["rules"][rule]["verdict"] for rule in ("S1","S2","S3")}
    lines=["[확인] Stage2 규칙별 판정: "+", ".join(f"{r}={labels[r]}" for r in ("S1","S2","S3"))+". 합친 방법 전체의 단일 우열 판정은 하지 않습니다.",
        "[확인] 코너·support·K·치수·모델·SubPix·cap·proper metric은 고정하고 가설 선택과 PnP fit 부분집합만 바꿨습니다. 나쁜 결과와 중단 gate도 아래에 남깁니다.",
        "[확인] REAL GT는 수동 2D 기반 기하 참조이며 독립적인 실제 6D 계측보다 신뢰도가 낮습니다. 기존 319 CONFIRMED와 재검토40의4 UNCLEAR를 구분합니다.",
        "", "# W/D 선택 규칙의 2단계 결과", "", "## 규칙별 판정과 실행 gate", "",
        "| 규칙 | 실제 판정 | 해석 |", "|---|---|---|"]
    for rule in ("S1","S2","S3"):
        interpretation="사후 session GT·target 제외 plane의 FEASIBILITY_ONLY; 일반화나 배포 성능 판정 아님" if rule=="S3" else "사전 고정 주 경로·혼동/성공 CI 및 SYNTH gate"
        lines.append(f"| {rule} | {labels[rule]} | {interpretation} |")
    lines += ["", "[확인] 주 경로는 N3_THEN_SUBPIX이고 변화는 규칙−S0입니다. 혼동은 R>45°·|yaw|≥60°, 성공은 T<5 cm·proper R<5°입니다. SYNTH의 주 경로가 WORSENED이면 REAL 평가를 실행하지 않고 모든319 ID를 skip receipt에 남깁니다. SUPPORTED 조건을 충족하지 않은 결과를 성능 향상으로 단정하지 않습니다. 세 규칙×두 지표의 다중 비교 보정은 하지 않았습니다. [VERDICT_STAGE2.json](VERDICT_STAGE2.json)"]
    for rule in ("S1","S2"):
        skipped=doc/f"SKIPPED_{rule}_REAL.json"
        if skipped.exists():lines += ["",f"[확인] {rule} REAL은 SYNTH gate로 미실행입니다. [{skipped.name}]({skipped.name})에 전체 예정 ID와 실제 F=0을 보존했습니다."]
    lines += ["", "## 주 경로의 실제 paired 변화", "", "![Stage2 혼동·성공 paired CI](STAGE2_figure.png)", ""]
    for (rule,pop),(path,result) in sorted(results.items()):
        if not result.get("primary"):continue
        primary=result["primary"];conf,success=primary["confusion_rate"],primary["success_rate"]
        methods=result["metrics"].get(V.PRIMARY_METHOD,{})
        frames=methods.get("frames","NA")
        lines += [f'[확인] {rule}/{pop}, 주 경로 n={frames}: 혼동 변화 {R.number(100*conf["delta"])} pp, CI {R.interval(conf["CI95"],100)}; '
            f'성공 변화 {R.number(100*success["delta"])} pp, CI {R.interval(success["CI95"],100)}. '
            f'혼동 seed별 {" / ".join(R.number(100*x) for x in conf["per_seed_delta"])} pp, '
            f'성공 seed별 {" / ".join(R.number(100*x) for x in success["per_seed_delta"])} pp. [{path.name}]({path.name})',""]
    lines += ["## 모든 경로·seed의 정확도", "", "[전체 정확도 표](STAGE2_table.md)는 네 경로의 S0/규칙별 seed1·2·3 및 seed mean을 모두 포함합니다. 평균·표본 분산·SD·중앙값·P90·95% CI·자세 산출 수를 제시하며, 큰 오차를 잘라내지 않습니다. 미산출은 수치 평균에서 제외하고 전체 비율 분모에 false로 남깁니다.",
        "", "## 모든 paired 차이", "",paired_table(results),"", "## 실패·손상·가설 변경·fallback", "",failures_table(results),
        "", "## S3의 범위와 제한", "",
        "[확인] session별 camera-world 외부 자세가 고정됐다는 기존 기록은 확인되지 않았습니다. 높이 SD≤5 cm·normal RMS≤2°로 고른 session은 GT를 이용한 사후 선택입니다. 해당 target frame의 GT는 plane 계산에서 제외하지만 다른 frame GT를 쓰므로 일반적인 GT-free 선택기라고 부르지 않습니다. SYNTH S3는 GT plane oracle 상한입니다. 세 개 이하 session의 REAL S3는 frame bootstrap 기술 CI만 사용하고 언제나 FEASIBILITY_ONLY입니다. [S3_ELIGIBILITY_REAL.json](S3_ELIGIBILITY_REAL.json)",
        "", "## 코드 봉인·재현·이후 깊이 gate", "",
        "[확인] 각 규칙의 새 선택/부분집합을 참조값 scoring 전에 봉인했습니다. 결과·실패 ID는 위 RESULTS 파일, 규칙별 최종 판정은 VERDICT_STAGE2에 보존됩니다. [고정 사전 방법](METHOD_KO.md), [재현](REPRODUCE.md). 별도 Square 연구 결과와 이 W/D 연구의 판정을 합치지 않습니다.",
        "", "[확인] Stage3는 공식 Metric3D-v2 Small로 고정합니다. REAL231의 주 경로 per-frame 세 seed 절대 상대오차 평균→231 median≤5% 및 누락0을 먼저 요구합니다. 누락이나 gate 악화이면 S4 미실행을 기록합니다. 깊이 정확도와 S4 결과를 아직 평가하지 않은 Stage2 보고서에서 성공을 주장하지 않습니다. [DEPTH_PREPARATION.json](DEPTH_PREPARATION.json)",
        "", "## 최초 게시한 Stage1 진단", "",
        f'[확인] 최초 Stage1 게시 commit `{snapshot["publication_commit"]}`의 문서·그림은 [원문 snapshot/hash](STAGE1_PUBLISHED_SNAPSHOT.json)로 보존합니다. [원래 보고서](STAGE1_REPORT_KO.md), [원래 방법](STAGE1_METHOD_KO.md), [원래 재현](STAGE1_REPRODUCE.md), [원래 표](FIRST_PUBLISHED_STAGE1_table.md), [원래 그림](FIRST_PUBLISHED_STAGE1_figure.png).',
        "", "첨부의 oracle 혼동0%/R4.17°/성공49.8% 기대는 실제 GT-parity 정의에서 재현되지 않았습니다. 실제 Stage1 주 경로 REAL oracle은 혼동6.583%/R9.199°/성공47.753%이며, S0 parity는 T/R 최대 차이0입니다. 아래는 최초 원문이며 이후 단계의 판정과 구분합니다.",
        "",(doc/"STAGE1_REPORT_KO.md").read_text()]
    R.write(doc/"REPORT_KO.md","\n".join(lines))
    reproduce=(doc/"STAGE1_REPRODUCE.md").read_text()
    reproduce += f'''\n\n# Stage2 재현\n\n원래1단계의 fresh 출력에서 source/방법 lock을 유지하고 아래 driver를 사용합니다. 네 경로의 SYNTH를 먼저 계산하고 사전 gate로 REAL 실행을 결정합니다. 첫 Stage1 실제 게시 commit을 보고서 snapshot에 연결합니다.\n\n```bash\n"$PALLET_PYTHON" -B -m {R.PREFIX}.run_stage2\n"$PALLET_PYTHON" -B -m {R.PREFIX}.stage2_report --doc "$PALLET_WD_OUTPUT" --stage1-commit {args.stage1_commit}\n```\n\n공개 산출물만으로 Stage2 문서를 재생성할 때는 두 번째 명령을 사용합니다. --documents-only는 PNG/index를 보존합니다. METHOD_KO는 Stage2 전에 고정된 사전 계약이므로 이 생성기는 수정하지 않습니다. 실제 Stage3는 Stage2 보고 완료 receipt 이후 root driver가 실행하며 학습은 없습니다.\n'''
    R.write(doc/"REPRODUCE.md",reproduce)
    hashes={name:C.sha(doc/name) for name in ("REPORT_KO.md","METHOD_KO.md","REPRODUCE.md","STAGE2_table.md","STAGE2_figure.png","STAGE2_FIGURE_INDEX.json")}
    receipt=dict(status="COMPLETE",phase="STAGE2_REPORT",stage1_publication_commit=args.stage1_commit,
        report_generator_sha256=C.sha(__file__),artifacts_sha256=hashes,
        results_sha256={p.name:C.sha(p) for p,result in results.values()},verdict_sha256=C.sha(doc/"VERDICT_STAGE2.json"),
        METHOD_preregister_unchanged=True,visual_inspection_status="PENDING_ROOT",models_constructed_by_report=0,model_forwards_by_report=0)
    R.write(doc/"STAGE2_REPORT_RECEIPT.json",json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps(dict(status="COMPLETE",phase="STAGE2_REPORT",rule_verdicts=labels,figures_generated=not args.documents_only)))


if __name__=="__main__":main()

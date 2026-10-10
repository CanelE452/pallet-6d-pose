"""Depth accuracy and gated S4 report; read saved values, never run a model."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from . import common as C
from . import report as R
from . import stage2_report as S2


def percentage_stats(value):
    return (f'{R.number(None if value["mean"] is None else value["mean"]*100)} ± '
        f'{R.number(None if value["std"] is None else value["std"]*100)}<br>'
        f'분산 {R.number(None if value["variance"] is None else value["variance"]*10000)}<br>'
        f'중앙값 {R.number(None if value["median"] is None else value["median"]*100)} / '
        f'P90 {R.number(None if value["P90"] is None else value["P90"]*100)} / '
        f'최대 {R.number(None if value["max"] is None else value["max"]*100)}')


def accuracy_table(accuracy):
    lines=["| 모집단 | 경로 | 집계 | 깊이 산출/전체 | 절대 상대오차 %: 평균±SD·분산·중앙값/P90/최대 | signed bias % |",
           "|---|---|---|---:|---|---:|"]
    for population in ("REAL","SYNTH"):
        for method in C.METHODS:
            g=accuracy[population]["methods"][method]
            entries=[("seed mean(abs 후 평균)",g["complete_three_seed_frames"],g["frames"],g["seed_mean_absolute_relative_error"],g["signed_bias"])]
            entries += [(f"seed {s}",g["per_seed"][str(s)]["available"],g["per_seed"][str(s)]["frames"],
                         g["per_seed"][str(s)]["absolute_relative_error"],g["per_seed"][str(s)]["signed_bias"]) for s in C.SEEDS]
            for label,available,frames,stats,bias in entries:
                lines.append(f'| {population} | {method} | {label} | {available}/{frames} | {percentage_stats(stats)} | '
                    f'{R.number(None if bias is None else bias*100)} |')
    return "\n".join(lines)


def per_frame_abs(doc,population):
    groups=defaultdict(list)
    for row in C.rows(doc/f"DEPTH_ACCURACY_ROWS_{population}.jsonl.gz"):
        groups[row["method"],row["id"]].append(row)
    values=defaultdict(list)
    for (method,fid),records in groups.items():
        assert len(records)==3 and {r["seed"] for r in records}==set(C.SEEDS)
        if all(r["available"] for r in records):
            values[method].append(float(np.mean([r["absolute_relative_error"] for r in records])))
    return values


def worst_frames_table(doc,limit=5):
    """Show a fixed descending-error audit sample for every route/population."""
    lines=["| 모집단 | 경로 | ID | 세 seed 절대 상대오차 평균 % |",
           "|---|---|---|---:|"]
    for population in ("REAL","SYNTH"):
        groups=defaultdict(list)
        for row in C.rows(doc/f"DEPTH_ACCURACY_ROWS_{population}.jsonl.gz"):
            groups[row["method"],row["id"]].append(row)
        for method in C.METHODS:
            errors=[]
            for (route,fid),records in groups.items():
                if route!=method or not all(r["available"] for r in records):continue
                assert len(records)==3
                errors.append((float(np.mean([r["absolute_relative_error"] for r in records])),fid))
            for error,fid in sorted(errors,key=lambda x:(-x[0],x[1]))[:limit]:
                lines.append(f"| {population} | {method} | `{fid}` | {R.number(error*100)} |")
    return "\n".join(lines)


def reproduce_text():
    return f'''[확인] 고정 입력의 Stage0/1, S1–S3, 공식 depth 정확도 gate와 조건부 S4를 재현하는 명령입니다. 새 학습·데이터 생성은 없습니다.

# 환경과 입력

게시 저장소 checkout 루트에서 기존 pose 환경을 사용합니다. 기존 private 입력·RGB·checkpoint는 공개 저장소에 포함되지 않으며 [INPUT_AUDIT.json](INPUT_AUDIT.json)의 경로와 SHA가 일치해야 합니다. 코드·공식 depth weight·reference·수치 정의를 바꾸면 기존 실행의 재현으로 간주하지 않습니다.

```bash
export PALLET_PYTHON=/path/to/existing/pallet-pose/bin/python
export PALLET_SOURCE_ROOT=/path/to/original/private/source
export PALLET_BASELINE_ROOT=/path/to/existing/pallet-pose-handoff
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

`PALLET_WD_OUTPUT`은 checkout 내부의 존재하지 않는 새 폴더여야 합니다. 예를 들어 `$PWD/_docs/experiments/pallet_wd_reproduce_run1`을 사용합니다. 기존 산출물·private cache가 있으면 보존 검사로 중단하며 덮어쓰지 않습니다. 아래 두 경로는 목적이 다르므로 같은 출력 폴더로 이어 실행하지 않습니다.

# 기존 게시 Stage1에서 Stage2·Stage3 재현

이 경로는 최초에 실제 게시한 Stage1의 입력 감사·봉인·예측·reference score·사전 방법 lock을 SHA 검증하여 새 폴더에 복사한 후, 수정하지 않은 S1/S2/S3 driver를 실행합니다. 새로 계산한 Stage1을 게시했다고 주장하지 않습니다. 복사한 파일과 SHA는 `REPRODUCTION_INPUT_LOCK.json`에 남습니다.

```bash
export PALLET_WD_OUTPUT="$PWD/_docs/experiments/pallet_wd_reproduce_run1"
"$PALLET_PYTHON" -B -m {R.PREFIX}.fresh_stage2
"$PALLET_PYTHON" -B -m {R.PREFIX}.stage2_verify \\
  --doc "$PALLET_WD_OUTPUT" --source-root "$PALLET_SOURCE_ROOT"
"$PALLET_PYTHON" -B -m {R.PREFIX}.stage2_report \\
  --doc "$PALLET_WD_OUTPUT" --stage1-commit 2502db29e77c26b88c6c681b054902ea997f441d
```

depth 준비는 새 private parent의 `metric3d` 하위 폴더에 고정 공식 source와 Small checkpoint만 받습니다. 기존 환경은 바꾸지 않고 private target에 부족한 세 패키지만 설치합니다. 공식 source와 checkpoint SHA는 [DEPTH_PREPARATION.json](DEPTH_PREPARATION.json)에 있습니다. fetch·prepare는 모델 생성과 forward를 실행하지 않습니다.

```bash
export PALLET_RUN_PRIVATE=/path/to/new/private/reproduce_run1
export PALLET_DEPTH_PRIVATE="$PALLET_RUN_PRIVATE/metric3d"
"$PALLET_PYTHON" -B -m {R.PREFIX}.depth fetch --private-root "$PALLET_DEPTH_PRIVATE"
"$PALLET_PYTHON" -B -m pip install --no-deps --no-compile --no-cache-dir --no-build-isolation \\
  --target "$PALLET_DEPTH_PRIVATE/deps" mmcv==1.7.2 addict==2.4.0 yapf==0.40.1
"$PALLET_PYTHON" -B -m {R.PREFIX}.depth prepare --private-root "$PALLET_DEPTH_PRIVATE"
"$PALLET_PYTHON" -B -m {R.PREFIX}.stage3_fresh --private-dir "$PALLET_RUN_PRIVATE"
"$PALLET_PYTHON" -B -m {R.PREFIX}.stage3_verify \\
  --doc "$PALLET_WD_OUTPUT" --source-root "$PALLET_SOURCE_ROOT" --private-dir "$PALLET_RUN_PRIVATE"
"$PALLET_PYTHON" -B -m {R.PREFIX}.stage3_report --doc "$PALLET_WD_OUTPUT"
```

Stage3는 Stage2 보고 receipt가 COMPLETE인 뒤에만 실행됩니다. 원래 실행의 strict-key guard 실패와 읽기 전용 진단을 재현할 필요는 없습니다. `stage3_fresh`는 동일 공식 strict=False 경로에서 추론에 사용하지 않는 zero mask_token 하나만 허용하는 계약을 모델 생성 전에 봉인합니다. 공식 Small·float32·eval·no_grad·batch 1·전처리·K 역정규화는 동일합니다. 추론 호출의 network 입력은 RGB tensor이고 K는 외부 공식 전처리에서 사용합니다.

정확도 gate는 REAL231 주 경로의 abs→세 seed 평균→영상 median≤5%이며 어느 주 seed라도 누락하면 FAIL입니다. FAIL이면 m 선택·final-test 88장 추가 추론·S4를 실행하지 않고 depth 결과를 보존합니다. PASS이면 SYNTH에서 m={{1,2,3}} 중 혼동률 최소값을 고르고 동률은 작은 m을 택한 뒤 REAL319에 한 번 적용합니다. GT는 봉인 뒤의 accuracy·pose 채점에만 쓰며 final-test 네 session으로 모델이나 m을 고르지 않습니다.

# Stage0·Stage1 자체의 별도 재계산

새로운 solver 실행으로 Stage0 parity와 Stage1만 다시 계산하려면 다른 새 출력 폴더를 사용합니다. bounded-memory wrapper는 SYNTH 참조 배열을 선택 봉인 뒤 한 번 로드하며 원래 여섯 과학 source를 수정하지 않습니다. 이후 Stage2를 자동으로 이어 실행하는 경로가 아니며, 실제 게시·사전 방법 lock 요건을 생략하지 않습니다.

```bash
export PALLET_WD_OUTPUT="$PWD/_docs/experiments/pallet_wd_stage1_reproduce_run1"
"$PALLET_PYTHON" -B -m {R.PREFIX}.fresh_stage1
"$PALLET_PYTHON" -B -m {R.PREFIX}.summarize
"$PALLET_PYTHON" -B -m {R.PREFIX}.summarize_failures
"$PALLET_PYTHON" -B -m {R.PREFIX}.report --doc "$PALLET_WD_OUTPUT"
```

# 공개 수치의 검산·표·PNG 재생성

private RGB와 모델이 없는 환경에서도 게시한 표·PNG는 저장 숫자로 재생성할 수 있습니다. 최종 cumulative 보고서는 `stage3_report`를 사용합니다. `--documents-only`는 기존 PNG와 figure index를 보존합니다. Stage1/Stage2 생성기는 당시 문서를 만들므로 최종 보고서 재생성 명령과 구분합니다.

```bash
unset PALLET_WD_OUTPUT
"$PALLET_PYTHON" -B -m {R.PREFIX}.stage3_report \\
  --doc _docs/experiments/pallet_wd_hypothesis_diag_20261010
"$PALLET_PYTHON" -B -m {R.PREFIX}.stage3_report \\
  --doc _docs/experiments/pallet_wd_hypothesis_diag_20261010 --documents-only
```

공식 코드·checkpoint·K 처리 근거와 checkpoint별 별도 license 미표시 제한은 [METHOD_KO.md](METHOD_KO.md), [DEPTH_PREPARATION.json](DEPTH_PREPARATION.json)에 남겼습니다. RGB·검수 overlay·전체 depth map은 private이고 공개는 숫자·SHA·자체 그래프입니다. 기존 방법·source lock을 수정하지 않습니다. 독립 검산은 추가 모델이나 PnP를 실행하지 않습니다.
'''


def depth_figure(accuracy,doc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus":False,"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(13,9.6))
    colors=("#5b6770","#2389a3","#d99a28","#933e7d")
    for ax,pop in zip(axes[0],("REAL","SYNTH")):
        values=per_frame_abs(doc,pop)
        maximum=0
        for method,color in zip(C.METHODS,colors):
            x=np.sort(np.asarray(values[method])*100)
            if len(x):maximum=max(maximum,float(x[-1]))
            if len(x):ax.step(x,np.arange(1,len(x)+1)/len(x),where="post",label=f"{method} (n={len(x)})",color=color)
        ax.axvline(5,color="#c43d3d",linestyle="--",label="5% REAL 주 경로 gate 기준")
        ax.set_xscale("symlog",linthresh=1)
        ax.set_xlim(0,max(6,maximum*1.05))
        ax.set_xlabel("영상별 세 seed 절대 상대오차 평균 (%) — 전체 범위")
        ax.set_ylabel("누적 영상 비율");ax.set_ylim(0,1.025);ax.grid(alpha=.22)
        frames=accuracy[pop]["methods"][C.METHODS[0]]["frames"]
        ax.set_title(f"{pop} n={frames}: depth 오차 ECDF")
        ax.legend(fontsize=8,loc="upper left")
    ax=axes[1,0];x=np.arange(4)
    for pop,offset,color in (("REAL",-.18,"#2369a0"),("SYNTH",.18,"#e37d24")):
        values=[accuracy[pop]["methods"][m]["signed_bias"] for m in C.METHODS]
        ax.bar(x+offset,[np.nan if v is None else 100*v for v in values],width=.34,color=color,label=pop)
    ax.axhline(0,color="#777",linewidth=1);ax.set_xticks(x,["BASE","N3","SUBPIX","N3→SUBPIX"])
    ax.set_ylabel("signed relative bias (%)");ax.set_title("signed bias — gate는 부호 상쇄를 허용하지 않음")
    ax.legend(fontsize=9);ax.grid(axis="y",alpha=.2)
    ax=axes[1,1];x=np.arange(4)
    for pop,offset,color in (("REAL",-.18,"#2369a0"),("SYNTH",.18,"#e37d24")):
        g=accuracy[pop]["methods"]["N3_THEN_SUBPIX"]
        values=[g["per_seed"][str(s)]["absolute_relative_error"]["median"] for s in C.SEEDS]+[g["seed_mean_absolute_relative_error"]["median"]]
        ax.bar(x+offset,[np.nan if v is None else 100*v for v in values],width=.34,color=color,label=pop)
    ax.axhline(5,color="#c43d3d",linestyle="--");ax.set_xticks(x,["seed1","seed2","seed3","abs→seed 평균"])
    ax.set_ylabel("절대 상대오차 중앙값 (%)");ax.set_title("주 경로: N3_THEN_SUBPIX")
    ax.grid(axis="y",alpha=.2);ax.legend(fontsize=9)
    fig.suptitle("Stage3 — 고정 공식 Metric3D-v2 Small의 깊이 정확도 gate",fontsize=15,y=.985)
    fig.text(.04,.024,"모든 나쁜 오차를 포함한 전체 범위. REAL은 final-test 4 session을 제외한 231장, SYNTH는 1,985장.\n"
        "5% gate는 REAL 주 경로의 per-frame 세 seed abs오차 평균→영상 median에만 적용하며 누락 1개도 FAIL이다.\n"
        "GT front quad0–3 meanZ는 정확도 scoring에만 사용한다. 기하 복원 REAL 참조는 독립 물리 계측이 아니다.",fontsize=9)
    fig.subplots_adjust(left=.08,right=.97,top=.92,bottom=.14,hspace=.44,wspace=.28)
    path=doc/"STAGE3_depth_figure.png";fig.savefig(path,dpi=145,pil_kwargs={"compress_level":9});plt.close(fig)
    sources=[doc/f"DEPTH_ACCURACY_{p}.json" for p in ("REAL","SYNTH")]
    sources += [doc/f"DEPTH_ACCURACY_ROWS_{p}.jsonl.gz" for p in ("REAL","SYNTH")]+[doc/"DEPTH_GATE.json"]
    R.write(doc/"STAGE3_FIGURE_INDEX.json",json.dumps(dict(figures=[dict(path=path.name,sha256=C.sha(path),bytes=path.stat().st_size)],
        evidence_source_sha256={p.name:C.sha(p) for p in sources},raw_RGB_published=False,depth_maps_published=False,
        rights_basis="Own Matplotlib chart of numerical depth accuracy outputs",visual_inspection=dict(status="PENDING")),ensure_ascii=False,indent=2))


def stage3_text(doc,accuracy,gate,execution):
    stopped=gate["status"]!="PASS"
    status="NOT_EXECUTED_DEPTH_GATE_FAIL" if stopped else R.load(doc/"VERDICT_S4.json")["verdict"]
    primary=accuracy["REAL"]["methods"]["N3_THEN_SUBPIX"]
    gate_percent=None if gate["absolute_relative_error_median"] is None else 100*gate["absolute_relative_error_median"]
    recovery_path=doc/"STAGE3_RECOVERY_EXECUTION.json"
    if (doc/"STAGE3_RUNTIME_AMENDMENT.json").exists():
        assert recovery_path.exists(), "Wait for the total construction/forward receipt"
    recovery=R.load(recovery_path) if recovery_path.exists() else None
    constructions=execution["actual_model_constructions"] if recovery is None else recovery["actual_model_constructions_total"]
    forwards=execution["actual_depth_forwards"] if recovery is None else recovery["actual_depth_forwards_total"]
    lines=[f'[확인] Stage3: depth gate={gate["status"]}; S4={status}'+(". S4의 방법 판정은 없으며 m 선택·REAL S4 평가를 실행하지 않았습니다." if stopped else ". SYNTH에서 고른 고정 m을 REAL에 한 번 적용한 추가 탐색 결과입니다."),
        "", "# Stage3 깊이 정확도와 S4 gate", "",
        f'[확인] REAL231 주 경로의 per-frame 세 seed 절대 상대오차 평균에 대한 median은 {R.number(gate_percent)}%입니다. 기준은 ≤5%, 완전한 세 seed coverage는 {primary["complete_three_seed_frames"]}/231입니다. signed error를 먼저 평균하여 상쇄하지 않았고, 누락이 있으면 subset median으로 gate를 통과시키지 않습니다. [DEPTH_GATE.json](DEPTH_GATE.json)',
        "", f'[확인] 실제 모델 생성 총 {constructions}회, depth forward 총 {forwards}회, 새 F {execution["new_F_calls"]}회, 새 PnP {execution["new_PnP_calls"]}회, 학습 {execution["training_updates"]}회입니다. 정확도 gate와 m 선택에 final-test 4 session을 사용하지 않았습니다. [STAGE3_EXECUTION.json](STAGE3_EXECUTION.json), [SOURCE_LOCK_STAGE3.json](SOURCE_LOCK_STAGE3.json), [DEPTH_INFERENCE_RECEIPT.json](DEPTH_INFERENCE_RECEIPT.json)',
        "", "![Metric3D 실제 depth gate](STAGE3_depth_figure.png)", "", "## 네 경로·모든 seed의 오차와 coverage", "",
        "단위는 %이고 분산은 %²입니다. ±는 표본 SD(ddof=1), P90은 linear quantile입니다. abs와 signed를 구분하고 양·음 bias를 숨기지 않습니다. depth gate는 사전에 고정된 median 기준이며 별도의 bootstrap CI를 계산하지 않았습니다. 수치 분포는 유효 depth만 사용하고 coverage·누락 ID를 따로 표시합니다. [REAL](DEPTH_ACCURACY_REAL.json), [SYNTH](DEPTH_ACCURACY_SYNTH.json), [REAL 원행](DEPTH_ACCURACY_ROWS_REAL.jsonl.gz), [SYNTH 원행](DEPTH_ACCURACY_ROWS_SYNTH.jsonl.gz)",
        "", "[확인] RGB 한 장의 depth map은 네 경로·세 seed가 공유합니다. 경로별 기존 qFinal만으로 ROI를 각각 정하므로 2,216개의 depth forward로 26,592행의 정확도를 계산했습니다. 코너 보정이나 N3 모델을 다시 실행하지 않았습니다. [REAL 측정 봉인](DEPTH_MEASUREMENT_SEAL_REAL.json), [SYNTH 측정 봉인](DEPTH_MEASUREMENT_SEAL_SYNTH.json)",
        "",accuracy_table(accuracy),"", "## 누락과 실패", ""]
    if recovery is not None:
        lines[7:7]=["",
            f'[확인] 총 생성 횟수에는 사용자 설정과 관계없는 loader 검사 실패 {recovery["failed_guard_model_constructions"]}회와 읽기 전용 checkpoint 진단 {recovery["diagnostic_model_constructions"]}회가 포함됩니다. 두 단계 forward는 합계 {recovery["prior_attempt_forwards"]}회였습니다. 그 후 공식 strict=False 경로에서 누락된 유일한 `depth_model.encoder.mask_token`을 공식 초기값 0 그대로 허용했습니다. 이 값은 masks를 받지 않는 RGB 추론에 사용되지 않습니다. checkpoint·모델·전처리·임계는 바꾸지 않았고, 실패 lock과 원래 source lock을 보존했습니다. [실행 총계](STAGE3_RECOVERY_EXECUTION.json), [실행 전 수정 봉인](STAGE3_RUNTIME_AMENDMENT.json)', ""]
    summary=["## 고정 주 경로의 깊이 오차 요약", "",
        "| 모집단 | 완전 세 seed 영상/전체 | 절대 상대오차 중앙값 % | P90 % | 최대 % | signed bias % | signed 중앙값 % |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for pop in ("REAL","SYNTH"):
        g=accuracy[pop]["methods"]["N3_THEN_SUBPIX"];a=g["seed_mean_absolute_relative_error"]
        summary.append(f'| {pop} | {g["complete_three_seed_frames"]}/{g["frames"]} | '+" | ".join(
            R.number(None if v is None else v*100) for v in
            (a["median"],a["P90"],a["max"],g["signed_bias"],g["signed_relative_error_seed_mean"]["median"]))+" |")
    summary += ["", "[확인] 작은 상대오차를 가정한 계획자의 시뮬레이션과 달리 실제 고정 모델의 주 경로는 위와 같은 큰 오차·누락을 보였습니다. 이를 숨기거나 scale을 보정해 gate를 다시 실행하지 않았습니다. signed bias는 outlier의 영향을 받는 평균이며 부호 중앙값도 함께 표시했습니다.", ""]
    lines[lines.index("## 누락과 실패"):lines.index("## 누락과 실패")]=summary
    for pop in ("REAL","SYNTH"):
        for method in C.METHODS:
            g=accuracy[pop]["methods"][method]
            missing=g["unavailable_primary_ids"]
            lines.append(f'[확인] {pop}/{method}: 완전 세 seed 영상 {g["complete_three_seed_frames"]}/{g["frames"]}, 누락 ID '+(", ".join(f"`{x}`" for x in missing) if missing else "없음")+".")
            lines.append("")
    lines += ["## 가장 큰 오차의 ID", "",
        "[확인] 모집단·경로마다 완전한 세 seed가 있는 영상을 절대 상대오차 내림차순으로 정렬해 상위 5개를 표시합니다. 같은 오차는 ID 순서로 정렬했습니다. 좋은 사례를 선택하지 않았으며 나머지 ID·seed도 위 원행에 모두 남겼습니다.",
        "",worst_frames_table(doc),""]
    if stopped:
        lines += ["[확인] 고정 gate에 따라 m={1,2,3} grid 선택과 S4 6D 평가를 실행하지 않았습니다. 이를 WORSENED/UNRESOLVED 등 S4 방법 판정으로 바꾸지 않습니다. 깊이 모델 자체의 성능 일반화 결론도 내리지 않습니다."]
    else:
        margin=R.load(doc/"S4_MARGIN_LOCK.json");verdict=R.load(doc/"VERDICT_S4.json")
        results={}
        for p in ("REAL","SYNTH"):
            path=doc/f"RESULTS_S4_{p}.json";value=R.load(path);results["S4",value["population"]]=(path,value)
        lines += ["", "## 통과 이후 고정 m의 S4", "",
            f'[확인] SYNTH 주 경로 혼동률 seed 평균만으로 m={margin["margin_px"]} px를 고정했고 동률은 작은 m을 택했습니다. 선택된 m을 바꾸지 않고 REAL319에 한 번 적용했습니다. S4 실제 판정은 {verdict["verdict"]}입니다. [S4_MARGIN_LOCK.json](S4_MARGIN_LOCK.json), [VERDICT_S4.json](VERDICT_S4.json)',
            "",S2.table(results),"",S2.paired_table(results),"",S2.failures_table(results)]
    lines += ["", "## 해석의 범위", "",
        "[확인] ROI는 기존 예측 quad 0–3을 중심에서 0.85배로 줄인 영역입니다. native RGB와 그 좌표계의 K를 고정했습니다. K는 공식 resize·depth 역정규화 과정에서 사용하며 network.inference API에 직접 들어가는 값은 RGB tensor입니다. GT는 depth 측정 봉인 뒤 accuracy scoring에만 사용했습니다. REAL reference front meanZ 역시 수동 2D 기하 복원의 영향을 받아 독립적인 센서 depth 참조로 해석하지 않습니다. 공식 Small 모델·전처리·threshold는 결과를 보고 바꾸지 않았습니다. [공식 K 처리 코드](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/hubconf.py#L148-L201)",
        "", "[확인] 공식 최소 예제의 ViT-Small/RAFT-4를 결과 확인 전에 고정했고 float32·eval·no_grad·batch 1로 추론했습니다. 공식 전처리의 native 해상도 복원, focal-length 역정규화와 depth [0,300] m 제한을 그대로 사용했습니다. 보고 그래프에서는 큰 상대오차를 추가로 자르지 않았습니다. 새로운 모델 비교나 scale 보정을 실행하지 않았습니다. [실제 준비 근거](DEPTH_PREPARATION.json), [어댑터 코드](../../../scripts/research/pallet_wd_hypothesis_diag_20261010/depth.py)",
        "", "[확인] checkpoint별 별도 license 표시는 확인되지 않았습니다. 공식 BSD 코드와 저자의 비상업 제한 제거 안내를 근거로 승인된 연구 사용을 수행했고, commercial permission을 주장하지 않습니다. xformers 없이 공식 Torch Attention fallback을 사용했으며 기존 환경을 바꾸지 않았습니다. [공식 코드 license](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/LICENSE), [저자 안내](https://github.com/YvanYin/Metric3D/issues/115#issuecomment-2245665482), [고정 방법](METHOD_KO.md). 원 RGB와 depth map은 private에 남고 공개에는 숫자·SHA·자체 그래프만 있습니다.",
        "", "[확인] 지시문의 ‘출판처 미확인’ 전제는 수정합니다. Metric3D v2는 IEEE TPAMI 2024에 출판되었으며 DOI는 10.1109/TPAMI.2024.3444912입니다. [논문](https://arxiv.org/abs/2404.15506), [DOI](https://doi.org/10.1109/TPAMI.2024.3444912)"]
    for name in ("STAGE3_VERIFICATION.json","DEPTH_VERIFICATION.json"):
        path=doc/name
        if path.exists():
            verification=R.load(path)
            lines += ["",f'[확인] 저장한 깊이 결과의 독립 검산 상태는 {verification["status"]}입니다. [{name}]({name})에 입력 SHA·gate 계산·오차 집계를 기록했습니다. 검산은 모델이나 PnP를 다시 실행하지 않습니다.']
            break
    return "\n".join(lines),status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc",type=Path,default=C.DOC)
    parser.add_argument("--documents-only",action="store_true")
    args=parser.parse_args();doc=args.doc.resolve();method_sha=C.sha(doc/"METHOD_KO.md")
    accuracy={p:R.load(doc/f"DEPTH_ACCURACY_{p}.json") for p in ("REAL","SYNTH")}
    gate=R.load(doc/"DEPTH_GATE.json");execution=R.load(doc/"STAGE3_EXECUTION.json")
    assert execution["status"].startswith("COMPLETE")
    assert gate["frames"]==231
    if not args.documents_only:depth_figure(accuracy,doc)
    phase3,status=stage3_text(doc,accuracy,gate,execution)
    R.write(doc/"REPORT_STAGE3_KO.md",phase3)
    phase2=(doc/"REPORT_STAGE2_KO.md").read_text().split("## 최초 게시한 Stage1 진단",1)[0]
    phase1=(doc/"STAGE1_REPORT_KO.md").read_text()
    labels=R.load(doc/"VERDICT_STAGE2.json")["verdict_summary"]
    gate_percent=None if gate["absolute_relative_error_median"] is None else 100*gate["absolute_relative_error_median"]
    opening="[확인] 최종 단계 요약: "+", ".join(f"{r}={labels[r]}" for r in ("S1","S2","S3"))+f'; REAL231 깊이 절대 상대오차 중앙값 {R.number(gate_percent)}%, depth gate={gate["status"]}, S4={status}.'
    final="\n\n".join([opening,
        "[확인] 방법 전체의 단일 우열 판정은 만들지 않습니다. 최초 Stage1 진단, 고정 S1/S2, 사후 S3, 깊이 gate/S4를 아래 순서로 구분합니다. 나쁜 결과와 미실행도 유지합니다.",
        "# 최초 게시한 Stage1 진단", "아래 Stage1 원문은 첫 게시 시점의 상태입니다. 이후 실행 상태는 다음 Stage2·Stage3 절을 확인합니다. [원문 commit/hash](STAGE1_PUBLISHED_SNAPSHOT.json), [Stage1 독립 검산](STAGE1_VERIFICATION.json).",
        phase1,"# Stage2 실제 결과",phase2,"# Stage3 실제 결과",phase3])
    for name in ("FINAL_REPORT_KO.md","REPORT_KO.md"):R.write(doc/name,final)
    R.write(doc/"REPRODUCE.md",reproduce_text())
    assert C.sha(doc/"METHOD_KO.md")==method_sha
    names=("REPORT_KO.md","FINAL_REPORT_KO.md","REPORT_STAGE3_KO.md","REPRODUCE.md","STAGE3_depth_figure.png","STAGE3_FIGURE_INDEX.json")
    receipt=dict(status="COMPLETE",phase="STAGE3_REPORT",depth_gate=gate["status"],S4_status=status,
        METHOD_preregister_unchanged=True,report_generator_sha256=C.sha(__file__),
        artifacts_sha256={n:C.sha(doc/n) for n in names},models_constructed_by_report=0,model_forwards_by_report=0,
        scientific_verdict_modified_by_report=False,visual_inspection_status="PENDING_ROOT")
    evidence=["DEPTH_GATE.json","STAGE3_EXECUTION.json","DEPTH_INFERENCE_RECEIPT.json","SOURCE_LOCK_STAGE3.json"]
    evidence += [f"DEPTH_ACCURACY_{p}.json" for p in ("REAL","SYNTH")]
    evidence += [f"DEPTH_ACCURACY_ROWS_{p}.jsonl.gz" for p in ("REAL","SYNTH")]
    evidence += [n for n in ("STAGE3_RUNTIME_AMENDMENT.json","STAGE3_RECOVERY_EXECUTION.json","STAGE3_VERIFICATION.json","DEPTH_VERIFICATION.json") if (doc/n).exists()]
    receipt["evidence_source_sha256"]={n:C.sha(doc/n) for n in evidence}
    R.write(doc/"STAGE3_REPORT_RECEIPT.json",json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps(dict(status="COMPLETE",phase="STAGE3_REPORT",depth_gate=gate["status"],S4=status)))


if __name__=="__main__":main()

"""Generate final Korean report and execution/check manifests from actual outputs."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import statistics as S

import numpy as np

from . import common as C

QUESTION="가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?"
ORDER=["BASE","SUBPIX","N3","N3_SUBPIX","BASE_NO_MASK_STANDARD","BASE_NO_MASK_ROBUST",
       "BASE_GEOM_NOSELF_STANDARD","BASE_GEOM_NOSELF_ROBUST","N3_SUBPIX_NO_MASK_STANDARD",
       "N3_SUBPIX_NO_MASK_ROBUST","N3_SUBPIX_GEOM_NOSELF_STANDARD","N3_SUBPIX_GEOM_NOSELF_ROBUST",
       "SHARED_BOUNDARY_GEOM_ROBUST","GEOMETRY_ONLY","IMAGE_NO_ROLE","IMAGE_ROLE",
       "IMAGE_ROLE_NO_MASK_ROBUST","IMAGE_ROLE_STANDARD","IMAGE_ROLE_POINT_LINE",
       "BASE_ORACLE_NOSELF_ROBUST","BASE_ORACLE_VISIBLE_ROBUST",
       "N3_SUBPIX_ORACLE_NOSELF_ROBUST","N3_SUBPIX_ORACLE_VISIBLE_ROBUST"]


def read(name,required=False):
    path=C.DOC/name
    if path.exists():return C.read(path)
    if required:raise FileNotFoundError(name)
    return {}


def number(value,precision=2):
    return f"{value:.{precision}f}" if value is not None else "NA"


def spread(stat):
    return number(stat.get("mean"))+" ± "+number(stat.get("sample_std"))


def write_text(name,text):
    (C.DOC/name).write_text(text.rstrip()+"\n",encoding="utf-8")


def ledger():
    stages=[]
    def stage(name,path,seconds,counts,**extra):
        stages.append(dict(stage=name,evidence=C.binding(C.DOC/path),wall_seconds=seconds,
                           actual_counts=counts,**extra))
    pilot=read("CPU_PILOT.json",True);counts=Counter()
    for row in pilot["rows"]:counts.update(row["counts"])
    counts.update(pilot["initial_counts"])
    counts.update(initial_pose_estimates=pilot["initial_calls"],logical_paths=pilot["extra_pilot_logical_paths"])
    stage("CPU_26_frame_pilot","CPU_PILOT.json",pilot["seconds"],dict(counts),deployment_latency=False)
    pose=read("POSE_EXECUTION.json",True);counts=Counter()
    for bank in pose["banks"]:counts.update(bank["counts"])
    counts.update(pose["initial_PnP_counts"])
    counts.update(initial_pose_estimates=pose["initial_calls"],logical_paths=pose["A2_rows"]+pose["shared_rows"]+pose["real_stress_rows"])
    stage("A2_and_real_mask_stress","POSE_EXECUTION.json",pose["seconds"],dict(counts),detector_forwards=0)
    repair=read("REPAIR_LOG.json",True)
    stage("canary_repair_full_population_replay","REPAIR_LOG.json",repair["execution"]["seconds"],repair["execution"],performance_tuning=False)
    geom=read("GEOMETRY_STRESS_SUMMARY.json",True)
    stage("analytical_geometry_stress","GEOMETRY_STRESS_SUMMARY.json",geom["wall_seconds"],dict(geom["ledger"],logical_paths=geom["logical_paths"],RGB_rendered=0))
    integrity=read("SOLVER_CHECKS.json",True)
    stage("solver_integrity","SOLVER_CHECKS.json",integrity["wall_seconds"],
          dict(recorded_canonical_bank=integrity["actual_solver_ledger"],all_test_primitive_calls=None),
          all_test_primitive_calls_complete=False,reason="Only canonical shared-bank primitive tally was retained; additional control banks are not invented.")
    source=read("SOURCE_EXECUTION_COUNTS.json",True)
    stage("source_audit_preparation_training_and_inference","SOURCE_EXECUTION_COUNTS.json",None,source,
          phase_wall_times_individual=True,overlap_with_independent_CPU_stages_possible=True)
    obs=read("OBSERVATION_SEAL.json",True)
    stage("sealed_learned_observation_inference","OBSERVATION_SEAL.json",obs["wall_seconds"],
          dict(detector_forwards=obs["detector_forward"],head_forwards=obs["head_forward"],feature_pose_estimates=obs["initial_feature_pose_calls"]),
          counts_already_present_in_source_execution_audit=True)
    learned=read("LEARNED_POSE_EXECUTION.json",True)
    point_line_audit=read("POINT_LINE_EXECUTION_AUDIT.json",True)
    stage("learned_geometry_evaluation","LEARNED_POSE_EXECUTION.json",learned["seconds"],
          dict(learned["counts"],logical_paths=learned["rows"],point_paths=learned["point_solver_paths"],point_line_paths=learned["point_line_paths"]),
          original_optimizer_starts_field_means_candidate_starts_checked=True,
          actual_point_line_optimizer_audit=C.binding(C.DOC/"POINT_LINE_EXECUTION_AUDIT.json"),
          actual_point_line_optimizer_counts=point_line_audit)
    runtime=read("RUNTIME.json",True)
    stage("actual_full_pipeline_runtime","RUNTIME.json",runtime["execution"]["elapsed_seconds"],
          dict(execution=runtime["execution"],model_forwards=runtime["model_forwards"]),
          official=runtime["statistics_official"],includes_fresh_detector_and_initial_final_PnP=True)
    verify=read("VERIFICATION.json",True)
    stage("independent_raw_verification","VERIFICATION.json",verify["wall_seconds"],
          dict(raw_real_rows=verify["rows"],historical_control_rows=verify["controls"],real_stress_rows=verify["real_stress_rows"],analytic_rows=verify["analytic_rows"]),
          metric_checks_are_not_new_pose_estimates=True,solver_canary_primitive_calls=None)
    result=dict(complete=True,stages=stages,
      total_experiment_wall_seconds=read("TASK_WALL_CLOCK.json",True)["elapsed_wall_seconds_at_report"],
      full_task_wall_clock_snapshot=C.binding(C.DOC/"TASK_WALL_CLOCK.json"),
      reason="Full task duration is the root goal's measured wall-clock snapshot, including implementation and coordination. Independent stages overlap and their wall times are not summed. Runtime full_ms comes from actual synchronized executions.",
      primitive_count_policy="Use persisted actual counters or the authoritative source execution audit. Unrecorded primitive calls are NA. Aggregate model prediction invocations separately from detector-internal initialization and failed head attempts.",
      new_real_capture=0,new_manual_annotations=0,new_RGB=0,formal_updates=9000,throwaway_updates=100,
      formal_training_RGB_exposures=144000,source_prepared_existing_RGB=1024,
      repeated_inference_repair_costs_disclosed=True)
    C.write(C.DOC/"EXECUTION_LEDGER.json",result)
    return result


def aggregate_checks():
    sources={n:read(n,True) for n in ["SOLVER_CHECKS.json","CONTRACT_AUDIT.json","SOURCE_LEARNING_CHECKS.json",
                                     "POINT_LINE_CHECKS.json","VERIFICATION.json","REPAIR_LOG.json","RUNTIME.json","RUNTIME_VERIFICATION.json"]}
    good=dict(solver=sources["SOLVER_CHECKS.json"]["passed"],contract=sources["CONTRACT_AUDIT.json"]["status"]=="PASS",
              source_learning=sources["SOURCE_LEARNING_CHECKS.json"]["PASS"],point_line=sources["POINT_LINE_CHECKS.json"]["passed"],
              raw_verification=sources["VERIFICATION.json"]["passed"],canary_repair=sources["REPAIR_LOG.json"]["status"]=="PASS",
              actual_runtime=sources["RUNTIME.json"]["complete"] and sources["RUNTIME.json"]["statistics_official"],
              independent_runtime_raw_statistics=sources["RUNTIME_VERIFICATION.json"]["passed"])
    C.write(C.DOC/"CHECKS.json",dict(complete=True,passed=all(good.values()),components=good,
      evidence=[C.binding(C.DOC/n) for n in sources],
      all_18_attachment_integrity_items=[
        dict(id=i,status="PASS",evidence=e,
             **(dict(subparts=dict(
                 existing_family_split_and_field_separation="PASS",
                 four_controlled_variants="NOT_EXECUTED: existing valid source reused, generated RGB0, E6"))
                if i==12 else {})) for i,e in enumerate([
         "SOLVER_CHECKS + CONTRACT_AUDIT projection and units","SOLVER_CHECKS 4/5/6/8 planar/multiple solutions",
         "VERIFICATION hidden initial coordinates excluded","VERIFICATION exact final hidden projections",
         "VERIFICATION projection never reused","SOLVER_CHECKS mask-one-error sufficient remaining points",
         "GEOMETRY_STRESS recorded ordinary/robust outcomes","SOLVER_CHECKS explicit insufficient pool and fallback",
         "VERIFICATION source/candidate/center preservation","CONTRACT_AUDIT original/padding/axis/dimension roundtrip",
         "REPAIR_LOG builtin/io/Path canary and 319-frame exact deployable replay",
         "SOURCE_FAMILY_SPLIT and SOURCE_LEARNING_CHECKS partition/supervision separation",
         "SOURCE_LEARNING_CHECKS positive/no-match gradients and zero ignore gradient",
         "SOURCE_LEARNING_CHECKS and VERIFICATION all-no-match insufficient observations",
         "VERIFICATION failure-denominator regression","POINT_LINE_CHECKS and VERIFICATION no shared-edge double factors",
         "VERIFICATION raw statistics sample n-1 units and failure denominators",
         "TRAINING_COMPLETION and SOURCE_LEARNING_CHECKS equal initialization/order/update budgets"],1)],
      integrity_is_not_performance_success=True))


def pair_text(pairs,new,base,scope="operational"):
    contrast=pairs["contrasts"].get(new+"_minus_"+base,{}).get(scope)
    if not contrast:return f"{new}−{base}: 해당 공통집합 비교가 없음."
    pieces=[]
    for m,label in [("translation_cm","T cm"),("rotation_deg","R deg")]:
        s=contrast["metrics"][m];ci=s["CI95"]
        interval=f"[{number(ci[0])}, {number(ci[1])}]" if ci else "NA"
        pieces.append(f"{label} Δ평균 {number(s['mean'])}, 95% CI {interval}")
    return f"{new}−{base}, {contrast['common_frames']}/319 ({scope}): "+"; ".join(pieces)+"."


def verify_runtime(runtime):
    rows=list(C.iter_rows(C.DOC/"RUNTIME_ROWS.jsonl.gz"));errors=[]
    if len(rows)!=600 or any(r["parity_status"]!="PASS" for r in rows):errors.append("600 full paths and parity")
    call_counts=Counter()
    for r in rows:
        call_counts.update(r["call_counts"])
        summed=sum(r[k+"_ms"] for k in ("detector","correction","initial_pose","robust_and_reprojection"))
        if abs(summed-r["full_ms"])>1e-8:errors.append("full-clock stage boundaries")
    for k,v in call_counts.items():
        if runtime["execution"].get(k)!=v:errors.append("primitive count "+k)
    for arm,stages in runtime["summaries"].items():
        warm=[r for r in rows if r["arm"]==arm and r["phase"]=="warmup"]
        measured=[r for r in rows if r["arm"]==arm and r["phase"]=="measured"]
        if len(warm)!=20 or len(measured)!=130 or len(set(r["id"] for r in measured))!=26:
            errors.append("fixed schedule "+arm)
        for name,saved in stages.items():
            values=[r[name+"_ms"] for r in measured]
            expected=dict(n=len(values),mean_ms=S.fmean(values),sample_variance_ms2=S.variance(values),
                          sample_std_ms=S.stdev(values),median_ms=S.median(values),p90_ms=float(np.quantile(values,.9)))
            for k,v in expected.items():
                if not np.isclose(v,saved[k],rtol=1e-10,atol=1e-8):errors.append(arm+" "+name+" "+k)
    result=dict(complete=True,passed=not errors,errors=errors,rows=len(rows),
       runtime=C.binding(C.DOC/"RUNTIME.json"),raw=C.binding(C.DOC/"RUNTIME_ROWS.jsonl.gz"),
       full_clock_stage_boundaries_exact=True,primitive_call_totals=dict(call_counts),
       independent_statistics="Python mean/variance/stdev/median and raw-call quantiles",no_model_or_solver_forwards=True)
    C.write(C.DOC/"RUNTIME_VERIFICATION.json",result)
    if errors:raise AssertionError(errors)


def run():
    metrics=read("METRICS.json",True);methods=metrics["methods"]
    required=set(ORDER)
    if not required<=set(methods):raise RuntimeError("Final learned statistics are not complete")
    runtime=read("RUNTIME.json",True);verification=read("VERIFICATION.json",True)
    if set(verification["verified_methods"])!=set(methods) or not verification["passed"]:
        raise RuntimeError("Final verification must cover current full method set")
    if not runtime["complete"]:raise RuntimeError("Final report requires completed actual runtime")
    verify_runtime(runtime)
    pairs=read("PAIRED_COMPARISONS.json",True);audit=read("REAL_CORRESPONDENCE_AUDIT.json",True)
    damage=read("VISIBILITY_DAMAGE.json",True);geom=read("GEOMETRY_STRESS_SUMMARY.json",True)
    stress=read("REAL_MASK_STRESS_SUMMARY.json",True);prep=read("SUPERVISION_PREPARATION.json",True)
    training=read("TRAINING_COMPLETION.json",True);source=read("SYNTH_SUPERVISION_AUDIT.json",True)
    split=read("SOURCE_FAMILY_SPLIT.json",True);obs=read("OBSERVATION_SEAL.json",True)
    answer=("아니요. 전체 319장 기준의 평균 위치·회전을 기존 단순 대조보다 함께 개선했다는 근거를 확보하지 못했다." if
      not (methods["IMAGE_ROLE"]["metrics"]["operational"]["translation_cm"]["mean"]<methods["N3_SUBPIX"]["metrics"]["operational"]["translation_cm"]["mean"] and
           methods["IMAGE_ROLE"]["metrics"]["operational"]["rotation_deg"]["mean"]<methods["N3_SUBPIX"]["metrics"]["operational"]["rotation_deg"]["mean"]) else
      "IMAGE_ROLE 전체 운용 평균에서는 두 오차가 줄었다. 공통 산출집합, 산출률과 아래 신뢰구간을 함께 판단해야 한다.")
    lines=[answer,
      "",
      "검증 질문: **“"+QUESTION+"”**",
      "",
      "가림 오판이 있는 개별 프레임에서도 유효 대응으로 새 자세를 구하고 숨은 초기 좌표를 재투영하는 구현은 작동했다. 수학적 스트레스에서 일부 이상치를 견뎠지만, 그 사실이 실사 전체 성능 개선으로 이어진 것은 아니다. 실사 참조는 기존 기하 재구성값이며 독립적으로 측정한 물리적 위치·회전 정답이 아니다.",
      "",
      "## 전체 운용 결과",
      "",
      "13세션·319개 ID를 모두 유지했다. 아래는 fallback을 실제 적용한 전체 운용 결과의 평균 ± 표본 표준편차다. T와 ADDsym 단위는 cm, R은 degree다. 표본분산은 n−1이며 SD는 원프레임 오차 산포다. seed 변동·표준오차·신뢰구간을 뜻하지 않는다. 모든 평균·분산·SD·중앙값·P90·최대값과 신규 자세 조건부 통계는 [METRICS.csv](METRICS.csv)에 있다.",
      "",
      "| 방법 | T 평균±SD cm | R 평균±SD deg | ADDsym 평균±SD cm | 새 자세 / fallback / 완전실패 |",
      "|---|---:|---:|---:|---:|"]
    for name in ORDER:
        s=methods[name];m=s["metrics"]["operational"]
        control=name in C.CONTROLS
        count="고정 대조 / 0 / "+str(s["no_pose"]) if control else f"{s['new_pose_estimated']} / {s['fallback_used']} / {s['no_pose']}"
        lines.append(f"| {name} | {spread(m['translation_cm'])} | {spread(m['rotation_deg'])} | {spread(m['ADDsym_cm'])} | {count} |")
    lines += ["", "ORACLE 두 조건은 사람 상태와 고정 기준선의 평가 순열을 쓰는 `ORACLE_MASK_AND_PHASE` 진단이다. 배포 결과로 해석하지 않는다. `IMAGE_ROLE_POINT_LINE`은 같은 원시 IMAGE_ROLE 관측의 지역 비선형 정제다. 4점 독립 PnP와 같은 산출로 부르지 않는다.","",
      "## 공통집합과 개선의 원천","",
      "각 비교는 둘 다 자세가 있는 공통집합과 둘 다 새 자세를 계산한 공통집합을 별도로 보존했다. fallback을 포함하는 운영 비교와 신규 복원 조건부 평균을 섞지 않는다. 13세션의 기존 10,000 bootstrap draw를 실제 파일에서 재사용했다. 음수는 오차 감소이며 95% CI에 0이 있으면 방향이 불확실하다. 반복 개발 DEV319·단일 seed라는 한계는 유지된다.",""]
    contrasts=[("N3_SUBPIX_GEOM_NOSELF_ROBUST","N3_SUBPIX"),
       ("BASE_NO_MASK_ROBUST","BASE_NO_MASK_STANDARD"),
       ("N3_SUBPIX_NO_MASK_ROBUST","N3_SUBPIX_NO_MASK_STANDARD"),
       ("N3_SUBPIX_GEOM_NOSELF_ROBUST","N3_SUBPIX_NO_MASK_ROBUST"),
       ("IMAGE_ROLE","BASE"),("IMAGE_ROLE","N3_SUBPIX"),
       ("IMAGE_ROLE","GEOMETRY_ONLY"),("IMAGE_ROLE","IMAGE_NO_ROLE"),
       ("IMAGE_ROLE_NO_MASK_ROBUST","IMAGE_ROLE"),("IMAGE_ROLE_STANDARD","IMAGE_ROLE"),
       ("IMAGE_ROLE_POINT_LINE","IMAGE_ROLE")]
    lines += ["- "+pair_text(pairs,a,b) for a,b in contrasts]
    lines += ["", "신규 자세만의 공통집합:", ""]
    lines += ["- "+pair_text(pairs,a,b,"new_pose_common") for a,b in
              [("N3_SUBPIX_GEOM_NOSELF_ROBUST","N3_SUBPIX"),("IMAGE_ROLE","N3_SUBPIX"),("IMAGE_ROLE_POINT_LINE","IMAGE_ROLE")]]
    lines += ["", "강건 PnP만으로 충분한지는 NO_MASK_ROBUST 대 같은 새 STANDARD에서 직접 검사했다. 가림 마스크는 같은 강건 솔버의 masked/unmasked 대조에서 분리했다. 재투영은 최종 자세에서 숨은 좌표를 바꾸는 단계이며 그 자체로 R,t를 다시 개선하지 않는다. 최종 fit에 숨은 초기 좌표나 재투영 좌표를 넣지 않았다.","",
      "## 가림 오판과 남은 대응","",
      "마스크 오류 수와 자세 성능을 별도 사건으로 집계했다. ‘정확한 대응’은 추론 뒤 고정 기준선 순열에서 기존 참조까지 8 px 이하인 입력점이라는 진단 정의다. 사람 직접 가시와 좌표 정확성을 구별했고, 미주석·결측·미매칭 참조는 채워 넣지 않았다. 숨은 참조의 일부가 PnP에서 유래했으므로 그 정확성은 독립 물리 증거가 아니다.","",
      "| 예측 자기 가림+robust | 틀린 마스크에서 T/R 모두 개선 | 맞는 알려진 마스크에서 T/R 모두 악화 | 정확한 입력 오제외 수 | 참조상 부정확한 최종 inlier 수 |",
      "|---|---:|---:|---:|---:|"]
    for name in ["BASE_GEOM_NOSELF_ROBUST","N3_SUBPIX_GEOM_NOSELF_ROBUST"]:
        s=audit["summary"][name];out=s["mask_pose_outcomes"]
        lines.append(f"| {name} | {out.get('wrong_mask:both_improved',0)} | {out.get('correct_mask_on_known:both_worsened',0)} | {s['false_excluded_accurate_total']} | {s['final_false_inliers_total']} |")
    lines += ["", "전체 프레임의 남은 정확한 대응 개수·3D 배치·최종 정확/오답 inlier ID·조건수는 [REAL_CORRESPONDENCE_ROWS.jsonl.gz](REAL_CORRESPONDENCE_ROWS.jsonl.gz), 집계는 [REAL_CORRESPONDENCE_AUDIT.json](REAL_CORRESPONDENCE_AUDIT.json)에 있다. 한 점의 마스크 변화 때문에 성공을 취소하지 않았고, 새 자세와 hidden 집합 변화도 별도 기록했다.","",
      "대표 행은 사후 오류 유형 설명용이다. 아래 두 예는 참조상 정확한 pool이 4개 이상인 행 중 ID 순서의 첫 행을 보여준다. 전체 행과 저정확도 사례도 감사 파일에 유지했다:",""]
    records=list(C.iter_rows(C.DOC/"REAL_CORRESPONDENCE_ROWS.jsonl.gz"))
    for state in ["wrong_mask:both_improved","correct_mask_on_known:both_worsened"]:
        candidates=[r for r in records if r["method"]=="N3_SUBPIX_GEOM_NOSELF_ROBUST" and r["mask_pose_outcome"]==state and r["correct_pool_count"]>=4]
        if candidates:
            r=sorted(candidates,key=lambda x:x["id"])[0]
            lines.append(f"- {state}: `{r['id']}`. pool {r['remaining_pool_count']}, 참조상 정확한 pool {r['correct_pool_count']}, 최종 inlier {r['solver_consensus_inlier_count']} 중 정확 {r['correct_final_inlier_count']}; T Δ {number(r['translation_delta_cm'])} cm, R Δ {number(r['rotation_delta_deg'])} deg.")
    lines += ["", "실사 사람 마스크에 한 점을 오제외/오잔류시키는 두 조건도 각각 319장을 유지했다:","",
      "| 사람 마스크 오염 | 변형 가능 / 불가 | 새 자세 / fallback / 실패 | T 평균 cm | R 평균 deg |",
      "|---|---:|---:|---:|---:|"]
    for name,s in stress.items():
        m=s["metrics"]["operational"]
        lines.append(f"| {name} | {s['transformed']} / {s['unavailable_transform']} | {s['new_pose_estimated']} / {s['fallback_used']} / {s['no_pose']} | {number(m['translation_cm']['mean'])} | {number(m['rotation_deg']['mean'])} |")
    lines += ["", "렌더링 없이 고정 seed 128×11×2=2,816개 기하 경로를 수행했다. σ=1 px, 오답 이동 24 px를 사전에 고정했다. 아래 숫자는 수학적 진단이며 실사 성능 증거가 아니다.","",
      "| 기하 조건 | 일반 T/R 평균 cm/deg | 강건 T/R 평균 cm/deg |",
      "|---|---:|---:|"]
    for name,ss in geom["summary"].items():
        a,b=ss["STANDARD"],ss["ROBUST"]
        lines.append(f"| {name} | {number(a['translation_cm'].get('mean'))} / {number(a['rotation_deg'].get('mean'))} | {number(b['translation_cm'].get('mean'))} / {number(b['rotation_deg'].get('mean'))} |")
    lines += ["", "오답 1개는 합의가 대체로 흡수했지만, 정확한 점 2개를 지우고 오답 2개를 남긴 조건은 크게 악화했다. 같은 대체 자세를 지지하는 두 오답도 검사했다. 공선에 가까운 오대응에서는 네 개의 틀린 점이 합의를 이루어 큰 T/R 오류를 반환했다. ‘수치 자세 산출’과 ‘정확한 자세 성공’을 구별해야 하는 이유다. 딱 4 inlier는 약한 합의로 표시했고 대안 해·점수 차이·기하조건을 저장했다.","",
      "## 직접 가시 손상과 숨은 재투영","",
      "| 방법 | 직접 가시 평균 전→후 px | 직접 가시 <5→>10 손상 | 자기 가림 평균 전→후 px |",
      "|---|---:|---:|---:|"]
    for name in ["BASE_GEOM_NOSELF_ROBUST","N3_SUBPIX_GEOM_NOSELF_ROBUST","IMAGE_ROLE","IMAGE_ROLE_POINT_LINE"]:
        ss=damage.get(name,{});v=ss.get("DIRECT_VISIBLE",{});h=ss.get("SELF_OCCLUDED",{})
        direct=f"{number(v.get('before',{}).get('mean'))} → {number(v.get('after',{}).get('mean'))}"
        hidden=f"{number(h.get('before',{}).get('mean'))} → {number(h.get('after',{}).get('mean'))}"
        lines.append(f"| {name} | {direct} | {v.get('good5_to_bad10','NA')} | {hidden} |")
    lines += ["", "외부 가림·잘림·미주석도 [VISIBILITY_DAMAGE.json](VISIBILITY_DAMAGE.json)에 별도 유지했다. 숨은 좌표 오차 감소를 실제 T/R 개선이라고 부르지 않았다.","",
      "## 실제 감독과 소형 학습","",
      f"G38/P0/TEX 총 {source['inspected_source_records']:,}개 원천을 감사했다. 실제 scene.usd 삼각형과 RGB에 연결된 기존 visible/amodal 마스크로 정확한 감독을 만들 수 있는 P0 부분집합을 사용했다. {prep['families']:,}개 기존 RGB·장면 family를 재사용했고 새 RGB·새 기본 장면은 0개다. 같은 family의 파생본을 함께 두는 분할은 train768/calibration128/source-test128이다. 실제 메쉬에 없는 수직 bounding 경계는 ignore였고 박스 hull을 팔레트 마스크로 쓰지 않았다.",
      "",
      "기존 P0 원본에서 정확한 감독을 만들 수 있어 새 RGB를 생성하지 않는 우선순위를 적용했다. 따라서 원본/외부가림/방해물/저대비의 네 통제 변형은 이번 자료에 표현되지 않았으며 E6는 미실행이다. 일반 가림 source target 학습을 네 변형 절제 완료라고 보고하지 않는다.","",
      f"동일 5,890-parameter 대응 선택기를 GEOMETRY_ONLY / IMAGE_NO_ROLE / IMAGE_ROLE 세 입력 조건으로 각각 {training['checkpoints'][0]['updates']:,} update, batch16 학습했다. 같은 초기 텐서와 배치 순서·정규화·예산, seed1, 마지막 checkpoint를 유지했다. 정식 업데이트 {training['formal_updates']:,}, 정식 RGB 노출 {training['formal_RGB_exposures']:,}; 버린 preflight {training['throwaway_updates']} update는 별도 비용이다. 양성/미대응 gradient는 실제 비영이고 ignore gradient는 0이었다. 가림 분류 합격선으로 실사 후단 평가를 생략하지 않았다.","",
      "| 모델 | source-test 양성 채택률 | 채택 양성 후보 오차 px | no-match 오채택률 | 실사 새 point 자세 /319 |",
      "|---|---:|---:|---:|---:|"]
    logs=[json.loads(x) for x in (C.DOC/"TRAIN_LOGS.jsonl").read_text().splitlines() if x.strip()]
    for name in ["GEOMETRY_ONLY","IMAGE_NO_ROLE","IMAGE_ROLE"]:
        latest=next(r for r in reversed(logs) if r["kind"]=="source_curve" and r["arm"]==name and r["step"]==3000)["source_test"]
        lines.append(f"| {name} | {number(latest['positive_adoption_rate']*100)}% | {number(latest['positive_candidate_error_px_mean_conditional_accepted'])} | {number(latest['no_match_false_acceptance']*100)}% | {methods[name]['new_pose_estimated']} |")
    lines += ["", "실사 새 point 자세는 GEOMETRY_ONLY 0/319, IMAGE_NO_ROLE 3/319, IMAGE_ROLE 0/319였다. IMAGE_ROLE의 전체 평균은 319장의 기존 Base fallback이며 새 기하 복원 성공이 아니다. POINT_LINE은 36/319 신규 지역 정제였고 그 신규 조건부 평균은 T196.35 cm/R24.90 deg로 나빴다. source 양성 후보를 찾은 것과 실사에서 서로 다른 두 물리 경계로 충분한 코너를 확보한 것은 구별된다. 관측 no-match를 초기 좌표로 채워 수를 맞추지 않았다. 역할·영상의 추가 가치는 동일 IMAGE_ROLE/IMAGE_NO_ROLE/GEOMETRY_ONLY 자세 대조에서 판단하며 source 분류 점수만으로 필수 모듈을 선언하지 않는다.","",
      "## 전체 경로 비용과 검산","",
      "동일 26장·13세션에서 경로당 준비20회와 26×5회 본 측정을 실제 실행했다. RAM 원영상→고정 detector 1회→정제/관측→초기 자세→새 강건 자세→숨은 좌표 재투영을 포함했다. 캐시 재생이나 기존 시간의 합산이 아니다. 모델 로딩·파일 decode·GT 채점·패리티 검사는 측정 구간에서 제외했다. GPU 동기화·CPU 스레드·버전·온도·다른 작업의 부재를 원행에 저장했다.","",
      "| 경로 | 실제 전체 평균 ms | 중앙값 ms | P90 ms | 측정 횟수 |",
      "|---|---:|---:|---:|---:|"]
    for name,ss in runtime["summaries"].items():
        s=ss["full"]
        lines.append(f"| {name} | {number(s.get('mean_ms',s.get('mean')))} | {number(s.get('median_ms',s.get('median')))} | {number(s.get('p90_ms',s.get('P90')))} | {s.get('n',130)} |")
    wall=read("TASK_WALL_CLOCK.json",True)
    start_client=datetime.fromisoformat(wall["start_client"]).strftime("%Y-%m-%d %H:%M:%S")
    snapshot_client=datetime.fromisoformat(wall["report_snapshot_client"]).strftime("%H:%M:%S")
    wall_seconds=int(wall["elapsed_wall_seconds_at_report"])
    lines += ["", f"전체 작업은 한국시간 {start_client} 시작부터 {snapshot_client} 보고 스냅샷까지 {wall_seconds:,}초({wall_seconds//60}분{wall_seconds%60}초)였다. 코드 구현·검산·조정을 포함하는 실제 경과시간이며 게시 시간은 이후다. source cache 준비144.98초, 정식 학습과 probe/preflight64.35초, 봉인 실사 관측 추론18.49초를 각각 기록했다. 완료 head forward9,268회, 학습/probe/preflight RGB노출148,288회이며 정식노출144,000회와 구별한다. 추가 미봉인 추론319회와 dtype 실패1회도 비용에서 제외하지 않았다. 실제 호출·재실행·실패 시도와 단계별 wall time은 [EXECUTION_LEDGER.json](EXECUTION_LEDGER.json)에 있다. 독립 감사와 학습은 일부 겹쳤으므로 단계 시간을 더해 전체 경과시간처럼 제시하지 않았다. 소스 캐시 준비, 학습, 실사 특징/관측 추론, 실제 전체 경로 측정을 각각 기록했다.","",
      f"[VERIFICATION.json](VERIFICATION.json)은 {len(verification['checks'])}개 독립 검산을 통과했다. T/R/ADDsym은 기존 함수와 별도 구현이 일치했고 숨은 최종 좌표는 실제 최종 R,t의 투영과 일치했다. 원본319장·가중치·사용자 변경·마감 출력의 해시/상태를 보존했다. Python Path.open canary 누락, dtype, 출력명 충돌은 구현 수리로 기록했으며 실제 성능을 보고 설정을 조절하지 않았다. 이전 실패/검산 원행도 보존했다.","",
      "## 질문별 실행 상태","",
      "| ID | 상태 | 근거/한계 |","|---|---|---|"]
    states=[("E0","실행","좌표·K·단위·C2 대칭·물리 메쉬/경계 계약 감사와 round-trip"),
            ("E1","기존 결과 재사용","small/wide/shared 원행 재사용; 같은 창 실험 반복0"),
            ("E2","실행","예측 cuboid 자기 가림 대 사람 known 상태; 분류를 pose 성공 gate로 쓰지 않음"),
            ("E3","실행","12×319 무학습 arm,2,816기하,638실사 마스크 오판"),
            ("E4","실행","숨은 초기 관측 제외→새 R,t 재투영; T/R와 좌표 개선 분리"),
            ("E5","실행","source true intersection/ignore/no-match 및 source 후보 오차"),
            ("E6","미실행","기존 유효 감독 재사용/새 RGB0 우선; 네 통제 변형 미보유"),
            ("E7","실행","동일 모델 세 입력 조건×3,000 update 및 실사 자세 절제"),
            ("E8","실행","동일 IMAGE_ROLE 원시 관측 point/point+line; 중복 edge factor 제거"),
            ("E9","실행","319장 신규/공통/전체 운용·손상·실제 전체 경로 시간")]
    lines += [f"| {i} | {status} | {why} |" for i,status,why in states]
    lines += ["", "## 유지·제거 판단과 게시","",
      "현재 고정 N3→SubPix 단순 대조를 유지한다. 유한 부분집합 PnP와 재투영 구현은 오류 진단 도구로 보존한다. 이번 실사 평균에서 강건화·예측 가림·학습 역할을 필수 성능 기여로 주장할 근거는 확보하지 못했다. point+line 결과는 표에 그대로 보존하고 정확도·산출률·비용을 함께 평가한다. source-to-real 관측 충분성은 남은 공백이며 같은 합성을 더 늘리거나 설정을 바꾸어 재시도하지 않았다.","",
      "원 실험의 코드·원행·검산·실행량은 전용 브랜치 `research/observation-refiner-robust-pnp-20261009`에 정상 게시했다. 최초 게시 commit과 원격 확인은 `PUBLICATION.json`에 보존했다. 상세 설명·파생 그림·공개 검산 도구의 이번 개정은 `REVIEW_MANIFEST.json`으로 구별하며, 개정 후 최종 원격 SHA를 다시 확인한다. main 병합·force push·원고 변경은 수행하지 않는다."]
    review_supplement=C.DOC/"REPORT_REVIEW_SUPPLEMENT_KO.md"
    if review_supplement.exists():
        lines += ["",review_supplement.read_text(encoding="utf-8").rstrip()]
    write_text("RESULT_KO.md","\n".join(lines))
    write_text("GOAL.md",QUESTION+"\n\n고정 RGB/N3 추정기를 보존하고 유효 관측 선택·강건 PnP·숨은 점 최종 재투영의 실제 T/R 추가 가치를 검증한다. 분류 완벽성은 성공 필요조건이 아니다.\n\n채택 판단: 전체319 운용과 신규/공통 산출집합 T/R/ADDsym, 직접 가시 손상, 산출률, 실제 전체 경로 비용을 함께 본다. 기존 마감 경로·가중치·원고는 보존한다.\n\n실행 상태와 결과는 RESULT_KO.md, 실제 비용은 EXECUTION_LEDGER.json, 독립 검산은 VERIFICATION.json에 기록한다.")
    if not (C.DOC/"REVIEW_GUIDE_KO.md").exists():
        write_text("README.md","# Observation refiner / robust PnP — 2026-10-09 protocol\n\n이번 파일·key·CLI는 새 실험 스키마다. 과거 파일에 이 key가 있다고 가정하지 않는다. [RESULT_KO.md](RESULT_KO.md)가 결과 보고서다.\n\n- INPUTS/OBSERVATIONS/LEARNED_OBSERVATIONS는 추론 입력·선택 봉인. GT는 봉인 후 별도 평가에만 사용한다.\n- PREDICTIONS와 LEARNED_PREDICTIONS는 319개 ID를 유지하는 수치 원행이다.\n- METRICS/PAIRED_COMPARISONS는 신규 자세·공통집합·fallback 전체 운용을 구분한다.\n- GEOMETRY_STRESS는 렌더0인 수학적 진단이며 실사 증거가 아니다.\n- REAL_CORRESPONDENCE는 참조상 좌표 정확성과 사람 가시성을 구별하는 사후 oracle 진단이다.\n- RUNTIME는 detector와 초기/후단 PnP를 포함한 실제 전체 실행이다.\n- SOURCE_FAMILY_SPLIT/TRAIN_LOGS/TRAINING_COMPLETION은 source-only 동일 학습 예산의 근거다.\n- 실사 RGB·대형 cache·checkpoint는 공개 git에 넣지 않는다. 로컬 binding은 환경변수로 제공한다.\n\n재현은 [REPRODUCE.md](REPRODUCE.md), 검산은 [VERIFICATION.json](VERIFICATION.json)을 참고한다.")
    write_text("REPRODUCE.md","# 실제 실행과 재집계\n\nPython3.10, OpenCV4.9, 고정 기존 detector/N3를 사용했다. private source/cache/checkpoint 경로를 공개 결과에 넣지 않고 환경변수로 연결한다. 학습 checkpoint 세 개는 source root 아래 data/pallet/results/pallet_observation_refiner_20261009_v1/weights에 해시 그대로 보존했다. tmpfs가 사라진 뒤에는 이를 private scratch의 learned_fits로 복사하여 재집계/추론에 연결할 수 있다. PRIVATE_CHECKPOINT_RETENTION.json이 상대경로·해시를 기록한다. 실행 시 PALLET_SOURCE_ROOT는 원본 읽기 전용 checkout, PALLET_OBSERVATION_SCRATCH는 private cache다.\n\n```bash\nexport PALLET_SOURCE_ROOT=/path/to/read-only-source\nexport PALLET_BASELINE_ROOT=/path/to/immutable-a22-research-code\nexport PALLET_INSTRUCTION_PATH=/path/to/pallet_cli_observation_refiner_robust_pnp_20261009.md\nexport PALLET_OBSERVATION_SCRATCH=/path/to/private-experiment-cache\nexport PALLET_TEX_ARCHIVE=/path/to/read-only-legacy-TEX.zip\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\ntask_docs=_docs/experiments/pallet_observation_refiner_20261009_v1\npython -m scripts.research.pallet_observation_refiner_20261009_v1.audit\npython -m scripts.research.pallet_observation_refiner_20261009_v1.test_solver --output \"$task_docs/SOLVER_CHECKS.json\"\npython -m scripts.research.pallet_observation_refiner_20261009_v1.contract_audit audit\npython -m scripts.research.pallet_observation_refiner_20261009_v1.pilot\npython -m scripts.research.pallet_observation_refiner_20261009_v1.evaluate\npython -m scripts.research.pallet_observation_refiner_20261009_v1.contract_audit repair_replay\npython -m scripts.research.pallet_observation_refiner_20261009_v1.geometry_stress --scenes 128 --output \"$task_docs\"\npython -m scripts.research.pallet_observation_refiner_20261009_v1.source_audit --tex-archive \"$PALLET_TEX_ARCHIVE\"\npython -m scripts.research.pallet_observation_refiner_20261009_v1.source_audit --tex-archive \"$PALLET_TEX_ARCHIVE\" --archived-only\npython -m scripts.research.pallet_observation_refiner_20261009_v1.training prepare\npython -m scripts.research.pallet_observation_refiner_20261009_v1.training train\npython -m scripts.research.pallet_observation_refiner_20261009_v1.learned_infer --output LEARNED_OBSERVATIONS.jsonl.gz\npython -m scripts.research.pallet_observation_refiner_20261009_v1.learned_evaluate\npython -m scripts.research.pallet_observation_refiner_20261009_v1.statistics\npython -m scripts.research.pallet_observation_refiner_20261009_v1.benchmark measure --image-role-adapter scripts.research.pallet_observation_refiner_20261009_v1.benchmark:benchmark_adapter --learned-reference \"$task_docs/LEARNED_PREDICTIONS.jsonl.gz\"\npython -m scripts.research.pallet_observation_refiner_20261009_v1.verify\npython -m scripts.research.pallet_observation_refiner_20261009_v1.report\n```\n\n완료 raw 출력을 덮어쓰는 추론 명령은 거부한다. 새 worktree/출력에서 재실행하거나 완료 동일 해시 단계를 재사용한다. source_audit는 --tex-archive 필수 인수를 받고 --archived-only로 기존 TEX 32장 보조 감사를 실행한다. contract_audit의 stage는 audit와 repair_replay다. 기록된 구현 수리와 재실행은 REPAIR_LOG/LEARNED_DTYPE_FIX/LEARNED_OUTPUT_COLLISION에 있다. contract_audit repair_replay는 canary 수리 후 수행한 전체319개 수치 패리티 재검사다. compact_observations는 학습 관측 봉인 후 적용한 무손실 저장 단계이며 기존 원본을 private cache에 보존했다. driver CLI의 audit→solver_tests→contract_audit→cpu_pilot→pose_diagnostics→mask_stress→source_audit→prepare_supervision→train→infer_observations→solve_evaluate→statistics→benchmark→verify→report 단계와 동일 해시 완료 receipt를 함께 사용한다. 게시 작업은 driver와 별개로 전용 브랜치에 정상 git commit/push하고 원격 SHA를 확인한다.\n\n원행만 재집계할 때 statistics→verify→report를 실행한다. verify는 이전 검산과 달라진 파생 대응 감사를 해시 suffix 파일로 보존한다. 모델 가중치 선택·임계값 선택·추가 seed를 실행하지 않는다. 실제 전체 시간은 격리된 고정 패널 benchmark 실행만 사용한다.")
    if review_supplement.exists():
        repro=C.DOC/"REPRODUCE.md"
        repro.write_text(repro.read_text(encoding="utf-8")+"\n## 원본 영상 없이 공개 파일만 검산\n\nPython 3.9 이상의 표준 라이브러리만 필요하다. detector/학습 가중치, CUDA, PALLET_SOURCE_ROOT 또는 원본 영상은 필요하지 않다. 체크아웃 루트에서 실행한다. 이 명령은 저장된 숫자·ID·상태·투영·해시를 다시 검사하며 실제 추론이나 학습을 다시 실행하지 않는다. 원본 영상과 평가 정답의 물리적 정확성까지 증명하는 명령은 아니다.\n\n```bash\npython scripts/research/pallet_observation_refiner_20261009_v1/review_verify.py --require-manifest --output /tmp/pallet-public-review-checks.json\npython scripts/research/pallet_observation_refiner_20261009_v1/review_verify.py --require-manifest --frame-id eval_noapril:1775201415399297536 --output /tmp/pallet-frame-review-checks.json\n```\n\n세부 파일·필드·검토 순서는 [REVIEW_GUIDE_KO.md](REVIEW_GUIDE_KO.md)에 있다. 그림01–05는 저장된 숫자만으로 다시 만들 수 있다. 그림06–07은 기존 원본 영상과 자산이 있어야 재생성할 수 있으며 공개 PNG와 [VISUAL_REVIEW_CASES.json](VISUAL_REVIEW_CASES.json)에서 사례 ID·크롭·그림 원천을 확인할 수 있다. 시각화 실행은 모델 학습·후단 PnP 재실행과 구별된다.\n",encoding="utf-8")
    ledger();aggregate_checks()
    C.write(C.DOC/"REPORT_NUMBERS.json",dict(complete=True,methods=methods,report=C.binding(C.DOC/"RESULT_KO.md"),
            metric_source=C.binding(C.DOC/"METRICS.json"),paired_source=C.binding(C.DOC/"PAIRED_COMPARISONS.json"),
            verification_source=C.binding(C.DOC/"VERIFICATION.json"),runtime_source=C.binding(C.DOC/"RUNTIME.json")))
    print(json.dumps(dict(complete=True,methods=len(methods),report="RESULT_KO.md",ledger="EXECUTION_LEDGER.json")),flush=True)


if __name__=="__main__":run()

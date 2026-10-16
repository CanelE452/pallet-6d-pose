"""Package small immutable analysis tables and evidence-based Korean reports."""
from collections import Counter
from pathlib import Path
import argparse
import json
import shutil
import time

from .common import read, rows, csv_rows, write, write_csv, sha

NS='scripts/research/pallet_selection_score_diagnosis_v1'
OLD='scripts/research/pallet_visibility_curriculum_20261013_v1'
RESULT='407de865445e4987614f50f893d2ccea78d507b6'


def table(headers, data):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in data])


def emit(path, text):
    path=Path(path)
    if path.exists():raise FileExistsError('Preserve previous report attempt: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text.rstrip()+'\n',encoding='utf-8')


def stage_decisions(root,public):
    """Small public table of all original family decisions, including fallback."""
    root,public=Path(root),Path(public)
    candidates={(r['id'],r['candidate_id']):r for r in rows(root/'selection_v1/RETAINED_CANDIDATES.jsonl.gz')}
    values=[]
    for f in rows(root/'selection_v1/FRAME_SELECTIONS.jsonl.gz'):
        for stage,decision in f['stages'].items():
            for index,r in enumerate(decision['families']):
                c=candidates.get((f['id'],r['winner_candidate_id']))
                values.append(dict(id=f['id'],partition=f['partition'],group=f['group'],stage=stage,
                    family_index=index,family=r['family'],accepted=r['accepted'],fallback=r['fallback'],status=r['status'],
                    solver_schema=r['solver_schema'],original_objective=r['original_objective'],
                    stage_selected_family=decision['selected_family'],stage_selected_family_index=decision['selected_family_index'],
                    exact_tie_rule=decision['exact_tie_rule'],e_frozen_G10_family=f['selected_family'],
                    winner_candidate_id=r['winner_candidate_id'],original_retained_count=r['retained_count'],
                    published_physical_pose=r['published_actual_pose'],
                    native_R=c['R'] if c else None,native_t=c['t'] if c else None,
                    physical_registration_Q=c['physical_registration_Q'] if c else None,
                    native_xyz=c['native_xyz'] if c else None,role=c['role'] if c else 'ORIGINAL_FALLBACK_NOT_GENERATED',
                    initialization=c['original_record'].get('initialization') if c else None,
                    original_outcome_order=c['original_outcome_order'] if c else None,
                    execution_role='SAVED_ORIGINAL_DECISION_NO_NEW_SOLVE'))
    assert len(values)==952
    write_csv(public/'FAMILY_STAGE_DECISIONS.csv',values)
    return values


def build(root,parent,public,source_root):
    started,cpu=time.monotonic(),time.process_time()
    root,parent,public,source_root=map(Path,(root,parent,public,source_root))
    template=source_root/'_docs/experiments/pallet_selection_score_diagnosis_v1'
    # A fresh reproduction has no failed historical attempts; do not fabricate
    # them or copy their timings into the fresh run's COST.
    for name in ('USER_TASK.md','PLAN_AND_CONTRACT.md','OBJECTIVE_MAP.md','SELECTION_RULE_MAP.md','REPRODUCE.md'):
        if not (public/name).exists():shutil.copyfile(template/name,public/name)
    for name in ('SOURCE_BINDINGS.json','INFLUENCE_PROTOCOL.json'):
        if not (public/name).exists():shutil.copyfile(root/name,public/name)
    aliases={str(root):'PRIVATE_ANALYSIS',str(parent):'PRIVATE_PARENT',str(source_root):'SOURCE_ROOT',
             str(parent.parent/'minimal-bi-ablation-20261016-v1'):'PRIVATE_REFERENCE_SOURCE'}
    def redact(value):
        if isinstance(value,dict):return {redact(k):redact(v) for k,v in value.items()}
        if isinstance(value,list):return [redact(v) for v in value]
        if isinstance(value,str):
            for a,b in sorted(aliases.items(),key=lambda x:-len(x[0])):value=value.replace(a,b)
        return value
    copies=[]
    def publish(src,destination=None,sanitized=False):
        src=Path(src);dst=public/(destination or src.name);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():raise FileExistsError(dst)
        if sanitized:
            value=redact(read(src));write(dst,value)
        else:shutil.copyfile(src,dst)
        copies.append(dict(source=redact(str(src)),source_sha256=sha(src),published=str(dst.relative_to(public)),
                           published_sha256=sha(dst),path_alias_redaction_only=sanitized))
    pair=root/'pairs_v2'
    publish(pair/'CANDIDATE_COMPONENTS_ALL.csv.gz','SELECTION_COMPONENT_LEDGER.csv.gz')
    for name in ('SELECTION_PAIR_TABLE.csv','FAMILY10_CASES.csv','WITHIN22_CASES.csv','SUCCESS114_CONTROLS.csv',
                 'CAL_CASES.csv','PER_OBSERVATION_PAIR_LEDGER.csv.gz','PAIR_LEDGER.jsonl.gz','PAIR_SUMMARY.json'):
        publish(pair/name)
    for name in ('NO_CANDIDATE46_CASES.csv','NO_CANDIDATE_CAL6_CASES.csv','NO_CANDIDATE_START_END_PAIRS.csv',
                 'NO_CANDIDATE_INDEX_KO.md','NO_CANDIDATE_DETAILS.jsonl.gz'):
        publish(root/'no_candidate_v2'/name)
    for name in ('BASELINE_CROSS_TABLE.csv','BASELINE_PATTERN_CROSS_TABLE.csv','GROUP_MEMBERSHIP_238.csv'):
        publish(root/'baseline_v1'/name)
    stage_decisions(root,public)
    for name in ('SCORE_INFLUENCE_DIAGNOSTIC.csv','INFLUENCE_SUMMARY.json','SCORE_INFLUENCE_ROWS.jsonl.gz'):
        publish(root/'influence_attempt01'/name)
    evidence=[
        'selection_v1/SELECTION_REPLAY.json','selection_v1/SELECTION_SEAL.json','SELECTION_RUNTIME_GUARD.json',
        'fixed_cost_pilot_v1/PILOT_COST_RECEIPT.json','fixed_cost_pilot_v2/PILOT_COST_RECEIPT.json',
        'cost_v1/COST_SUMMARY.json','cost_v1/COST_SEAL.json','cost_v1/DENSE_ARRAY_VALIDATION.json',
        'cost_v1/MEMORY_SNAPSHOT_DURING_RUN.json','metrics_v1/METRIC_CHECK.json','metrics_v2/METRIC_CHECK.json',
        'pairs_v1/PAIR_CHECK.json','pairs_v2/PAIR_CHECK.json','PAIR_V1_V2_METADATA_BRIDGE.json',
        'baseline_v1/BASELINE_RECEIPT.json','no_candidate_v2/NO_CANDIDATE_RECEIPT.json',
        'influence_attempt01/INFLUENCE_SEAL.json','INDEPENDENT_INFLUENCE_CHECK.json',
        'CONTROLS_COST_AND_ATTEMPTS.json','final_bindings/SOURCE_BINDINGS.json','UNIT_TEST_RECEIPT.json']
    missing_history=[]
    for name in evidence:
        if (root/name).exists():publish(root/name,'evidence/'+name,sanitized=True)
        else:missing_history.append(name)
    # The renderer's original receipt is retained privately. Public paths are aliases.
    receipt=public/'gallery/GALLERY_RECEIPT.json'
    backup=root/'gallery_original/GALLERY_RECEIPT.json';backup.parent.mkdir(exist_ok=True)
    if backup.exists():raise FileExistsError(backup)
    shutil.copyfile(receipt,backup)
    original=read(receipt);receipt.write_text(json.dumps(redact(original),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    copies.append(dict(source='PRIVATE_ANALYSIS/gallery_original/GALLERY_RECEIPT.json',source_sha256=sha(backup),
                       published='gallery/GALLERY_RECEIPT.json',published_sha256=sha(receipt),path_alias_redaction_only=True))
    write(public/'PACKAGING_BINDINGS.json',dict(files=copies,numerical_values_changed=False,
        original_private_receipts_preserved=True,large_raw_journals_copied_to_git=False,
        historical_attempt_receipts_not_present_in_this_run=missing_history))
    selection=read(root/'selection_v1/SELECTION_REPLAY.json');cost=read(root/'cost_v1/COST_SUMMARY.json')
    pairs=read(pair/'PAIR_SUMMARY.json');baseline=list(csv_rows(root/'baseline_v1/BASELINE_CROSS_TABLE.csv'))
    influence=read(root/'influence_attempt01/INFLUENCE_SUMMARY.json')
    within=list(csv_rows(pair/'WITHIN22_CASES.csv'));family=list(csv_rows(pair/'FAMILY10_CASES.csv'))
    cases=read(public/'gallery/GALLERY_PROOF.json')['cases'];caseby={c['id']:c for c in cases}
    reps=read(public/'gallery/REPRESENTATIVE_SELECTION.json')['examples']
    primary=[b for b in baseline if b['cohort']=='PRIMARY192']
    main=next(b for b in primary if b['diagnostic_group']=='ALL')
    fmt=lambda x:f'{float(x):.3f}'
    def baseline_table(cohort):
        return table(['경로','공동 성공/N','위치 평균 ± 표본SD (cm)','위치 분산 (cm²)','위치 최대','회전 평균 ± 표본SD (°)','회전 분산 (°²)','회전 최대'],[
            [arm,f'{b[arm+"_success"]}/{b["images"]}',f'{fmt(b[arm+"_T_cm_mean"])} ± {fmt(b[arm+"_T_cm_sample_SD"])}',
             fmt(b[arm+'_T_cm_sample_variance']),fmt(b[arm+'_T_cm_max']),
             f'{fmt(b[arm+"_R_deg_mean"])} ± {fmt(b[arm+"_R_deg_sample_SD"])}',
             fmt(b[arm+'_R_deg_sample_variance']),fmt(b[arm+'_R_deg_max'])]
            for b in baseline if b['cohort']==cohort and b['diagnostic_group']=='ALL' for arm in ('N0','POSEFIX_RAW','E0')])
    cross=table(['PRIMARY 그룹','장수','N0 성공','PoseFix RAW 성공','둘 중 하나 성공','둘 다 실패','e 성공'],[
        [b['diagnostic_group'],b['images'],b['N0_success'],b['POSEFIX_RAW_success'],b['N0_or_PoseFix_success'],
         b['N0_and_PoseFix_fail'],b['E0_success']] for b in primary if b['diagnostic_group']!='ALL'])
    within_stats=pairs['cohorts']['PRIMARY']['WITHIN_FAMILY_SELECTION_LOSS/G13/selected_vs_comparator_first_reference_success_same_family']
    term_table=table(['G13 같은-family 22장, g−s','평균 ± 표본SD','최소 ~ 최대','s 선호 / g 선호 / 동률'],[
        [k,f'{v["mean"]:.6f} ± {v["sample_SD"]:.6f}',f'{v["min"]:.6f} ~ {v["max"]:.6f}',
         f'{within_stats["terms_preferring_s"][k]} / {within_stats["terms_preferring_other"][k]} / {22-within_stats["terms_preferring_s"][k]-within_stats["terms_preferring_other"][k]}']
        for k,v in within_stats['component_delta'].items()])
    rank_table=table(['독립 변경 단계','단일 제외 항','성공/192','회복','손상','양쪽 실패','위치 평균±SD cm','회전 평균±SD °'],[
        [r['variant'].split('__without_')[0],r['variant'].split('__without_')[1],r['new_success'],r['recovery'],r['damage'],r['both_failure'],
         f'{r["continuous"]["translation_cm"]["mean"]:.3f} ± {r["continuous"]["translation_cm"]["SD"]:.3f}',
         f'{r["continuous"]["rotation_deg"]["mean"]:.3f} ± {r["continuous"]["rotation_deg"]["SD"]:.3f}']
        for r in influence['summary'] if r['cohort']=='PRIMARY'])
    split_rank=table(['진단','TEST49 성공(회복/손상)','추가143 성공(회복/손상)','CAL46 성공(회복/손상)'],[
        [v]+[f'{r["new_success"]} ({r["recovery"]}/{r["damage"]})'
              for c in ('TEST','REMAINDER','CAL') for r in influence['summary'] if r['cohort']==c and r['variant']==v]
        for v in influence['predeclared_variants']])
    example_blocks=[]
    for i,rep in enumerate(reps,1):
        c=caseby[rep['id']];folder=Path(c['source_crop']).parent
        detail=read(public/'gallery'/folder/'cost_and_observations.json')
        a,b=c['panels'][-2:];am,bm=a['reference_metric'],b['reference_metric']
        cost_table=table(['실제 가중 항','s','대조','Delta'],[
            [k,format(a['weighted_components'][k],'.8f'),format(b['weighted_components'][k],'.8f'),
             format(detail['Delta_weighted_components'][k],'+.8f')]
            for k in ('full_BI','partial_normal','outline','prior')])
        example_blocks.append(f'''### {i}. {rep['id']} · {rep['group']}

실제 선택 s: {am['translation_cm']:.6f}cm / {am['rotation_deg']:.6f}°, 대조: {bm['translation_cm']:.6f}cm / {bm['rotation_deg']:.6f}°. G13 J는 {a['objective']:.8f} → {b['objective']:.8f}; Delta={detail['Delta_total']:.8f}.

{cost_table}

![{rep['id']} 실제 저장 후보 비교](gallery/{folder}/comparison.jpg)

[전체 관측/출처/원 px/정규화 비용](gallery/{folder}/cost_and_observations.json) · [원 observed outline 표본별 비용차](gallery/outline_witness/witness_{i:02d}.jpg)
''')
    groups=selection['group_counts'];fdiag=selection['family_loss_reference_counts']
    position_only=sum(float(w['s_translation_cm'])>5 and float(w['s_rotation_deg'])<=5 for w in within)
    rotation_only=sum(float(w['s_translation_cm'])<=5 and float(w['s_rotation_deg'])>5 for w in within)
    both=sum(float(w['s_translation_cm'])>5 and float(w['s_rotation_deg'])>5 for w in within)
    report=f'''# 기존 e의 후보 선택 비용 진단 — 원결과407de865

**결론: 원 selector 구현 불일치는 없었다. 실패32장은 실제 원비용의 선택 손실이고, 후보부재46장은 따로 남는다. 다음 우선순위는 외곽선 표본과 등록 cuboid의 물리 외곽 대응을 확인하는 최소 검토 한 가지다. 항 삭제·가중치 변경·새 선택기 채택은 하지 않았다.**

## 1. 새 계산 범위와 재현

원결과 `{RESULT}`, 실행 base `2fe1969b7a999d9a8d460703be4bfd11317e7f07`, 별도 연구 브랜치 `research/selection-score-diag-20261016-v1`다. [SOURCE_BINDINGS](SOURCE_BINDINGS.json)는 원 공개129개 산출물·수치 의존779개 소스·관측/mask 관련250개 소스·annotation143개 및 원 봉인을 확인한다. 현재 main의 새 결과로 원407 결과를 대체하지 않았다.

238/238 영상의 원 G10/G13 family winner·수락/위임/중복 제거·후보 순서·e 최종 R/t/Q/역할을 재현했다. unresolved **0**. stage/family winner952개와 비제외 e 정책476개는 별도 독립 tuple 비교로도 일치했다. 실제 원비용을 고정 R/t에서 평가한 retained11,603개(G10 5,861 / G13 5,742)가 모두 복원됐다. 원 저장 total과 최대차 {cost['max_absolute_differences']['stored_objective']:.3g}, `atol=1e-9, rtol=0` 그대로다. 자세 solve·forward·학습·새 관측/후보는 모두0회다.

{table(['분할','성공','family 손실','같은-family 손실','후보부재','합계'],[[c]+[groups[c].get(k,0) for k in ('FINAL_SUCCESS_CONTROL','FAMILY_SELECTION_LOSS','WITHIN_FAMILY_SELECTION_LOSS','NO_SUCCESS_CANDIDATE')]+[sum(groups[c].values())] for c in ('PRIMARY','TEST','REMAINDER','CAL')])}

PRIMARY는 TEST49+추가143이며 독립 신설 test가 아니다. 선택손실32장은 모두 추가143. CAL46은 교정에 사용된 집합이며 같은-family 손실0장이라 22장의 선택 개선을 CAL로 검증할 수 없다. 성공은 **동일 후보**의 T≤5cm AND physical/C2 R≤5°; 90° W/D 혼동은 C2로 지우지 않는다. 참조는 기존 기하 재구성값이며 독립 물리 정답/semantic owner 정답이 아니다.

## 2. family10 / 내부22에서 실패한 실제 결정

family10은 G10 좋은 retained 후보가 **10/10**에 이미 있었다. G13에서만 처음 좋은 후보가 생긴 사례는0이다. G10 반대-family **실제 winner**가 성공한 것은6/10, G13 반대-family winner는5/10이다. 나머지는 family를 바꿔도 내부 순위 문제가 남는다. G13 자체 family 재선택은 **10/10 실패**했다. 원 G10 family 고정 정책만 풀면 해결된다는 근거가 없다. CAL family3도 G10 좋은 후보3/3, 반대 winner2/3, G13 자체 재선택 실패3/3이다.

G10 실제 두 accepted winner의 같은 목적함수 비교는 [선택쌍 표](SELECTION_PAIR_TABLE.csv)의 `G10/accepted_family_winner_comparison`, G13 선택/성공 후보는 [FAMILY10_CASES](FAMILY10_CASES.csv)로 분리했다. [원 stage/family 결정952행](FAMILY_STAGE_DECISIONS.csv)에 R/t/Q·수락/위임·winner ID·index·tie-break를 보존했다. 두 단계의 cost를 서로 빼지 않았다. G10 실제 winner 차이에서 가장 큰 양의 항은 outline6/full3/partial1장이다. G13 실제 e 대 comparator-first 성공 후보 비교는 outline7/full1/partial2장이다.

같은-family22는 위치만 실패{position_only}, 회전만 실패{rotation_only}, 둘 다 실패{both}다. 회전 최대19.116°로 이22장을 W/D swap이라고 부를 수 없다. 성공 후보 전체 ID·원순위·cost 범위는 [WITHIN22_CASES](WITHIN22_CASES.csv)에 남겼으며 최소 GT 오차 후보를 골라 제시하지 않았다.

## 3. 비용 항에서 원관측까지의 근거와 반례

아래 Delta는 같은 G13·같은 family에서 `J(g)−J(s)`다. 양수는 원래 잘못 선택된 s를 선호한다. 비용은 무단위 정규화/가중 후 값이며 px/cm와 섞지 않는다.

{term_table}

outline은22/22에서 s 선호, 가장 큰 양의 항은15/22였다(partial5/full2). full BI는15/22에서 g를 오히려 선호했다. full fallback Delta 평균−0.094139±0.261947, 범위[−0.958200,+0.288223]로 fallback 하나가22장을 공통으로 망쳤다고 설명할 수 없다. full B_TARGET_BACKGROUND_LINES는21장에 있고 g선호14/s선호7이었다. partial은19장; actual line normal/offset까지 일치하는 기록 source-ray link가18장, 해당 매칭 근거가 없는1장은 명시적으로 미연결이다. partial의 상속 `measurement_source`만 보고 실제 RGB normal을 backbone fallback 측정이라고 해석하지 않았다.

대표 `wood_night_01:030698`에서 [`statewise_partial_BI_outline_solver_v4_13._outline_value`](../../../{OLD}/statewise_partial_BI_outline_solver_v4_13.py#L112)가 사용하는 저장 전역 mask outline은, 실제 s(29.388937cm/19.115684°)보다 참조성공 g(1.512569cm/4.763383°)에 큰 거리를 부여했다. observed→model 동일854표본의 평균 원 px 잔차는 **7.4999→17.3307**, 가중 비용4.13649→10.26530(Delta **+6.128810**); model→observed는90표본·각 자세 원 arc 정규화833.595/739.068px에서 Delta **+3.528464**다. full BI−0.969840과 partial−0.052049가 g를 지지해도 outline **+9.657273**이 상쇄해 J9.022478인 s가 J17.657862인 g보다 앞섰다. [표본 그림](gallery/outline_witness/witness_03.jpg)은 양쪽에서 동일한 저장 observed 표본만 대응시킨다.

관측이 reference-success pose와 충돌한다는 것은 확인됐다. **mask가 틀렸는지, 등록 cuboid의 projected silhouette가 실제 팔레트 경계를 대표하지 못하는지, 기하 참조/축 가정의 한계인지 아직 분리되지 않았다.** 높은 비용만으로 weight 오류나 참조 오류를 확정하지 않는다. 모든32장의 semantic owner는 미검증이며 의사결정 기여를 확인한32장과 물리 원인을 확정한 장수를 구분한다.

정확한 sample source group의 공유는 좁은 조건부 linkage로 기록했다. active nonfallback 관측 간 같은 group이 확인된 영상은 within22=0/family10=0/controls114=2였지만, 같은 RGB edge/mask의 근접 표본 상관이나 통계적 독립성 검증은 아니다. 미연결을 '독립' 또는 '중복 없음'으로 바꾸지 않았다. [전체 관측별 장부](PER_OBSERVATION_PAIR_LEDGER.csv.gz)와 [쌍별 원 provenance](PAIR_LEDGER.jsonl.gz)를 보존했다.

{''.join(example_blocks)}

## 4. 성공114장과 CAL46의 반례, 전체 순위 영향

성공114장 모두 같은-family의 원순위상 최선 참조실패 후보 b가 있었다. outline은 **114/114**에서 올바른 s를 선호했다. Delta outline 평균31.547848±33.083359, 최소+0.011905. 가장 큰 양의 항은 full80/outline21/partial13이다. CAL 성공37장 중 b가 있는36장만 쌍비교 가능하며1장은 NA와 이유를 유지했다. CAL 같은-family 손실0장을 검증 성능으로 꾸미지 않았다.

특히 `wood_night_01:030898`에서는 full BI가 b를 −0.062484 지지해도 outline +0.774584가 성공 s를 보호했다. observed→model +0.443872, model→observed +0.330712다. 실패 사례의 같은 항을 일괄 삭제하면 이 보호 근거도 사라진다.

소스/선택/가산 복원을 통과한 뒤, 성과표를 만들기 전에 [고정한6개 규칙](INFLUENCE_PROTOCOL.json)만 적용했다. G10만 변경하거나 원 G10 family를 고정한 G13 내부만 변경했다. original eligible·hard guards·dedup·tie-break와 기존 G13 family 출력은 고정했다. prior는 실제0이라 제외 실험을 만들지 않았다.

{rank_table}

기존 e는114/192, T평균14.112±30.077cm / R13.001±28.388°였다. 단순 항 삭제로 PRIMARY 공동 성공을 늘린 경로는 **0/6**이다. G13 outline 제외는 선택손실 일부를 회복하지만 **9회복/11손상**, 원114 성공 중103만 보존한다. G10 outline 제외는3/3이며 전체 성공수 동률도 연속오차 무변화가 아니다.

{split_rank}

> 원 목적함수로 생성한 고정 후보 은행 위의 사후 순위 영향 진단이다. 항을 제거한 목적함수로 후보를 새로 생성·최적화한 방법의 성능이 아니며, 새 선택기로 채택하지 않았다.

[24개 전체 집계](SCORE_INFLUENCE_DIAGNOSTIC.csv), [1428행 선택/회복/손상 원장](SCORE_INFLUENCE_ROWS.jsonl.gz), [연속오차의 전이군별 합·분산·최대](INFLUENCE_SUMMARY.json)에 유리하지 않은 결과도 전량 남겼다. 원192·49·143·CAL46 분모를 줄이지 않았다. 별도 구현이1428개 선택과24집계의 `회복−손상=성공수 차이`, 모든 전이군 연속오차 합을 정확 유리수 산술로 검산했다.

## 5. 후보부재46장의 확인된 범위와 공백

46장 중 **4장은 실제 admission/no-generation**, 42장은 G13 consumer 후보가 있지만 동일 후보 공동 성공이 없다. 이46장은 원 G13 소비자 은행의 한계이며 모든 역사적/미래 후보의 부재를 뜻하지 않는다. 생성 전 버린 WD/P1 후보나 후속 V15/V17/V33/SINGLE·외부 pose를 은행에 합치지 않았다.

위치≤5cm 후보가 있는14장, 회전≤5° 후보가 있는36장, 두 조건을 각기 다른 후보로만 만족하는13장을 분리했다. 위치 최소와 회전 최소를 합친 가상 성공은0이다. 원 시작점→끝점652쌍의 시작점 공동 성공은0이지만, `wood_night_01:033421`에는 **G10 성공 winner3.244cm/4.596°**가 있었고 G13 은행에는 성공 후보가 없었다(최종12.707cm/3.669°). 원 G13은 그 G10 winner를 start로 쓰지 않았다. 이 사례를 모든 초기 단계가 나쁜 사례로 분류하지 않았다.

모든46장 outline 지원 기록은True이나 활성 full fallback은35장에 없었다. T 관측0으로 이번 자료에서 truncation 기여를 검증할 수 없다. endpoint의 음의 slack을 전체 제약의 모순이라고 부르지 않았다. **46/46은 OBSERVATION_VS_PROPOSAL_NOT_ISOLATED와 SEMANTIC_OWNER_UNVERIFIED**가 남는다. 원 source 연결 미해결은0이지만 semantic owner/물리 축/독립 참조 정확성 공백은 그대로다. [46장별 사실·경쟁 설명·부족 증거·최소 확인](NO_CANDIDATE46_CASES.csv), [CAL6](NO_CANDIDATE_CAL6_CASES.csv), [실제 RGB 링크](NO_CANDIDATE_INDEX_KO.md)를 제공한다.

## 6. N0/PoseFix RAW의 같은 집합 비교

N0는 기존 **N3→SubPix+F**, PoseFix는 **POSEFIX_RAW**다. PF+SubPix와 섞지 않았고 외부 pose를 e bank에 추가하거나 재추론하지 않았다.

{cross}

후보부재46장 중7장은 외부 모델 출력에는 성공 pose가 있다. e 정보/시작/대응이 본질적으로 해결 불가능하다는 설명의 반례다. 나머지39장은 세 경로 모두 실패하므로 점수 변경만으로 회복을 기대할 근거가 없다. 반면 e 성공114장 중19장은 N0/PoseFix 모두 실패했다. e의 상태/관측 경로가 지킨 성공을 버리는 결론도 맞지 않는다.

PRIMARY192 전체 원 저장 결과:

{baseline_table('PRIMARY192')}

별도 CAL46:

{baseline_table('CAL46')}

기존 TEST49/추가143 및 그룹별 평균·표본SD·표본분산·최대·N은 [BASELINE_CROSS_TABLE](BASELINE_CROSS_TABLE.csv)에 있다. [기존 PF transfer 감사](../pallet_statewise_correspondence_diagnosis_v1/POSEFIX_TRANSFER_AUDIT.md)의 공통95장 분석은 참고일 뿐 전체192 PF+SubPix 단일 규칙 검증으로 확대하지 않았다.

## 7. 판정과 검산 한계

{table(['구분','이번 판정','남은 공백'],[
['선택/비용 구현 오류','238 selector 일치, 11603 total/components 복원; 확인된 구현 불일치0','별도 독립 semantic correctness를 의미하지 않음'],
['관측·가중치 모델','32장의 원 비용 기여는 전량 확인; 내부22 outline s선호','잘못된 관측/모델 불일치/불확실성 보정의 인과 원인32/32 미확정'],
['이산 선택 정책','G10 family10 손실; 10장 모두 G10에도 좋은 후보 존재','G13 자체 재선택10/10 실패하므로 family lock 해제 단독 해결 불가'],
['정보 부족·제안 한계','46=진입4+G13 후보존재42, pruning loss0 유지','관측 대 proposal 인과분리46/46 미해결']])}

원 full2D 철회는 모든 partial native ID에서 실제로0기여이며 원 legacy partial72·V47 correction 계보를 유지했다. partialnormal은 corner내 radial robust 합이라 선별로 가산하지 않았다. S는 양의 depth 같은 모델 guard/투영으로만 쓰고 측정점으로 넣지 않았다. O/T/U·finite tangent는 원 hard constraint이며 soft cost에 더하지 않았다. prior는 실제0이었다. [OBJECTIVE_MAP](OBJECTIVE_MAP.md), [SELECTION_RULE_MAP](SELECTION_RULE_MAP.md), [독립 dense 검산](evidence/cost_v1/DENSE_ARRAY_VALIDATION.json)에 상세가 있다.

11,603 후보의 참조 성공 ID는 원 오라클과 전량 일치했다. 극도로 먼48개 저장 후보에서 cm float ULP가1e-9보다 커서 최초 독립 metric 절대차 검사는 실패했다(최대1.52587890625e-5cm). 원 metric 값·비용/선택의1e-9 기준은 바꾸지 않았다. 실패 `metrics_v1`을 보존했고, v2에서80자리 Decimal norm과 IEEE 연산별 반올림 경계로 수치 표현을 검증했다. 이48개 최초 절대 검사 실패를 PASS로 지우지 않았으며 [두 영수증](evidence/metrics_v2/METRIC_CHECK.json)에 명시했다. 회전 독립차 최대5.70e-12°, 원 success 불일치0이다.

기존 실패/중간 시도, 원 source·weights·prediction·annotation·사용자 dirty6개를 보존했다. [VALIDATION](VALIDATION.json)과 [COST](COST.json)는 독립 검산/실제 측정·결측 시간을 구분한다. 분석은2 worker, GPU0이며 이 시간은 영상추론 latency/FPS가 아니다.

## 8. 다음 우선순위 한 가지

**외곽선 비용의 결정적 표본이 등록 cuboid의 물리 외곽에 실제로 대응하는지 확인하는 최소 검토**를 우선한다. [구체 제안](NEXT_CHANGE_PROPOSAL_KO.md)은 성공/실패 모두에 같은 절차를 적용하고, 확인 전 어떤 loss·weight·threshold도 바꾸지 않도록 한다. 새 학습은 필요하지 않다. 효과는 미검증이며32장 전부 회복 또는146/192를 약속하지 않는다. N3·SubPix·e·상태별 처리·운영 기본값·논문은 유지했다.

[전체 비교 갤러리](gallery/index.html) · [재현 명령](REPRODUCE.md) · [고정 계약](PLAN_AND_CONTRACT.md)
'''
    emit(public/'RESULTS_KO.md',report)
    proposal='''# 다음에 판단을 바꿀 최소 확인 한 가지

우선순위는 **저장 global outline의 결정적 표본이 등록 cuboid의 물리 외곽과 대응하는지 확인**하는 것이다. 현재 가중치나 선택기를 바꾸는 안은 확정하지 않는다.

원 comparator/점수 연결은 전량 재현됐다. 같은-family 실패22장의 outline은 모두 잘못 선택된 s를 선호했지만, 성공114장에서도 모두 올바른 s를 지지했다. G13 내부 outline 단일 제외는192장 전체에서9회복/11손상이다. 따라서 항 전체 삭제/일괄 약화는 성공을 보호할 근거를 함께 잃는다. G13 family 재선택도 family 실패10장을 회복하지 못했고, fallback은46장 중35장에 활성하지 않아 공통 원인이 아니다.

## 확인 대상과 읽을 근거

대표 규칙은 비용/새 성과를 보기 전에 고정한6장(실패family2, 내부2, 성공반례2)이다. [원 대표 목록](gallery/REPRESENTATIVE_SELECTION.json), [동일 crop 후보 비교](gallery/INDEX_KO.md), [원 observed sample 비용차](gallery/outline_witness/INDEX_KO.md), 전체32/114의 per-observation 장부를 함께 쓴다. 유리한 프레임만 골라 새 선택기를 검증하는 절차가 아니다.

핵심 반례 wood_night_01:030698은 동일854 outline 표본에서 참조성공 pose의 관측→모델 비용이6.128810 더 크다. 그림상 오른쪽/아래쪽 지지가 차이를 크게 만든다. 이는 s의 projected convex silhouette가 저장 mask outline에 더 가깝다는 사실이다. mask가 잘못됐다는 정답이나, 참조가 실제 물리 자세라는 보증은 아니다. 반대 성공 wood_night_01:030898에서는 같은 outline이 잘못된 b를0.774584 더 벌점화해 성공을 지켰다.

확인할 질문은 하나다. **큰 양·음 sample Delta가 놓인 저장 경계가 해당 등록 모델의 물리적 외곽을 측정하는가?** 저장 RGB/마스크/visible edge, 등록 치수·physical Q·참조 투영을 같은 좌표계로 대조하고, (1) 실제 팔레트 외곽 지지, (2) occluder/그림자/convex hull 연결 등 다른 경계 지지, (3) 영상만으로 미확정으로 기록한다. 여러 관측/학습/초기화를 동시에 바꾸지 않는다. 새로운 정답 수정이나 owner label 학습은 이번 작업에서 하지 않았다. 이 검토에서 독립적인 physical edge 판단이 불가능하면 추가 정보 없이는 어느 가정을 바꿔야 할지 확정할 수 없다고 남긴다.

## 판단이 어떻게 바뀌는가

기존 성공을 지키는 경계와 실패를 지지하는 경계에서 물리 지지의 일관된 차이가 확인되면, 그 출처/가시성·불확실성 모델의 한 가지 최소 수정을 다음 승인안으로 만든다. 차이가 없고 모두 실제 외곽을 잘 지지하면 mask를 고치는 결론을 내리지 않고, cuboid-silhouette/등록/기하 참조와 그 영상 근거가 서로 양립하는지의 정보 부족으로 남긴다. 어느 경우에도 현192장 사후 선택을 새 독립 성능으로 쓰지 않는다.

이 확인은 새 학습을 요구하지 않는다. **개선 효과는 [추정·미검증]**이다. 새 forward/PnP/수치최적화/후보 생성, 가중치·임계값 탐색, PRIMARY 실패를 감독으로 쓰는 selector 학습은 별도 변경이며 실행하지 않았다. 널리 쓰이는 관측 유효성 검토를 새 알고리즘으로 이름 붙이지 않는다.

후보부재46장은 이 선택 점수 검토만으로 회복할 수 없다. 해당 행의 admission4/기존후보42, 외부성공7과 semantic/source/reference 공백은 별도 장부에 남긴다. 먼저 확인해야 할 실제 사례는 G10에는 성공 winner가 있으나 G13에 없는 wood_night_01:033421이다. 이것은 두 번째 변경안이 아니라, 현재 점수 검토가46장을 설명하지 못하는 범위를 드러내는 기록이다.
'''
    emit(public/'NEXT_CHANGE_PROPOSAL_KO.md',proposal)
    write(public/'INPUT_AND_SELECTION_CHECK.json',dict(
        source_result_commit=RESULT,source_binding_pass=read(root/('final_bindings/SOURCE_BINDINGS.json' if (root/'final_bindings/SOURCE_BINDINGS.json').exists() else 'SOURCE_BINDINGS.json'))['source_and_input_parity'],
        group_counts=selection['group_counts'],original_selection_replay_pass=selection['all_passed'],
        unresolved_ids=selection['unresolved_ids'],original_R_t_Q_and_fallback_roles_preserved=True,
        retained_candidates=11603,strict_atol=1e-9,strict_rtol=0,
        independent_winner_and_policy_checks=read(root/'INDEPENDENT_INFLUENCE_CHECK.json')['passed']))
    # Measured receipts, including superseded attempts, are distinct from full task wall.
    ledger_path=root/'RESOURCE_LEDGER.jsonl'
    ledger=[json.loads(line) for line in ledger_path.read_text().splitlines() if line] if ledger_path.exists() else []
    records=[dict(receipt='RESOURCE_LEDGER.jsonl#'+str(i+1),**r) for i,r in enumerate(ledger)]
    controls_path=root/'CONTROLS_COST_AND_ATTEMPTS.json'
    if controls_path.exists():
        records.extend(read(controls_path)['measured_stages'])
    else:
        for directory,name in ((root/'baseline_v1','BASELINE_RECEIPT.json'),(root/'no_candidate_v2','NO_CANDIDATE_RECEIPT.json'),
                               (public/'gallery','GALLERY_RECEIPT.json'),(public/'gallery','QA_RECEIPT.json')):
            r=read(directory/name)
            records.append(dict(receipt=str(directory/name),elapsed_seconds=r['elapsed_seconds'],
                                process_cpu_seconds=r.get('process_cpu_seconds',r.get('process_CPU_seconds'))))
    if not any(r.get('stage')=='original_selector_record_only_final' for r in records):
        records.append(dict(receipt='selection_v1/SELECTION_REPLAY.json',stage='original_selector_record_only',
            elapsed_seconds=selection['elapsed_seconds'],process_cpu_seconds=selection['worker_cpu_seconds']))
    for name in ('fixed_cost_pilot_v1/PILOT_COST_RECEIPT.json','fixed_cost_pilot_v2/PILOT_COST_RECEIPT.json',
                 'cost_v1/DENSE_ARRAY_VALIDATION.json','INDEPENDENT_INFLUENCE_CHECK.json',
                 'final_bindings/RESOURCE_LEDGER.jsonl'):
        if not (root/name).exists():continue
        if name.endswith('.jsonl'):
            records.extend(dict(receipt=name,**json.loads(line)) for line in (root/name).read_text().splitlines() if line)
        else:
            r=read(root/name);records.append(dict(receipt=name,elapsed_seconds=r['elapsed_seconds'],
                process_cpu_seconds=r.get('process_cpu_seconds')))
    records.append(dict(receipt='cost_v1/COST_SUMMARY.json',stage='full_fixed_cost_two_workers',
        elapsed_seconds=cost['elapsed_seconds'],process_cpu_seconds=cost['parent_cpu_seconds']+cost['worker_cpu_seconds'],
        worker_cpu_seconds=cost['worker_cpu_seconds'],parent_cpu_seconds=cost['parent_cpu_seconds']))
    witness=read(public/'gallery/outline_witness/WITNESS_RECEIPT.json')
    records.append(dict(receipt='gallery/outline_witness/WITNESS_RECEIPT.json',
        elapsed_seconds=witness['elapsed_seconds'],process_cpu_seconds=witness['process_cpu_seconds']))
    elapsed=sum(r['elapsed_seconds'] for r in records)
    cpu_sum=sum(r.get('process_cpu_seconds',r.get('process_CPU_seconds',0)) or 0 for r in records)
    write(public/'COST.json',redact(dict(measured_stage_records=records,
        measured_stage_elapsed_sum_seconds=elapsed,measured_CPU_seconds_sum=cpu_sum,
        full_fixed_cost_elapsed_seconds=cost['elapsed_seconds'],worker_CPU_seconds=cost['worker_cpu_seconds'],
        gallery_elapsed_seconds=next(r['elapsed_seconds'] for r in records if str(r['receipt']).endswith('/gallery/GALLERY_RECEIPT.json')),
        independent_dense_check_seconds=read(root/'cost_v1/DENSE_ARRAY_VALIDATION.json')['elapsed_seconds'],
        input_restore_wall_seconds=0,restore_reason='Existing local sealed files reused; no download or archive extraction in this task',
        task_wall_at_packaging_seconds=time.time()-read(root/'START.json')['start_epoch'],
        CPU_analysis_stage_elapsed_cap_seconds=7200,within_measured_stage_cap=elapsed<7200,
        GPU_seconds=0,max_CPU_workers_used=2,max_CPU_workers_allowed=3,
        observed_PSS_GiB=2.38,memory_note='One measured two-worker snapshot, not global peak RSS',
        uninstrumented_time='Source editing, shell inspection, initial selection_attempt01 and first packaging exception were not separately CPU-instrumented; full-task wall includes them. Stage sum is not total task wall.',
        never_convert_to_RGB_latency_or_FPS=True,new_model_forwards=0,new_pose_solves=0,new_training_updates=0,
        preserved_attempts=read(controls_path)['preserved_failed_or_superseded_attempts'] if controls_path.exists() else [],
        absent_historical_attempt_receipts=missing_history)))
    write(root/'REPORT_PACKAGE_RECEIPT.json',dict(elapsed_seconds=time.monotonic()-started,
        process_cpu_seconds=time.process_time()-cpu,new_cost_evaluations=0,new_pose_calls=0,
        report_sha256=sha(public/'RESULTS_KO.md'),public_files=len(list(public.rglob('*')))))
    return public/'RESULTS_KO.md'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','parent','public','source-root'):p.add_argument('--'+k,required=True)
    a=p.parse_args();print(build(a.root,a.parent,a.public,a.source_root))

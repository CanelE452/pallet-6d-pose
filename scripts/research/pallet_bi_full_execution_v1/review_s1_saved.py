"""Read-only saved-output audit and S1 gallery/report preparation.

Standard library only. No image/reference decoding, metric/cost evaluator,
model forward, bootstrap rerun, optimizer or scientific dispatcher.
"""
from pathlib import Path
import argparse
from collections import Counter
from copy import deepcopy
import math
from . import f_core_resume as F

SEEDS=(20261018,20261019,20261020)


def checked(root,name):
    root=Path(root);seal=F.read(root/name)
    if not seal.get('complete',seal.get('PASS',False)):raise ValueError('Completed saved receipt required')
    for path,value in seal.get('artifacts',{}).items():
        if F.sha(root/path)!=value:raise ValueError('Saved artifact changed: '+path)
    return seal


def run(campaign,d1,public):
    campaign,d1,public=map(Path,(campaign,d1,public));folder=Path(__file__).parent
    bank=campaign/'POST_BANK_S1_VALIDATION_V2';rank=campaign/'S1_SELECTION_RANK_DIAGNOSTIC_V2';contrast=campaign/'POST_CONTRASTS_S1_VALIDATION_V2'
    bs=checked(bank,'ANALYSIS_SEAL.json');rs=checked(rank,'SELECTION_DIAGNOSTIC_SEAL.json');cs=checked(contrast,'CONTRASTS_SEAL.json')
    if rs['bank_analysis_seal_sha256']!=F.sha(bank/'ANALYSIS_SEAL.json'):raise ValueError('Rank labels belong to different bank coverage')
    if 'INPUT_BINDINGS.json'not in bs['artifacts']:raise ValueError('Authenticated coverage input closure required')
    for name,seal in (('post_selection.py',rs),('post_contrasts.py',cs)):
        if F.sha(folder/name)!=seal['source_sha256']:raise ValueError('Saved V2 receipt/source snapshot differs')
    fixed=F.read(campaign/'P0_FIXED_SETS/FIXED_SETS.json')
    if F.sha(campaign/'P0_FIXED_SETS/FIXED_SETS.json')!=bs['fixed_sets_sha256']or rs['fixed_sets_sha256']!=bs['fixed_sets_sha256']:raise ValueError('Fixed cohort ancestry differs')
    inp=F.read(bank/'INPUT_BINDINGS.json')['S1_UNION'];binding=rs['prediction_bindings']['S1_UNION']
    if inp['prediction']!=binding or F.sha(binding['path'])!=binding['sha256']or F.sha(Path(binding['path']).parent/'PREDICTION_SEAL.json')!=binding['seal_sha256']:
        raise ValueError('Actual saved action/label binding differs')
    cover={r['id']:r for r in F.rows(bank/'BANK_COVERAGE_ROWS.jsonl.gz')};pred={r['id']:r for r in F.rows(binding['path'])}
    saved={(r['id'],r['seed']):r for r in F.rows(rank/'ACTION_RANK_ROWS.jsonl.gz')}
    if len(cover)!=238 or set(cover)!=set(pred)or len(saved)!=714:raise ValueError('Full saved denominator differs')
    new3=[];checks=0;primary_counts={str(seed):Counter()for seed in SEEDS}
    for fid,row in cover.items():
        for seed in SEEDS:
            p=pred[fid]['predictions']['S1_UNION_s'+str(seed)];actual=saved[(fid,seed)];actions=p['actions'];good=set(row['good_eligible_ids'])
            if any(not math.isfinite(a['score'])for a in actions):raise ValueError('Nonfinite action logit')
            order=sorted(range(len(actions)),key=lambda i:(-actions[i]['score'],i))
            ranks=[(i,a)for i,a in enumerate(order,1)if ({actions[a]['candidate_id']}|set(actions[a].get('source_aliases',[])))&good]
            if actions:
                chosen=actions[order[0]]
                if p['selected_candidate_id']!=chosen['candidate_id']or p['selected_score']!=chosen['score']:raise ValueError('Saved selected argmax/tie differs')
                selected_good=bool(({chosen['candidate_id']}|set(chosen.get('source_aliases',[])))&good)
                if selected_good!=row['selected'][str(seed)]['metric']['success']:raise ValueError('Saved selected good-alias label differs from actual selected success')
            best_id=actions[ranks[0][1]]['candidate_id']if ranks else None;best_rank=ranks[0][0]if ranks else None
            margin=actions[ranks[0][1]]['score']-p['selected_score']if ranks else None
            if actual['best_good_action_id']!=best_id or actual['best_good_action_rank']!=best_rank or actual['best_good_logit_minus_selected']!=margin:
                raise ValueError('Saved rank differs from independent ordered-position expression')
            if margin is not None and margin>0:raise ValueError('Good action score exceeds selected finite argmax')
            if row['partition']in ('TEST','REMAINDER'):primary_counts[str(seed)][actual['failure_detail']]+=1
            if fid in fixed['sets']['PF_new_good']:new3.append({k:actual[k]for k in ('id','seed','best_good_action_id','best_good_action_rank','best_good_logit_minus_selected','selected_score','failure_detail')})
            checks+=1
    byte_parity={}
    for old,new,names in [('S1_SELECTION_RANK_DIAGNOSTIC','S1_SELECTION_RANK_DIAGNOSTIC_V2',rs['artifacts']),
        ('POST_CONTRASTS_S1_VALIDATION','POST_CONTRASTS_S1_VALIDATION_V2',cs['artifacts'])]:
        for name in names:
            if F.sha(campaign/old/name)!=F.sha(campaign/new/name):raise ValueError('Hardened metadata review unexpectedly changed saved numerical artifacts')
            byte_parity[old+'/'+name]=True
    paired=F.read(contrast/'INDEPENDENT_PAIRED_CONTRASTS.json');ci={}
    for seed in SEEDS:
        pair=paired['PRIMARY192']['S1_UNION_s'+str(seed)];ci[str(seed)]=pair['deltas']['success_rate']['recording_cluster_CI']
        if ci[str(seed)]['cluster_count']!=6 or ci[str(seed)]['resamples']!=10000 or ci[str(seed)]['seed']!=20260917 or len(pair['leave_one_cluster_out'])!=6:
            raise ValueError('Registered cluster sensitivity schedule differs')
    ds=checked(campaign/'D_FIXED_GAMMA_POSTSEAL_EVALUATION','EVALUATION_SEAL.json')
    checked(campaign/'D_FIXED_GAMMA_POSTSEAL_EVALUATION','SUPPLEMENT_SEAL.json')
    d=F.read(campaign/'D_FIXED_GAMMA_POSTSEAL_EVALUATION/SUMMARY.json');dall=F.read(campaign/'D_FIXED_GAMMA_POSTSEAL_EVALUATION/ALL238_AGGREGATE.json')
    gamma=F.read(campaign/'D_FIXED_GAMMA_POSTSEAL_EVALUATION/GAMMA_CORRECTION_COMPARISON.json');strict=F.read(campaign/'D_FIXED_GAMMA_POSTSEAL_EVALUATION/STRICT_SOURCE_SHARED_SUBSET.json')
    manifest=deepcopy(F.read(campaign/'POST_BANK_S1_VALIDATION_MANIFEST.json'))
    manifest.update(schema='POSTSEAL_S1_DERIVED_GALLERY_MANIFEST_V1',fixed_examples=[dict(arm='S1_UNION',id=fid,reason='PF_new_good3')for fid in fixed['sets']['PF_new_good']],
        historical_baseline=dict(arm_prefix='D1_H1_MLP',prediction_dir=str(d1),seed_arms=['D1_H1_MLP_s'+str(s)for s in SEEDS],
            renderer_contract='Default baseline from explicit d1 argument;comparison_baseline override intentionally absent'),
        diagnostic_evidence=dict(selection_diagnostic_seal=str(rank/'SELECTION_DIAGNOSTIC_SEAL.json'),selection_diagnostic_seal_sha256=F.sha(rank/'SELECTION_DIAGNOSTIC_SEAL.json'),
            fixed_sets=str(campaign/'P0_FIXED_SETS/FIXED_SETS.json'),fixed_sets_sha256=bs['fixed_sets_sha256']),
        rendering_status='PENDING_ROOT_RESOURCE_GO;manifest preparation only')
    mp=campaign/'S1_STAGE_GALLERY_MANIFEST.json';F.write(mp,manifest)
    sources={name:F.sha(folder/name)for name in ('post_selection.py','post_contrasts.py','f_frontend.py','f_core.py','f_core_resume.py','stage_gallery.py')}
    review=dict(complete=True,component='read-only source/saved-output audit and manifest/report drafting',source_snapshots=sources,
        rank_rows_checked=checks,rank_position_and_saved_argmax_tie_exact=True,current_selected_alias_success_labels_consistent=True,
        source_alias_limitation='Automatic numeric-action aliases,not independent physical ownership. Near-tolerance action aliases cannot certify exact-pose success in arbitrary future banks;actual714 selected-label checks pass.',
        original_rank_consistency_receipt=F.read(campaign/'S1_SELECTION_RANK_DIAGNOSTIC/RANK_CONSISTENCY.json'),
        primary_failure_counts={k:dict(v)for k,v in primary_counts.items()},PF_new_good3_rows=new3,
        authenticated_input_bindings_sha256=bs['artifacts']['INPUT_BINDINGS.json'],rank_seal_sha256=F.sha(rank/'SELECTION_DIAGNOSTIC_SEAL.json'),
        bank_seal_sha256=F.sha(bank/'ANALYSIS_SEAL.json'),contrasts_seal_sha256=F.sha(contrast/'CONTRASTS_SEAL.json'),
        hardened_outputs_byte_identical=byte_parity,bootstrap_rerun=False,PRIMARY_recording_CIs=ci,
        contrasts_validation_scope='1056 shared canonical fields;independent additionally emits percluster/leave-one-cluster fields. Canonical both_failure_tail intentionally skipped by independent_check.compare,not independently revalidated by this paired-only receipt.',
        remaining_code_findings=[dict(severity='low',file='post_contrasts.py',finding='Duplicate contrast names can overwrite a dictionary output while len(contrasts) remains larger;actual S1 plan names unique.'),
            dict(severity='low',file='post_selection.py',finding='Finite actions are enforced;selected_score has no separate finite check. Production strict JSON export and actual714 exact argmax checks exclude the issue here.'),
            dict(severity='resolved in separate adapter',file='f_core.py',finding='Dynamic immutable controller/resource fields and completed aggregator re-entry prevent straightforward restart;new f_core_resume.py preserves the running source/receipts.')],
        f_frontend_review='New invocation receipts and downstream full1241 audit avoid rewriting completed production manifests;current own-CAL/fullpopulation byte ancestry is checked by f_front_audit. No new accuracy inferred.',
        S1_gallery_manifest=str(mp),S1_gallery_manifest_sha256=F.sha(mp),rendering_status='NOT_RUN',new_model_forwards=0,new_metric_calls=0,new_bootstrap_calls=0,new_cost_calls=0,new_pose_calls=0,new_fits=0,
        references_decoded_in_this_module=0,images_opened=0,whole_campaign_complete=False,
        separate_manual_inspection_incidental_CAL_decode=dict(
            path=str(campaign/'S2_CALIBRATION/ISOLATED_CURRENT_CAL_REFERENCES.json'),
            sha256=F.sha(campaign/'S2_CALIBRATION/ISOLATED_CURRENT_CAL_REFERENCES.json'),
            CAL_reference_objects=46,count_source='existing S2_CALIBRATION/SUMMARY.json CAL_frames46;not recounted by reopening values',
            purpose='overly broad JSON metadata inventory;only top-level keys printed',
            PRIMARY_reference_containers_opened=0,reference_values_copied_to_public_outputs=False,
            scientific_use=False,method_or_rule_change=False,new_fit_or_repair_required=False,
            constructed_tests_remain_actual_zero_read=True))
    F.write(campaign/'POSTPROCESS_READ_ONLY_REVIEW_S1_D.json',review)
    write_report(campaign,public,paired,new3,d,dall,gamma,strict)
    return dict(review=str(campaign/'POSTPROCESS_READ_ONLY_REVIEW_S1_D.json'),manifest=str(mp),report=str(public/'S1_D_EVIDENCE_DRAFT.md'),rank_checks=checks)


def write_report(run,public,paired,new3,d,dall,gamma,strict):
    s=F.read(run/'S1_EVALUATION/SUMMARY.json');e=F.read(run/'S1_EXECUTION_SUMMARY.json');fixed=F.read(run/'P0_FIXED_SETS/FIXED_SETS.json')
    lines=['# S1·D 완료 근거 초안','',
        '상태: **DRAFT — S1 실행 및 D 진단 근거만 정리**. 전체 캠페인 완료 보고서가 아니다. 수치는 저장된 예측·평가·독립 검증 산출물에서 옮겼으며 이 문서 작성 중 새 추론·학습·pose solve·metric 평가·bootstrap을 실행하지 않았다.',
        '', '## 평가 범위','',
        '실제 real238 전체를 유지한다. PRIMARY192 = TEST49 + REMAINDER143이며 CAL46은 별도 보고한다. PRIMARY는 반복 사용된 개발 데이터로 독립 실험의 표본 수는 0이다. 성공은 기존 C2 회전 정의와 5cm/5° 기준을 사용한다. 이진 분모에는 모든 프레임을 유지하고 연속 오차는 available-only, 표본 SD(ddof=1), tail trimming 없음으로 보고한다.',
        '', '## S1: PF complete-action union과 H1 실제 선택','',
        '기존 D1의 모든 방문 endpoint·alias와 관측/비용 문맥을 재사용하고, frozen PF 두 anchor의 실제 complete pose를 공통 proper-rotation/positive-depth 규칙으로 admission했다. 새 real 평가 reference를 열기 전에 real238×3 seed 예측을 먼저 봉인했다. 기존 bank의 개별 26 scalar와 31 validity mask를 정확히 재사용하며 상대 5 gap, bank summary, TRAIN normalizer와 188D H1 입력을 새 bank에 맞춰 구성했다. 기존 후보의 admission은 변하지 않아 matched baseline 재학습은 필요하지 않았으며 기존 H1 세 seed 결과를 정확히 재사용했다.',
        '', '| Population | 원래 H1 (seed18/19/20) | S1 union H1 (seed18/19/20) |', '|---|---:|---:|']
    for pop in ('PRIMARY192','TEST49','REMAINDER143','CAL46'):
        old='/'.join(str(s[pop]['D1_H1_MLP_s'+str(seed)]['successes'])for seed in SEEDS)
        new='/'.join(str(s[pop]['S1_UNION_s'+str(seed)]['successes'])for seed in SEEDS);lines.append(f'| {pop} | {old} | {new} |')
    lines+=['','| Seed | PRIMARY recovery / damage | Δsuccess (pp) | Recording-cluster 95% descriptive CI (pp) |', '|---|---:|---:|---:|']
    for seed in SEEDS:
        p=paired['PRIMARY192']['S1_UNION_s'+str(seed)];ci=p['deltas']['success_rate']['recording_cluster_CI']['CI95']
        lines.append(f"| {seed} | {p['counts']['RECOVERY']} / {p['counts']['DAMAGE']} | {p['success_difference_percentage_points']:+.4f} | [{100*ci[0]:+.4f}, {100*ci[1]:+.4f}] |")
    lines+=['','원래 PRIMARY bank generated/eligible upper bound는 151/192이고 PF union은 154/192다. 새로 good 후보가 생긴 세 프레임 모두에서 세 seed H1 선택은 실패했다. 따라서 후보 upper bound 증가를 실제 selector 성능 향상으로 보고하지 않는다. 원래 seed18로 고정한 selection-failure31의 선택 성공은 두 시스템 모두 seed18/19/20에서 0/0/1이고, PF가 이미 good이던 old12 역시 실제 선택 성공은 0/0/1이다. 고정 cohort 자체를 새 seed 결과에 맞춰 다시 정의하지 않았다.',
        '', '### 새 good PF 3개와 저장된 logit 순위','',
        '다음 순위는 같은 이미지·같은 H1 seed 안에서 저장된 finite action logits를 정렬한 posthoc 진단이다. good label은 이미 봉인된 bank 평가에서 가져왔으며 logit과 sigmoid는 calibrated confidence가 아니다. seed18 대표 순위는 8/2/17이며 세 seed 전체는 아래와 같다.',
        '', '| Frame | Good PF action | Rank18 / Rank19 / Rank20 | Good logit − selected logit (18 / 19 / 20) |', '|---|---|---:|---:|']
    for fid in fixed['sets']['PF_new_good']:
        a=sorted([r for r in new3 if r['id']==fid],key=lambda r:r['seed'])
        lines.append('| '+fid+' | '+a[0]['best_good_action_id']+' | '+' / '.join(str(r['best_good_action_rank'])for r in a)+' | '+' / '.join(f"{r['best_good_logit_minus_selected']:.6f}"for r in a)+' |')
    lines+=['','real238×3=714 저장 action 목록에서 선택 argmax/tie order, good 순위, logit 차이를 별도의 정렬 위치식으로 확인했다. construction gap은 0이다. PRIMARY seed18은 성공119 / wrong-rank35 / generated-absent38이다. 모든 seed의 분모와 세부 분류는 private `S1_SELECTION_RANK_DIAGNOSTIC_V2/SUMMARY.json`에 남긴다.',
        '', '### 연속 오차와 tails (PRIMARY192)','',
        '| Arm/seed | T mean±sample SD (cm) | T median / P90 / P95 / max | R mean±sample SD (deg) | R median / P90 / P95 / max | T>50 / T>100 / R>45 |', '|---|---:|---:|---:|---:|---:|']
    for prefix in ('D1_H1_MLP','S1_UNION'):
        for seed in SEEDS:
            q=s['PRIMARY192'][prefix+'_s'+str(seed)];t=q['translation_cm'];r=q['rotation_deg']
            lines.append(f"| {prefix}/{seed} | {t['mean']:.4f}±{t['sample_std']:.4f} | "+' / '.join(f'{t[k]:.4f}'for k in ('median','P90','P95','max'))+f" | {r['mean']:.4f}±{r['sample_std']:.4f} | "+' / '.join(f'{r[k]:.4f}'for k in ('median','P90','P95','max'))+f" | {q['T_gt50cm']} / {q['T_gt100cm']} / {q['R_gt45deg']} |")
    lines+=['','세 seed의 이진 효과는 −1/+1/−1 프레임으로 일관된 회복이 아니다. Recording-cluster bootstrap은 원래 6개 cluster를 whole-cluster로 복원추출한 frame-weighted paired ratio, 10,000회, seed20260917의 기술통계다. 8개 session을 단위로 한 민감도와 6개 leave-one-cluster 결과도 별도 보존한다. 이를 독립 검증·동등성·population generalization 주장에 사용하지 않는다.',
        '', 'S1은 real238+synthetic1003 PF tensor cache를 정확히 재사용하여 새 neural forward는 0회다. synthetic 두 anchor F 호출2006회와 SubPix corner 호출7453회, 새 개별 비용1352회(새 PF1348 + old context 복구4), H1 세 fit/1176 optimizer step을 기록했다. 별도 기존 S1 postseal gallery 진단 비용130회와 constructed optimizer step1은 과학 학습/producer 비용과 구분한다. 계측 단계 wall 합438.8669초는 전체 캠페인 경과 시간이 아니며 phase 밖 import·review 시간은 포함하지 않는다.',
        '', '## D: COMMON Γ 고정 q/Σ 진단','',
        '이전 D의 covariance donor는 Σ와 `fit_reliability_weight` Γ를 함께 옮겼다. audit에서 18프레임/19factor의 Γ 차이를 확인했으므로 이전 결과는 Σ+Γ donor sensitivity로 보존한다. 수정 실행은 COMMON 논리 graph와 COMMON Γ를 고정하고 q/Σ 교환을 수행했다. 두 WD family의 실제 readout과 방문 endpoint를 유지한 진단이며 원래 120-success 시스템의 BI 효과나 새 H1 전체 시스템 성능을 뜻하지 않는다. 관측 seed는 하나다.',
        '', '| Fixed-COMMON-Γ cell | PRIMARY192 | TEST49 | REMAINDER143 | CAL46 | ALL238 supplementary |', '|---|---:|---:|---:|---:|---:|']
    for cell in ('Q0S0','Q0S1','Q1S0','Q1S1'):
        lines.append('| '+cell+' | '+' | '.join(str(d[p][cell]['successes'])for p in ('PRIMARY192','TEST49','REMAINDER143','CAL46'))+' | '+str(dall['summary'][cell]['successes'])+' |')
    lines+=['','Q0/Q1은 COMMON/BI q, S0/S1은 COMMON/BI covariance donor를 나타내며 Γ는 COMMON 값으로 고정된다. Γ 수정으로 S1 donor 두 cell 각각에서 pose dictionary16개( PRIMARY15 + CAL1)가 바뀌었으나 모든 보고 population의 success delta는 0이다. 이는 이전 진단의 혼입이 없었다는 뜻이 아니며, 이번 cutoff에서 성공 분류가 같았다는 결과다.',
        '', f"엄격한 동일 physical-source association 교차 조건을 만족하는 범위는 {strict['valid_counterfactual_n']}/238이고 나머지 {strict['excluded_cross_n']}는 source-changing cross로 별도 표시한다. 이 135개를 algorithm failure로 바꾸지 않는다. 엄격 subset의 네 cell은 성공46/103으로 같고 q/Σ 개입의 pose 효과도 0이다. 전체 fixed-COMMON-graph192/238 수치는 이 엄격 source 보존 subset과 구분해야 하며, 전체 graph 조건부 donor 변화가 순수한 동일 측정의 q 대 Σ 인과 분해를 인증하지 않는다.",
        '', '수정 실행은 actual family optimizer64회, family dispatch72회, exact packet reuse916건을 기록했고 wrapper wall42.9771초다. 원래 Γ-mixed 실행과 수정 실행의 예측·평가·supplement seal을 각각 보존했다. 전체238 supplementary와 PRIMARY/TEST/REMAINDER/CAL, recording 및 family-readout 분해를 private audit에 남긴다.',
        '', '저장된 PRIMARY paired decomposition에서 q-only(Q1S0−Q0S0)는 이전에 선택했던 family 안의 성공 +2가 family readout 기여 −2와 상쇄되어 전체 0이다(family switch7개). Σ-only(Q0S1−Q0S0)는 family switch5개, 성공 −1이다. 이는 저장된 family readout 사이의 대수적 오차 분해이며 독립 물리 owner 인증이나 일반적 인과 효과를 뜻하지 않는다.',
        '', '별도 CAL noise-model 진단(S2_CALIBRATION)에서 recording-group held-out nominal80% coverage는 conditional C0 0.4264에서 pooled CP 0.7381로 변했고 full CAL pooled floor는115.8798px²였다. q는 고정하고 모델의 total-error calibration을 검사한 결과다. floor 수치와 coverage 개선으로 아직 완료되지 않은 S2 end-to-end pose/H1 성능 향상을 단정하지 않는다.',
        '', '## F 관측 완료 범위 표기','',
        'F의 raw COMMON/BI 관측 선택·source/q/conditional-covariance 문맥은 real230/238에서 다르다. 각 own-q에 같은 pooled CAL recipe를 적용한 뒤 effective solver packet이 다른 real 프레임은237/238이다. 후자는 calibration covariance 차이를 포함하므로 raw source/q 교체가237개라는 뜻이 아니다. F 관측 및 own CAL packet 완료와 최종 optional union/system 선택·H1·real accuracy 완료를 구분한다. 이 초안은 새 S2/S3/S4/F accuracy를 보고하지 않는다.',
        '', '## 구조 proxy와 물리 해석 한계','',
        'reference-free asset audit에서 synthetic VAL235 전부의 named GLB가 없어 고정 structured proxy를 쓴 사실을 확인했다. 이는 확인된 train/VAL geometry·appearance 분포 제한이다. 아직 새 accuracy 실패의 원인으로 입증된 것은 아니다. 렌더러는 CPU llvmpipe/EGL software rendering이며 hardware GPU rendering으로 표시하지 않는다. 실제 pallet instance CAD identity, signed axes 및 material-owner association은 독립 인증되지 않았고 이 물리 인증을 위한 human review는0이다. 화면을 그린 virtual native corner를 인증된 실제 mesh corner로 취급하지 않는다.',
        '', '## 근거 및 display 후속','',
        'Private run root는 `pallet-bi-full-execution-20261019-v1`이다. 다음은 원본 reference R/t나 RGB를 포함하지 않는 경로 표기이며 raw GT/원본 RGB를 public 산출물로 복사하지 않았다.',
        '', '- S1: `S1_EXECUTION_SUMMARY.json`, `S1_EVALUATION/{EVALUATION_SEAL,SUMMARY,PAIRED}.json`, `S1_INDEPENDENT_FINAL/INDEPENDENT_VERIFICATION.json`.',
        '- Input closure: `POST_BANK_S1_VALIDATION_V2/ANALYSIS_SEAL.json`이 `INPUT_BINDINGS.json`까지 SHA로 bind한다. 기존 V1 receipt와 V2를 보존한다.',
        '- Rank: `S1_SELECTION_RANK_DIAGNOSTIC_V2/{SELECTION_DIAGNOSTIC_SEAL,SUMMARY,FIXED_COHORT_ACTIONS}.json`; V1의 `RANK_CONSISTENCY.json`; V1/V2의 세 numerical artifacts는 byte-identical이다.',
        '- Paired: `POST_CONTRASTS_S1_VALIDATION_V2/CONTRASTS_SEAL.json`의 canonical shared1056항목 최대차4.44e-16. 독립 percluster/leave-one-cluster 항목은 추가 산출물이다. canonical both-failure tail block 자체는 이 paired 비교기의 검증 대상에서 제외된다.',
        '- D: `D_GAMMA_AUDIT.json`, `D_FIXED_COMMON_GRAPH_FIXED_GAMMA/{SUMMARY,PREDICTION_SEAL}.json`, `D_FIXED_GAMMA_POSTSEAL_EVALUATION/{EVALUATION_SEAL,SUPPLEMENT_SEAL,GAMMA_CORRECTION_COMPARISON,STRICT_SOURCE_SHARED_SUBSET,ALL238_AGGREGATE}.json`.',
        '- F: `F_FRONT_AUDIT/{SUMMARY,INTERVENTION_PATH_COUNTS}.json`, `F_CALIBRATION/PACKET_MANIFEST.json`; accuracy 주장 없음.',
        '- Noise/asset limits: `S2_CALIBRATION/{SUMMARY,MODEL}.json`, `s3_s4_archive_report/PROXY_ASSET_DOMAIN_COUNTS.json`. CAL-only metadata inventory 중 isolated CAL46 container를 불필요하게 decode한 1건은 private `POSTPROCESS_READ_ONLY_REVIEW_S1_D.json`에 경로·SHA와 함께 공개적으로 기록했다. PRIMARY reference read·새 과학 사용은 없고 constructed test의 실제 zero-read receipt와 구분한다.',
        '- New gallery manifest: `S1_STAGE_GALLERY_MANIFEST.json`. 원래 H1의 동일 seed를 baseline으로 하고 PF_new_good3 전부 및 모든 PRIMARY recovery/damage를 요구한다. 렌더링은 아직 실행하지 않았으며 root resource GO와 실제 derived-pixel privacy QA가 남아 있다. Crop은 N0/native projection 및 initial query envelope로만 정하고, 저장 automatic outline이 밖으로 나가면 full context를 유지한다.',
        '', '이 문서는 review용 공개 보고서 초안 위치에만 저장되었으며 commit·merge·배포·최종 보고 완료를 주장하지 않는다.']
    public.mkdir(parents=True,exist_ok=True);path=public/'S1_D_EVIDENCE_DRAFT.md';text='\n'.join(lines)+'\n'
    if path.exists()and path.read_text()!=text:raise ValueError('Preserved draft differs;write a new revision')
    path.write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('campaign','d1','public'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(run(a.campaign,a.d1,a.public))

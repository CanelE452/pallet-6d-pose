"""Diagnose the original no-success bank using saved logs only.

This module never imports a metric scorer, optimizer, initializer, or model.
Per-candidate reference numbers are copied only from the sealed old oracle's
start/end pairs.  Stage minima stay explicitly separate: they cannot be joined
into a synthetic successful pose.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import time
from urllib.parse import quote

from .baseline import (group, read_rows, read_metrics, sha, write_json, write_csv)

ADMISSION_SOURCE = 'scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_u_region_solver_v4_5.py:333'
OUTLINE_ADMISSION_SOURCE = 'scripts/research/pallet_visibility_curriculum_20261013_v1/statewise_global_outline_solver_v4_10.py:192'
OLD_GALLERY = {'wood_night_01:029428':1,'plastic_day_01:011640':3,
               'wood_day_01:005249':4,'wood_night_01:031426':5,'wood_night_01:031344':6}


def finite_slack(record):
    value=record.get('minimum_constraint_slack',record.get('minimum_joint_slack'))
    return float(value) if value is not None and math.isfinite(float(value)) else None


def pair_summary(pairs,stage):
    p=[r for r in pairs if r['stage']==stage]
    return {
        'paired_starts':len(p),
        'starts_joint_success':sum(r['before_metric']['success_le5'] is True for r in p),
        'ends_joint_success':sum(r['after_metric']['success_le5'] is True for r in p),
        'start_T_le5_count':sum(r['before_metric']['translation_cm'] is not None and r['before_metric']['translation_cm']<=5 for r in p),
        'start_R_le5_count':sum(r['before_metric']['rotation_deg'] is not None and r['before_metric']['rotation_deg']<=5 for r in p),
        'end_T_le5_count':sum(r['after_metric']['translation_cm'] is not None and r['after_metric']['translation_cm']<=5 for r in p),
        'end_R_le5_count':sum(r['after_metric']['rotation_deg'] is not None and r['after_metric']['rotation_deg']<=5 for r in p),
        'joint_success_to_failure':sum(r['joint_success_to_failure'] is True for r in p),
        'scope':'original paired unrefined-to-numerical-endpoint records; not every historical seed and not optimizer iterates',
    }


def make_case(o,packet,bank,endpoints,metrics,permission):
    ident=o['id'];s=packet['solver_input'];a=packet['activation'];m=metrics[ident]
    gen=o['stages']['generated'];zero=gen['candidate_count']==0
    if gen['joint_success_exists'] is not False or o['final_joint_success'] is not False:
        raise ValueError('No-success group must not contain success '+ident)
    fallback=bank['original_e']['fallback_audit']
    counts=Counter(s['states']);visible=s['visible'];partial=set(a['all_partial_ids']);complete=set(a['complete_ids'])
    if len(complete)+len(partial)!=len({v['native_corner']for v in visible}):
        raise ValueError('Full/partial partition mismatch '+ident)
    if zero:
        if fallback.get('reason')!='fewer_than_two_BI' or len(visible)>=2:
            raise ValueError('Admission reason must be proven from source and packet '+ident)
    pairrows=[];familyrows=[]
    for st,stage in bank['stages'].items():
        for fam in stage['families']:
            es=[r for r in endpoints if r['stage']==st and r['family']==fam['name']]
            es_slack=[finite_slack(r['original_endpoint_record'])for r in es]
            es_known=[x for x in es_slack if x is not None]
            familyrows.append(dict(stage=st,family=fam['name'],solver_schema=fam['solver_schema'],
                solver_status=fam['solver_status'],accepted=fam['accepted'],
                original_outcomes=len(fam['original_outcomes']),retained_feasible=len(fam['feasible_samples']),
                logged_numeric_endpoints=len(es),finite_minimum_slack_count=len(es_known),
                negative_minimum_slack_count=sum(x<0 for x in es_known),
                minimum_recorded_slack=min(es_known)if es_known else None,
                maximum_recorded_slack=max(es_known)if es_known else None,
                original_search=fam['original_search'],original_parent_search=fam['original_parent_search'],
                selected_pose_recorded=fam['selected_solver_pose'] is not None,
                original_objective=fam['original_objective'],
                constraint_scope='saved minimum aggregate slack; negative endpoints do not prove all observations mutually inconsistent'))
    for p in o['refinement_pairs']:
        pairrows.append(dict(id=ident,stage=p['stage'],family=p['family'],branch=p['branch'],
            initialization=p['initialization'],before_candidate_id=p['before_candidate_id'],
            after_candidate_id=p['after_candidate_id'],
            before_T_cm=p['before_metric']['translation_cm'],before_R_deg=p['before_metric']['rotation_deg'],
            before_joint_success=p['before_metric']['success_le5'],
            after_T_cm=p['after_metric']['translation_cm'],after_R_deg=p['after_metric']['rotation_deg'],
            after_joint_success=p['after_metric']['success_le5'],joint_success_to_failure=p['joint_success_to_failure']))
    full_fallback=[v['native_corner']for v in visible if v['native_corner']in complete and v.get('source')=='VISIBLE_BACKBONE_FALLBACK']
    all_fallback=[v['native_corner']for v in visible if v.get('source')=='VISIBLE_BACKBONE_FALLBACK']
    full_non_fallback=[v['native_corner']for v in visible if v['native_corner']in complete and v.get('source')!='VISIBLE_BACKBONE_FALLBACK']
    has_T=gen['translation_min_cm'] is not None and gen['translation_min_cm']<=5
    has_R=gen['rotation_min_deg'] is not None and gen['rotation_min_deg']<=5
    g10good=o['original_stage_oracles']['G10']['generated']['joint_success_exists']
    n0pf=any(m[k]['success'] is True for k in ('N0','POSEFIX_RAW'))
    if zero:
        confirmed=f"가시 B/I 입력 {len(visible)}개이므로 원 코드 fewer_than_two_BI 조건에서 진입하지 않았다. generated=0이고 coarse fallback은 별도 출력이다."
        competing='B/I 상태 판단 자체의 적절성, 실제 차폐와 관측 부족, 초기 자세에 조건화된 상태 오류를 이 로그만으로 구분하지 못한다.'
        minimum='저장 RGB·mask·초기 자세에서 B/I로 인정 가능한 실제 지지가 있는지와 원 0~1개 B/I의 physical owner를 수동 검토한다. 새 검출이나 후보 생성은 하지 않았다.'
        status='ADMISSION_NO_GENERATION_CONFIRMED'
    else:
        confirmed=f"최종 G13 consumer generated {gen['candidate_count']}개·retained {o['stages']['feasible']['candidate_count']}개에 동일 pose 공동 성공은 없다. complete {len(complete)}개(예측 fallback {len(full_fallback)}), partial {len(partial)}개가 원 활성 관측이다."
        if g10good:
            confirmed+=' G10 당시에는 공동 성공 후보 및 선택이 있었으나 최종 G13 패키지에는 남지 않았다.'
        competing='관측·공분산·선 physical owner·초기화/제안 범위를 동시에 바꾸지 않았으므로 OBSERVATION_VS_PROPOSAL_NOT_ISOLATED다. 실패 후보에서 음의 slack이 있어도 전체 제약의 모순을 뜻하지 않는다.'
        minimum=('G10 성공 pose와 G13 원 시작/선 association의 차이를 고정 packet별로 확인한다.' if g10good else
                 '원 full fallback·partial 지지의 physical owner와 초기 후보의 물리 축을 저장 RGB에 대조해 최소 반례를 확인한다.')
        if n0pf:
            minimum+=' 이 영상에서 성공한 N0/PoseFix 출력은 외부 대조군이며 e 은행에 합치지 않는다.'
        status='OBSERVATION_VS_PROPOSAL_NOT_ISOLATED'
    prior=permission.get(packet['image_sha256'])
    # RGB inventory membership proves prior publication in the verified Release
    # snapshot; it does NOT imply the original data path is a tracked Git blob.
    public_rgb=('https://github.com/CanelE452/pallet-6d-pose/blob/2fe1969b7a999d9a8d460703be4bfd11317e7f07/'
        '_docs/experiments/pallet_visibility_curriculum_20261013_v1/publication_followup_20261015_v11/repro_inventory_DOWNLOADS_KO_v1.md')if prior else None
    oldnum=OLD_GALLERY.get(ident)
    previous_gallery=('../pallet_statewise_correspondence_diagnosis_v1/gallery/index.html#case'+str(oldnum))if oldnum else None
    image_link='gallery/no_candidate_rgb/'+ident.replace(':','__')+'.jpg'if prior else None
    detail=dict(id=ident,partition=o['partition'],session=o['session'],diagnostic_status=status,
        original_oracle_stage=o['earliest_observed_failure_stage'],admission_no_generation=zero,
        admission_condition_source=ADMISSION_SOURCE if zero else None,
        outline_early_return_source=OUTLINE_ADMISSION_SOURCE if zero else None,
        admission_fallback_reason=fallback.get('reason'),states=s['states'],state_counts=dict(counts),
        active_complete_ids=sorted(complete),active_partial_ids=sorted(partial),
        typed_partial_ids=a['typed_partial_ids'],legacy_partial_ids=a['legacy_partial_ids'],
        complete_prediction_fallback_ids=full_fallback,complete_nonfallback_ids=full_non_fallback,
        all_archived_prediction_fallback_ids=all_fallback,
        full2D_revoked_for_all_partial=True,S_image_terms=0,OTU_point_terms=0,
        stored_full_covariance_and_reliability=[{k:v.get(k)for k in('native_corner','source','fit_covariance_px2','fit_reliability_weight')}for v in visible if v['native_corner']in complete],
        outline_supported=s['outline_observation'].get('supported'),outline_reason=s['outline_observation'].get('reason'),
        domains_count=len(s['domains']),U_regions_count=len(s['u_regions']),
        generated_summary=gen,retained_summary=o['stages']['feasible'],original_stage_oracles=o['original_stage_oracles'],
        generated_translation_pass_exists=has_T,generated_rotation_pass_exists=has_R,
        separate_T_R_pass_candidates_but_no_joint=has_T and has_R,
        G10_joint_success_exists_but_G13_none=g10good,
        paired_start_end_summary={st:pair_summary(o['refinement_pairs'],st)for st in('G10','G13')},
        family_solver_log_summary=familyrows,reference_registration_audit=o['reference_and_registration_audit'],
        semantic_owner_status='SEMANTIC_OWNER_UNVERIFIED',initial_S_state_is_conditioned_on_existing_coarse_pose=True,
        no_observed_negative_slack_is_global_inconsistency_proof=True,
        N0=m['N0'],POSEFIX_RAW=m['POSEFIX_RAW'],E0=m['E0'],
        source_image_sha256=packet['image_sha256'],existing_public_image_permission=prior is not None,
        existing_image_link=image_link,previous_diagnostic_gallery_link=previous_gallery,
        existing_public_RGB_archive_restore_link=public_rgb,existing_public_RGB_archive_member=prior,
        confirmed_facts=confirmed,competing_explanations=competing,
        missing_evidence='독립 semantic owner·실제 physical axis 정답 없음. 참조는 기존 기하 재구성값이다. 원 역사적 WD/P1 rejected proposal 및 optimizer 내부 iterate는 consumer 은행 밖이며 미완전하다. 관측 대 제안의 인과분리는 수행하지 않았다.',
        next_minimal_check=minimum)
    summary={k:detail[k] for k in('id','partition','session','diagnostic_status','admission_no_generation','admission_fallback_reason')}
    summary.update(visible_BI=len(visible),complete=len(complete),partial=len(partial),
        complete_prediction_fallback=len(full_fallback),outline_supported=detail['outline_supported'],
        O_count=counts['O'],T_count=counts['T'],U_count=counts['U'],S_count=counts['S'],
        generated_count=gen['candidate_count'],retained_count=o['stages']['feasible']['candidate_count'],
        generated_T_min_cm=gen['translation_min_cm'],generated_R_min_deg=gen['rotation_min_deg'],
        separate_T_R_pass_candidates_but_no_joint=has_T and has_R,G10_joint_success_exists_but_G13_none=g10good,
        N0_success=m['N0']['success'],POSEFIX_RAW_success=m['POSEFIX_RAW']['success'],E0_success=False,
        N0_translation_cm=m['N0']['T_cm'],N0_rotation_deg=m['N0']['R_deg'],
        POSEFIX_RAW_translation_cm=m['POSEFIX_RAW']['T_cm'],POSEFIX_RAW_rotation_deg=m['POSEFIX_RAW']['R_deg'],
        E0_translation_cm=m['E0']['T_cm'],E0_rotation_deg=m['E0']['R_deg'],
        facts=confirmed,competing_explanations=competing,missing_evidence=detail['missing_evidence'],
        next_minimal_check=minimum,existing_image_link=image_link)
    return summary,detail,pairrows


def build(old_private,source_root,output):
    from scripts.research.pallet_minimal_bi_ablation_v1.gallery import public_image_inventory
    start=time.monotonic();cpu=time.process_time();old,root,out=map(Path,(old_private,source_root,output))
    out.mkdir(parents=True,exist_ok=False)
    paths={
        'oracle':old/'oracle_v1/ORACLE_ROWS.jsonl.gz',
        'metrics':old/'score_v2/ROW_METRICS.csv',
        'packets':old/'e_trace_v2/TYPED_PACKETS.jsonl.gz',
        'bank':old/'original_candidate_archive_v1/ORIGINAL_E_CANDIDATE_BANK.jsonl.gz',
        'endpoints':old/'GENERATED_CANDIDATES.jsonl.gz'}
    oracle=[r for r in read_rows(paths['oracle'])if group(r)=='NO_SUCCESS_CANDIDATE'];ids={r['id']for r in oracle}
    packets={r['id']:r for r in read_rows(paths['packets'])if r['id']in ids}
    banks={r['id']:r for r in read_rows(paths['bank'])if r['id']in ids}
    endpoints={r['id']:r['candidates']for r in read_rows(paths['endpoints'])if r['id']in ids}
    metrics=read_metrics(paths['metrics']);permission=public_image_inventory(root)
    summaries=[];details=[];pairs=[]
    for o in oracle:
        ss,dd,pp=make_case(o,packets[o['id']],banks[o['id']],endpoints[o['id']],metrics,permission)
        summaries.append(ss);details.append(dd);pairs.extend(pp)
    primary=[r for r in summaries if r['partition']!='CAL'];cal=[r for r in summaries if r['partition']=='CAL']
    if len(primary)!=46 or len(cal)!=6 or sum(r['admission_no_generation']for r in primary)!=4:
        raise ValueError('Original no-candidate 46/6 and primary admission4 required')
    write_csv(out/'NO_CANDIDATE46_CASES.csv',primary);write_csv(out/'NO_CANDIDATE_CAL6_CASES.csv',cal)
    write_csv(out/'NO_CANDIDATE_START_END_PAIRS.csv',pairs)
    with gzip.open(out/'NO_CANDIDATE_DETAILS.jsonl.gz','xt')as stream:
        for r in details:stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    index=['# 원 e 후보부재 46장 장부','',
      '원 최종 G13 consumer 은행에서 동일 pose의 위치≤5cm AND 회전≤5°를 만족하는 후보가 없는 집단이다. 새 자세·후보·관측·참조를 만들지 않았다.',
      '4장은 fewer_than_two_BI로 새 후보가 없고, 42장은 실제 후보가 있지만 공동 성공이 없다. 아래 이미지 링크는 검증된 기존 공개 Release의 같은 RGB에서 만든 작은 crop이다. 새 사진 공개나 새 자세 추론은 없다. 상세 JSONL은 기존 진단 갤러리·Release 복원 안내·원 멤버 경로도 보존한다. 각 행의 정량 CSV와 상세 JSONL은 전체 실패를 보존한다.',
      '', '| 영상 | 구분 | B/I full·partial | 후보 generated·retained | N0/PoseFix | 기존 이미지 |',
      '|---|---|---:|---:|---|---|']
    for r in primary:
        index.append(f"| {r['id']} | {'admission'if r['admission_no_generation']else'후보 있음·공동 성공 없음'} | {r['complete']}·{r['partial']} | {r['generated_count']}·{r['retained_count']} | {r['N0_success']}/{r['POSEFIX_RAW_success']} | [기존 RGB]({r['existing_image_link']}) |")
    (out/'NO_CANDIDATE_INDEX_KO.md').write_text('\n'.join(index)+'\n')
    receipt={'schema':'NO_SUCCESS_CANDIDATE_SAVED_LOG_DIAG_V1','primary_images':46,'CAL_images':6,
        'PRIMARY_admission_no_generation':4,'PRIMARY_existing_candidate_no_joint_success':42,
        'PRIMARY_T_pass_candidate_exists':sum(r['generated_T_min_cm']is not None and r['generated_T_min_cm']<=5 for r in primary),
        'PRIMARY_R_pass_candidate_exists':sum(r['generated_R_min_deg']is not None and r['generated_R_min_deg']<=5 for r in primary),
        'PRIMARY_separate_T_and_R_pass_but_no_joint':sum(r['separate_T_R_pass_candidates_but_no_joint']for r in primary),
        'PRIMARY_G10_joint_success_but_G13_none_ids':[r['id']for r in primary if r['G10_joint_success_exists_but_G13_none']],
        'PRIMARY_N0_or_PoseFix_success':sum(r['N0_success']or r['POSEFIX_RAW_success']for r in primary),
        'PRIMARY_N0_and_PoseFix_fail':sum(not r['N0_success']and not r['POSEFIX_RAW_success']for r in primary),
        'all_published_RGB_links_previously_permitted':all(d['existing_public_image_permission']for d in details),
        'new_metric_scoring_calls':0,'new_NN_PnP_optimizer_candidate_training_calls':0,
        'elapsed_seconds':time.monotonic()-start,'process_CPU_seconds':time.process_time()-cpu,
        'inputs':{k:{'path':str(v),'sha256':sha(v)}for k,v in paths.items()},
        'artifacts':{p.name:sha(p)for p in out.iterdir()}}
    write_json(out/'NO_CANDIDATE_RECEIPT.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old-private',required=True);p.add_argument('--source-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(build(a.old_private,a.source_root,a.output),ensure_ascii=False))

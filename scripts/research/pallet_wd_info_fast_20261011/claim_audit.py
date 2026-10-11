"""Audit published claims without rescoring protected images; join allowed residuals."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import numpy as np
from . import common as C
from . import geometry_analysis as A


def main():
    p=argparse.ArgumentParser();p.parse_args()
    source=C.PUBLIC/A.WD_REL/'STAGE1_METRICS.json';old=A.read(source)
    numbers_path=C.PUBLIC/'_docs/experiments/pallet_paper_tables_245_20261011/PAPER_NUMBERS.json'
    numbers=A.read(numbers_path);depth_path=C.PUBLIC/A.WD_REL/'DEPTH_GATE.json';depth=A.read(depth_path)
    claims=[]
    def add(claim,path,definition,population,raw,status,interpretation):
        claims.append(dict(user_claim=claim,source=C.binding(path),definition=definition,
            population=population,raw_value=raw,display_rounding='Only display rounding; raw values retained',
            status=status,permitted_interpretation=interpretation))
    add('reference margin median9.4px; RECT245 44%<8px',numbers_path,
        'Alternative mean reprojection residual minus chosen mean residual of manual-corner PnP fits, not point-cloud distance or official weighted score gap',
        'Claim RECT245 intersects protected146; no new protected raw-row computation',None,'NOT_REVERIFIED_PROTECTED',
        'Do not claim independent verification. New REAL173 paired diagnostics are separately named below.')
    add('Base/Two-stage corner error median5.9/5.0px',numbers_path,
        'Corner pooling/support/symmetry matching not specified in claim; require matching source definition',
        'Claim RECT245; historical public result only',None,'NOT_REVERIFIED_PROTECTED',
        'New manual-visible fixed-index residual is a different statistic, not a replacement for this median.')
    for label,claimed in [('<2','N20 confusion45.0→30.0%'),('2–4','N35 confusion25.7→15.2%'),('>=16','N64 confusion4.7→4.7%')]:
        base=old['REAL_DEV']['groups']['YOLO::BASE']['bins']['reference_margin_px'][label]
        seq=old['REAL_DEV']['groups']['YOLO::N3_THEN_SUBPIX']['bins']['reference_margin_px'][label]
        raw=dict(existing_RECT319_N=base['frames'],Base_confusion=base['seed_mean']['S0']['rates']['confusion_rate']['rate'],
                 Two_stage_confusion=seq['seed_mean']['S0']['rates']['confusion_rate']['rate'])
        add('margin '+label+': '+claimed,source,
            'Legacy confusion = R>45deg AND |yaw|>=60deg; binary per seed then frame/seed mean',
            'Existing RECT319/13sessions, differs from claimed RECT245',raw,'NOT_REVERIFIED_PROTECTED_POPULATION_MISMATCH',
            'The public319 values do not independently validate245. New permitted173 bins retain all adverse cases.')
    low=numbers['altitude_confusion_bins']['bins'][0]
    add('Elevation<5deg GT-parity oracle confusion50.8→4.8%',numbers_path,
        'Legacy rotation/yaw confusion; GT parity supplied, not best-R or best-T',
        'Existing RECT245 low-elevation21frames; 63 fixed seed predictions',
        {k:v for k,v in low.items() if k!='frame_ids'},'REUSED_CORROBORATED',
        'Public aggregate matches rounding; no protected rescoring. Does not prove runtime inferability or complete6D recovery.')
    base=old['REAL_DEV']['groups']['YOLO::BASE'];seq=old['REAL_DEV']['groups']['YOLO::N3_THEN_SUBPIX']
    add('plastic/wood confusion RECT24516.3/12.9%; RECT31925.8/15.2%',source,
        'Legacy confusion; registered type includes material, ratio, sessions and viewpoints',
        'Existing RECT319 plastic194/wood125; RECT245 claim not recomputed',
        dict(RECT319_BASE={k:dict(frames=v['frames'],rate=v['seed_mean']['S0']['rates']['confusion_rate']['rate']) for k,v in base['bins']['material'].items()},
             RECT319_TWO_STAGE={k:dict(frames=v['frames'],rate=v['seed_mean']['S0']['rates']['confusion_rate']['rate']) for k,v in seq['bins']['material'].items()}),
        'REUSED_CORROBORATED_RECT319_BASE_ONLY',
        '25.8/15.2% matches Base, not Two-stage. This is material/type/viewpoint confounding, not a ratio intervention.')
    metadata_path=C.PUBLIC/'_docs/experiments/pallet_dim_conditioned_p_v1/METADATA_SENSITIVITY.json'
    meta=A.read(metadata_path)
    add('dimension ablation median0.6/P902.4px; seed0.75–0.92; symmetry0.25–0.30; movement1.7px',metadata_path,
        'Different-checkpoint ablation and within-checkpoint metadata intervention are different estimands',
        'Historical METADATA_SENSITIVITY model arm names',
        dict(arms_first_population=list(next(iter(meta['results'].values())).keys()),claimed_N3_raw_verified=False),
        'CONTRACT_MISMATCH_N4_NOT_N3',
        'This file uses N4_META_SYM; do not replace N3 sensitivity. New B results are reported under their own permitted population.')
    for pop,claim in [('REAL_DEV','real0.73'),('SYNTH_HELDOUT','synthetic0.61')]:
        g=old[pop]['groups']['YOLO::N3_THEN_SUBPIX']['score_gap_auc']
        add('Score-gap error-detection ROC-AUC '+claim,source,
            'Positive legacy rotation/yaw confusion; ROC score=-official alternative-minus-selected weighted score gap',
            dict(population=pop,frames=319 if pop=='REAL_DEV' else 1985,seeds=3),g,
            'REUSED_CORROBORATED','Pooled descriptive repeated-seed AUC, not an independent957/5955-image sample; not ADDsym AUC/AURC.')
    add('S1/S2/S3/S4 status and depth error17.2%',depth_path,
        'Absolute relative error per seed, mean three seeds per image, median231images',
        'Original depth231 non-final4 source; no new depth evaluation',
        {k:depth[k] for k in ['status','frames','absolute_relative_error_median','threshold','S4_executed']},
        'REUSED_CORROBORATED','Specific Metric3D gate failed; no S4 policy run or calibration bypass. S3 is privileged GT plane feasibility.')
    claims.append(dict(user_claim='S1/S2/S3 actual execution',
        source_bindings=[C.binding(C.PUBLIC/A.WD_REL/f'EXECUTION_{r}_{pop}.json') for r,pop in [('S1','REAL'),('S1','SYNTH'),('S2','SYNTH'),('S3','REAL'),('S3','SYNTH')]],
        raw_value={f'{r}_{pop}':A.read(C.PUBLIC/A.WD_REL/f'EXECUTION_{r}_{pop}.json')['status'] for r,pop in [('S1','REAL'),('S1','SYNTH'),('S2','SYNTH'),('S3','REAL'),('S3','SYNTH')]},
        status='REUSED_EXECUTION_RECEIPTS',definition='S2_REAL skipped after synth worsening; not unrun success',
        population='Original historical populations; no new protected processing',
        permitted_interpretation='S1 unresolved; S2 synth worsened; S3 feasibility; S4 gate blocked. Preserve adverse results.'))
    A.write('CLAIM_AUDIT.json',dict(status='DONE_WITH_PROTECTED_LIMITATIONS',claims=claims,
        reference_margin_equation='m_ref=mean_i||project(X_alt,R_alt,t_alt)-q_manual_i||_2 - mean_i||project(X_chosen,R_chosen,t_chosen)-q_manual_i||_2, same builder usable corners',
        official_gap_equation='score_alternative(q_pred,K,dims)-score_selected(q_pred,K,dims); nominal pixel scale includes penalties',
        newly_recomputed_protected_frames=0,correlated_statistic_size_is_not_identifiability_test=True))
    lines=['사용자 수치는 보호 자료를 다시 채점하지 않고 기존 공개 집계와 대조했으며, 새 원행 검산은 허용 REAL173으로 한정했다.','',
           '[확인] physical FINAL146장은 신규 특징·채점에서 제외한다. 기존319/245 집계 재사용은 독립 재검산이 아니다. [INPUT_CONTRACT.json](INPUT_CONTRACT.json)','',
           '| USER_CLAIM | 상태 | 정의·모집단 및 실제 값 | 허용 해석 |','|---|---|---|---|']
    for r in claims:
        raw=json.dumps(A.clean(r.get('raw_value')),ensure_ascii=False,separators=(',',':'))
        if len(raw)>1800:raw=raw[:1800]+'… (전체 JSON 참조)'
        lines.append('| '+r['user_claim']+' | '+r['status']+' | '+r['definition']+'; '+raw.replace('|','\\|')+' | '+r['permitted_interpretation']+' |')
    lines.extend(['','[확인] 참조 margin은 같은 수동 코너의 두 PnP 적합 평균 잔차 차이이다. 공식 gap은 예측 코너의 penalty 포함 weighted score 차이이다. 점 집합 간 거리와 서로 바꾸어 부르지 않는다.',
                  '', '[확인] 자세 혼동 `R>45° AND |yaw|≥60°`과 직접 긴면/짧은면 오선택률은 별도 지표다. 축이 같아도 자세가 실패할 수 있으며, GT parity는 best-T/best-R/best-success와 다르다.',
                  '', '[확인] 원수치·SHA·분모·source binding 전체는 [CLAIM_AUDIT.json](CLAIM_AUDIT.json)에 있다. 치수·seed 중앙값이 비슷하다는 이유로 효과0이나 식별 불가능을 선언하지 않는다.'])
    (A.DOC/'CLAIM_AUDIT.md').write_text('\n'.join(lines)+'\n')
    allow=set(A.read(A.DOC/'INPUT_CONTRACT.json')['allowed_ids'])
    manifest=A.read(C.SOURCE/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json')
    annotations={}
    for item in manifest['items']:
        if item['frame_id'] not in allow:continue
        obj=A.read(C.SOURCE/item['gt_v2_path'])['objects'][0]
        assert obj['gt_source']=='manual'
        kp=obj['keypoint_annotations'][:8]
        coords=np.asarray([k['xy'] for k in kp],float)
        mask=np.array([k['visibility']==2 and k['in_frame'] and k['source'] not in ('pnp_projected','extrapolated') for k in kp])
        assert coords.shape==(8,2)
        annotations[item['frame_id']]=(coords,mask)
    paired=[]
    for r in A.rows(C.PUBLIC/A.WD_REL/'STAGE1_ROWS_REAL.jsonl.gz'):
        if r['id'] not in allow:continue
        q=np.asarray(r['qFinal'],float);target,mask=annotations[r['id']]
        valid=mask & np.isfinite(q[:8]).all(-1)&np.isfinite(target).all(-1)
        residual=np.linalg.norm(q[:8][valid]-target[valid],axis=1)
        pose=r['pose']['S0']
        paired.append(dict(id=r['id'],session=r['session'],method=r['method'],seed=r['seed'],
            manual_visible_corners=int(valid.sum()),observed_fixed_index_corner_error=A.basic(residual),
            reference_margin_px=r['reference_margin_px'],selected=r['hypS0'],
            axis_wrong=A.axis_wrong(r['hypS0'],r['hypOracle'],r['id']),
            wd_confusion=int(pose.get('available') and pose['rotation_deg']>45 and abs(pose['yaw_deg'])>=60)))
    A.write_rows('A1_PAIRED_MARGIN_RESIDUAL_ROWS.jsonl.gz',paired)
    A.write('A1_PAIRED_MARGIN_RESIDUAL.json',dict(status='DONE',frames=173,rows=len(paired),
        residual_definition='Fixed camera-facing index, legacy manual object, visible2 and in-frame, excludes pnp_projected/extrapolated; no GT remapping',
        legacy_source_unknown='Legacy migrated human visible source unknown retained when object gt_source=manual; source filter stated explicitly',
        comparison_to_claimed_pooled_symmetry_median='Different support/estimand; not substituted',
        reference_margin_173=A.basic([r['reference_margin_px'] for r in paired if r['method']=='BASE' and r['seed']==1]),
        no_predictor_GT_input=True,new_PnP_calls=0,reference_coordinates_published=False))
    print('A1 DONE claims',len(claims),'paired_rows',len(paired),flush=True)


if __name__=='__main__':main()

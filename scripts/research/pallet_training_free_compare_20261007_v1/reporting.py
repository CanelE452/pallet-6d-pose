"""Pure saved-row TF comparison; no model/F/optimizer or manuscript I/O."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import numpy as np

from scripts.research.pallet_pose_target_6d_20261006_v1 import reporting as R
from scripts.paper.pose_metric_closure_v1.symmetry_aware_pose_metrics import pose_auc
from .common import ROOT, DOC, ARMS, read

OLD = R.OLD
NEW = ARMS
CONTROLS = ('BASE','N3_seed1','PoseFix_seed1','N0_seed1')
VISIBILITY = ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json'
CATEGORIES = ('DIRECT_VISIBLE','SELF_OCCLUDED','EXTERNAL_OCCLUDED','OUT_OF_FRAME','UNKNOWN','UNANNOTATED')
TOL = dict(R.TOLERANCES)


def bind(path):
    path=Path(path)
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),sha256=R.sha(path),bytes=path.stat().st_size)


def save(name,value,text=False):
    path=DOC/name
    assert path.resolve().is_relative_to(DOC.resolve())
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.pending')
    temp.write_text(value if text else json.dumps(R.clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def canonical_observed(row):
    c=row['corner']
    if not c['evaluable']:return np.zeros(8,bool)
    valid=np.asarray(c['canonical_valid'],bool)
    if 'canonical_observed' in row or 'canonical_observed' in c:
        observed=np.asarray(row.get('canonical_observed',c.get('canonical_observed')),bool)
        assert observed.shape==(8,) and not (observed & ~valid).any()
    else:
        # Proved for these frozen old rows: all reference points observed iff matched.
        observed=valid & bool(c['matched'])
    values=[c['canonical_errors'][i] for i in np.flatnonzero(observed)]
    assert sorted(values)==sorted(c['observed_errors']), 'canonical observed mask cannot be proven'
    return observed


def auc_summary(rows,curve=True):
    values=np.asarray([r['pose']['ADDsym_normalized'] if r['pose']['available'] else np.inf for r in rows])
    assert not np.isnan(values).any() and (values>=0).all()
    finite=values[np.isfinite(values)]
    output=dict(full=pose_auc(values,1.),conditional=pose_auc(finite,1.) if len(finite) else None,
        full_frames=len(rows),failed_frames=int(np.isinf(values).sum()),
        definition='existing normalized corresponding8corner proper-group ADD, thresholds0..0.1/1001,trapz; full includes failures=inf; not surface ADD-S')
    if curve:
        thresholds=np.linspace(0,.1,1001)
        output.update(thresholds_normalized=thresholds.tolist(),full_accuracy=[float(np.mean(values<=t)) for t in thresholds])
    return output


def summarize(rows,curve=True):
    output=R.summarize(rows)
    if rows and rows[0].get('method') in NEW:
        output['NoOp']=sum(all(r['correction']['corner_unchanged8']) for r in rows)
        output['NoOp_scope']='all initial corner0..7 exactly unchanged; YOLO object-selection index is not a correction action'
        output['detection_selected_index_distribution']=output.pop('selected_action_distribution')
    output['ADDsym_AUC']=auc_summary(rows,curve)
    return output


def mean_seed_statistics(summaries):
    def combine(values):
        if isinstance(values[0],dict):
            return {k:combine([v[k] for v in values]) for k in values[0] if k not in ('thresholds_normalized','full_accuracy','percentiles','selected_action_distribution')}
        if all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in values):
            return dict(mean=float(np.mean(values)),seed_min=float(min(values)),seed_max=float(max(values)),seed_SD=float(np.std(values,ddof=1)))
        return values[0] if all(v==values[0] for v in values) else values
    return dict(seeds=[1,2,3],data_frames=319,statistic='mean of per-seed statistics; seed SD not a data-generalization interval',values=combine(summaries))


def compare(new,base,boots):
    output=R.compare(new,base,boots)
    a,b=summarize(new,False),summarize(base,False)
    output['difference_of_statistics']={k:{stat:a['pose'][k][stat]-b['pose'][k][stat]
        if a['pose'][k][stat] is not None and b['pose'][k][stat] is not None else None
        for stat in ('mean','median','P90')} for k in R.METRICS}
    output['corner_difference_of_statistics']={k:a['corner'][k]-b['corner'][k] for k in ('matched_pooled_corner8_median_px','matched_pooled_corner8_P90_px','gross20')}
    output['full_PCK10_difference']=a['corner']['PCK']['10']-b['corner']['PCK']['10']
    output['full_ADDsym_AUC_difference']=a['ADDsym_AUC']['full']-b['ADDsym_AUC']['full']
    output['pose_failures_difference']=a['pose']['failures']-b['pose']['failures']
    flags={f'{k}_{stat}_nonworse':d is not None and d<=TOL[k] for k,stats in output['difference_of_statistics'].items() for stat,d in stats.items()}
    flags.update(failures_nonincrease=output['pose_failures_difference']<=0,
        corner_median_nonworse=output['corner_difference_of_statistics']['matched_pooled_corner8_median_px']<=1e-9,
        corner_P90_nonworse=output['corner_difference_of_statistics']['matched_pooled_corner8_P90_px']<=1e-9,
        full_PCK10_nondecrease=output['full_PCK10_difference']>=-1e-9,
        gross20_nonincrease=output['corner_difference_of_statistics']['gross20']<=1e-9)
    output['observed_nonworse_flags']=flags
    output['statistic_note']='CI: mean same-frame paired differences; paired median distinct from difference of marginal medians; repeated-use DEV, no equivalence margin/multiplicity correction'
    return output


def visibility_layers(methods,labels,population='DEV319'):
    ids=[r['id'] for r in methods['BASE']];hist=Counter();unknown_hist=Counter();states_count=Counter()
    for row in methods['BASE']:
        c=row['corner'];valid=c.get('canonical_valid',[False]*8)
        states=[labels.get((population,row['id'],i),'UNANNOTATED') for i,v in enumerate(valid) if v]
        hist[states.count('DIRECT_VISIBLE')]+=1;unknown_hist[states.count('UNANNOTATED')]+=1;states_count.update(states)
    output={}
    for name,rows in methods.items():
        assert [r['id'] for r in rows]==ids
        groups={state:[] for state in CATEGORIES}
        for row,raw in zip(rows,methods['BASE']):
            c,rc=row['corner'],raw['corner'];assert c['evaluable']==rc['evaluable']
            if not c['evaluable']:continue
            assert c['canonical_valid']==rc['canonical_valid'];obs=canonical_observed(row)
            for i,valid in enumerate(c['canonical_valid']):
                if valid:groups[labels.get((population,row['id'],i),'UNANNOTATED')].append((row['id'],c['canonical_errors'][i],bool(obs[i]),rc['canonical_errors'][i]))
        metrics={}
        for state,values in groups.items():
            full=np.asarray([v[1] for v in values],float);obs=np.asarray([v[1] for v in values if v[2]],float);raw=np.asarray([v[3] for v in values],float)
            metrics[state]=dict(reference_corners=len(full),observed_corners=len(obs),frames=len({v[0] for v in values}),
                unobserved_reference_corners=len(full)-len(obs),observed_error_px=R.distribution(obs),
                full_PCK10=float(np.mean(full<=10)) if len(full) else None,full_gross20=float(np.mean(full>20)) if len(full) else None,
                RAW_good5_to_bad10=int(((raw<5)&(full>10)).sum()),RAW_bad20_to_good10=int(((raw>20)&(full<10)).sum()))
        assert sum(v['reference_corners'] for v in metrics.values())==R.M.summary([r['corner'] for r in rows])['corners']
        output[name]=metrics
    return dict(population=population,methods=output,reference_state_counts=dict(states_count),
        direct_visible_count_histogram={str(i):hist[i] for i in range(9)},unannotated_reference_count_histogram={str(i):unknown_hist[i] for i in range(9)},
        definition='human state attached to fixed canonical GT after full-object branch selection; no per-stratum branch search; xy is not visibility; UNKNOWN/unannotated/occluded distinct; observed quantiles conditional/PCK full-reference',causal_claim=False)


def motion_summary(rows):
    moves=[];unchanged=0;fallback=Counter();status=Counter();frames=Counter()
    for row in rows:
        info=row['correction'];d=np.asarray(info['displacement_px8'],float)
        assert d.shape==(8,) and np.isfinite(d).all() and (d>=0).all()
        assert np.array_equal(np.asarray(info['corner_unchanged8'],bool),d==0)
        moves.extend(d.tolist());unchanged+=int((d==0).sum());frames[str(int((d>0).sum()))]+=1
        fallback.update(info['diagnostics']['fallback_counts']);status.update(info['diagnostics']['status_counts'])
    return dict(eligible_slots=len(moves),modified_corners=len(moves)-unchanged,unchanged_corners=unchanged,
        displacement_px=R.distribution(moves),modified_corners_per_frame_histogram=dict(frames),
        fallback_reasons=dict(fallback),corner_status_counts=dict(status),
        definition='all initial corner0..7 image-pixel displacements; center8 fixed; unsupported/nonfinite/missing initial unattempted, not function-error fallback; original collect swallows some CV-family errors, wrapper sees escaping errors only')


def historical_sources():
    paths=[ROOT/'data/pallet/results/multiteacher_corner_distill_v1/final/CORNER_EVIDENCE_RESULT.md',
        ROOT/'_docs/experiments/pallet_clean_to_pose_transfer_v1/REPORT_KO.md',ROOT/'_docs/experiments/pallet_oracle_mechanism_followup_v1/PRIOR_ATTEMPTS.md',
        ROOT/'_docs/experiments/pallet_quick_pose_loss_20261006_v1/TRAIN_PROGRESS_EXPECT6D.json']
    p=read(paths[-1]);assert p['step']==6000
    return dict(classification='EXISTING_RESULT_QUOTATION; no rerun/pooling',source_bindings=[bind(path) for path in paths],
        classical='old visible==2 supervised1594 corners; BASE median/P90 6.36/43.89px, selector7.07/45.03px; source calibrated square patch x/y radius12, not Euclidean12 limit',
        pseudo_following='old clean78 native REF-target mean R0 3.3586px to REF OCC2.6490px; partial imitation, not physical-reference accuracy or augmented loss',
        selector_only='same REF OCC S42 coordinates/two poses natural99: D9 T/R12.5120cm/6.1832deg to oldGEO11.3696cm/5.2635deg; TP90unchanged128.6450cm; Clean T/R worsen',
        agreement='old217 AGREE1218/1627 points, all six>20px corrections removed; large recovery0; oldDEV194/159, not319',
        occlusion='clean78 S42 scheduled1282/2560, applied418/2560=16.328125%; planned and actual differ',
        EXPECT6D=dict(final100_NoOp_fraction=p['window']['NoOp_fraction'],final100_mean_gradient_norm=p['window']['mean_gradient_norm'],eval_NoOp='previous SYNTH1985+REAL319 all2304=RAW',scope='pre-update repeated exposures; observed collapse of that training, not cause of all refiners/selftraining'),
        current_N3_constraint='initial object/box/score/center8/missing masks fixed; comparisons are whole-package, not causal component ablations')


def link_N0(old,ids):
    """Late availability-only link; original pre-evaluation protocol is preserved."""
    directory=ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1'
    pp,cp=directory/'PAPER_POSE_METRICS.json',directory/'REAL_DEV_METRICS.json'
    poses=read(pp)['REAL_DEV'];corners=read(cp);raw={r['id']:r for r in old['rows']['RAW']}
    checks={};rows={}
    for seed in (1,2,3):
        old_n3={r['id']:r for r in old['rows'][f'N3_seed{seed}']};name=f'N3_DIM_SYM_seed{seed}'
        assert len(poses[name])==len(corners[name])==319
        pose_equal=sum(r==old_n3[r['id']]['pose'] for r in poses[name])
        corner_equal=sum(all(r[k]==old_n3[r['id']]['corner'][k] for k in old_n3[r['id']]['corner']) for r in corners[name])
        assert pose_equal==corner_equal==319
        name=f'N0_BASE_REPLAY_seed{seed}';ps={r['id']:r for r in poses[name]};cs={r['id']:r for r in corners[name]}
        assert len(ps)==len(cs)==319 and set(ps)==set(cs)==set(ids)
        assert all(cs[i]['canonical_valid']==raw[i]['corner']['canonical_valid'] for i in ids)
        scored=[dict(id=i,session=raw[i]['session'],method=f'N0_seed{seed}',
            corner={k:cs[i][k] for k in raw[i]['corner']},pose=ps[i],selected_index=None,
            final_hypothesis=None,generating_hypothesis=None) for i in ids]
        summary=R.M.summary([r['corner'] for r in scored]);assert (summary['corners'],summary['matched'],summary['observed_corners'])==(2499,311,2445)
        rows[f'N0_seed{seed}']=scored
        checks[str(seed)]=dict(N3_pose_exact_rows=pose_equal,N3_all_shared_corner_fields_exact_rows=corner_equal,
            N3_only_extra_corner_fields=['group','object','ratio_bin'],N0_full_frames=319,
            N0_pose_available=sum(r['pose']['available'] for r in scored),N0_reference_corners=2499,N0_matched_frames=311,N0_observed_corners=2445)
    receipt=dict(status='PASS',classification='availability-only existing control link after frozen TF evaluation; no performance-based row/control selection',
        original_protocol_NOT_LINKED_preserved=True,source_bindings=[bind(pp),bind(cp)],checks=checks,
        target_reference_proof='same files N3 three seeds reproduce all a22 pose and shared corner fields exactly; N0 joins identical319IDs/canonicalGTvalid/session mapping',
        final_hypothesis='not loaded for this optional N0 reference; pose/corner saved scores reused',new_model_forwards=0,new_final_F=0,new_optimizer_updates=0)
    save('N0_LINK_RECEIPT.json',receipt)
    return rows,[bind(pp),bind(cp),bind(DOC/'N0_LINK_RECEIPT.json')]


def runtime_summary():
    path=DOC/'RUNTIME.json'
    if not path.exists():return dict(status='BLOCKED_RUNTIME',reason='new matched runtime unavailable'),[]
    rt=read(path)
    if rt.get('status')=='DONE':assert rt['complete'] and rt['GT_inference_access'] is False
    return rt,[bind(path)]


def native_cap_equality(methods):
    output={}
    for algorithm in ('SUBPIX','CVRANK'):
        a,b=methods[algorithm+'_NATIVE'],methods[algorithm+'_CAP1']
        assert [r['id'] for r in a]==[r['id'] for r in b]
        output[algorithm]=dict(frames=len(a),native_points_exact_equal=sum(np.array_equal(np.asarray(x['native_points'],float),np.asarray(y['native_points'],float),equal_nan=True) for x,y in zip(a,b)),
            stored_pose_and_hypothesis_exact_equal=sum(x['pose']==y['pose'] and x.get('final_hypothesis')==y.get('final_hypothesis') for x,y in zip(a,b)),scope='two retained deterministic output variants; not two independent trials')
    return output


def square_summary(labels):
    path=DOC/'SQUARE_PREDICTIONS.json'
    if not path.exists():return dict(status='BLOCKED_INPUT',reason='no completed direct-verified square119 packet; no replacement clean pool'),[]
    packet=read(path)
    assert packet['complete'] and packet['direct_input_verified'] and packet['new_F_calls']==0
    relative=Path(packet['raw_rows_file'])
    rows_path=ROOT/relative if relative.parts[0]=='_docs' else DOC/relative
    assert rows_path.resolve().is_relative_to(DOC.resolve())
    assert R.sha(rows_path)==packet['raw_rows_sha256'] and rows_path.stat().st_size==packet['raw_rows_bytes']
    assert R.sha(DOC/'SQUARE_PROTOCOL.json')==packet['square_protocol_sha256']
    with gzip.open(rows_path,'rt') as stream:flat=[json.loads(line) for line in stream if line.strip()]
    assert len(flat)==476
    result=dict(status='DONE',frames=119,sessions=1,new_F_calls=0,CI='NA: single correlated session; no session-generalization interval',modes={})
    for mode,count in [('manual_declared',602),('manual_in_frame',600)]:
        def pick(row):
            return dict(id=row['id'],session=row['session'],corner=row['corners'][mode],canonical_observed=row['canonical_observed'][mode])
        methods={name:[pick(r) for r in rows] for name,rows in packet['baselines'].items()}
        ids=[r['id'] for r in methods['BASE']];assert len(ids)==119 and len(set(ids))==119
        for name in NEW:methods[name]=R.ordered([pick(r) for r in flat if r['method']==name],ids)
        summaries={name:R.M.summary([r['corner'] for r in rows]) for name,rows in methods.items()}
        assert all(s['corners']==count for s in summaries.values())
        result['modes'][mode]=dict(summaries=summaries,RAW_canonical_damage={name:R.M.damage([r['corner'] for r in methods['BASE']],[r['corner'] for r in rows]) for name,rows in methods.items()},visibility=visibility_layers(methods,labels,'GREEN0918'))
    result['PoseFix_status']=packet['posefix_status']
    return result,[bind(path),bind(rows_path),bind(DOC/'SQUARE_PROTOCOL.json'),bind(DOC/'SQUARE_PROOF.json')]


def report():
    started=time.monotonic();old_path=OLD/'results/A_REAL_DEV_BASELINES.json';old=read(old_path)
    ids=[r['id'] for r in old['rows']['RAW']];assert len(ids)==len(set(ids))==319
    methods={'BASE':R.ordered(old['rows']['RAW'],ids)}
    for family in ('N3','PoseFix'):
        for seed in (1,2,3):methods[f'{family}_seed{seed}']=R.ordered(old['rows'][f'{family}_seed{seed}'],ids)
    n0,n0_bindings=link_N0(old,ids);methods.update(n0)
    pp=DOC/'PREDICTIONS.json';p=read(pp);protocol=read(DOC/'PROTOCOL.json');rp=DOC/'PREDICTIONS.jsonl.gz'
    assert p['complete'] and p['GT_selection'] is False and p['methods']==list(NEW) and p['full_denominator_per_arm']==319
    assert p['protocol_sha256']==R.sha(DOC/'PROTOCOL.json') and protocol['baseline']['sha256']==R.sha(old_path)
    target_path=ROOT/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json';assert old['target_sha256']==R.sha(target_path)
    assert p['raw_rows_file']==rp.name and p['raw_rows_sha256']==R.sha(rp) and p['raw_rows_bytes']==rp.stat().st_size
    with gzip.open(rp,'rt') as stream:flat=[json.loads(line) for line in stream if line.strip()]
    assert len(flat)==1276 and set(r['method'] for r in flat)==set(NEW)
    for name in NEW:
        methods[name]=R.ordered([r for r in flat if r['method']==name],ids)
        for row,raw in zip(methods[name],methods['BASE']):
            assert row['session']==raw['session'] and row['corner']['canonical_valid']==raw['corner']['canonical_valid']
            assert row['F_attempt'] and row['F_complete'] and row['F_available']==row['pose']['available']
            assert row['reference_sha256']==protocol['baseline']['sha256'] and row['fixed_metadata']['preserved']
    assert len({r['session'] for r in methods['BASE']})==13
    boots=dict(session=R.SharedBootstrap(ids,[r['session'] for r in methods['BASE']]))
    summaries={name:summarize(rows) for name,rows in methods.items()};c=summaries['BASE']['corner']
    assert (c['corners'],c['matched'],c['observed_corners'])==(2499,311,2445)
    comparisons={f'{new}_minus_{base}':compare(methods[new],methods[base],boots) for new in NEW for base in CONTROLS}
    families={};seed_comparisons={}
    for family in ('N3','PoseFix','N0'):
        bases=[methods[f'{family}_seed{s}'] for s in (1,2,3)]
        families[family]=mean_seed_statistics([summaries[f'{family}_seed{s}'] for s in (1,2,3)])
        for name in NEW:
            seed_comparisons[f'{name}_minus_{family}']=dict(per_seed={str(s):compare(methods[name],methods[f'{family}_seed{s}'],boots) for s in (1,2,3)},paired_seed_mean=R.seed_mean_compare([methods[name]]*3,bases,boots),
                difference_from_mean_per_seed_statistics={k:{stat:summaries[name]['pose'][k][stat]-float(np.mean([summaries[f'{family}_seed{s}']['pose'][k][stat] for s in (1,2,3)])) for stat in ('mean','median','P90')} for k in R.METRICS},deterministic_TF_runs=1,data_frames=319,bootstrap_units=13,seed_replicated_as_data=False)
    vs=read(VISIBILITY);labels={(r['population'],r['frame_id'],r['corner_id']):r['category'] for r in vs['rows']}
    sessions={session:{name:summarize([r for r in rows if r['session']==session],False) for name,rows in methods.items()} for session in sorted({r['session'] for r in methods['BASE']})}
    runtime,rt_bindings=runtime_summary();square,sq_bindings=square_summary(labels)
    paths=[old_path,pp,rp,DOC/'PROTOCOL.json',target_path,VISIBILITY,Path(__file__),Path(R.__file__),Path(R.M.__file__),Path(__file__).with_name('common.py'),ROOT/'scripts/paper/pose_metric_closure_v1/symmetry_aware_pose_metrics.py']
    output=dict(schema='pallet_training_free_compare_20261007_v1',status='DONE',full_frames=319,populations=dict(pose_full=319,reference_corners=2499,matched_2D_frames=311,observed_corners=2445,sessions=13),summaries=summaries,
        comparisons=comparisons,three_seed_comparisons=seed_comparisons,existing_seed_family_statistics=families,
        visibility=visibility_layers(methods,labels),sessions=sessions,RAW_canonical_damage={name:R.M.damage([r['corner'] for r in methods['BASE']],[r['corner'] for r in rows]) for name,rows in methods.items()},
        correction={name:motion_summary(methods[name]) for name in NEW},native_cap_equality=native_cap_equality(methods),runtime=runtime,square119=square,bootstrap={key:b.meta() for key,b in boots.items()},bootstrap_primary='session',
        historical_sources=historical_sources(),source_bindings=[bind(path) for path in paths]+rt_bindings+sq_bindings+n0_bindings,
        method_provenance=protocol['methods'],accuracy_execution=p['execution'],accuracy_replay_seconds=p['elapsed_seconds'],reference_scope='reused DEV reconstructed from2D/registered dimensions; independent physical same-relative-pose pairs0/BLOCKED_DATA; exploratory CI',
        execution=dict(new_model_forwards=0,new_final_F=0,new_optimizer_updates=0,new_pose_costs=0,seconds=time.monotonic()-started))
    save('RESULTS.json',output);save('RESULT_KO.md',render(output),True);save('README_KO.md',readme(),True)
    return output


def fmt(v,n=4):return 'NA' if v is None else f'{v:.{n}f}'


def render(result):
    s=result['summaries'];better=[m for m in NEW if all(result['comparisons'][f'{m}_minus_N3_seed1']['difference_of_statistics'][k]['median'] < -TOL[k] for k in ('translation_cm','rotation_deg'))]
    n3,sub=s['N3_seed1'],s['SUBPIX_NATIVE']
    mixed='[확인] SUBPIX NATIVE는 N3 seed1 대비 T 중앙값 '+fmt(n3['pose']['translation_cm']['median'],3)+'→'+fmt(sub['pose']['translation_cm']['median'],3)+'cm, ADD 중앙값 '+fmt(n3['pose']['ADDsym_m']['median'],6)+'→'+fmt(sub['pose']['ADDsym_m']['median'],6)+'m이다. R 중앙값은 '+fmt(n3['pose']['rotation_deg']['median'],3)+'→'+fmt(sub['pose']['rotation_deg']['median'],3)+'degree, 2D 중앙값은 '+fmt(n3['corner']['matched_pooled_corner8_median_px'],3)+'→'+fmt(sub['corner']['matched_pooled_corner8_median_px'],3)+'px다. 위치·ADD 이득과 회전·코너 손상이 공존해 전체 대체나 동등성을 입증하지 않는다.'
    lines=['# 팔레트 비학습 보정 고정 비교','',
        '[확인] 비학습 보정이 N3를 대체할 만큼 좋아졌는가: '+('N3 seed1보다 T/R 중앙값이 함께 낮은 출력은 '+', '.join(better)+'이다. 손상·꼬리·시간·탐색적 CI를 함께 판단한다.' if better else '이번 두 알고리즘·네 출력에서 N3 seed1보다 T/R 중앙값이 함께 낮은 방법은 없다.')+' 전체 방법군이나 동등성 결론으로 확대하지 않는다.','',
        mixed,'',
        '[확인] 같은 초기 YOLO319장·13세션. 추가 보정 학습0이며 전체 시스템에는 기존 YOLO/N3/PoseFix의 학습이 있다. 2D 참조2499점·매칭311장·관측2445점, 자세분모319장. NATIVE/CAP1 네 출력을 모두 보존했다.','',
        '[확인] CVRANK의 반경12는 기존 지시문 기본값이고 x/y 사각 창이다. 후보 설정은 과거 SOURCE_DEV 합성 val에서 cov5 최대·후보수 최소 grid로 선택했고 네 rank 항은 동일 비중으로 수동 고정했다. 현재 새 설정 선택·가중치 학습은0이다. SUBPIX win5/count40/epsilon0.001과CAP1=원영상대각1%를 사전 고정했다.','',
        '| 방법 | 2D median/P90 px | full PCK10/gross20 | T median/P90 cm | R median/P90 degree | ADD median/P90 m | F성공/319 | full ADD AUC |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in (*CONTROLS,*NEW):
        p=s[name]['pose'];c=s[name]['corner'];fields=[fmt(c['matched_pooled_corner8_median_px'])+'/'+fmt(c['matched_pooled_corner8_P90_px']),fmt(c['PCK']['10'],6)+'/'+fmt(c['gross20'],6)]
        fields+=['/'.join(fmt(p[k][stat],6 if k=='ADDsym_m' else 4) for stat in ('median','P90')) for k in R.METRICS]
        lines.append('| '+name+' | '+' | '.join(fields)+f" | {p['available']}/319 | {fmt(s[name]['ADDsym_AUC']['full'],6)} |")
    lines+=['','[확인] PCK/gross/AUC는0..1분율. 2D quantile은관측조건부,PCK/gross는전체참조+결측페널티. ADD는정준8점proper-group대응평균이며표면ADD-S가아니다. 기존정규화ADD AUC는직경대비0..0.1/1001임계값trapz이고실패inf를포함한다.','',
        '| TF−대조 | T paired mean [session95CI] cm | R paired mean [session95CI] degree | ADD paired mean [session95CI] m | T/R difference-of-medians | F실패Δ |','|---|---:|---:|---:|---:|---:|']
    for name,c in result['comparisons'].items():
        vals=[]
        for metric in R.METRICS:
            st=c['statistics'][metric]['session'];ci=st['CI95'];vals.append(fmt(st['mean_paired_difference'],6)+(f' [{fmt(ci[0],6)}, {fmt(ci[1],6)}]' if ci else ' [NA]'))
        med='/'.join(fmt(c['difference_of_statistics'][k]['median']) for k in ('translation_cm','rotation_deg'))
        lines.append('| '+name+' | '+' | '.join(vals)+f" | {med} | {c['pose_failures_difference']} |")
    lines+=['','[확인] A=TF−BASE, B=TF−N3/PoseFix. 음수는낮은오차방향이며CI가0을포함하면확정개선이아니다. 모든비교·seed가원13세션동일draw10000회/seed20260917을공유한다. paired차이중앙값과marginal중앙값차이는다르다. 전체mean/P90·각seed/seed통계평균·13세션결과는RESULTS에보존했고TF를가짜seed3개로복제하지않았다. TF NoOp는8코너정확무이동프레임수이며보존된검출객체선택index0을actionNoOp로세지않는다.','',
        '[재집계] N0는 동일319장 saved pose/2D를 뒤늦게 연결한 참고행이다. 같은 원파일의N3 3seed가a22참조를완전히재현함을검산했고원PROTOCOL의NOT_LINKED를보존했다. 성능에따른선별이나추가N0 F/NN은0이며[N0_LINK_RECEIPT](N0_LINK_RECEIPT.json)에근거를남겼다.','',
        '| 방법 | 수정/초기유지 코너 | 이동 mean/median px | RAW good<5→bad>10 / bad>20→good<10 | fallback |','|---|---:|---:|---:|---|']
    for name in NEW:
        m=result['correction'][name];d=result['RAW_canonical_damage'][name]
        lines.append(f"| {name} | {m['modified_corners']}/{m['unchanged_corners']} | {fmt(m['displacement_px']['mean'])}/{fmt(m['displacement_px']['median'])} | {d['good5_to_bad10']}/{d['bad20_to_good10']} | {m['fallback_reasons']} |")
    lines+=['', '[확인] NATIVE/CAP1 좌표·저장 자세의 정확 일치 집계: '+str(result['native_cap_equality'])+'. 같은 출력은 독립 반복이나 두 번의 성공으로 세지 않는다.']
    lines+=['','| 방법 | 가시성 | GT/관측 코너 | 2D median/P90 px | full PCK10 | RAW 양호손상/큰오류복구 |','|---|---|---:|---:|---:|---:|']
    for name in (*CONTROLS,*NEW):
        for state,g in result['visibility']['methods'][name].items():
            if not g['reference_corners']:continue
            d=g['observed_error_px'];lines.append(f"| {name} | {state} | {g['reference_corners']}/{g['observed_corners']} | {fmt(d['median'])}/{fmt(d['P90'])} | {fmt(g['full_PCK10'],6)} | {g['RAW_good5_to_bad10']}/{g['RAW_bad20_to_good10']} |")
    lines+=['',f"[확인] 직접가시0..8점 영상히스토그램: {result['visibility']['direct_visible_count_histogram']}. 미주석/UNKNOWN/가림/실패를구분한다. 전체객체대칭대응후GT정준상태를붙였고층마다대칭을재선택하지않았다. 가림인과효과나독립참조가아니다.",'',
        '[확인] 시간은새RUNTIME의동일환경패널만사용하며과거14.47/25.24ms에새CPU비용을더하지않는다. 보정단독과RAM원영상→전처리/YOLO→보정→실제F를구분한다.']
    rt=result['runtime']
    if rt.get('status')=='DONE':
        lines+=['','| 경로 | 전체 median/P90 ms | 보정단독 median/P90 ms |','|---|---:|---:|']
        for name,v in rt['summaries'].items():
            p,t=v['full_pipeline'],v['stage_only'];stage='NA' if t is None else fmt(t['median_ms'])+'/'+fmt(t['p90_ms'])
            lines.append(f"| {name} | {fmt(p['median_ms'])}/{fmt(p['p90_ms'])} | {stage} |")
        e=rt['execution'];w=rt['warmup_accounting']['BASE'];nn=rt['execution_model_forwards']
        lines+=['',f"[확인] 시간 계측 pipeline/F {e['pipeline_calls_started']}회: 이전 실패 준비 {e['previous_failed_warmup_consumed']}회와 유효 {e['new_pipeline_calls_started']}회. 측정 {e['full_measured_rows']}행(7경로×130); BASE 유효 준비 {w['valid']}회·실패 소비 {w['failed_consumed']}회, 나머지6경로 준비 각20회. detector predict {e['detector_calls']}회, N3/PoseFix head 각 {nn['N3']}/{nn['PoseFix']}회, 추가 stage F {e['stage_only_additional_F_calls']}회. 이전 실패·CPU fallback은 표 통계에 합치지 않았으며 내부 초기화 forward·수치 parity·원측정은 RUNTIME.json에 구분했다."]
    else:lines+=['','[확인] 전체시간상태: '+rt.get('status','BLOCKED_RUNTIME')+'. 추정비용으로대체하지않는다.']
    square=result['square119']
    lines+=['','[확인] 정사각형119 상태: '+square['status']+'. 단일세션2D보조이며새clean집합으로대체하지않는다.']
    if square['status']=='DONE':
        lines+=['','| square119 모드 | 방법 | GT/관측 코너 | 2D median/P90 px | full PCK10/gross20 |','|---|---|---:|---:|---:|']
        for mode,data in square['modes'].items():
            for name,c in data['summaries'].items():
                lines.append(f"| {mode} | {name} | {c['corners']}/{c['observed_corners']} | {fmt(c['matched_pooled_corner8_median_px'])}/{fmt(c['matched_pooled_corner8_P90_px'])} | {fmt(c['PCK']['10'],6)}/{fmt(c['gross20'],6)} |")
        lines+=['','[확인] declared602/inframe600를분리하고한세션일반화CI·자세평가는없다. 기존N3seed별/가시성/hist는RESULTS.square119에보존한다.']
    h=result['historical_sources'];lines+=['','[기존 결과 인용] '+h['classical']+'. 현재2499참조와합치지않고과거원인단정을복사하지않는다.','',
        '[기존 결과 인용] E/F: '+h['pseudo_following']+'. '+h['selector_only']+'. '+h['agreement']+'. '+h['occlusion']+'. pseudo추종/평가오차,D9→GEO선택변경,planned/applied를구분하고99/128/217의조건을현재319과합치지않는다.','',
        '[기존 결과 인용] EXPECT6D final100 NoOp='+fmt(h['EXPECT6D']['final100_NoOp_fraction'],6)+', mean gradientnorm='+str(h['EXPECT6D']['final100_mean_gradient_norm'])+'. 이전2304NoOp=RAW는그학습퇴화관측이며이번TF나모든보정실패의공통원인증명이아니다.','',
        '[확인] 반복DEV탐색적구간이고다중비교보정·사전실용동등성폭이없다. 유의하지않음은동등성입증이아니다. 2D/등록치수재구성참조이며독립물리실측0쌍/BLOCKED_DATA다. opticalcorner와semanticcorner는일치보장이없다. 원설정CV는source calibration이력이있으며사후rank/가중/반경변경으로구제하지않았다. 무이동=BASE를개선으로부르지않고이번두영상기반방법의범위만판단한다. 렌더링/CAD/다중시점/기초모델방식전반에대한결론이아니다.','',
        '[확인] 현재N3는 객체/박스/점수/중심8/결측을 고정한다. 정확도 F '+str(result['accuracy_execution']['new_F_accuracy'])+'회 + BASE parity '+str(result['accuracy_execution']['new_F_BASE_parity'])+'회; 정확도 detector/N3/PoseFix 재추론0, 새 학습0. 보고 CPU 집계 '+fmt(result['execution']['seconds'],3)+'초, 보고 자체 NN/F/optimizer0. 추가 pose-cost 생성·논문/LaTeX/PDF/bib 수정·빌드0.','',
        '[RESULTS](RESULTS.json) · [PROTOCOL](PROTOCOL.json) · [PREDICTIONS](PREDICTIONS.json) · [RUNTIME](RUNTIME.json) · [CHECKS](CHECKS.json)','']
    return '\n'.join(lines)


def readme():
    return '\n'.join(['# 비학습 보정 비교','',
        '[RESULT_KO.md](RESULT_KO.md)의결론과[RESULTS.json](RESULTS.json)의전체/seed/가시성/세션/paired값을읽는다. 같은YOLO319장과두알고리즘네출력의고정평가다.','',
        '추가보정학습0. 현재F·기존N3/PoseFix3seed원행·원본YOLOcache·가시성상태를읽기전용재사용한다. a22baseline은별도불변worktree또는PALLET_BASELINE_ROOT이며개인원영상/가중치는공개물에포함하지않는다.','',
        '```bash','PYTHONDONTWRITEBYTECODE=1 /home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B -m scripts.research.pallet_training_free_compare_20261007_v1.reporting','```','',
        'reporting은완료된PREDICTIONS와기존원행만집계하며모델/F를실행하지않는다. 입력/방법/예산은PROTOCOL,수정좌표·F는PREDICTIONS,새동일환경시간은RUNTIME,검산·호출수는CHECKS. 보고전실제선행평가가필요하고원고/LaTeX/PDF/bib변경·빌드0.',''])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args(argv)
    result=report();print(json.dumps(dict(status=result['status'],execution=result['execution']),ensure_ascii=False))


if __name__=='__main__':main()

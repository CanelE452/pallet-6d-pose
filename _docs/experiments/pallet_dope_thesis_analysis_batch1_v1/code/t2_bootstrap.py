"""T2: fixed paired frame/recording bootstrap from derived score JSON only.

10,000 multinomial resamples, PCG64 seed20261026, percentile95(linear).
All comparisons within a panel/unit share one sample-weight matrix. No original
annotation/RGB/model is opened. Recording IDs come exclusively from T0 metadata.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

import numpy as np
import common as C
from t1_cached_bias import EXPECTED,ROLES,ensure_new,require,frame_signature

B=10000
METRICS=('PCK5','PCK10','PCK20','coverage','conditional_precision10')
COMPARISONS=(
    ('flip_vs_control_R1','flip_R1','control_R1'),
    ('flip_vs_control_R2','flip_R2','control_R2'),
    ('flip_vs_loo_R1','flip_R1','loo_R1'),
    ('flip_vs_loo_R2','flip_R2','loo_R2'),
    ('flip_R2_vs_R1','flip_R2','flip_R1'),
    ('control_R2_vs_R1','control_R2','control_R1'),
)


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def count_matrix(frames,panel):
    rows=[]
    for frame in frames:
        points=frame[panel];errors=[p['error_px'] for p in points if p['error_px'] is not None]
        require(all(math.isfinite(e) and e>=0 for e in errors),'Invalid stored error')
        rows.append((len(points),len(errors),sum(e<=5 for e in errors),sum(e<=10 for e in errors),sum(e<=20 for e in errors)))
    return np.asarray(rows,dtype=np.int64)


def metric_values(counts):
    """Point-weighted ratios from sums [reference,observed,correct5,10,20]."""
    n,observed,c5,c10,c20=counts.T
    def ratio(num,den):
        return np.divide(num,den,out=np.full(num.shape,np.nan,dtype=float),where=den!=0)
    return np.column_stack((ratio(c5,n),ratio(c10,n),ratio(c20,n),ratio(observed,n),ratio(c10,observed)))


def load_panels():
    core=C.read(C.NOFLIP/'SCORES.json');extra=C.read(C.NOFLIP/'PLASTIC184_SCORES.json')
    summary=C.read(C.NOFLIP/'SUMMARY.json');summary184=C.read(C.NOFLIP/'PLASTIC184_SUMMARY.json')
    require(summary['scores_sha256']==C.sha(C.NOFLIP/'SCORES.json'),'Core score SHA differs')
    require(summary184['scores_sha256']==C.sha(C.NOFLIP/'PLASTIC184_SCORES.json'),'PLASTIC184 score SHA differs')
    panels=[]
    for kind,(nf,nr) in EXPECTED.items():
        panels.append(dict(id='PRIMARY_'+kind,family='PRIMARY',kind=kind,n_frames=nf,n_references=nr,
                           panel='all_annotated_reference',payload=core,source=C.NOFLIP/'SCORES.json'))
    panels.append(dict(id='GREEN_MANUAL',family='GREEN_MANUAL',kind='GREEN',n_frames=119,n_references=600,
                       panel='verified_manual_only',payload=core,source=C.NOFLIP/'SCORES.json'))
    panels.append(dict(id='PLASTIC184',family='PLASTIC184',kind='PLASTIC',n_frames=184,n_references=1413,
                       panel='all_annotated_reference',payload=extra,source=C.NOFLIP/'PLASTIC184_SCORES.json'))
    for item in panels:
        roles=C.roles(item['kind']);matrices={};baseline_frames=None
        for role in ROLES:
            require(roles[role] in item['payload']['models'],'Missing fixed model alias: '+roles[role])
            frames=[r for r in item['payload']['models'][roles[role]]['frames'] if C.kind_of(r)==item['kind']]
            require(len(frames)==item['n_frames'] and len({r['id'] for r in frames})==len(frames),'Wrong panel frame population')
            require(sum(len(r[item['panel']]) for r in frames)==item['n_references'],'Wrong panel reference denominator')
            if item['panel']=='verified_manual_only':
                require(all(p['source']=='manual_click' for f in frames for p in f[item['panel']]),'Manual panel contains another reference source')
            if baseline_frames is None:baseline_frames=frames
            else:
                require([r['id'] for r in frames]==[r['id'] for r in baseline_frames],'Paired panel frame order differs')
                require(all(frame_signature(a,item['panel'])==frame_signature(b,item['panel']) for a,b in zip(frames,baseline_frames)),
                        'Paired reference identity/denominator differs')
            matrices[role]=count_matrix(frames,item['panel'])
        item['frames']=baseline_frames;item['matrices']=matrices
        item.pop('payload')
    return panels


def recording_plan(frames,metadata):
    registry=metadata['frame_records'];missing=[];source_by_id={};recording_by_frame=[]
    for frame in frames:
        row=registry.get(frame['id']);rid=None if row is None else row.get('recording_id')
        require(rid is None or isinstance(rid,str),'Recording ID must be explicit string/null')
        if rid is None or not rid.strip():missing.append(frame['id']);recording_by_frame.append(None);continue
        require(isinstance(row.get('recording_source'),str) and bool(row['recording_source']),
                'Known recording has no metadata source: '+frame['id'])
        recording_by_frame.append(rid);source_by_id[frame['id']]=row['recording_source']
    known=sorted({x for x in recording_by_frame if x is not None})
    if missing:status='NOT_COMPUTED_MISSING_RECORDING_ID'
    elif len(known)<3:status='NOT_COMPUTED_FEWER_THAN3_RECORDINGS'
    else:status='READY'
    return dict(status=status,known_recordings=known,n_known_recordings=len(known),missing_frame_ids=missing,
        n_missing_recording_ids=len(missing),recording_by_frame=recording_by_frame,source_by_frame=source_by_id,
        body_primary_unit=('unavailable_due_to_missing_recording_metadata' if missing else 'recording' if len(known)>=5 else 'frame'))


def resample_weights(unit_ids):
    require(C.SEED==20261026 and B==10000,'Predeclared seed/replicate count changed')
    count=len(unit_ids);require(count>0,'No bootstrap units')
    # Reset the same predeclared seed for each distinct population/unit. No result-dependent seed.
    rng=np.random.Generator(np.random.PCG64(C.SEED))
    weights=rng.multinomial(count,np.full(count,1/count),size=B).astype('<i8',copy=False)
    require(np.all(weights.sum(axis=1)==count),'Invalid bootstrap sample sizes')
    identity=json.dumps(unit_ids,ensure_ascii=False,separators=(',',':')).encode()
    proof=dict(algorithm='numpy.Generator(PCG64(seed)).multinomial(n_units, equal_probabilities, size10000)',seed=C.SEED,
        seed_reset_policy='Same fixed seed reset for each population/unit; never selected from results.',replicates=B,
        n_units=count,ordered_unit_ids=list(unit_ids),ordered_unit_ids_sha256=hashlib.sha256(identity).hexdigest(),
        weights_sha256=hashlib.sha256(weights.tobytes(order='C')).hexdigest(),weights_encoding='C-order little-endian int64',
        numpy_version=np.__version__,paired_weights_shared_by_all_models_and_comparisons=True,no_resampling_of_undefined_replicates=True)
    return weights,proof


def bootstrap_panel(item,metadata):
    matrices=item['matrices'];frames=item['frames'];plan=recording_plan(frames,metadata)
    point_metrics={role:metric_values(matrix.sum(axis=0,keepdims=True))[0] for role,matrix in matrices.items()}
    rows=[];sampling=[]
    for unit in ('frame','recording'):
        status='READY' if unit=='frame' else plan['status']
        generated=status=='READY'
        if generated:
            if unit=='frame':
                unit_ids=[r['id'] for r in frames];unit_matrices=matrices
            else:
                unit_ids=plan['known_recordings'];index={rid:i for i,rid in enumerate(unit_ids)}
                unit_matrices={}
                for role,matrix in matrices.items():
                    summed=np.zeros((len(unit_ids),matrix.shape[1]),dtype=np.int64)
                    for frame_index,rid in enumerate(plan['recording_by_frame']):summed[index[rid]]+=matrix[frame_index]
                    require(np.array_equal(summed.sum(axis=0),matrix.sum(axis=0)),'Recording grouping lost frames/points')
                    unit_matrices[role]=summed
            weights,proof=resample_weights(unit_ids);proof.update(panel=item['id'],unit=unit)
            sampling.append(proof)
            sampled_metrics={role:metric_values(weights@matrix) for role,matrix in unit_matrices.items()}
        for comparison,candidate,comparator in COMPARISONS:
            a_count=matrices[candidate].sum(axis=0);b_count=matrices[comparator].sum(axis=0)
            require(a_count[0]==b_count[0]==item['n_references'],'Point denominator differs between paired models')
            for metric_index,metric in enumerate(METRICS):
                estimate=point_metrics[candidate][metric_index]-point_metrics[comparator][metric_index]
                low=high=None;defined=undefined=ca_undefined=co_undefined=0;decision='미계산';ci_scope='not_computed'
                if generated:
                    a=sampled_metrics[candidate][:,metric_index];b=sampled_metrics[comparator][:,metric_index]
                    valid=np.isfinite(a)&np.isfinite(b);delta=a[valid]-b[valid]
                    defined=int(valid.sum());undefined=B-defined
                    ca_undefined=int((~np.isfinite(a)).sum());co_undefined=int((~np.isfinite(b)).sum())
                    if defined:
                        bounds=np.percentile(delta,[2.5,97.5],method='linear');low,high=map(float,bounds)
                        decision='높았다' if low>0 else '낮았다' if high<0 else '차이 불명확'
                        ci_scope='all_replicates' if undefined==0 else 'conditional_on_defined_replicates_no_redraw'
                    else:ci_scope='no_defined_paired_replicates';decision='미정의'
                rows.append(dict(panel=item['id'],family=item['family'],kind=item['kind'],unit=unit,comparison=comparison,
                    candidate_role=candidate,comparator_role=comparator,candidate_model=C.roles(item['kind'])[candidate],
                    comparator_model=C.roles(item['kind'])[comparator],metric=metric,n_frames=item['n_frames'],
                    n_reference_points=item['n_references'],n_known_recordings=plan['n_known_recordings'],
                    n_missing_recording_ids=plan['n_missing_recording_ids'],body_primary_unit=plan['body_primary_unit'],
                    unit_is_recommended_for_body=(unit==plan['body_primary_unit']),
                    candidate_observed=int(a_count[1]),comparator_observed=int(b_count[1]),
                    candidate_point_estimate=finite_or_none(point_metrics[candidate][metric_index]),
                    comparator_point_estimate=finite_or_none(point_metrics[comparator][metric_index]),
                    delta_fraction=finite_or_none(estimate),delta_pp=None if not np.isfinite(estimate) else float(100*estimate),
                    ci_low_fraction=low,ci_high_fraction=high,ci_low_pp=None if low is None else 100*low,
                    ci_high_pp=None if high is None else 100*high,decision=decision,
                    bootstrap_status=status if not generated else 'COMPUTED' if defined else 'NO_DEFINED_REPLICATES',
                    bootstrap_requested=B,bootstrap_generated=B if generated else 0,bootstrap_defined=defined,
                    bootstrap_undefined=undefined,candidate_undefined_replicates=ca_undefined,
                    comparator_undefined_replicates=co_undefined,CI_scope=ci_scope,seed=C.SEED,
                    source_score_file=str(item['source']),multiple_comparison_adjustment='none'))
    return rows,dict(panel=item['id'],kind=item['kind'],recording_plan=plan,sampling=sampling,
        point_metrics={role:{metric:finite_or_none(values[i]) for i,metric in enumerate(METRICS)} for role,values in point_metrics.items()})


def markdown(family,rows):
    def fmt(v):return 'NA' if v is None else f'{v:.4f}'
    text=f'# T2 {family} paired bootstrap\n\n'
    text+='모든 변화량은 후보−비교군의 percentage point(%p)다. 영상/녹화 단위로 같은 복원추출 가중치를 두 모델에 적용하고, 선택된 모든 기준점·관측점·정답점의 합으로 비율을 계산했다. 영상당 비율을 단순 평균하지 않는다.\n\n'
    text+='10000회, seed20261026, PCG64 multinomial, percentile95 및 linear quantile. 같은 panel×unit 안의 모든 모델·비교는 동일 표본을 사용한다. 다중 비교 보정은 하지 않았다. CI하한>0은 “높았다”, 상한<0은 “낮았다”, 그 외는 “차이 불명확”이다.\n\n'
    text+='조건부 정밀도는 error≤10 정답점 수/관측점 수다. 0분모 반복은 미정의로 기록하며 재추출하지 않는다. 미정의가 있으면 정의된 반복만의 조건부 percentile을 표시하고 유효/미정의 반복 수를 함께 보고한다. 녹화 ID가 하나라도 없거나 확인된 녹화가3개 미만이면 녹화CI를 계산하지 않는다. 완전한 녹화가5개 이상이면 녹화CI가 본문 기준이다.\n\n'
    text+=C.mdtable(['Panel','Unit','Known rec','Missing IDs','Pair','Metric','Δ %p','CI low','CI high','유효/미정의','판정','상태'],
        [[r['panel'],r['unit'],r['n_known_recordings'],r['n_missing_recording_ids'],r['comparison'],r['metric'],fmt(r['delta_pp']),
          fmt(r['ci_low_pp']),fmt(r['ci_high_pp']),str(r['bootstrap_defined'])+'/'+str(r['bootstrap_undefined']),r['decision'],r['bootstrap_status']] for r in rows])
    text+='\n반복 DEV·단일 optimizer seed의 기술적 비교다. 영상 단위CI는 같은 녹화의 연속 프레임 의존성을 반영하지 못한다. GREEN manual600 및 PLASTIC184는 주 평가와 분리된 보조 패널이며 독립 표본으로 합산하지 않는다. PLASTIC184의 추가56은 FINAL역할 자료의 이미 계산된 점수만 사용했고 원본 영상/주석은 열지 않았다.\n'
    return text


def main():
    output_paths=[]
    for family in ('PRIMARY','GREEN_MANUAL','PLASTIC184'):
        output_paths.extend((C.OUT/f'tables/T2_{family}_CI.csv',C.OUT/f'tables/T2_{family}_CI.md'))
    cache=C.OUT/'cache/T2_BOOTSTRAP.json';output_paths.append(cache);ensure_new(output_paths)
    metadata_path=C.OUT/'cache/METADATA_RECORDINGS.json';t1_path=C.OUT/'cache/T1_NATIVE_BIAS.json'
    require(metadata_path.is_file(),'T0 recording metadata must finish before T2')
    require(t1_path.is_file() and C.read(t1_path)['complete'] is True,'T1 must finish before T2')
    inputs=[C.NOFLIP/name for name in ('SCORES.json','SUMMARY.json','PLASTIC184_SCORES.json','PLASTIC184_SUMMARY.json')]+[metadata_path,t1_path]
    C.start('T2',inputs);metadata=C.read(metadata_path)
    require(isinstance(metadata['frame_records'],dict),'Unexpected T0 recording metadata schema')
    all_rows=[];proofs=[]
    for item in load_panels():
        rows,proof=bootstrap_panel(item,metadata);all_rows.extend(rows);proofs.append(proof)
    require(len(all_rows)==5*2*6*5,'Expected5populations×2units×6pairs×5metrics')
    for family in ('PRIMARY','GREEN_MANUAL','PLASTIC184'):
        rows=[r for r in all_rows if r['family']==family]
        C.csvwrite(C.OUT/f'tables/T2_{family}_CI.csv',rows)
        (C.OUT/f'tables/T2_{family}_CI.md').write_text(markdown(family,rows))
    C.write(cache,dict(schema='T2_fixed_paired_bootstrap_v1',complete=True,seed=C.SEED,replicates=B,
        comparisons=[dict(name=a,candidate=b,comparator=c) for a,b,c in COMPARISONS],metrics=METRICS,
        quantile='numpy.percentile2.5,97.5; method=linear',multiple_comparison_adjustment='none',
        denominator='Ratio of total correct/observed/reference counts over sampled units; point weighted, not mean frame ratios.',
        undefined_policy='No redraw. Undefined paired replicates excluded only from percentile and counted; conditional scope explicitly reported.',
        proofs=proofs,rows=all_rows,original_GT_images_weights_opened=False,new_forward_count=0))
    C.log('step_complete',task='T2',rows=len(all_rows),new_forward_count=0,
        sampling_hashes=[dict(panel=p['panel'],unit=s['unit'],weights_sha256=s['weights_sha256'],ordered_unit_ids_sha256=s['ordered_unit_ids_sha256']) for p in proofs for s in p['sampling']],
        outputs={str(p):C.sha(p) for p in output_paths})


if __name__=='__main__':main()

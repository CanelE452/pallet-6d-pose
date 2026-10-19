"""Postprediction bank coverage and fixed failure-cohort attribution.

This module reads references only after all requested prediction seals exist.
It never imports a producer or generates/reoptimizes a pose. Generated-pose
coverage is a retrospective upper bound, separate from actual H1 selection.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import gzip
import json
import time

from . import s1_union as U, independent_check as I
from .evaluation import sealed_predictions, POPULATIONS, EXPECTED

SEEDS=(20261018,20261019,20261020)


def pose_of(candidate):
    if 'actual_pose' in candidate:
        return candidate['actual_pose']
    if 'R_physical' in candidate and 'centroid' in candidate:
        return dict(available=True,R_physical=candidate['R_physical'],centroid=candidate['centroid'])
    raise ValueError('Stored physical pose required; coordinate conversion is not inferred')


def declared_extras(source,ids):
    """Use only the exact complete actions declared by the evaluated union.

    A frozen PF archive also contains POSEFIX_RAW, which is not one of PF2.
    Explicit method filters prevent that unused estimator from inflating the
    generated bound. Omitting a filter preserves the historical full archive.
    """
    from .extra_sources import verify
    path=Path(source['path']);binding=verify(path,source.get('seal'))
    methods=source.get('methods')
    if methods is not None:
        if not isinstance(methods,list)or not methods or any(not isinstance(m,str)or not m for m in methods)or len(set(methods))!=len(methods):
            raise ValueError('Explicit unique nonempty complete-action method filter required')
        allowed=set(methods)
    else:allowed=None
    selected={};actual=set()
    for row in U.rows(path):
        if allowed is not None and row.get('method')not in allowed:continue
        selected.setdefault(row['id'],[]).append(row)
        if allowed is not None:actual.add(row['method'])
    if set(selected)!=set(ids)or(allowed is not None and actual!=allowed):
        raise ValueError('Every declared complete-action parent must cover the exact evaluated frame population')
    if allowed is not None:
        if any({r['method']for r in rr}!=allowed for rr in selected.values()):
            raise ValueError('Every frame must preserve every declared action, including unavailable endpoints')
        binding.update(declared_methods=list(methods),
            selected_rows_payload_sha256=U.payload_sha([r for rr in selected.values()for r in rr]),
            selected_rows=sum(map(len,selected.values())),
            unused_parent_methods_excluded=True)
    return selected,binding


def generated_rows(spec,features,d1,base_predictions,binding):
    """Read every stored endpoint; zero new projection or pose solver calls."""
    n=len(features)
    if spec['backend']=='fullnormal':
        root=Path(spec['result_root'])
        feature_seal_path=Path(spec.get('generated_feature_seal',spec['feature_seal']))
        source_seal=U.read(feature_seal_path)
        if not source_seal.get('complete')or source_seal.get('rows')!=238:
            raise ValueError('Complete generated-bank feature seal required')
        if 'frame_seal_sha256'not in source_seal:
            raise ValueError('Fullnormal union requires explicit generated_feature_seal for its underlying core bank')
        binding['generated_feature_seal']=dict(path=str(feature_seal_path),sha256=U.sha(feature_seal_path))
        binding['generated_frame_seals']={}
        own_seal=U.read(spec['feature_seal'])
        union_rows={r['id']:r for r in own_seal.get('frame_seals',[])}
        for i,frame in enumerate(features):
            folder=root/f'{i:04d}';seal=U.read(folder/'FRAME_SEAL.json')
            if U.sha(folder/'FRAME_SEAL.json')!=source_seal['frame_seal_sha256'][f'{i:04d}']:
                raise ValueError('Generated row belongs to a different feature bank')
            binding['generated_frame_seals'][f'{i:04d}']=U.sha(folder/'FRAME_SEAL.json')
            if feature_seal_path.resolve()!=Path(spec['feature_seal']).resolve():
                if own_seal.get('schema')!='OWN_FULL_NORMAL_KERNEL_EXTRA_FEATURE_SEAL_V1':
                    raise ValueError('Explicit supported core-to-union ancestry required')
                union_entry=union_rows[frame['id']];union_path=Path(union_entry['seal_path'])
                if U.sha(union_path)!=union_entry['sha256']:
                    raise ValueError('Union row seal changed')
                dep=U.read(union_path)['dependencies']
                if dep['core_frame_seal_sha256']!=U.sha(folder/'FRAME_SEAL.json')or dep['result_sha256']!=U.sha(folder/'RESULT.json.gz'):
                    raise ValueError('Union features and generated core result belong to different banks')
            if U.sha(folder/'RESULT.json.gz')!=seal['artifacts']['RESULT.json.gz']:
                raise ValueError('Generated endpoint archive changed')
            with gzip.open(folder/'RESULT.json.gz','rt')as stream:result=json.load(stream)
            if result['id']!=frame['id']:raise ValueError('Endpoint identity changed')
            yield list(result['candidates'])
    elif spec['backend']=='original_union':
        feature_seal_path=d1/'d1_real_candidates_v1/FEATURE_SEAL.json';source_seal=U.read(feature_seal_path)
        binding['generated_feature_seal']=dict(path=str(feature_seal_path),sha256=U.sha(feature_seal_path))
        binding['generated_frame_seals']={}
        for i,frame in enumerate(features):
            folder=d1/'d1_real_candidates_v1/frames'/f'{i:04d}'
            seal=U.read(folder/'FRAME_SEAL.json')
            if U.sha(folder/'FRAME_SEAL.json')!=source_seal['per_frame_execution_seal_sha256'][f'{i:04d}']:
                raise ValueError('Original generated row ancestry changed')
            binding['generated_frame_seals'][f'{i:04d}']=U.sha(folder/'FRAME_SEAL.json')
            if U.sha(folder/'E_RUN.jsonl.gz')!=seal['e_run_sha256']:
                raise ValueError('Original generated endpoint archive changed')
            raw=next(U.rows(folder/'E_RUN.jsonl.gz'))
            if raw['id']!=frame['id']:raise ValueError('Original endpoint identity changed')
            extra=[]
            for method in U.METHODS:
                p=base_predictions[frame['id']]['predictions'][method]['actual_pose']
                if p.get('available'):extra.append(dict(candidate_id=method,actual_pose=p))
            yield list(raw['candidates'])+extra
    else:raise ValueError('Unknown explicit bank backend')


def run(manifest,d1,reference_dir,fixed_sets,output):
    manifest,d1,reference_dir,fixed_sets,output=map(Path,(manifest,d1,reference_dir,fixed_sets,output))
    started=time.monotonic();plan=U.read(manifest);output.mkdir(parents=True,exist_ok=True)
    if not plan.get('arms'):raise ValueError('Explicit arm population required')
    # Authenticate complete predictions and feature files before any numeric
    # reference is decoded. Labels and candidates are never modified here.
    sources={};bindings=[];input_bindings={}
    for name,spec in plan['arms'].items():
        pred,binding=sealed_predictions(spec['prediction_dir']);bindings.append(binding)
        features=list(U.rows(spec['features']));fs=U.read(spec['feature_seal'])
        if len(features)!=238 or len({r['id']for r in features})!=238:
            raise ValueError('Every real238 image must remain in the bank analysis')
        if U.sha(spec['features']) not in (fs.get('head_features_sha256'),fs.get('features_sha256')):
            raise ValueError('Feature archive is not authenticated')
        byid={r['id']:r for r in pred}
        if set(byid)!={r['id']for r in features}:raise ValueError('Bank/prediction population differs')
        for seed in SEEDS:
            if any(name+'_s'+str(seed)not in r['predictions']for r in pred):
                raise ValueError('All three own-bank head seeds are required')
        sources[name]=(spec,features,byid)
        input_bindings[name]=dict(features=dict(path=spec['features'],sha256=U.sha(spec['features'])),
            feature_seal=dict(path=spec['feature_seal'],sha256=U.sha(spec['feature_seal'])),prediction=binding,extras=[])
    base,binding=sealed_predictions(d1);base={r['id']:r for r in base};bindings.append(binding)
    rp=reference_dir/'FIXED_REFERENCES.json';rs=U.read(reference_dir/'SCORE_SEAL.json')
    if U.sha(rp)!=rs['artifacts'][rp.name]:raise ValueError('Canonical reference seal changed')
    U.write(output/'REFERENCE_ACCESS.json',dict(purpose='all sealed predictions: posthoc generated/eligible/selected coverage only',
        prediction_bindings=bindings,reference_sha256=U.sha(rp),source_sha256=U.sha(__file__),new_pose_calls=0))
    refs=U.read(rp)['references'];sets=U.read(fixed_sets)['sets'];result={};ledger=[];failures=[]
    for name,(spec,features,predictions)in sources.items():
        extras=[]
        for source in spec.get('extra_rows',[]):
            xx,binding=declared_extras(source,[x['id']for x in features])
            input_bindings[name]['extras'].append(binding)
            extras.append(xx)
        arm_rows=[]
        for index,(frame,raw)in enumerate(zip(features,generated_rows(spec,features,d1,base,input_bindings[name]))):
            fid=frame['id'];p=predictions[fid];ref=refs[frame['partition']][fid]
            if any(p[k]!=frame[k]for k in ('partition','session','cluster_id')):raise ValueError('Metadata identity changed')
            for archive in extras:
                raw+=archive[fid]
            gen_good=[c.get('candidate_id',c.get('method','UNNAMED'))for c in raw if I.metric(pose_of(c),ref)['success']]
            good=[c['candidate_id']for c in frame['candidates']if I.metric(c['actual_pose'],ref)['success']]
            eligible=[c['candidate_id']for c in frame['candidates']if c.get('eligible')and c['candidate_id']in good]
            row=dict(arm=name,id=fid,partition=frame['partition'],session=frame['session'],cluster_id=frame['cluster_id'],
                generated_good=bool(gen_good),unique_good=bool(good),eligible_good=bool(eligible),
                generated_candidates=len(raw),unique_candidates=len(frame['candidates']),eligible_candidates=sum(bool(c.get('eligible'))for c in frame['candidates']),
                good_generated_ids=gen_good,good_unique_ids=good,good_eligible_ids=eligible,selected={})
            for seed in SEEDS:
                key=name+'_s'+str(seed);prediction=p['predictions'][key];metric=I.metric(prediction['actual_pose'],ref)
                before=I.metric(base[fid]['predictions']['D1_H1_MLP_s'+str(seed)]['actual_pose'],ref)
                transition='BOTH_SUCCESS'if before['success']and metric['success']else'RECOVERY'if metric['success']else'DAMAGE'if before['success']else'BOTH_FAILURE'
                category='SUCCESS'if metric['success']else'GENERATED_ABSENT'if not gen_good else'ADMISSION_LOSS'if not eligible else'SELECTION_LOSS'
                selected_id=prediction.get('selected_candidate_id')
                candidate=next((c for c in frame['candidates']if c['candidate_id']==selected_id),None)
                row['selected'][str(seed)]=dict(metric=metric,baseline_metric=before,transition=transition,failure_stage=category,
                    selected_candidate_id=selected_id,source_aliases=candidate.get('source_aliases',[])if candidate else[],
                    fallback=prediction.get('fallback'),prediction_status=prediction.get('prediction_status'))
                if not metric['success']:
                    failures.append(dict(arm=name,id=fid,partition=frame['partition'],cluster_id=frame['cluster_id'],seed=seed,
                        translation_cm=metric['translation_cm'],rotation_deg=metric['rotation_deg'],transition=transition,
                        failure_stage=category,selected_candidate_id=selected_id,generated_good=bool(gen_good),eligible_good=bool(eligible)))
            arm_rows.append(row);ledger.append(row)
        byid={r['id']:r for r in arm_rows};populations={}
        for pop,parts in POPULATIONS.items():
            poprows=[r for r in arm_rows if r['partition']in parts]
            if len(poprows)!=EXPECTED[pop]:raise AssertionError('Fixed denominator lost')
            populations[pop]=dict(n=len(poprows),generated_good=sum(r['generated_good']for r in poprows),unique_good=sum(r['unique_good']for r in poprows),
                eligible_good=sum(r['eligible_good']for r in poprows),seeds={str(s):dict(selected_good=sum(r['selected'][str(s)]['metric']['success']for r in poprows),
                    failure_stages=dict(Counter(r['selected'][str(s)]['failure_stage']for r in poprows)),
                    transitions={k:[r['id']for r in poprows if r['selected'][str(s)]['transition']==k]for k in ('RECOVERY','DAMAGE','BOTH_SUCCESS','BOTH_FAILURE')})for s in SEEDS})
        cohorts={k:dict(n=len(ids),generated_good=sum(byid[fid]['generated_good']for fid in ids),eligible_good=sum(byid[fid]['eligible_good']for fid in ids),
            seeds={str(s):dict(selected_good=sum(byid[fid]['selected'][str(s)]['metric']['success']for fid in ids),
                transitions={cat:[fid for fid in ids if byid[fid]['selected'][str(s)]['transition']==cat]for cat in ('RECOVERY','DAMAGE','BOTH_SUCCESS','BOTH_FAILURE')})for s in SEEDS})for k,ids in sets.items()}
        result[name]=dict(populations=populations,fixed_cohorts=cohorts)
    U.save(output/'BANK_COVERAGE_ROWS.jsonl.gz',ledger);U.write(output/'SUMMARY.json',result)
    U.write(output/'INPUT_BINDINGS.json',input_bindings)
    fields=['arm','id','partition','cluster_id','seed','translation_cm','rotation_deg','transition','failure_stage','selected_candidate_id','generated_good','eligible_good']
    with (output/'ALL_FAILURES.csv').open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(failures)
    seal=dict(complete=True,manifest_sha256=U.sha(manifest),source_sha256=U.sha(__file__),reference_sha256=U.sha(rp),
        fixed_sets_sha256=U.sha(fixed_sets),actual_new_pose_calls=0,retrospective_upper_bound_is_not_selected_performance=True,
        PRIMARY_development_n=192,independent_real_test_n=0,wall_seconds=time.monotonic()-started,
        artifacts={n:U.sha(output/n)for n in ('BANK_COVERAGE_ROWS.jsonl.gz','SUMMARY.json','ALL_FAILURES.csv','INPUT_BINDINGS.json')})
    U.write(output/'ANALYSIS_SEAL.json',seal);return seal


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('manifest','d1','reference-dir','fixed-sets','output'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();print(run(a.manifest,a.d1,a.reference_dir,a.fixed_sets,a.output))

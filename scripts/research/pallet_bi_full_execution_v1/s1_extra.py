"""Generic completed-action endpoint union around the sealed S1 score algebra.

The original D1 packet stays explicit: new S3/S4 support produces actual new
poses, while this selector uses the same original downstream features in both
arms. New support packets must be archived separately by their producer.
"""
from __future__ import annotations
import argparse,time
from collections import Counter
from copy import deepcopy
from pathlib import Path
import cv2
from . import s1_union as U

def union_frame(frame,erun,base_features,bundle,extra_rows,config):
    """Return (features,costs), one row/candidate with id,method,candidate_id,pose.

    producer_available=False is a rejected endpoint even when a proposal's
    numerical matrix is present. All original and extra endpoint payloads are
    retained by batch_features independently of admission and deduplication.
    """
    prepared=[];groups={};seen=set();unavailable=set()
    for row in extra_rows:
        if row['id']!=frame['id']:raise ValueError('Extra endpoint frame ID differs')
        cid=row['candidate_id'];method=row['method']
        if cid in seen or cid in ('N0','E0','E0_REGENERATED'):raise ValueError('Unique non-baseline extra candidate IDs required')
        seen.add(cid);groups[cid]=method;adapted=deepcopy(row);adapted['method']=cid
        if not row.get('producer_available',row['actual_pose'].get('available',False)):
            adapted['actual_pose']['available']=False;unavailable.add(cid)
        prepared.append(adapted)
    result,costs=U.union_frame(frame,erun,base_features,bundle,prepared,config)
    info=result.pop('s1_union');info['new_unique_complete_actions']=info.pop('new_unique_PF_actions')
    info['new_complete_action_rejections']=info.pop('PF_rejections');info['new_complete_action_minima']=info.pop('new_PF_minima')
    for rejection in info['new_complete_action_rejections']:
        rejection['candidate_id']=rejection.pop('method');rejection['method']=groups[rejection['candidate_id']]
        if rejection['candidate_id']in unavailable or rejection['reason']=='ORIGINAL_F_UNAVAILABLE':rejection['reason']='PRODUCER_UNAVAILABLE'
    for duplicate in info['complete_action_duplicates']:
        duplicate['extra_candidate_id']=duplicate.pop('method');duplicate['method']=groups[duplicate['extra_candidate_id']]
    for c in result['candidates']:
        aliases=set(c.get('source_aliases',[]));stage_aliases={groups[k]for k in aliases if k in groups}
        if stage_aliases:
            c['source_aliases']=sorted(aliases|stage_aliases);c['complete_action_stage_aliases']=sorted(stage_aliases)
    info.update(extra_visited_endpoints=len(extra_rows),base='explicit original downstream packet and feature bank',
        admission='all new completed estimators: producer available, finite proper R, original-native all-positive depth',
        complete_action_methods=sorted(set(groups.values())),new_support_packets_used_as_H1_features=False)
    result['extra_union']=info
    costs['new_complete_action_scores']=costs.pop('new_PF_scores');costs['new_complete_action_bank']=costs.pop('new_PF_bank')
    for c in costs['new_complete_action_bank']:c['source']='COMPLETED_ACTION';c['method']=groups[c['candidate_id']]
    return result,costs

def batch_features(inputs_path,base_features_path,base_frame_dir,extra_rows_path,config_path,output,run):
    """Full ordered bank union. base_frame_dir contains frames/0000/E_RUN*."""
    U.check_plan(run);started=time.monotonic();cpu=time.process_time();cv2.setNumThreads(1)
    inputs=list(U.rows(inputs_path));base=list(U.rows(base_features_path));extra={}
    for row in U.rows(extra_rows_path):extra.setdefault(row['id'],[]).append(row)
    if [f['id']for f in inputs]!=[f['id']for f in base]or set(extra)!={f['id']for f in inputs}:raise ValueError('Exact full ordered base/extra/input IDs required')
    config=U.read(config_path);output=Path(output);output.mkdir(parents=True,exist_ok=True);base_frame_dir=Path(base_frame_dir)
    from scripts.research.pallet_boundary_action_repair_v1.d1_batch import source_bindings
    closure=source_bindings();closure.update({str(Path(U.__file__).resolve()):U.sha(U.__file__),str(Path(__file__).resolve()):U.sha(__file__)})
    U.write(output/'SOURCE_DEPENDENCY_CLOSURE.json',closure);result=[];counts=Counter()
    for i,(frame,baseframe)in enumerate(zip(inputs,base)):
        src=base_frame_dir/'frames'/f'{i:04d}';out=output/'frames'/f'{i:04d}';out.mkdir(parents=True,exist_ok=True)
        erun_path=src/'E_RUN.jsonl.gz'
        if not erun_path.exists():erun_path=src/'ORIGINAL_E_RUN_EXACT.jsonl.gz'
        bundle_path=src/'BOUNDARY_BUNDLE.jsonl.gz'
        key=dict(input_payload_sha256=U.payload_sha(frame),base_features_payload_sha256=U.payload_sha(baseframe),
                 E_RUN_sha256=U.sha(erun_path),boundary_bundle_sha256=U.sha(bundle_path),extras_payload_sha256=U.payload_sha(extra[frame['id']]),
                 config_sha256=U.sha(config_path),source_dependency_closure_sha256=U.payload_sha(closure))
        sealpath=out/'SEAL.json';fp=out/'FEATURES.jsonl.gz'
        if sealpath.exists():
            s=U.read(sealpath)
            if s['dependencies']!=key or s['features_sha256']!=U.sha(fp):raise ValueError('Completed extra union cache differs')
            result.extend(U.rows(fp));counts['completed_row_cache_hits']+=1;continue
        erun=next(U.rows(erun_path));bundle=next(U.rows(bundle_path));tick=time.monotonic()
        if erun['id']!=frame['id']:raise ValueError('Original endpoint ID differs')
        features,costs=union_frame(frame,erun,baseframe,bundle,extra[frame['id']],config)
        U.save(out/'ORIGINAL_E_RUN_EXACT.jsonl.gz',[erun]);U.save(out/'ALL_EXTRA_ENDPOINTS.jsonl.gz',extra[frame['id']]);U.save(out/'INDIVIDUAL_COSTS.jsonl.gz',[costs]);U.save(fp,[features])
        info=features['extra_union'];s=dict(complete=True,id=frame['id'],dependencies=key,features_sha256=U.sha(fp),
             original_E_RUN_payload_sha256=U.payload_sha(erun),extra_endpoint_payload_sha256=U.payload_sha(extra[frame['id']]),
             base_unique_count=len(baseframe['candidates']),unique_count=len(features['candidates']),new_unique_complete_actions=info['new_unique_complete_actions'],
             old_cost_reconstructions=info['old_cost_reconstructions_for_missing_context'],
             visited_extra_endpoints=len(extra[frame['id']]),new_pose_optimizer_calls=0,wall_seconds=time.monotonic()-tick)
        U.write(sealpath,s);result.append(features);counts['new_rows']+=1;counts['exact_base_E_RUN_reuses']+=1;counts['new_unique_cost_evaluations']+=s['new_unique_complete_actions'];counts['old_cost_reconstructions']+=s['old_cost_reconstructions']
        if i%100==0:print(U.canonical(dict(stage='EXTRA_UNION_FEATURES',done=i+1,total=len(inputs),wall=time.monotonic()-started)),flush=True)
    U.save(output/'HEAD_FEATURES.jsonl.gz',result)
    seal=dict(schema='NEW_CAMPAIGN_EXTRA_UNION_FEATURE_SEAL_V1',complete=True,rows=len(result),head_features_sha256=U.sha(output/'HEAD_FEATURES.jsonl.gz'),
        ordered_ids=[r['id']for r in result],input_sha256=U.sha(inputs_path),base_features_sha256=U.sha(base_features_path),extra_rows_sha256=U.sha(extra_rows_path),
        source_dependency_closure_sha256=U.payload_sha(closure),plan_sha256=U.sha(Path(run)/'CAMPAIGN_PLAN.json'),actual_calls=dict(counts),
        e0_independently_executed=True,original_full_bank_and_extra_visited_endpoints_preserved=True,current_real_reference_reads=0,
        wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu)
    U.write(output/'FEATURE_SEAL.json',seal);return seal

def main():
    p=argparse.ArgumentParser()
    for k in ('inputs-path','base-features-path','base-frame-dir','extra-rows-path','config-path','output','run'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=batch_features(a.inputs_path,a.base_features_path,a.base_frame_dir,a.extra_rows_path,a.config_path,a.output,a.run)
    print(U.canonical({k:r[k]for k in ('complete','rows','head_features_sha256','actual_calls','wall_seconds')}),flush=True)
if __name__=='__main__':main()

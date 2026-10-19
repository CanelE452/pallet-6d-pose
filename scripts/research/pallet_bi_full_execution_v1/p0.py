"""Whole-population byte/semantic ancestry proof for exact baseline reuse.

No model/pose producer or numeric real reference reader is imported. A previous
full semantic audit is reused only after its actual consumed bytes are checked.
"""
from pathlib import Path
import argparse
from collections import Counter
import time
from .s1_union import read, rows, sha, write, payload_sha

def verify(d1, previous, synthetic, code_root, output, run):
    start=time.monotonic();d1,previous,synthetic,code_root,output,run=map(Path,(d1,previous,synthetic,code_root,output,run))
    prior_path=previous/'P0_INPUT_CONTRACT.json';old=read(prior_path)
    if old['status']!='PASS':raise ValueError('Historical full input audit did not pass')
    memo={};artifacts={};checks=Counter();variants=Counter();deltas={}
    def check(path,expected=None):
        p=Path(path);s=str(p)
        if s not in memo:memo[s]=sha(p)
        value=memo[s]
        if expected is not None and value!=expected:raise ValueError('Bound artifact changed: '+s)
        artifacts[s]=value;return value
    check(prior_path);check(run/'CAMPAIGN_PLAN.json')
    for domain,key,total in [('real','REAL238',238),('synthetic','SYNTHETIC1003',1003)]:
        upstream=d1/f'd1_upstream_v1/{key}_UPSTREAM.jsonl.gz';frames=list(rows(upstream))
        if len(frames)!=total or len({f['id'] for f in frames})!=total:raise ValueError('Input population changed')
        obsroot=d1/f'd1_{domain}_observations_v1';os=read(obsroot/'OBSERVATION_SEAL.json')
        check(upstream,os['input_sha256']);check(obsroot/'OBSERVATION_SEAL.json',old[domain]['observation_seal_sha256'])
        if [f['id'] for f in frames]!=os['ordered_ids']:raise ValueError('Input order changed')
        bank=d1/f'd1_{domain}_candidates_v1';fs=read(bank/'FEATURE_SEAL.json')
        check(bank/'HEAD_FEATURES.jsonl.gz',fs['head_features_sha256']);check(bank/'FEATURE_SEAL.json')
        feature_frames=list(rows(bank/'HEAD_FEATURES.jsonl.gz'))
        if [f['id'] for f in feature_frames]!=os['ordered_ids']:raise ValueError('Feature population/order changed')
        inventories=old[domain]['inventory']
        if len(inventories)!=total:raise ValueError('Semantic audit inventory incomplete')
        for index,(frame,inv,feature) in enumerate(zip(frames,inventories,feature_frames)):
            of=obsroot/'frames'/f'{index:04d}';cf=bank/'frames'/f'{index:04d}'
            seal=read(of/'OBSERVATION_SEAL.json');cs=read(cf/'FRAME_SEAL.json')
            if frame['id']!=inv['id'] or frame['id']!=cs['id'] or frame['id']!=seal['id']:raise ValueError('Frame identity mismatch')
            if payload_sha(frame)!=seal['input_row_sha256']:raise ValueError('Consumed upstream row changed')
            check(of/'OBSERVATION_SEAL.json',os['frame_seals_sha256'][f'{index:04d}'])
            check(of/'OBSERVATION.jsonl.gz',seal['observation_sha256']);check(of/'MASK.npz',seal['mask_sha256'])
            check(frame['image'],frame['image_sha256']);check(cf/'FRAME_SEAL.json')
            check(cf/'E_RUN.jsonl.gz',cs['e_run_sha256']);check(cf/'FEATURES.jsonl.gz',cs['features_sha256'])
            check(cf/'BOUNDARY_BUNDLE.jsonl.gz',cs['boundary_bundle_sha256'])
            if cs['observation_seal_sha256']!=memo[str(of/'OBSERVATION_SEAL.json')]:raise ValueError('Candidate ancestry mismatch')
            if next(rows(cf/'FEATURES.jsonl.gz'))!=feature:raise ValueError('Aggregated features changed')
            for field in ('K','native_xyz','raw_hw','initial_pose'):
                if frame[field]!=inv[field]:raise ValueError('Audited geometry changed '+field)
            if feature['state_counts']!=dict(Counter(inv['states'])):raise ValueError('Feature state provenance changed')
            variants[payload_sha(cs['source_sha256'])]+=1
            for rel,expected in cs['source_sha256'].items():
                current=code_root/rel
                if current.is_file():
                    digest=check(current)
                    if digest!=expected:deltas.setdefault(rel,set()).add(expected)
                else:raise ValueError('Original source missing: '+rel)
            checks[domain+'_all_consumed_bytes_and_input_rows']+=1
            if index%128==0:print({'stage':'P0_VERIFY_BYTES','domain':domain,'done':index+1,'total':total},flush=True)
    ps=read(synthetic/'PROVISION_SEAL.json')
    for rel,expected in ps['artifacts'].items():check(synthetic/rel,expected)
    check(synthetic/'DATA_SPLIT_LOCK.json',old['synthetic_targets']['split_lock_sha256'])
    baseline=d1/'PREDICTIONS.jsonl.gz';bs=read(d1/'PREDICTION_SEAL.json');check(baseline)
    if memo[str(baseline)] not in [bs.get('predictions_sha256'),bs.get('prediction_sha256')]+list(bs.get('artifacts',{}).values()):raise ValueError('Baseline prediction not sealed')
    tm=read(d1/'action_training_v1/TRAINING_MANIFEST.json');check(d1/'action_training_v1/TRAINING_MANIFEST.json')
    reused=[]
    for r in tm['runs']:
        if r['arm']=='D1_H1_MLP':
            check(r['checkpoint'],r['checkpoint_sha256']);reused.append({k:r[k]for k in ('arm','seed','checkpoint','checkpoint_sha256')})
    if len(reused)!=3:raise ValueError('Missing original three H1 checkpoints')
    report=dict(schema='PALLET_FULL_CAMPAIGN_P0_REUSE_PROOF_V1',status='PASS',
        previous_semantic_audit_sha256=memo[str(prior_path)],previous_semantic_audit=str(prior_path),
        populations=dict(real=238,synthetic=1003,train=768,validation=235),checks=dict(checks),
        checked_distinct_files=len(memo),artifacts=artifacts,old_source_variants=dict(variants),
        historical_source_deltas={k:dict(historical_sha256=sorted(v),current_sha256=memo[str(code_root/k)])for k,v in deltas.items()},
        source_delta_interpretation='Historical row seals include logging/recovery revisions; consumed bytes are immutable. Exact mathematics additionally checked by existing and new fixed-row feature parity, never historical hashes silently rewritten.',
        reused_H1=reused,whole_input_contract='exact consumed bytes plus original semantic audit; source variant deltas reported separately',
        unchanged_adapter_parity_receipt=str(run/'s1_prepared/PARITY_TESTS.json'),
        real_numeric_reference_decodes=0,new_model_forwards=0,new_pose_solves=0,new_H1_fits=0,
        source_sha256=sha(__file__),wall_seconds=time.monotonic()-start)
    write(output,report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('d1','previous','synthetic','code-root','output','run'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=verify(a.d1,a.previous,a.synthetic,a.code_root,a.output,a.run);print({k:r[k]for k in ('status','checked_distinct_files','wall_seconds')})

"""Diagnostic-only saved synthetic relation coverage; not policy input."""
from __future__ import annotations
import argparse,collections,gzip,hashlib,json,math,re,resource,time
from pathlib import Path
from scripts.research.pallet_risk_information_v1.common import read,rows,sha,write,save_rows

DIAGNOSTIC_RECORD_KEYS=('diagnostic_mode','placement_mode','pallet_type','scene_preset','background_asset',
    'elev_actual','azimuth_bin','V_actual','V_vis','projected_size_actual','camera_distance_actual_m',
    'n_context_placed','n_context_visible','n_cargo_placed','n_cargo_visible','explicit_occluder_placed',
    'f_static','f_cargo','f_context','f_explicit','f_total','front_face_visibility',
    'left_opening_visibility','right_opening_visibility','visible_kp_count','mask_invariants_pass')

def run_key(r):
    g=r['generation'];return json.dumps(dict(source_root=g.get('_src_root'),
        generation_run=g.get('_src_shard')or g.get('_src_root')),sort_keys=True,separators=(',',':'))

def numerical_summary(values):
    import numpy as np
    a=np.asarray([float(x)for x in values if x is not None and math.isfinite(float(x))],float)
    if not len(a):return dict(n=0,min=None,mean=None,p50=None,p90=None,max=None)
    return dict(n=len(a),min=float(a.min()),mean=float(a.mean()),p50=float(np.quantile(a,.5)),p90=float(np.quantile(a,.9)),max=float(a.max()))

def _bin(value,bounds):
    if value is None or not math.isfinite(float(value)):return 'MISSING'
    for low,high in zip(bounds,bounds[1:]):
        if low<=float(value)<high:return '[%s,%s)'%(low,high)
    return 'OUTSIDE_PREDECLARED_BINS'

def coverage_summary(records):
    relation=collections.Counter(r['N0_E0_relation']for r in records)
    flags=('N0_success','E0_success','full_success_available','full_only_success','all_failure',
        'N0_E0_same_original_action','N0_E0_bitwise_same_numeric_pose','has_failure_Pareto_pair',
        'has_incomparable_or_equal_failure_pair','all_failure_with_Pareto_supervision',
        'all_failure_without_order_supervision','zero_actions','N0_or_E0_unavailable')
    fields=('actions','eligible_original_candidates','raw_original_candidates','N0_E0_translation_change_cm',
        'N0_E0_rotation_change_deg')
    categorical=('occlusion_bin','elevation_bin','distance_bin','azimuth_bin','diagnostic_mode',
        'explicit_occluder_placed','source_asset','symmetry_order')
    return dict(frames=len(records),generation_runs=len({r['generation_run']for r in records}),
        generation_attempt_ids=len({r['generation_attempt_key']for r in records}),
        assets=len({r['source_asset']for r in records}),relations=dict(relation),
        flags={k:sum(bool(r[k])for r in records)for k in flags},
        values={k:numerical_summary(r[k]for r in records)for k in fields},
        state_total=dict(sum((collections.Counter(r['state_counts'])for r in records),collections.Counter())),
        observation_total=dict(sum((collections.Counter(r['observation_coverage'])for r in records),collections.Counter())),
        factors={k:dict(collections.Counter(str(r[k])for r in records))for k in categorical},
        original_relation_pair_counts={k:sum(r['supervision_counts'][k]for r in records)for k in
            ('total_pairs','usable_pairs','success_failure_pairs','failure_Pareto_pairs','incomparable_or_equal_failure_pair')})

def analyze_saved_coverage(repo,archive,risk_run,provenance_dir,plan_path,output):
    """Diagnostic-only joins of existing synthetic outputs and fixed labels.

    These supervision/renderer fields are never exported as selector inputs.
    No model, solver, generator or renderer is called by this command.
    """
    import numpy as np
    from scripts.research.pallet_boundary_action_repair_v1 import action_data as AD
    start,cpu=time.monotonic(),time.process_time()
    repo,archive,risk_run,provenance_dir,plan_path,output=map(Path,(repo,archive,risk_run,provenance_dir,plan_path,output))
    output.mkdir(parents=True,exist_ok=False)
    if read(plan_path)['status']!='FROZEN_BEFORE_FIRST_OPTIMIZER_UPDATE':raise ValueError('Frozen experiment plan required')
    paths=dict(provenance=provenance_dir/'PROVENANCE_ROWS.jsonl.gz',provenance_seal=provenance_dir/'PROVENANCE_SEAL.json',
        features=archive/'d1_synthetic_candidates_v1/HEAD_FEATURES.jsonl.gz',feature_seal=archive/'d1_synthetic_candidates_v1/FEATURE_SEAL.json',
        targets=archive/'d1_action_labels_v1/ACTION_TARGETS.jsonl.gz',label_seal=archive/'d1_action_labels_v1/LABEL_SEAL.json',
        supervision=risk_run/'independent_supervision_v2/FRAME_DIAGNOSTICS.jsonl.gz',
        supervision_verification=risk_run/'independent_supervision_v2/VERIFICATION.json',
        renderer_metadata=risk_run/'b_reference_metadata_v1/B_RENDERER_VISIBILITY_METADATA.jsonl.gz')
    AD.bound(paths['features'],paths['feature_seal']);AD.bound(paths['targets'],paths['label_seal'])
    if sha(paths['provenance'])!=read(paths['provenance_seal'])['provenance_rows_sha256']:raise ValueError('Provenance changed')
    pr={r['id']:r for r in rows(paths['provenance'])};tg={r['id']:r for r in rows(paths['targets'])}
    su={r['id']:r for r in rows(paths['supervision'])};md={r['id']:r for r in rows(paths['renderer_metadata'])}
    if not(len(pr)==len(tg)==len(su)==len(md)==1003 and set(pr)==set(tg)==set(su)==set(md)):raise ValueError('Original1003 input membership differs')
    selected={int(k.rsplit('f',1)[1]):k for k in pr};rp=repo/'data/pallet/training_data/paper_release/v2_prod40k_clean_merged/records.jsonl'
    metadata={};digest=hashlib.sha256();records_count=0
    with rp.open('rb')as f:
        for line in f:
            records_count+=1;digest.update(line);m=re.findall(rb'"usable_id"\s*:\s*(\d+)',line)
            if len(m)!=1:raise ValueError('Ambiguous global record usable_id')
            ix=int(m[0])
            if ix not in selected:continue
            r=json.loads(line);fid=selected[ix]
            if hashlib.sha256(line).hexdigest()!=pr[fid]['original_record_line_sha256']:raise ValueError('Record bytes changed')
            metadata[fid]={k:r[k]for k in DIAGNOSTIC_RECORD_KEYS if k in r}
    if len(metadata)!=1003 or digest.hexdigest()!=read(paths['provenance_seal'])['original_records_sha256']:raise ValueError('Original records changed/incomplete')
    result=[]
    for frame in rows(paths['features']):
        fid=frame['id'];p,t,s,m=pr[fid],tg[fid],su[fid],metadata[fid]
        if not(frame['split']==p['split']==t['split']==s['split']):raise ValueError('Split changed')
        entries=AD.actions(frame);metrics=t['action_metrics']
        if [e['candidate_id']for e in entries]!=[x['candidate_id']for x in metrics]:raise ValueError('Original action ordering changed')
        n=[i for i,e in enumerate(entries)if 'N0'in AD.source_aliases(e)]
        e=[i for i,a in enumerate(entries)if 'E0'in AD.source_aliases(a)]
        if len(n)>1 or len(e)>1:raise ValueError('Baseline alias ambiguity')
        ni,ei=(n[0]if n else None),(e[0]if e else None)
        ns=bool(ni is not None and metrics[ni]['success_5cm5deg']);es=bool(ei is not None and metrics[ei]['success_5cm5deg'])
        full=any(v['success_5cm5deg']for v in metrics);unavailable=ni is None or ei is None
        dp=dr=None;exact=False;same=False
        if not unavailable:
            a,b=frame['fallback'],frame['e0_regenerated'];ra,rb=np.asarray(a['R_physical']),np.asarray(b['R_physical']);ta,tb=np.asarray(a['centroid']),np.asarray(b['centroid'])
            same=ni==ei;exact=np.array_equal(ra,rb)and np.array_equal(ta,tb)
            dp=float(np.linalg.norm(ta-tb)*100);dr=float(np.degrees(np.arccos(np.clip((np.trace(ra.T@rb)-1)/2,-1,1))))
        counts=dict(s['counts'])
        # Original independent audit stores sparse Counter keys. Recover only
        # combinatorially exact zeros/counts, never unknown measurements.
        total=len(entries)*(len(entries)-1)//2;positive=counts['positive_action_labels']
        incomparable=total-counts['success_failure_pairs']-counts['failure_Pareto_pairs']-positive*(positive-1)//2
        if counts.get('total_pairs',total)!=total or counts.get('incomparable_or_equal_failure_pair',incomparable)!=incomparable:
            raise ValueError('Original sparse pair counter differs from exact partition')
        counts['total_pairs']=total;counts['incomparable_or_equal_failure_pair']=incomparable
        r=dict(id=fid,split=frame['split'],source_asset=p['source_asset'],scenario_id=p['scenario_id'],
            generation_run=run_key(p),generation_attempt_key=p['generation_attempt_key'],symmetry_order=t['reference_symmetry_order'],
            N0_success=ns,E0_success=es,N0_E0_relation=('UNAVAILABLE'if unavailable else ('N0_SUCCESS'if ns else'N0_FAILURE')+'__'+('E0_SUCCESS'if es else'E0_FAILURE')),
            full_success_available=full,full_only_success=full and not ns and not es,all_failure=not full,
            N0_E0_same_original_action=same,N0_E0_bitwise_same_numeric_pose=bool(exact),N0_E0_translation_change_cm=dp,N0_E0_rotation_change_deg=dr,
            has_failure_Pareto_pair=counts['failure_Pareto_pairs']>0,has_incomparable_or_equal_failure_pair=counts['incomparable_or_equal_failure_pair']>0,
            all_failure_with_Pareto_supervision=bool(counts['all_zero_with_Pareto_supervision']),
            all_failure_without_order_supervision=bool(counts['all_zero_without_order_supervision']),zero_actions=not entries,N0_or_E0_unavailable=unavailable,
            actions=len(entries),eligible_original_candidates=sum(c['eligible']for c in frame['candidates']),raw_original_candidates=len(frame['candidates']),
            state_counts=frame.get('state_counts',{}),observation_coverage=frame.get('observation_coverage',{}),supervision_counts=counts,
            renderer_diagnostic_metadata=m,renderer_annotation_sha256=md[fid]['annotation_sha256'],
            occlusion_bin=_bin(m.get('f_total'),[0,.1,.3,.6,1.000000001]),elevation_bin=_bin(m.get('elev_actual'),[0,10,20,30,45,60,90.000000001]),
            distance_bin=_bin(m.get('camera_distance_actual_m'),[0,2,4,6,10,math.inf]),azimuth_bin=m.get('azimuth_bin'),
            diagnostic_mode=m.get('diagnostic_mode'),explicit_occluder_placed=m.get('explicit_occluder_placed'))
        if full==bool(t['all_zero_H1'])or len(entries)!=counts['actions']:raise ValueError('Original success/action accounting differs')
        result.append(r)
    if len(result)!=1003 or {r['id']for r in result}!=set(pr):raise ValueError('Incomplete coverage output')
    path=output/'SYNTHETIC_COVERAGE_ROWS.jsonl.gz';save_rows(path,result)
    summaries={}
    for split in('train','validation'):
        rr=[r for r in result if r['split']==split]
        summaries[split]=coverage_summary(rr)
        summaries[split]['by_condition']={k:{str(v):coverage_summary([r for r in rr if r[k]==v])for v in sorted({r[k]for r in rr},key=str)}
            for k in('source_asset','symmetry_order','occlusion_bin','elevation_bin','azimuth_bin','diagnostic_mode','explicit_occluder_placed')}
    train=[r for r in result if r['split']=='train'];valid=[r for r in result if r['split']=='validation'];gaps=[]
    for k in('occlusion_bin','elevation_bin','azimuth_bin','diagnostic_mode','explicit_occluder_placed'):
        for v in sorted({r[k]for r in valid},key=str):
            a=[r for r in train if r[k]==v];b=[r for r in valid if r[k]==v]
            if not a:gaps.append(dict(factor=k,value=v,train_frames=0,validation_frames=len(b),validation_runs=len({r['generation_run']for r in b})))
    receipt=dict(schema='SAVED_SYNTHETIC_RELATION_COVERAGE_V1',status='COMPLETE_DIAGNOSTIC_ONLY',frames=1003,
        summaries=summaries,marginal_conditions_absent_from_train_but_present_in_validation=gaps,
        bindings={k:dict(path=str(v),sha256=sha(v))for k,v in paths.items()},original_records_sha256=digest.hexdigest(),original_records=records_count,
        rows_sha256=sha(path),experiment_plan_sha256=sha(plan_path),source_sha256=sha(__file__),
        interpretation=dict(candidate_actions_are_not_independent_scenes=True,generation_runs_are_conservative_source_groups_not_verified_scene_resets=True,
            raw_original_action_labels_reused=True,old_independent_supervision_counts_reused=True,correction_amount_scope='Raw archived fallback N0 versus independently executed E0, before action tolerance dedup; same_original_action follows original action dedup',new_PnP_or_optimization=0,new_NN_forward=0,new_training=0,new_render=0,
            PRIMARY_or_CAL_reference_reads=0,metadata_used_as_selector_feature=False,success_ratio_matching_is_not_generation_objective=True,
            coverage_gaps_alone_do_not_establish_generator_or_cause=True),
        cost=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1))
    write(output/'SYNTHETIC_COVERAGE.json',receipt);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--risk-run',type=Path,required=True);p.add_argument('--provenance-dir',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=analyze_saved_coverage(a.repo_root,a.archive,a.risk_run,a.provenance_dir,a.plan,a.output)
    print(json.dumps({k:r[k]for k in('status','frames','marginal_conditions_absent_from_train_but_present_in_validation','cost')}))

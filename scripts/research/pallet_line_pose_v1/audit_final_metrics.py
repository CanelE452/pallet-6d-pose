"""Independent stored-artifact metric/identity/runtime and primary CI audit.

No inference, selection, original-source edits, image reads, or GPU use. The
primary session bootstrap uses explicit repetition and NumPy quantiles, without
importing aggregate_results or its weighted-quantile implementation.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS=('image_joint','geometry_joint','image_line_only')
KP=('keypoint_location_median_px','keypoint_location_p90_px')
POSE=('rotation_median_deg','translation_median_cm','iou3d_median','add_sym_auc')
METRICS=KP+POSE
LOWER=set(KP+POSE[:2])
FIELDS=dict(zip(POSE,('rotation_error_deg','translation_error_cm','iou3d','add_sym_m')))


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def vsha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):
    tmp=p.with_suffix('.pending.json');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def check(a,b,tol=1e-12):
    if not np.allclose(a,b,rtol=0,atol=tol):raise ValueError(f'Independent numeric mismatch: {a} versus {b}, tolerance={tol}')
def summary(v):
    v=np.asarray(v,float)
    return dict(seeds=v.tolist(),mean=float(v.mean()),std=float(v.std(ddof=1)))


def collect(run_dir):
    real=read(run_dir/'REAL_EVALUATION_COMPLETE.json');runtime=read(run_dir/'RUNTIME.json')
    assert real['complete'] and real['PASS'] and real['expected_runs']==len(real['runs'])==9
    assert real['runtime']['sha256']==sha(run_dir/'RUNTIME.json')
    assert real['selection_sha256']==sha(run_dir/'SELECTION.json')
    assert runtime['complete'] and runtime['PASS'] and runtime['parity_PASS']
    assert runtime['baseline_and_integrated_accuracy_prediction_max_abs_delta_px']==0
    cache=read(run_dir/'baseline/FULL_CANDIDATES.json');assert real['baseline_cache_sha256']==sha(run_dir/'baseline/FULL_CANDIDATES.json')
    negative={k:v for k,v in cache['frames'].items() if cache['frame_metadata'][k]['kind']=='negative'}
    assert len(negative)==2689
    boxscore=lambda f:{k:[dict(score=c['score'],box_xyxy=c['box_xyxy']) for c in cs] for k,cs in f.items()}
    negative_sha,box_sha=vsha(negative),vsha(boxscore(cache['frames']))
    results={};sources={};runtime_checks={};baseline=None;references=None
    for row in real['runs']:
        arm,seed=row['arm'],row['seed'];label=f'{arm}_seed{seed}';d=run_dir/'evaluation'/label
        assert (arm,seed) not in results and arm in ARMS and seed in (1,2,3)
        for name,key in [('RESULTS.json','result_sha256'),('COMPLETION.json','completion_sha256'),('IDENTITY_AUDIT.json','identity_audit_sha256')]:assert sha(d/name)==row[key]
        r=read(d/'RESULTS.json');c=read(d/'COMPLETION.json');audit=read(d/'IDENTITY_AUDIT.json')
        assert r['complete'] and c['complete'] and c['PASS'] and audit['complete'] and audit['PASS']
        for file,digest in c['output_sha256'].items():assert sha(d/file)==digest
        for ref in r['reference_sources'].values():assert sha(ref['path'])==ref['sha256']
        old2=read(r['reference_sources']['two_d']['path'])['metrics']['box_and_keypoint_2d']
        oldpose=read(r['reference_sources']['main_6d']['path'])['paths']['MAIN']
        base={k:old2[k] for k in KP}|{k:oldpose['ALL'][k] for k in POSE}
        if baseline is None:baseline=base;references=r['reference_sources']
        else:assert baseline==base and references==r['reference_sources']
        paper2=read(d/'PAPER_2D.json')['metrics']['box_and_keypoint_2d']
        paperpose=read(d/f'POSE_EVALUATION_{label}.json')['paths']['MAIN']
        for metric in KP:check(r['two_d'][metric],paper2[metric],0);check(r['reference_values']['two_d'][metric],old2[metric],0)
        for metric in POSE:check(r['main_6d'][metric],paperpose['ALL'][metric],0);check(r['reference_values']['main_6d'][metric],oldpose['ALL'][metric],0)
        for metric in ('box_ap50','box_ap50_95','candidate_count','keypoint_matched_frame_count_iou50','keypoint_supervision_count'):check(r['two_d'][metric],old2[metric],0)
        assert r['two_d_labeled_points']==2756 and r['main_6d']['n']==319 and r['main_6d_coverage']==1
        predictions=read(d/'PREDICTIONS.json')['frames']
        assert list(predictions)==list(cache['frames'])
        assert {k:predictions[k] for k in negative}==negative
        assert vsha(boxscore(predictions))==box_sha
        for key in predictions:
            old=cache['frames'][key];new=predictions[key]
            top=int(np.argmax([p['score'] for p in old])) if old else None
            for index,(a,b) in enumerate(zip(new,old)):
                if index!=top or key in negative:assert a==b
                elif b['keypoints_xy'] is not None:check(a['keypoints_xy'][8],b['keypoints_xy'][8],0)
        assert audit['negative_raw_candidates_sha256']==negative_sha and audit['all_box_score_sha256']==box_sha
        assert audit['all_boxes_scores_order_unchanged'] and not audit['negative_reinference_performed']
        rows=[v for v in runtime['records'] if v['evaluation_arm']==label]
        assert len(rows)==78 and len({v['image_key'] for v in rows})==26 and len({v['session_id'] for v in rows})==13
        assert set(v['repeat'] for v in rows)=={0,1,2}
        assert all(n==1 for n in Counter((v['image_key'],v['repeat']) for v in rows).values())
        for v in rows:check(v['added_ms'],v['integrated_ms']-v['baseline_ms'],0)
        for field,column in [('baseline_ms','baseline_ms'),('integrated_ms','integrated_ms'),('paired_added_ms','added_ms')]:
            vals=np.asarray([v[column] for v in rows]);saved=runtime['by_run'][label][field]
            assert saved['n']==78 and np.isfinite(vals).all()
            for k,value in [('mean',vals.mean()),('median',np.median(vals)),('p90',np.percentile(vals,90))]:check(saved[k],value,0)
        runtime_checks[label]=runtime['by_run'][label]
        results[arm,seed]=r
        for file in ('RESULTS.json','PAPER_2D.json',f'POSE_EVALUATION_{label}.json','PREDICTIONS.json','IDENTITY_AUDIT.json','COMPLETION.json','PAPER_2D_per_frame.csv','POSE_PER_FRAME_BY_ARM.json'):sources[str(d/file)]=sha(d/file)
    assert len(results)==9
    means={arm:{k:summary([results[arm,s]['two_d' if k in KP else 'main_6d'][k] for s in (1,2,3)]) for k in METRICS} for arm in ARMS}
    return dict(baseline=baseline,reference_sources=references,arms=means,runtime=runtime_checks,
        identity=dict(all_nine_box_ap_and_candidate_counts_exact=True,all_2689_negative_candidates_recompared_exact=True,
            all_nine_candidate_orders_and_nonselected_instances_preserved=True,all_selected_centroids_preserved=True,
            negative_raw_candidates_sha256=negative_sha,all_box_score_sha256=box_sha),
        source_sha256=sources,coverages={f'{a}_seed{s}':r['main_6d_coverage'] for (a,s),r in results.items()})


def keypoints(path):
    with Path(path).open() as h:
        return {r['frame_id']:dict(session=r['session_id'],values=np.fromstring(r['top_keypoint_supervised_errors_px'],sep=';'))
                for r in csv.DictReader(h) if r['kind']=='POSITIVE'}


def primary_session_bootstrap(run_dir,raw):
    path=run_dir/'INDEPENDENT_PRIMARY_SESSION_BOOTSTRAP.json'
    bound={p:d for p,d in raw['source_sha256'].items() if 'image_joint_seed' in p and p.endswith(('PAPER_2D_per_frame.csv','POSE_PER_FRAME_BY_ARM.json'))}
    reference_csv=Path(raw['reference_sources']['two_d']['path']).with_name('R0_per_frame.csv');bound[str(reference_csv)]=sha(reference_csv)
    if path.exists():
        result=read(path);assert result['input_sha256']==bound and result['PASS'];return result
    kps=[keypoints(run_dir/'evaluation'/f'image_joint_seed{s}'/'PAPER_2D_per_frame.csv') for s in (1,2,3)]
    kps.append(keypoints(reference_csv));poses=[];ref=None
    for s in (1,2,3):
        value=read(run_dir/'evaluation'/f'image_joint_seed{s}'/'POSE_PER_FRAME_BY_ARM.json')['per_frame']
        poses.append({r['frame_id']:r for r in value[f'image_joint_seed{s}']})
        current={r['frame_id']:r for r in value['R0']}
        if ref is None:ref=current
        else:assert ref==current
    poses.append(ref);output={}
    for metric in METRICS:
        stores=kps if metric in KP else poses
        keys=sorted(set.intersection(*(set(x) for x in stores)))
        if metric in KP:keys=[k for k in keys if all(len(s[k]['values']) for s in stores)]
        get_session=lambda row:row['session' if metric in KP else 'session_id']
        sessions=sorted({get_session(stores[0][k]) for k in keys});assert len(sessions)==13
        values=[];groups=[];diameters=None
        for store in stores:
            vv=[];gg=[]
            for k in keys:
                assert get_session(store[k])==get_session(stores[0][k])
                v=store[k]['values'] if metric in KP else [store[k][FIELDS[metric]]]
                vv.extend(v);gg.extend([sessions.index(get_session(store[k]))]*len(v))
            values.append(np.asarray(vv));groups.append(np.asarray(gg))
        assert all(np.array_equal(groups[0],g) for g in groups)
        if metric=='add_sym_auc':
            diameters=np.asarray([stores[0][k]['diameter_m'] for k in keys])
            unique=np.unique(diameters);candidates=np.unique((unique[:,None]+unique[None,:])/2)
            contributions={}
            for diameter in candidates:
                thresholds=np.linspace(0,.1*diameter,1001)
                contributions[float(diameter)]=[np.trapz((v[:,None]<=thresholds).astype(float),x=thresholds,axis=1)/thresholds[-1] for v in values]
        rng=np.random.default_rng(20260902 if metric in KP else 20260903)
        # The registered implementation consumes frame-level draws first.
        for start in range(0,10000,128):rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=min(128,10000-start))
        counts=rng.multinomial(13,np.full(13,1/13),size=10000)
        draws=np.empty(10000)
        for i,count in enumerate(counts):
            weights=count[groups[0]]
            if metric=='add_sym_auc':
                diameter=float(np.median(np.repeat(diameters,weights)))
                statistics=[float(np.dot(v,weights)/weights.sum()) for v in contributions[diameter]]
            else:
                statistics=[float(np.quantile(np.repeat(v,weights),.9 if metric==KP[1] else .5)) for v in values]
            draws[i]=np.mean(statistics[:3])-statistics[3]
        lo,hi=np.quantile(draws,[.025,.975])
        output[metric]=dict(low=float(lo),high=float(hi),confirmed_benefit=bool(hi<0 if metric in LOWER else lo>0),
            paired_frames=len(keys),paired_sessions=13,resamples=10000,seed=20260902 if metric in KP else 20260903)
        print(f'Independent session bootstrap {metric}: [{lo:.9g}, {hi:.9g}]',flush=True)
    result=dict(complete=True,PASS=True,metrics=output,input_sha256=bound,
        method='Explicit np.repeat integer session multiplicities plus NumPy linear quantiles. ADDsym AUC uses independently precomputed trapezoid integrals of each observation over the exact1001 thresholds; mean follows by linearity. Same13sessions and3seeds share each resample.',
        raw_csv_serialization_note='2D source CSV has6 decimal places. Full-precision final means come directly from original evaluator JSON.',
        inference_executed=False)
    write(path,result);return result


def run(run_dir):
    raw=collect(run_dir)
    write(run_dir/'INDEPENDENT_RAW_METRICS.json',dict(complete=True,PASS=True,**raw))
    bootstrap=primary_session_bootstrap(run_dir,raw)
    if not (run_dir/'SUMMARY.json').exists() or not (run_dir/'VERDICT.json').exists():
        print('Raw metric/identity/runtime and independent CI audit ready; final SUMMARY/VERDICT not yet present.',flush=True);return
    summary_json,verdict=read(run_dir/'SUMMARY.json'),read(run_dir/'VERDICT.json')
    assert summary_json['complete'] and summary_json['PASS'] and verdict['complete'] and verdict['PASS']
    train,decision=read(run_dir/'TRAIN_PROTOCOL.json'),read(run_dir/'DECISION_PROTOCOL.json')
    assert decision['primary_arm']=='image_joint' and decision['train_protocol_sha256']==sha(run_dir/'TRAIN_PROTOCOL.json')
    assert tuple(train['evaluation']['primary_2d'])==KP and tuple(train['evaluation']['primary_6d'])==POSE
    criteria={};ci_max_abs_delta=0.
    for arm in ARMS:
        for metric in METRICS:
            for field in ('mean','std','seeds'):check(summary_json['arms'][arm]['metrics'][metric][field],raw['arms'][arm][metric][field],0)
            check(summary_json['baseline']['two_d' if metric in KP else 'main_6d'][metric],raw['baseline'][metric],0)
    for metric in METRICS:
        ci=summary_json['comparisons']['image_joint_vs_R0']['metrics'][metric]
        independent=bootstrap['metrics'][metric]
        for bound in ('low','high'):
            check(ci['session_cluster'][bound],independent[bound],1e-10)
            ci_max_abs_delta=max(ci_max_abs_delta,abs(ci['session_cluster'][bound]-independent[bound]))
        assert ci['resamples']==10000 and ci['paired_sessions']==13 and ci['paired_frames']==independent['paired_frames']
        benefit=independent['confirmed_benefit'];assert ci['session_cluster']['confirmed_benefit']==benefit
        mean=raw['arms']['image_joint'][metric]['mean'];old=raw['baseline'][metric]
        improved=mean<old if metric in LOWER else mean>old
        criteria[metric]=dict(seed_mean_improved=bool(improved),session_ci_supports_improvement=benefit,confirmed=bool(improved and benefit))
    keypoint=all(criteria[k]['confirmed'] for k in KP)
    coverage=all(v>=1 for k,v in raw['coverages'].items() if k.startswith('image_joint'))
    pose=all(criteria[k]['confirmed'] for k in POSE) and coverage
    assert verdict['metric_criteria']==criteria and verdict['keypoint_gain_confirmed']==keypoint and verdict['pose_gain_confirmed']==pose
    assert verdict['pose_coverage_preserved']==coverage and verdict['negative_detection_preserved']
    assert verdict['overall_accuracy_improved']==bool(keypoint and pose)
    assert summary_json['identity']['negative_raw_candidates_sha256']==raw['identity']['negative_raw_candidates_sha256']
    runtime=read(run_dir/'RUNTIME.json');assert summary_json['runtime']==runtime
    repair_path=run_dir/'repairs/csv_precision_001/REPAIR_MANIFEST.json'
    repair=read(repair_path);contract=read(run_dir/'DRIVER_CONTRACT.json')
    assert repair['PASS'] and repair['complete'] and sha(run_dir/'DRIVER_CONTRACT.json')==repair['original_driver_contract_sha256']
    changed=[]
    for path,digest in contract['source_sha256'].items():
        if sha(path)!=digest:
            assert repair['source_before_sha256'][path]==digest and repair['source_after_sha256'][path]==sha(path)
            changed.append(path)
    assert len(changed)==1 and Path(changed[0]).name=='aggregate_results.py'
    assert sha(repair['source_backup'])==repair['source_before_sha256'][changed[0]]
    before,after=(ast.parse(Path(p).read_text()) for p in (repair['source_backup'],changed[0]))
    for tree in (before,after):tree.body=[node for node in tree.body if not(isinstance(node,ast.FunctionDef) and node.name=='close')]
    assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
    for item in repair['preserved_completed_stages']:assert sha(item['artifact'])==item['sha256']
    for path,digest in repair['additional_bound_artifacts'].items():assert sha(path)==digest
    report=dict(schema='pallet_line_pose_independent_final_metric_audit_v1',complete=True,PASS=True,
        original_evaluator_json_extraction_and_seed_mean_std_exact=True,seed_std_ddof=1,
        baseline=raw['baseline'],arms=raw['arms'],identity=raw['identity'],runtime=raw['runtime'],
        runtime_raw_timing_recalculation_all_nine_exact=True,runtime_accuracy_parity_PASS=True,
        primary_session_ci_independent_reproduction=bootstrap['metrics'],independent_verdict_criteria=criteria,
        independent_vs_summary_primary_ci_max_abs_delta=ci_max_abs_delta,
        keypoint_gain_confirmed=keypoint,pose_gain_confirmed=pose,overall_accuracy_improved=bool(keypoint and pose),
        full_precision_metric_values_unchanged_by_csv_serialization_repair=True,
        input_sha256={str(run_dir/name):sha(run_dir/name) for name in ('SUMMARY.json','VERDICT.json','TRAIN_PROTOCOL.json','DECISION_PROTOCOL.json','SELECTION.json','REAL_EVALUATION_COMPLETE.json','RUNTIME.json','INDEPENDENT_RAW_METRICS.json','INDEPENDENT_PRIMARY_SESSION_BOOTSTRAP.json')},
        reporting_repair=dict(manifest=str(repair_path),sha256=sha(repair_path),allowed_changed_source=changed[0],
            other_original_driver_sources_unchanged=13,computation_ast_outside_validation_close_function_unchanged=True,
            completed_training_selection_real_predictions_runtime_hashes_preserved=True,
            time_scope='The earlier training-matrix audit verified14 unchanged files before this later reporting-only serialization check repair. The final audit verifies13 unchanged files plus the one explicitly archived/authorized reporting repair.'),
        audit_source_sha256=sha(Path(__file__)),gpu_used=False,new_model_inference_executed=False,source_files_modified_by_this_audit=False,
        limits=['Reused319-frame DEV and13sessions; no independent final real-generalization claim.','Conditional session intervals average3fixed training seeds and do not represent all future training seeds.','Latency excludes image decoding and PnP. Allnegative candidates are preserved by copy, with no newnegative forward.'])
    write(run_dir/'INDEPENDENT_FINAL_METRIC_AUDIT.json',report)
    print(json.dumps(dict(PASS=True,overall_accuracy_improved=report['overall_accuracy_improved'],criteria=criteria),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    run(p.parse_args().run_dir.resolve())

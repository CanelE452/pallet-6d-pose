"""No-training diagnostic on14 fixed saved frames x3 old joint seeds.

Candidate generation and fixed selection finish BEFORE GT is read for each
record. GT is used only by evaluation and the separately named oracle bound.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from . import geometry as G

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'data/pallet/results/pallet_dht_joint_v1'
CASE='eval_pallet07:1778652166837872128'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(x) for x in value]
    return value


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(serial(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def errors(points,gt,mask):
    e=np.linalg.norm(np.asarray(points)-gt,axis=-1)
    use=e[mask]
    return dict(errors_px=e,supervised_count=int(mask.sum()),median_px=float(np.median(use)) if len(use) else None,
        mean_px=float(use.mean()) if len(use) else None)


def pooled_summary(records,corner_only=False):
    by_method={k:[] for k in ('baseline','fixed_selection','oracle')}
    good={k:[0,0] for k in ('fixed_selection','oracle')}
    supervised=matched_supervised=0;matches=0
    for record in records:
        mask=np.array(record['gt_supervised'],bool)
        if corner_only:mask[8]=False
        supervised+=int(mask.sum())
        if not record['iou50_match']:continue
        matches+=1;matched_supervised+=int(mask.sum())
        base=np.array(record['evaluation']['baseline']['errors_px'])[mask]
        for method in by_method:
            e=np.array(record['evaluation'][method]['errors_px'])[mask]
            by_method[method].extend(e.tolist())
            if method in good:
                good[method][0]+=int(((base<=10)&(e>10)).sum())
                good[method][1]+=int((base<=10).sum())
    result={}
    for method,values in by_method.items():
        a=np.array(values)
        if len(a):
            result[method]=dict(median_px=float(np.median(a)),p90_px=float(np.percentile(a,90)),mean_px=float(a.mean()),
                PCK10=float((a<=10).mean()),PCK20=float((a<=20).mean()),rate_gt20=float((a>20).mean()),
                rate_gt50=float((a>50).mean()),rate_gt100=float((a>100).mean()),point_count=len(a))
            if method in good:
                bad,total=good[method]
                result[method].update(good_le10_to_gt10_count=bad,baseline_good_le10_count=total,
                    good_le10_to_gt10_rate=bad/total if total else None)
        else:result[method]=dict(point_count=0)
    return dict(n_frames=len(records),matched_frames=matches,frame_coverage=matches/len(records),
        supervised_point_denominator=supervised,matched_supervised_points=matched_supervised,
        point_coverage=matched_supervised/supervised if supervised else None,metrics=result)


def run(run_dir):
    run_dir=Path(run_dir).resolve()
    inputs={}
    def bind(path,expected=None):
        path=Path(path).resolve();value=sha(path)
        if expected is not None and value!=expected:raise ValueError(f'Bound input changed: {path}')
        if str(path) in inputs and inputs[str(path)]!=value:raise ValueError(f'Input changed during diagnostic: {path}')
        inputs[str(path)]=value
        return path
    def read(path,expected=None):return json.loads(bind(path,expected).read_text())
    protocol=read(run_dir/'TRAIN_PROTOCOL.json')
    if protocol['geometry']!=G.DEFAULT_CONFIG:raise ValueError('Protocol differs from fixed geometry configuration')
    if protocol['diagnostic']['existing_seeds']!=[1,2,3] or protocol['diagnostic']['n_frames_per_seed']!=14:
        raise ValueError('Only the predeclared14frames x3 source seeds are allowed')
    if protocol['diagnostic']['ground_truth_selection'] or not protocol['evaluation']['original_semantic_ids']:
        raise ValueError('Diagnostic forbids GT-based deployed selection or relabeling')
    source_paths=[Path(__file__).resolve(),Path(G.__file__).resolve()]
    source_hashes={str(p):sha(p) for p in source_paths}
    manifest=read(protocol['data']['real_manifest'],protocol['data']['real_manifest_sha256'])
    lookup={x['frame_id']:x for x in manifest['items']}
    selected={CASE}
    sessions={}
    for item in manifest['items']:sessions.setdefault(item['session_id'],[]).append(item['frame_id'])
    for values in sessions.values():selected.add(min(values,key=lambda x:hashlib.sha256(x.encode()).hexdigest()))
    assert len(selected)==14
    records=[];csv_rows=[];max_csv_delta=0.
    for seed in (1,2,3):
        directory=SOURCE/'evaluation'/f'hough_joint_seed{seed}'
        completion=read(directory/'COMPLETION.json')
        if not (completion['complete'] and completion['PASS']):raise ValueError('Source evaluation incomplete')
        prediction=read(directory/'PREDICTIONS.json',completion['output_sha256']['PREDICTIONS.json'])
        evidence=read(directory/'LINE_EVIDENCE.json',completion['output_sha256']['LINE_EVIDENCE.json'])
        assert set(evidence['selected_frame_ids'])==selected
        sample_lookup={x['frame_id']:x for x in evidence['samples']}
        assert set(sample_lookup)==selected and not evidence['expected_absent']
        csv_path=bind(directory/'PAPER_2D_per_frame.csv',completion['output_sha256']['PAPER_2D_per_frame.csv'])
        with csv_path.open() as f:csv_lookup={r['frame_id']:r for r in csv.DictReader(f) if r['frame_id'] in selected}
        assert set(csv_lookup)==selected
        for frame_id in sorted(selected):
            sample=sample_lookup[frame_id];item=lookup[frame_id]
            assert sample['image_key']==item['image_path']
            candidates=prediction['frames'][item['image_path']]
            if not candidates:raise ValueError('No source detection; explicit missing-case support required')
            top=max(candidates,key=lambda x:x['score'])
            with np.load(bind(sample['path'],sample['sha256']),allow_pickle=False) as a:
                geometry=G.build_candidates(a['logits'],a['theta_radians'],a['rho_values'],a['valid'],
                    a['feature_shape_hw'],a['input_shape_hw'],a['raw_to_input_affine'],
                    top['keypoints_xy'],top['keypoints_conf'],G.DEFAULT_CONFIG)
            fixed=G.fixed_select(geometry)
            # GT becomes available only after the deployed calculation above.
            annotation=read(ROOT/item['gt_v2_path'])
            obj=annotation['objects'][0]
            assert obj['keypoint_frame']=='camera_dynamic_0123_v4'
            gt=np.array([p['xy'] for p in obj['keypoint_annotations']],float)
            mask=np.array([p['visibility']>0 for p in obj['keypoint_annotations']],bool)
            assert gt.shape==(9,2) and np.isfinite(gt).all()
            baseline=np.asarray(top['keypoints_xy'],float)
            oracle_points=baseline.copy();oracle_indices=np.zeros(8,int)
            distance=np.linalg.norm(geometry['candidates_xy']-gt[:8,None,:],axis=-1)
            for corner in range(8):
                if mask[corner]:
                    oracle_indices[corner]=np.argmin(np.where(geometry['candidate_valid'][corner],distance[corner],np.inf))
                    oracle_points[corner]=geometry['candidates_xy'][corner,oracle_indices[corner]]
            evaluation={name:errors(value,gt,mask) for name,value in
                [('baseline',baseline),('fixed_selection',fixed['points_xy']),('oracle',oracle_points)]}
            assert np.all(evaluation['oracle']['errors_px'][mask]<=evaluation['baseline']['errors_px'][mask]+1e-10)
            row=csv_lookup[frame_id]
            assert float(row['top_score'])==top['score']
            matched=row['top_iou50_match']=='True'
            saved=np.array([float(x) for x in row['top_keypoint_supervised_errors_px'].split(';') if x])
            if len(saved):
                delta=float(abs(saved-evaluation['baseline']['errors_px'][mask]).max())
                if delta>5.1e-7:raise ValueError('Original candidate/GT does not reproduce source official errors')
                max_csv_delta=max(max_csv_delta,delta)
            for name in ('fixed_selection','oracle'):
                old=evaluation['baseline']['errors_px'];new=evaluation[name]['errors_px']
                evaluation[name]['improved_supervised_count']=int(((new<old-1e-9)&mask).sum())
                evaluation[name]['worsened_supervised_count']=int(((new>old+1e-9)&mask).sum())
            record=dict(frame_id=frame_id,session_id=item['session_id'],image_path=item['image_path'],seed=seed,
                source_evaluation=str(directory),checkpoint_sha256=completion['checkpoint_sha256'],
                image_sha256=prediction['frame_metadata'][item['image_path']]['image_sha256'],
                baseline_points_xy=baseline,selected_points_xy=fixed['points_xy'],
                gt_points_xy=gt,gt_supervised=mask,original_candidate_count=len(candidates),score=top['score'],
                original_box_xyxy=top['box_xyxy'],iou50_match=matched,
                geometry=geometry,selection=fixed,evaluation=evaluation,
                oracle=dict(indices=oracle_indices,points_xy=oracle_points,deployable=False,
                    type='same_ID_per_corner_best_available_candidate_with_GT',
                    note='Independent per-corner upper bound, not a globally consistent cuboid or learned selection result.'))
            records.append(record)
            csv_rows.append(dict(frame_id=frame_id,seed=seed,iou50_match=matched,
                baseline_median_px=evaluation['baseline']['median_px'],fixed_median_px=evaluation['fixed_selection']['median_px'],
                oracle_median_px=evaluation['oracle']['median_px'],
                original_fallback_count=int((fixed['indices']==0).sum()),
                valid_intersections=geometry['diagnostics']['generated_valid_intersections']))
    assert len(records)==42
    summaries={str(seed):dict(nine_points=pooled_summary([r for r in records if r['seed']==seed]),
        corners_only=pooled_summary([r for r in records if r['seed']==seed],True)) for seed in (1,2,3)}
    for p,value in inputs.items():assert sha(p)==value,'Input changed during diagnostic: '+p
    for p,value in source_hashes.items():assert sha(p)==value,'Source changed during diagnostic: '+p
    csv_output=run_dir/'DIAGNOSTIC_PER_FRAME.csv'
    with csv_output.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    output=dict(schema='decoder_probe_no_training_diagnostic_v1',complete=True,PASS=True,
        created_at_utc=datetime.now(timezone.utc).isoformat(),no_new_model_forwards=True,no_training=True,
        actual_records=42,unique_real_frames=14,model_seeds=[1,2,3],source_arm='hough_joint',
        geometry_config=G.DEFAULT_CONFIG,feature_names=list(G.FEATURE_NAMES),selection_uses_gt=False,
        oracle_is_not_deployable=True,per_seed=summaries,records=records,
        caveats=['Exploratory14fixedDEVframes, not an independent final/generalization result.',
            'Same14frames across3seeds:42records are not42 independent images.',
            'Candidate generation is discrete/frozen; this diagnostic does not train or validate a differentiable learned selector.',
            'All12same-role distributions used at inference; v>0 labels are evaluation supervision, not a physical-edge visibility mask.',
            'Independent corner selection does not enforce that two endpoints select the same line mode or a common3D cuboid.',
            'Weak original-point prior may retain semantic swaps; full distribution evidence is not calibrated correctness.',
            'Even uniform mass on a finite Hough footprint induces a location-dependent mixture cost; original-point fallback does not guarantee identity for uninformative line evidence.',
            'Peak multiplicity may create repeated candidate coordinates; extraction order is fixed and no GT deduplication/selection occurs.'],
        original_csv_rounding_max_abs_delta_px=max_csv_delta,input_sha256=inputs,source_sha256=source_hashes,
        output_sha256={str(csv_output):sha(csv_output)})
    write(run_dir/'DIAGNOSTIC.json',output)
    print(json.dumps(dict(complete=True,PASS=True,actual_records=42,per_seed=summaries),indent=2))
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();run(args.run_dir)


if __name__=='__main__':main()

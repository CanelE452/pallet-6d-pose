"""Independent saved-output/checkpoint audit. No CNN or decoder forward.

Actual metric arithmetic is recomputed here instead of calling evaluate.py.
No PASS artifact is written before every required actual output is present.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
ARMS=('baseline','point_only','line_fusion','wrong_image_line')
CASE='eval_pallet07:1778652166837872128'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def tensor_sha(state):
    h=hashlib.sha256()
    for name,value in state.items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def close(actual,expected,name,atol=1e-9):
    if actual is None or expected is None:
        if actual is not expected:raise AssertionError(name)
        return
    if not np.allclose(actual,expected,rtol=0,atol=atol):raise AssertionError(f'{name}: {actual} != {expected}')


def decoded_points(payload):
    mask=np.asarray(payload['point_valid'],bool)
    assert mask.shape==(9,) and len(payload['points'])==9
    points=np.asarray([[0.,0.] if p is None else p for p in payload['points']],float)
    assert points.shape==(9,2) and np.isfinite(points).all()
    assert all(p is not None or not ok for p,ok in zip(payload['points'],mask))
    points[~mask]=0.
    return points,mask


def audit(run_dir):
    run=Path(run_dir).resolve();inputs={}
    def bind(path,expected=None):
        path=Path(path).resolve();value=sha(path)
        if expected is not None:assert value==expected,f'Hash mismatch: {path}'
        if str(path) in inputs:assert inputs[str(path)]==value
        inputs[str(path)]=value;return path
    def read(path,expected=None):return json.loads(bind(path,expected).read_text())
    def verify_maps(payload,directory):
        if isinstance(payload,dict):
            for key,value in payload.items():
                if key in ('source_sha256','input_sha256','output_sha256','array_sha256','frame_sha256') and isinstance(value,dict):
                    for p,digest in value.items():
                        if isinstance(digest,str) and len(digest)==64:
                            path=Path(p);bind(path if path.is_absolute() else directory/path,digest)
                elif isinstance(value,(dict,list)):verify_maps(value,directory)
        elif isinstance(payload,list):
            for value in payload:verify_maps(value,directory)
    protocol=read(run/'TRAIN_PROTOCOL.json');protocol_sha=sha(run/'TRAIN_PROTOCOL.json')
    assert protocol['stage']=='main' and protocol['training']['steps']==1000
    assert protocol['model']['parameter_count']==55118
    manifest=read(run/'MANIFEST.json');cache=read(run/'CACHE_MANIFEST.json')
    cached=read(run/'CACHE_RECORDS.json');cache_done=read(run/'CACHE_COMPLETION.json')
    extraction=read(run/'EXTRACTION_COMPLETE.json');training=read(run/'TRAINING_COMPLETION.json')
    evaluation=read(run/'EVALUATION_COMPLETION.json');predictions=read(run/'PREDICTIONS.json')
    result=read(run/'RESULTS.json');frames=read(run/'FRAME_METRICS.json');diagnostic=read(run/'DIAGNOSTIC.json')
    frozen=read(run/'SOURCE_FREEZE.json')
    for payload in (cache_done,extraction,training,evaluation,predictions,result,diagnostic):
        assert payload.get('complete') is True and payload.get('PASS',True) is True
        verify_maps(payload,run)
    verify_maps(frozen,run)
    assert manifest['protocol_sha256']==protocol_sha
    assert training['protocol_sha256']==protocol_sha
    cache_counts={k:len(v) for k,v in manifest['populations'].items()}
    assert cache_counts==dict(synth_train=2048,synth_val=512,real_dev=319)
    assert len(manifest['records'])==len(cached['records'])==cache_done['frames']==2879
    assert {r['index'] for r in manifest['records']}==set(range(2879))
    train_indices=set(manifest['populations']['synth_train']);val_indices=set(manifest['populations']['synth_val'])
    real_indices=set(manifest['populations']['real_dev'])
    assert not (train_indices&val_indices or train_indices&real_indices or val_indices&real_indices)
    assert extraction['frozen_state_sha_before']==extraction['frozen_state_sha_after']
    assert extraction['real_GT_read'] is False and cache_done['real_GT_read'] is False
    common={k:np.load(spec['path'],mmap_mode='r') for k,spec in cache['arrays']['common'].items()}
    assert not np.asarray(common['gt_valid'][list(real_indices)]).any()
    # Verify actual checkpoint tensors and the entire optimizer-index trace.
    import torch
    cells=[];initials=[];traces=[]
    for arm in ('point_only','line_fusion'):
        directory=run/'runs'/arm;done=read(directory/'COMPLETION.json');setup=read(directory/'SETUP.json')
        verify_maps(done,directory)
        assert done['complete'] and done['PASS'] and done['optimizer_steps']==1000 and done['parameters']==55118
        assert done['train_frames']==2048 and done['val_frames']==512 and done['real_samples_used_for_optimization_or_validation']==0
        checkpoint=torch.load(bind(done['checkpoint'],done['checkpoint_sha256']),map_location='cpu',weights_only=False)
        initial=torch.load(bind(directory/'INITIAL_STATE.pth',setup['initial_state_file_sha256']),map_location='cpu',weights_only=False)
        assert checkpoint['optimizer_steps']==1000 and checkpoint['parameters']==55118 and checkpoint['complete']
        assert sum(v.numel() for v in checkpoint['state_dict'].values())==55118
        assert tensor_sha(initial)==done['initial_state_tensor_sha256']==setup['initial_state_tensor_sha256']
        assert tensor_sha(checkpoint['state_dict'])==done['final_state_tensor_sha256']
        changed=sum(int((initial[k]!=checkpoint['state_dict'][k]).sum()) for k in initial)
        assert changed==done['changed_parameter_values'] and changed>0
        trace_path=bind(directory/'BATCH_TRACE.jsonl',done['trace_sha256'])
        trace=[json.loads(s) for s in trace_path.read_text().splitlines()]
        assert [x['step'] for x in trace]==list(range(1,1001))
        assert all(len(x['indices'])==64 and set(x['indices'])<=train_indices for x in trace)
        for row in trace:
            assert int(np.asarray(common['loss_valid'][row['indices']]).sum())==row['supervised_corners']>0
        assert sum(x['supervised_corners'] for x in trace)==done['supervised_corner_exposures']
        initials.append(done['initial_state_tensor_sha256']);traces.append(done['trace_sha256'])
        cells.append(dict(arm=arm,updates=1000,parameters=55118,changed_parameter_values=changed,
            checkpoint_sha256=done['checkpoint_sha256'],trace_sha256=done['trace_sha256']))
    assert len(set(initials))==len(set(traces))==1
    assert training['actual_optimizer_steps']==2000
    # Read the precise model input allow-list; target arrays are never inputs.
    model_source=ROOT/'scripts/research/pallet_dht_decoder_probe_v1/model.py'
    tree=ast.parse(bind(model_source).read_text());input_keys=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INPUT_KEYS' for t in node.targets):
            input_keys=ast.literal_eval(node.value)
    assert input_keys is not None and not any('gt' in k or 'target' in k or 'loss' in k for k in input_keys)
    # All final output coordinates are checked against independent canonical GT.
    source=read(protocol['data']['source_manifest'],protocol['data']['source_manifest_sha256'])
    source_by_id={r['id']:r for r in source['records']}
    positive=read(protocol['data']['real_manifest'],protocol['data']['real_manifest_sha256'])
    gt_items={r['frame_id']:r for r in positive['items']}
    prediction_by_id={r['id']:r for r in predictions['records']}
    frame_by_id={r['id']:r for r in frames['records']}
    assert len(prediction_by_id)==len(predictions['records'])==len(frame_by_id)==831
    assert Counter(r['population'] for r in predictions['records'])==dict(real_dev=319,synth_val=512)
    assert predictions['n_actual_cached_head_forwards']==2493 and predictions['real_GT_used_in_model'] is False
    checkpoints={r['arm']:r['checkpoint_sha256'] for r in cells}
    independent=[];case=None
    for fid,pred in prediction_by_id.items():
        stored=frame_by_id[fid];population=pred['population']
        if population=='real_dev':
            item=gt_items[fid];annotation=read(ROOT/item['gt_v2_path'])['objects'][0]
            gt=np.array([p['xy'] for p in annotation['keypoint_annotations']],float)
            mask=np.array([p['visibility']>0 for p in annotation['keypoint_annotations']],bool)
        else:
            original=source_by_id[fid];assert len(original['targets'])==1
            target=np.asarray(original['targets'][0]['keypoints_normalized'],float)
            gt=target[:,:2]*np.array(original['prepared_shape_hw'][::-1])-original['reflect_pad_px']
            mask=target[:,2]>0
        close(stored['gt_points'],gt,f'{fid}/GT',1e-9)
        assert np.array_equal(stored['gt_supervised'],mask)
        baseline,valid=decoded_points(pred['baseline'])
        local=dict(id=fid,population=population,session_id=pred['session_id'],mask=mask,errors={})
        for arm in ARMS:
            payload=pred['baseline'] if arm=='baseline' else pred['arms'][arm]
            point,pv=decoded_points(payload)
            assert np.array_equal(valid,pv)
            assert np.array_equal(point[8],baseline[8])
            if arm!='baseline':
                expected_arm='line_fusion' if arm=='wrong_image_line' else arm
                assert payload['actual_cached_head_forward'] and payload['checkpoint_sha256']==checkpoints[expected_arm]
                delta=np.asarray(payload['decoder']['delta'],np.float32)
                expected_xy=baseline[:8].astype(np.float32)+delta
                assert np.array_equal(expected_xy[pv[:8]].astype(float),point[:8][pv[:8]])
            observed=mask&pv&stored['baseline_match_iou50']
            error=np.linalg.norm(point-gt,axis=-1)
            expected=[float(e) if ok else None for e,ok in zip(error,observed)]
            saved=stored['arms'][arm]['errors_px']
            for j,(a,b) in enumerate(zip(expected,saved)):close(a,b,f'{fid}/{arm}/{j}')
            local['errors'][arm]=expected
        independent.append(local)
        if fid==CASE:case=dict(frame_id=fid,gt_supervised_indices=np.flatnonzero(mask).tolist(),
            baseline_points_xy=baseline.tolist(),line_fusion_points_xy=pred['arms']['line_fusion']['points'],
            original_semantic_ids=True,centroid_unchanged=True)
    assert case is not None
    checked_summaries=0
    for collection,indices in [('summaries',range(9)),('corner_only_summaries',range(8))]:
        for row in result[collection]:
            rr=[r for r in independent if r['population']==row['population']]
            values=[r['errors'][row['arm']][i] for r in rr for i in indices if r['errors'][row['arm']][i] is not None]
            arr=np.array(values);denom=sum(int(r['mask'][list(indices)].sum()) for r in rr)
            assert len(values)==row['n_observed_points'] and denom==row['n_gt_points']
            close(row['point_coverage'],len(values)/denom,'coverage')
            for key,value in dict(median_px=np.median(arr),p90_px=np.quantile(arr,.9),mean_px=arr.mean(),
                pck10_observed=np.mean(arr<=10),pck20_observed=np.mean(arr<=20),
                pck10_all_supervised=np.sum(arr<=10)/denom,pck20_all_supervised=np.sum(arr<=20)/denom,
                above20_fraction=np.mean(arr>20),above50_fraction=np.mean(arr>50),above100_fraction=np.mean(arr>100)).items():
                close(row[key],value,f'{collection}/{row["population"]}/{row["arm"]}/{key}')
            checked_summaries+=1
    # One prespecified representative paired session bootstrap, independently.
    real=[r for r in independent if r['population']=='real_dev'];deltas=[]
    for r in real:
        pair=[a-b for a,b in zip(r['errors']['line_fusion'],r['errors']['point_only']) if a is not None and b is not None]
        if pair:deltas.append((r['session_id'],float(np.mean(pair))))
    sessions=sorted({s for s,_ in deltas})
    sums=np.array([sum(d for s,d in deltas if s==session) for session in sessions])
    counts=np.array([sum(s==session for s,_ in deltas) for session in sessions])
    draws=np.random.default_rng(20260909).multinomial(len(sessions),np.full(len(sessions),1/len(sessions)),size=20000)
    sampled=(draws*sums).sum(1)/(draws*counts).sum(1)
    interval=np.quantile(sampled,[.025,.975])
    reported=next(r for r in result['paired'] if r['left']=='line_fusion' and r['right']=='point_only')
    close(interval,reported['ci95'],'line-control paired CI',1e-9)
    close(sums.sum()/counts.sum(),reported['mean_frame_delta_px'],'line-control paired mean')
    # Check the continuation decision from measured values, not headline text.
    real_stats={r['arm']:r for r in result['summaries'] if r['population']=='real_dev'}
    damage={}
    for arm in ('point_only','line_fusion'):
        pairs=[(a,b) for r in real for a,b in zip(r['errors']['baseline'],r['errors'][arm])
               if a is not None and b is not None and a<=10]
        damage[arm]=sum(b>10 for a,b in pairs)/len(pairs) if pairs else None
    continuation=all(real_stats['line_fusion']['median_px']<real_stats[ref]['median_px']
        and real_stats['line_fusion']['p90_px']<=real_stats[ref]['p90_px']
        and real_stats['line_fusion']['n_observed_points']>=real_stats[ref]['n_observed_points']
        for ref in ('point_only','baseline'))
    continuation=continuation and interval[1]<0 and all(v is not None for v in damage.values()) \
        and damage['line_fusion']<=damage['point_only']
    assert bool(continuation)==result['continuation']['continuation_signal']
    assert diagnostic['actual_records']==42 and diagnostic['unique_real_frames']==14 and not diagnostic['selection_uses_gt']
    assert result['six_d_evaluated'] is False and result['independent_final'] is False
    for path,value in inputs.items():assert sha(path)==value,'Input changed during audit: '+path
    output=dict(schema='pallet_dht_decoder_independent_audit_v1',complete=True,PASS=True,
        created_at_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=protocol_sha,
        scope='Execution and saved2D metric audit only; PASS is not an accuracy improvement, independent FINAL, or6D verdict.',
        no_model_forwards=True,no_training=True,cache_counts=cache_counts,
        training_cells=cells,identical_initial_state=True,identical_all1000_batch_traces=True,
        input_allow_list=list(input_keys),GT_targets_absent_from_model_inputs=True,
        all831_frame_errors_independently_recomputed=True,summary_tables_checked=checked_summaries,
        actual_cached_head_forwards=2493,saved_delta_reconstructs_actual_output_points=True,
        synthetic_unmatched_GT_kept_in_denominator=True,original_centroids_and_masks_preserved=True,
        continuation_criterion_independently_recomputed=bool(continuation),
        paired_bootstrap=dict(contrast='line_fusion-point_only',resamples=20000,unit='session',ci95=interval.tolist(),recomputed_independently=True),
        supplied_case=case,input_sha256=inputs,source_sha256={str(Path(__file__).resolve()):sha(__file__)})
    target=run/'INDEPENDENT_AUDIT.json'
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True,PASS=True,summary_tables_checked=checked_summaries,protocol_sha256=protocol_sha)))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    audit(parser.parse_args().run_dir)

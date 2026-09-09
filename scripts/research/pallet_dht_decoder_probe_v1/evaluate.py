"""Cached small-head inference and original-coordinate metrics; no detector forward.

Original semantic IDs and supervised points remain fixed. The real population is
reused DEV319; this single-seed, frozen-network experiment is an exploratory pilot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ('baseline', 'point_only', 'line_fusion', 'wrong_image_line')
CASE = 'eval_pallet07:1778652166837872128'
N_BOOTSTRAP = 20000
BOOTSTRAP_SEED = 20260909


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None:
            require(digest == expected, f'Input changed: {path}')
        self.hashes[str(path)] = digest
        return path

    def read(self, path, expected=None):
        return read(self.bind(path, expected))

    def verify(self):
        for path, digest in self.hashes.items():
            require(sha(path) == digest, f'Input changed during evaluation: {path}')


def finite_points(value, valid=None):
    if value is None:
        return np.zeros((9, 2), dtype=float), np.zeros(9, dtype=bool)
    points = np.array([[np.nan, np.nan] if p is None else p for p in value], dtype=float)
    require(points.shape == (9, 2), 'Every prediction retains nine original point IDs')
    mask = np.isfinite(points).all(axis=1)
    if valid is not None:
        require(np.shape(valid) == (9,), 'Malformed predicted point mask')
        mask &= np.asarray(valid, dtype=bool)
    points[~mask] = 0.
    return points, mask


def summarize(rows, arm, point_indices=tuple(range(9))):
    values, supervised, observed, moves, frame_errors = [], 0, 0, [], []
    for row in rows:
        gt_mask = np.asarray(row['gt_supervised'], bool)[list(point_indices)]
        errors = [row['arms'][arm]['errors_px'][i] for i in point_indices]
        supervised += int(gt_mask.sum())
        local = [float(e) for e in errors if e is not None]
        observed += len(local)
        values.extend(local)
        if local:
            frame_errors.append(float(np.mean(local)))
        moves.extend(row['arms'][arm]['move_px'][i] for i in point_indices if row['arms'][arm]['move_px'][i] is not None)
    a = np.asarray(values, dtype=float)
    return dict(n_frames=len(rows), n_gt_points=supervised, n_observed_points=observed,
        point_coverage=observed/supervised if supervised else None,
        n_frames_with_observations=len(frame_errors), n_missing_or_unmatched_points=supervised-observed,
        mean_px=float(a.mean()) if len(a) else None, median_px=float(np.median(a)) if len(a) else None,
        p90_px=float(np.quantile(a,.9)) if len(a) else None,
        frame_mean_px=float(np.mean(frame_errors)) if frame_errors else None,
        pck10_observed=float(np.mean(a<=10)) if len(a) else None,
        pck20_observed=float(np.mean(a<=20)) if len(a) else None,
        pck10_all_supervised=float(np.sum(a<=10)/supervised) if supervised else None,
        pck20_all_supervised=float(np.sum(a<=20)/supervised) if supervised else None,
        above20_fraction=float(np.mean(a>20)) if len(a) else None,
        above50_fraction=float(np.mean(a>50)) if len(a) else None,
        above100_fraction=float(np.mean(a>100)) if len(a) else None,
        mean_move_px=float(np.mean(moves)) if moves else None)


def point_difficulty(rows, arm):
    output = []
    for name in ('easy_le10', 'moderate_10_20', 'hard_gt20'):
        before, after, frame_ids = [], [], set()
        for row in rows:
            for base, actual in zip(row['arms']['baseline']['errors_px'],row['arms'][arm]['errors_px']):
                if base is None or actual is None:
                    continue
                label = 'easy_le10' if base<=10 else 'moderate_10_20' if base<=20 else 'hard_gt20'
                if label == name:
                    before.append(base);after.append(actual);frame_ids.add(row['id'])
        before,after = np.asarray(before),np.asarray(after)
        output.append(dict(arm=arm,difficulty=name,n_points=len(before),n_frames=len(frame_ids),
            baseline_mean_px=float(before.mean()) if len(before) else None,
            after_mean_px=float(after.mean()) if len(after) else None,
            mean_delta_px=float(np.mean(after-before)) if len(before) else None,
            improved_fraction=float(np.mean(after<before)) if len(before) else None,
            worsened_fraction=float(np.mean(after>before)) if len(before) else None,
            easy_to_over10_rate=float(np.mean(after>10)) if len(before) and name=='easy_le10' else None,
            scope='GT-defined posthoc point difficulty; never a deployable selection rule.'))
    return output


def paired(rows, left, right, resamples=N_BOOTSTRAP):
    """Paired per-frame mean error, resampling complete capture sessions."""
    records=[]
    for row in rows:
        pairs=[(a,b) for a,b in zip(row['arms'][left]['errors_px'],row['arms'][right]['errors_px']) if a is not None and b is not None]
        if pairs:
            records.append((row['session_id'],float(np.mean([a-b for a,b in pairs]))))
    if not records:
        return dict(left=left,right=right,n_frames=0,n_sessions=0,mean_frame_delta_px=None,ci95=[None,None],confirmed_mean_benefit=False)
    sessions=sorted({s for s,_ in records})
    sums=np.array([sum(v for s,v in records if s==k) for k in sessions])
    counts=np.array([sum(s==k for s,_ in records) for k in sessions])
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    weights=rng.multinomial(len(sessions),np.full(len(sessions),1/len(sessions)),size=resamples)
    draws=(weights@sums)/(weights@counts)
    low,high=np.quantile(draws,[.025,.975])
    return dict(left=left,right=right,n_frames=len(records),n_sessions=len(sessions),
        mean_frame_delta_px=float(sums.sum()/counts.sum()),ci95=[float(low),float(high)],
        confirmed_mean_benefit=bool(high<0),resamples=resamples,seed=BOOTSTRAP_SEED,
        estimand='Mean of original-ID per-frame mean-error differences on common observed points; whole-session paired bootstrap.',
        interpretation='Exploratory single-trained-seed estimate conditional on reused capture sessions, not independent FINAL or stable-gain evidence.')


def continuation(summaries, paired_rows, difficulty):
    stats={r['arm']:r for r in summaries}
    pairs={(r['left'],r['right']):r for r in paired_rows}
    good={r['arm']:r for r in difficulty if r['difficulty']=='easy_le10'}
    checks={}
    for reference in ('point_only','baseline'):
        a,b=stats['line_fusion'],stats[reference]
        checks[reference]=dict(median_improved=a['median_px'] is not None and b['median_px'] is not None and a['median_px']<b['median_px'],
            p90_not_worse=a['p90_px'] is not None and b['p90_px'] is not None and a['p90_px']<=b['p90_px'],
            point_coverage_preserved=a['n_observed_points']>=b['n_observed_points'])
    checks['paired_line_vs_control_ci_upper_below_zero']=pairs['line_fusion','point_only']['confirmed_mean_benefit']
    a,b=good['line_fusion']['easy_to_over10_rate'],good['point_only']['easy_to_over10_rate']
    checks['good_point_damage_no_higher_than_control']=a is not None and b is not None and a<=b
    passed=all(all(v.values()) if isinstance(v,dict) else v for v in checks.values())
    return dict(continuation_signal=bool(passed),criteria=checks,stable_gain_claim=False,
        meaning='Prespecified exploratory continuation signal only; one decoder-training seed and reused DEV319.',
        headline_ko='점·선 명시 결합의 탐색적 후속 진행 기준을 충족했습니다.' if passed else '점·선 명시 결합의 탐색적 후속 진행 기준은 충족하지 못했습니다.')


def infer_cached(run_dir, inputs, device='cpu', batch_size=64):
    """Actual small-decoder forward only; frozen detector/P4 cache is reused."""
    import torch
    from . import cache as C
    from . import model as M
    protocol=inputs.read(run_dir/'TRAIN_PROTOCOL.json')
    require(protocol['schema']=='pallet_dht_decoder_probe_protocol_v1' and protocol['stage']=='main','Main pilot protocol required')
    require(protocol['training']['steps']==1000 and protocol['training']['seed']==1,'Registered1000-step single-seed budget differs')
    require(protocol['evaluation']['bootstrap_draws']==N_BOOTSTRAP and protocol['evaluation']['bootstrap_seed']==BOOTSTRAP_SEED,'Registered paired bootstrap differs')
    cache_done=inputs.read(run_dir/'CACHE_COMPLETION.json')
    require(cache_done['complete'] and cache_done['PASS'] and cache_done['stage']=='main' and cache_done['real_GT_read'] is False,'Completed GT-free real feature cache required')
    inputs.bind(run_dir/'CACHE_MANIFEST.json',cache_done['cache_manifest_sha256'])
    records_payload=inputs.read(run_dir/'CACHE_RECORDS.json',cache_done['cache_records_sha256'])
    manifest=inputs.read(run_dir/'MANIFEST.json',records_payload['manifest_sha256'])
    source_records={r['id']:r for r in manifest['records']}
    require(cache_done['bindings']['protocol_sha256']==sha(run_dir/'TRAIN_PROTOCOL.json'),'Cache protocol changed')
    for path,digest in cache_done['array_sha256'].items():inputs.bind(path,digest)
    for path,digest in cache_done['bindings']['source_sha256'].items():inputs.bind(path,digest)
    require(len(cache_done.get('candidate_evidence_sha256',{}))==cache_done['frames'],'Completed candidate evidence bindings required')
    for path,digest in cache_done['candidate_evidence_sha256'].items():inputs.bind(path,digest)
    selected=[r for r in records_payload['records'] if r['population'] in ('synth_val','real_dev')]
    require(len(selected)==831 and sum(r['population']=='real_dev' for r in selected)==319,'Actual evaluation needs512synth_val+319real')
    require(all(r['wrong_line_available'] for r in selected),'Wrong-line ablation unavailable on an evaluation row')
    by_id={}
    for record in selected:
        require(record['id'] not in by_id,'Duplicate saved cache frame ID')
        inputs.bind(record['image'],record['image_sha256'])
        base=record['baseline'];points,valid=finite_points(base['points'],base['point_valid'])
        if not base['detected']:valid[:]=False
        public={**base,'points':[p.tolist() if v else None for p,v in zip(points,valid)],'point_valid':valid.tolist(),
            'box_xyxy':base['box_xyxy'] if base['detected'] else None}
        row=dict(id=record['id'],index=record['index'],population=record['population'],image=record['image'],
            image_sha256=sha(record['image']),image_key=record['image_key'],width=record['width'],height=record['height'],
            session_id=record['session_id'] or record['population'],baseline=public,arms={})
        if record['population']=='synth_val':
            source=source_records[record['id']]['source_record']
            require(len(source['targets'])==1,'Registered synthetic validation has exactly one target per image')
            inputs.bind(source['label'],source['label_sha256'])
            normalized=np.asarray(source['targets'][0]['keypoints_normalized'],float)
            prepared_h,prepared_w=source['prepared_shape_hw']
            target_points=normalized[:,:2]*[prepared_w,prepared_h]-source['reflect_pad_px']
            row['evaluation_target']=dict(points=target_points.tolist(),valid=(normalized[:,2]>0).tolist(),matched=record['loss_matched'],
                scope='Full source GT denominator, including unmatched frames; model INPUT_KEYS never receive target fields.')
        by_id[record['id']]=row
    training=inputs.read(run_dir/'TRAINING_COMPLETION.json')
    require(training['complete'] and training['PASS'] and training['stage']=='main'
        and training['expected_runs']==2 and training['actual_optimizer_steps']==2000
        and training['identical_initial_state'] and training['identical_full_batch_trace'] and training['no_checkpoint_selection'],
        'Two matched completed1000-update arms required')
    require(training['protocol_sha256']==sha(run_dir/'TRAIN_PROTOCOL.json') and training['cache_completion_sha256']==sha(run_dir/'CACHE_COMPLETION.json'),
        'Training protocol/cache binding differs')
    torch.set_num_threads(4)
    checkpoints={}
    for mode in ('point_only','line_fusion','wrong_image_line'):
        trained_arm='line_fusion' if mode=='wrong_image_line' else mode
        cell=run_dir/'runs'/trained_arm
        completion=inputs.read(cell/'COMPLETION.json')
        require(completion['complete'] and completion['PASS'],'Completed actual decoder training required')
        path=inputs.bind(cell/'checkpoint_final.pth',completion['checkpoint_sha256'])
        model,saved=M.load_refiner(path,device=device)
        require(saved['optimizer_steps']==saved['expected_optimizer_steps']==1000 and saved['arm']==trained_arm
            and saved['seed']==1 and saved['stage']=='main' and saved['complete'],'Final1000-update checkpoint identity differs')
        require(saved['bindings']==completion['bindings'] and saved['bindings']['protocol_sha256']==sha(run_dir/'TRAIN_PROTOCOL.json')
            and saved['bindings']['manifest_sha256']==sha(run_dir/'MANIFEST.json')
            and saved['bindings']['cache_completion_sha256']==sha(run_dir/'CACHE_COMPLETION.json'),'Checkpoint/source bindings differ')
        for source,digest in saved['bindings']['source_sha256'].items():inputs.bind(source,digest)
        inputs.bind(cell/'history.json',saved['history_sha256'])
        inputs.bind(cell/'BATCH_TRACE.jsonl',saved['trace_sha256'])
        checkpoints[mode]=dict(path=str(path),sha256=sha(path),training_arm=trained_arm)
        for parameter in model.parameters():parameter.requires_grad_(False)
        arrays=C.load_model_arrays(run_dir,mode=mode)
        require(not any('gt' in key or 'loss' in key for key in M.INPUT_KEYS),'GT/target key entered inference arguments')
        for start in range(0,len(selected),batch_size):
            subset=selected[start:start+batch_size];indices=np.array([r['index'] for r in subset],dtype=int)
            with torch.inference_mode():
                output,diagnostics=model(M.to_device(arrays,indices,device),return_diagnostics=True)
            output=output.detach().cpu().numpy()
            details={k:v.detach().cpu().numpy() for k,v in diagnostics.items()}
            require(np.isfinite(output).all() and output.shape==(len(subset),9,2),'Malformed actual decoder output')
            for j,record in enumerate(subset):
                row=by_id[record['id']];index=record['index'];valid=np.array(row['baseline']['point_valid'],bool)
                original=np.asarray(arrays['points'][index])
                require(np.array_equal(output[j,8],original[8]),'Decoder changed centroid8')
                require(np.array_equal(output[j,~valid],original[~valid]),'Decoder changed missing/unavailable points')
                geometry_path=inputs.bind(record['candidate_evidence_npz'])
                with np.load(geometry_path,allow_pickle=False) as a:
                    geo={key.split('__',1)[1]:a[key].tolist() for key in a.files if key.startswith(mode+'__')}
                require(np.allclose(np.asarray(geo['candidates_xy'],np.float32),arrays['candidate_xy'][index],atol=0,rtol=0),'Visual candidate coordinates differ from actual model input')
                require(np.array_equal(geo['candidate_valid'],arrays['candidate_valid'][index]),'Candidate valid mask differs')
                row['arms'][mode]=dict(points=[p.tolist() if v else None for p,v in zip(output[j],valid)],point_valid=valid.tolist(),
                    decoder=dict(candidate_weights=details['attention'][j].tolist(),gate=details['signed_gate'][j,:,0].tolist(),
                        delta=details['delta'][j].tolist(),residual=details['residual'][j].tolist(),
                        weighted_candidate_xy=details['weighted_candidate_xy'][j].tolist(),
                        semantics='Actual normalized candidate selector weights and signed tanh gate; not general attention, causal attribution or calibrated correctness.'),
                    geometry=geo,donor_id=record['donor_id'] if mode=='wrong_image_line' else None,
                    actual_cached_head_forward=True,checkpoint_sha256=checkpoints[mode]['sha256'])
        del model,arrays
    for file in (Path(__file__),Path(C.__file__),Path(M.__file__)):inputs.bind(file)
    inputs.verify()
    payload=dict(schema='pallet_dht_decoder_predictions_v1',complete=True,PASS=True,records=list(by_id.values()),
        checkpoints=checkpoints,n_new_training_seeds=1,n_actual_cached_head_forwards=831*3,
        real_GT_used_in_model=False,no_new_full_detector_forward=True,device=str(device),batch_size=batch_size,
        source_sha256=inputs.hashes,scope='Final small-head inference on frozen image/point/line cache; original detector outputs retained as baseline.')
    write(run_dir/'PREDICTIONS.json',payload)
    return payload


def evaluate_records(predictions, inputs):
    """Attach GT only after all GT-free small-head predictions are saved."""
    pos=inputs.read(ROOT/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json')
    require(pos['expected_count']==319 and pos['role']=='DEV','Canonical reused DEV319 required')
    items={r['frame_id']:r for r in pos['items']}
    strata=inputs.read(ROOT/'data/pallet/results/pallet_dht_joint_v1/VIEW_STRATA.json')
    views={r['frame_id']:r for r in strata['records']}
    rows=[]
    for record in predictions['records']:
        if record['population']=='real_dev':
            item=items[record['id']]
            require(Path(record['image']).resolve()==(ROOT/item['image_path']).resolve(),'Real original-image join differs')
            annotation=inputs.read(ROOT/item['gt_v2_path'])['objects'][0]
            xy=np.asarray([p['xy'] for p in annotation['keypoint_annotations']],float)
            mask=np.asarray([p['visibility']>0 for p in annotation['keypoint_annotations']],bool)
            # Preserve the canonical all-eight-corner box, including amodal and
            # truncated locations; the paper evaluator does not clip this box.
            from challenge.evaluation_v2 import paper_real_eval as paper
            inputs.bind(Path(paper.__file__))
            target_box=np.r_[xy[:8].min(0),xy[:8].max(0)]
            candidate_box=record['baseline']['box_xyxy']
            matched=candidate_box is not None and paper._box_iou(np.asarray(candidate_box),target_box)>=.5
            view={k:views[record['id']][k] for k in ('elevation_bin','frontness_bin')}
        else:
            require(record['population']=='synth_val','Training population cannot enter reported held-out decoder metrics')
            xy=np.asarray(record['evaluation_target']['points'],float)
            mask=np.asarray(record['evaluation_target']['valid'],bool)
            matched=bool(record['evaluation_target']['matched'])
            view={}
        require(xy.shape==(9,2) and mask.shape==(9,) and np.isfinite(xy).all(),'Malformed supervised target')
        baseline,bvalid=finite_points(record['baseline']['points'],record['baseline']['point_valid'])
        arms={}
        for arm in ARMS:
            payload=record['baseline'] if arm=='baseline' else record['arms'][arm]
            point,valid=finite_points(payload['points'],payload['point_valid'])
            require(np.array_equal(valid,bvalid),'Decoder changed missing/detected point coverage')
            require(np.array_equal(point[8],baseline[8]),'Original centroid changed')
            observed=mask&valid&matched
            distance=np.linalg.norm(point-xy,axis=1)
            move=np.linalg.norm(point-baseline,axis=1)
            arms[arm]=dict(errors_px=[float(v) if ok else None for v,ok in zip(distance,observed)],
                move_px=[float(v) if ok else None for v,ok in zip(move,valid&bvalid)],points=payload['points'])
        rows.append(dict(id=record['id'],population=record['population'],session_id=record['session_id'],
            gt_points=xy.tolist(),gt_supervised=mask.tolist(),baseline_match_iou50=bool(matched),view=view,arms=arms))
    real=[r for r in rows if r['population']=='real_dev']
    require(len(real)==319 and {r['id'] for r in real}==set(items),'All real319 including missing observations must remain')
    require(len({r['session_id'] for r in real})==13,'Expected13 original capture sessions')
    require(len([r for r in rows if r['population']=='synth_val'])==512,'Decoder synthetic validation denominator differs')
    summaries=[];paired_rows=[];difficult=[];corner_summaries=[]
    for population in ('real_dev','synth_val'):
        selected=[r for r in rows if r['population']==population]
        for arm in ARMS:
            summaries.append(dict(population=population,arm=arm,**summarize(selected,arm)))
            corner_summaries.append(dict(population=population,arm=arm,**summarize(selected,arm,tuple(range(8)))))
            difficult.extend(dict(population=population,**r) for r in point_difficulty(selected,arm))
        if population=='real_dev':
            for left,right in [('point_only','baseline'),('line_fusion','baseline'),('line_fusion','point_only'),('wrong_image_line','line_fusion')]:
                paired_rows.append(dict(population=population,**paired(selected,left,right)))
    view_rows=[]
    for axis in ('elevation_bin','frontness_bin'):
        for group in sorted({r['view'][axis] for r in real}):
            selected=[r for r in real if r['view'][axis]==group]
            for arm in ARMS:
                view_rows.append(dict(axis=axis,group=group,arm=arm,**summarize(selected,arm)))
    criterion=continuation([r for r in summaries if r['population']=='real_dev'],paired_rows,[r for r in difficult if r['population']=='real_dev'])
    result=dict(schema='pallet_dht_decoder_results_v1',complete=True,PASS=True,
        n_new_training_seeds=1,n_real_frames=319,n_real_sessions=13,n_synthetic_validation_frames=512,
        summaries=summaries,corner_only_summaries=corner_summaries,paired=paired_rows,difficulty=difficult,views=view_rows,
        continuation=criterion,headline_ko=criterion['headline_ko'],metric_scope='Original semantic IDs. Supervised nine points on unchanged highest-score baseline IoU>=.5 matches; missing/unmatched remain in coverage denominator.',
        six_d_evaluated=False,negative_evaluated=False,independent_final=False,
        limitations=['Single new decoder-training seed; the diagnostic14 existing three backbone seeds are not three newly trained decoders.',
            'Reused DEV319/13 sessions; session bootstrap is an exploratory conditional approximation.',
            'Wrong-image line is a distribution perturbation; degradation is not a causal proof.',
            'Ground-truth difficulty and oracle diagnostics never select operational candidates.',
            'Centroid8 is copied from the original network and separately included in nine-point reporting.'])
    return result,dict(complete=True,records=rows)


def evaluate(run_dir, device='cpu', phase='all', batch_size=64):
    run_dir=Path(run_dir).resolve();inputs=Inputs()
    require((run_dir/'PURPOSE.md').is_file(),'Purpose-declared new output root required')
    if phase in ('infer','all') and not (run_dir/'PREDICTIONS.json').is_file():
        predictions=infer_cached(run_dir,inputs,device,batch_size)
    else:
        predictions=inputs.read(run_dir/'PREDICTIONS.json')
        require(predictions['complete'] and predictions['PASS'],'Actual cached decoder predictions incomplete')
        for path,digest in predictions['source_sha256'].items():inputs.bind(path,digest)
    if phase=='infer':return
    inputs.bind(run_dir/'PREDICTIONS.json')
    result,frames=evaluate_records(predictions,inputs)
    inputs.verify()
    real={r['arm']:r for r in result['summaries'] if r['population']=='real_dev'}
    display=lambda value:'missing' if value is None else f'{value:.3f}'
    result['discord_lines_ko']=[result['headline_ko'],
        f"재사용 DEV319·새 학습1seed: median baseline {display(real['baseline']['median_px'])} / point {display(real['point_only']['median_px'])} / line {display(real['line_fusion']['median_px'])}px.",
        f"P90 baseline {display(real['baseline']['p90_px'])} / point {display(real['point_only']['p90_px'])} / line {display(real['line_fusion']['p90_px'])}px. 독립 final·6D 미평가."]
    write(run_dir/'RESULTS.json',result);write(run_dir/'FRAME_METRICS.json',frames)
    write(run_dir/'EVALUATION_COMPLETION.json',dict(complete=True,PASS=True,n_real_frames=319,n_new_training_seeds=1,
        input_sha256=inputs.hashes,output_sha256={name:sha(run_dir/name) for name in ('RESULTS.json','FRAME_METRICS.json','PREDICTIONS.json')},
        source_sha256={str(Path(__file__).resolve()):sha(__file__)},six_d_evaluated=False,independent_final=False))
    print(json.dumps({k:result[k] for k in ('complete','PASS','headline_ko')},ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--phase',choices=('infer','metrics','all'),default='all')
    parser.add_argument('--batch-size',type=int,default=64)
    args=parser.parse_args();evaluate(args.run_dir,args.device,args.phase,args.batch_size)

"""Independent CPU arithmetic/provenance audit of completed v2 saved outputs.

Never trains, predicts, selects a model, or changes an accuracy artifact. The
older independent audit's CSV/EMA helpers are reused; aggregate/statistic
functions are not called. Runtime strict failures remain failures.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from scripts.research.pallet_dht_joint_v1 import audit_outputs as D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEW = ('balanced', 'pcgrad', 'balanced_pcgrad', 'incidence')
REFERENCE = ('point_only', 'hough_features', 'hough_joint')
ARMS = REFERENCE + NEW
SEEDS = (1, 2, 3)
DIAGNOSTIC_STEPS = (2, 16, 256, 1750, 3499, 3500, 5249, 6998)
METRICS, KEYPOINT, POSE_FIELDS = D.METRICS, D.KEYPOINT, D.POSE_FIELDS
require, close, sha = D.require, D.close, D.sha


def direct_session_bootstrap(stores, metric):
    """100k independently repeated-session medians with the frozen v2 stream."""
    is_point = metric in KEYPOINT
    keys = sorted(set.intersection(*(set(s) for s in stores)))
    if is_point:
        keys = [k for k in keys if all(len(s[k]['errors']) for s in stores)]
    if not keys:
        return dict(difference=None,paired_frames=0,paired_sessions=0,resamples=0,
                    session_cluster=dict(low=None,high=None),
                    session_simultaneous=dict(low=None,high=None))
    session = lambda row: row['session'] if is_point else row['session_id']
    sessions = sorted({session(stores[0][k]) for k in keys})
    require(all(len({session(s[k]) for s in stores}) == 1 for k in keys), 'Paired session mismatch')
    series = []
    for store in stores:
        values, groups = [], []
        for k in keys:
            v = store[k]['errors'] if is_point else [store[k][POSE_FIELDS[metric]]]
            values.extend(v); groups.extend([sessions.index(session(store[k]))] * len(v))
        series.append((np.asarray(values), np.asarray(groups, dtype=int)))
    observed = np.mean([np.median(v) for v, _ in series[:3]]) - np.mean([np.median(v) for v, _ in series[3:]])
    # V2 deliberately uses a separate session stream; do not consume frame draws.
    rng = np.random.default_rng(20260903 if is_point else 20260904)
    draws = np.empty(100000)
    for start in range(0, len(draws), 128):
        counts = rng.multinomial(len(sessions), np.full(len(sessions), 1 / len(sessions)),
                                size=min(128, len(draws)-start))
        for i, weights in enumerate(counts):
            medians = [np.median(np.repeat(v, weights[g])) for v, g in series]
            draws[start+i] = np.mean(medians[:3]) - np.mean(medians[3:])
    result = dict(difference=float(observed), paired_frames=len(keys), paired_sessions=len(sessions), resamples=len(draws))
    for name, alpha in [('session_cluster', .05), ('session_simultaneous', .05/48)]:
        low, high = np.quantile(draws, [alpha/2, 1-alpha/2])
        result[name] = dict(low=float(low), high=float(high))
    return result


def runtime_audit(inputs, root, old):
    value = inputs.marker(root/'RUNTIME.json', allow_recorded_runtime_failure=True)
    require(value['timing_collection_complete'] is True and value['n_models'] == 21
            and value['n_timing_observations'] == 1638, 'Incomplete 21-model timing')
    require(value['reference_models_retimed'] == 9 and value['new_models_timed'] == 12,
            'Timing reuse/new counts differ')
    require(value['parity_policy'] == dict(atol=1e-4, rtol=0, criterion_changed=False), 'Runtime tolerance changed')
    require(len(value['runs']) == 21 and {(r['arm'], r['seed']) for r in value['runs']}
            == {(a,s) for a in ARMS for s in SEEDS}, 'Runtime model identities differ')
    frames = value['frames']
    require(len(frames) == len(set(frames)) == 26 and value['repeats'] == 3, 'Timing frame denominator differs')
    expected = {(f,i) for f in frames for i in range(3)}
    fields = ('score','box_xyxy','keypoints_xy','keypoints_conf')
    failed, maxima = set(), dict.fromkeys(fields, 0.)
    for row in value['runs']:
        obs = row['observations']
        require(row['reference_model'] is (row['arm'] in REFERENCE), 'Retimed model provenance differs')
        require(row['n'] == len(obs) == 78 and {(o['frame_id'],o['repeat']) for o in obs} == expected, 'Timing observations differ')
        times = [o['milliseconds'] for o in obs]
        require(all(math.isfinite(v) and v > 0 for v in times), 'Invalid timing')
        for actual,key in [(np.mean(times),'mean_ms'),(np.median(times),'median_ms'),(np.percentile(times,90),'p90_ms')]:
            close(actual,row[key], 'timing/'+key)
        row_max = 0.
        for o in obs:
            delta = o['max_abs_delta_by_field']
            require(set(delta) == set(fields) and all(math.isfinite(v) and v >= 0 for v in delta.values()), 'Invalid parity deltas')
            passing = all(v <= 1e-4 for v in delta.values())
            require(o['strict_parity_pass'] is passing, 'Timing numeric failure relabeled')
            if not passing: failed.add((row['arm'],row['seed'],o['frame_id'],o['repeat']))
            for k,v in delta.items(): maxima[k] = max(maxima[k],v); row_max = max(row_max,v)
        close(row_max,row['max_coordinate_or_score_delta'],'timing maximum')
    recorded = value['strict_failures']
    require(len(recorded) == len(failed) and {(x['arm'],x['seed'],x['frame_id'],x['repeat']) for x in recorded} == failed,
            'Missing or duplicated strict timing failures')
    references = {}
    for failure in recorded:
        label = f"{failure['arm']}_seed{failure['seed']}"
        if label not in references:
            owner = old if failure['arm'] in REFERENCE else root
            references[label] = inputs.read(owner/'evaluation'/label/'PREDICTIONS.json')['frames']
        reference = references[label][failure['key']]
        require(reference == failure['reference_candidates'], 'Runtime stored reference differs from accuracy cache')
        actual = failure['actual_candidates']; require(len(actual) == len(reference), 'Runtime candidate count changed')
        delta = dict.fromkeys(fields,0.)
        for a,b in zip(actual,reference):
            for key in fields:
                if a[key] is None or b[key] is None:
                    require(a[key] is b[key], 'Runtime availability changed'); continue
                x,y = np.asarray(a[key]),np.asarray(b[key])
                require(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), 'Runtime finite shape differs')
                delta[key] = max(delta[key],float(np.max(np.abs(x-y))))
        require(any(v > 1e-4 for v in delta.values()), 'Recorded runtime failure does not reproduce')
        for k,v in delta.items(): close(v,failure['max_abs_delta_by_field'][k], 'runtime saved failure/'+k)
    passing = not failed
    require(value['PASS'] is passing and value['parity_PASS'] is passing, 'Runtime failure converted to PASS')
    require(value['status'] == ('COMPLETE' if passing else 'COMPLETE_WITH_STRICT_PARITY_FAILURE'), 'Runtime status hides failure')
    return dict(observed_timing_count=1638, strict_parity_PASS=passing, strict_failure_count=len(failed),
        criterion_changed=False, original_atol=1e-4, original_rtol=0, maximum_abs_delta_by_field=maxima,
        meaning='Timing arithmetic/failure records audited; strict numeric failures remain failures.')


def gradient_audit(inputs, path, arm, coefficient):
    data = inputs.marker(path)
    require(data['arm'] == arm, 'Gradient audit arm mismatch')
    surgery = arm in ('pcgrad','balanced_pcgrad')
    require(data['surgery_enabled'] is surgery, 'Unexpected surgery arm')
    rows = data['surgery_steps']; diagnostics = data['actual_task_diagnostics']
    expected = list(range(1,6999)) if surgery else list(DIAGNOSTIC_STEPS)
    require([r['optimizer_step'] for r in rows] == expected, 'Actual gradient capture/update count differs')
    require([r['optimizer_step'] for r in diagnostics] == list(DIAGNOSTIC_STEPS), 'Fixed diagnostic steps differ')
    masks = data['shared_masks']
    for digest,names in masks.items():
        require(len(names) == len(set(names)) and hashlib.sha256(('\n'.join(names)+'\n').encode()).hexdigest() == digest, 'Shared-mask binding differs')
        require(all(not n.startswith('model.23.') or '.hough.' in n for n in names), 'Task-private point head entered surgery mask')
        require(all('.hough.outputs.' not in n and '.hough.spatial_mix.' not in n for n in names), 'Aux-absent feedback projection entered surgery mask')
    def check_stats(s):
        dot,a,b = s['dot'],s['norm_a'],s['norm_b']
        require(all(math.isfinite(v) for v in (dot,a,b)) and a >= 0 and b >= 0, 'Invalid gradient norm/dot')
        close(s['cosine'],dot/(a*b) if a*b>0 else None,'gradient cosine',2e-6)
        require(s['conflict'] is (dot<0 and a>0 and b>0), 'Gradient conflict flag differs')
        if s['cosine'] is not None: require(abs(s['cosine']) <= 1+2e-6,'Gradient cosine outside unit interval')
    for row in rows:
        check_stats(row)
        require(row['applied'] is (surgery and row['conflict']), 'Surgery application flag differs')
        require(row['shared_tensors'] == len(masks[row['shared_mask_sha256']]), 'Shared tensor count differs')
        epoch = 1 if row['optimizer_step'] <= 3499 else 2
        require(row['epoch'] == epoch,'Gradient epoch differs')
        o2m = .8 if epoch == 1 else .1
        close(row['one2many'],o2m,'stock one2many weight')
        close(row['one2one'],1-o2m,'stock one2one weight')
        close(row['line_coefficient'],.1*o2m/.8 if arm in ('balanced','balanced_pcgrad') else .1,'line coefficient')
        close(row['incidence_coefficient'],coefficient if arm == 'incidence' else 0.,'incidence coefficient')
    for row in diagnostics:
        for family in ('full_stock_vs_weighted_line','pose_location_RLE_vs_weighted_line'):
            for stats in row[family].values(): check_stats(stats)
    return dict(actual_diagnostic_steps=len(diagnostics), actual_surgery_or_capture_rows=len(rows),
        conflict_rows=sum(r['conflict'] for r in rows), applied_rows=sum(r['applied'] for r in rows),
        shared_mask_sha256=list(masks), raw_gradient_vectors_recomputed=False,
        scope='Stored actual training statistics, masks, schedules and arithmetic; no extra backward replay. Private preservation refers to before global clipping.')


def audit(run_dir):
    root = Path(run_dir).resolve(); inputs = D.Inputs()
    protocol = inputs.marker(root/'TRAIN_PROTOCOL.json')
    require(protocol['arms'] == list(NEW) and protocol['training']['seeds'] == list(SEEDS), 'Frozen arms/seeds differ')
    require(protocol['training']['epochs'] == 2 and protocol['training']['batch'] == 16, 'Training budget differs')
    for name in protocol['preflight_required']:
        preflight = inputs.marker(root/name)
        if name == 'GRADIENT_DIAGNOSIS.json':
            require(len(preflight['source_results']) == 3,'Three actual gradient measurements required')
            for path,digest in preflight['source_results'].items():
                inputs.bind(path,digest); inputs.marker(path)
    controls = inputs.marker(root/'REUSED_CONTROLS.json'); old = Path(controls['control_run_dir'])
    require(controls['n_control_cells'] == len(controls['controls']) == 9,'Reference count differs')
    for control in controls['controls']:
        require(control['reused'] is True and control['new_v2_accuracy_forwards'] == 0,'Reference reuse mislabeled')
        for artifact in [control['checkpoint'],*control['artifacts'].values()]: inputs.bind(artifact['path'],artifact['sha256'])
    summary = inputs.marker(root/'SUMMARY.json'); verdict = inputs.marker(root/'VERDICT.json')
    inputs.marker(root/'AGGREGATE_COMPLETE.json'); inputs.marker(root/'MATCHED_CONTROL_AUDIT.json')
    new_training = inputs.marker(root/'TRAINING_AUDIT.json'); old_training = inputs.marker(old/'TRAINING_AUDIT.json')
    training_rows = {(r['arm'],r['seed']):r for a in (new_training,old_training) for r in a['runs']}
    require(len(training_rows) == 21,'Actual training audit cell count differs')
    require(summary['n_completed_evaluations'] == 12 and summary['n_verified_reference_evaluations'] == 9,'New/reference evaluation counts differ')
    require(verdict['selected_winning_arm'] is None and verdict['real_model_selection_performed'] is False,'Real-data model selected')
    runtime = runtime_audit(inputs,root,old)
    render = inputs.marker(root/'REPORT_RENDER.json')
    text = inputs.bind(root/'index.html',render['html_sha256']).read_text()
    embedded = re.search(r'<script id="report-data" type="application/json">(.*?)</script>',text,flags=re.S)
    require(embedded is not None,'HTML report data missing')
    gallery = json.loads(embedded.group(1)); frames = {x['id']:x for x in gallery['frames']}
    labels = {f'{a}_seed{s}' for a in ARMS for s in SEEDS}
    require(len(frames) == len(gallery['frames']) == 319 and all(set(f['runs']) == labels for f in frames.values()),'HTML 319×21 denominator differs')
    require(gallery['edges'] == [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]],'HTML cuboid topology changed')
    summary_rows = {(r['arm'],r['seed']):r for r in summary['runs']}
    require(len(summary_rows) == 21,'SUMMARY 21-run denominator differs')
    results, checks, traces, budgets, ema_values, point_stores, pose_stores = {},[],{},[],[],{},{}
    for arm in ARMS:
        point_stores[arm],pose_stores[arm] = [],[]
        owner = old if arm in REFERENCE else root
        for seed in SEEDS:
            label = f'{arm}_seed{seed}'; folder = owner/'evaluation'/label; cell = owner/'runs'/label
            print('Independent saved-output audit:',label,flush=True)
            result = inputs.marker(folder/'RESULTS.json'); done = inputs.marker(folder/'COMPLETION.json')
            trained = inputs.marker(cell/'COMPLETION.json'); config = inputs.read(cell/'CELL_CONFIG.json')
            require(trained['stage']=='main' and trained['epochs_completed']==2 and trained['optimizer_steps']==6998
                    and trained['train_frames']==55980 and trained['val_frames']==4020,'Actual full training budget differs')
            require(all(x['arm']==arm and x['seed']==seed for x in (result,done,trained)), 'Cell identity differs')
            require(summary_rows[arm,seed]['reused'] is (arm in REFERENCE),'SUMMARY reuse label differs')
            checkpoint = inputs.bind(trained['checkpoint'],trained['checkpoint_sha256'])
            require(done['checkpoint_sha256']==result['checkpoint_sha256']==trained['checkpoint_sha256'],'Training/prediction checkpoint differs')
            ema = D.ema_hash(checkpoint)
            require(ema==training_rows[arm,seed]['ema_tensor_sha256'],'Actual EMA tensor SHA differs')
            ema_values.append(ema)
            trace = inputs.bind(cell/'BATCH_TRACE.jsonl',trained['batch_trace_sha256'])
            traces.setdefault(seed,[]).append(sha(trace))
            require(sha(trace)==controls['original_trace_sha256_by_seed'][str(seed)],'Actual augmented batch trace differs from control')
            budgets.append({k:config[k] for k in ('epochs','batch','lr','optimizer','amp','train_frames','val_frames','training_recipe')})
            for receipt in (result,done):
                require(receipt['actual_positive_forwards']==319 and receipt['actual_negative_forwards']==2689
                        and receipt['baseline_candidate_copying'] is False,'Saved full-model inference receipt differs')
            payload = inputs.read(folder/'PREDICTIONS.json')
            require(payload['complete'] is True and payload['weights_sha256']==trained['checkpoint_sha256'],'Prediction checkpoint binding differs')
            meta,pred,observations = payload['frame_metadata'],payload['frames'],payload['observations']
            require(len(meta)==len(pred)==len(observations)==3008 and set(meta)==set(pred)=={o['image'] for o in observations},'Forward observation denominator differs')
            require(Counter(o['kind'] for o in observations)==Counter(positive=319,negative=2689),'Positive/negative forwards differ')
            negatives=[]
            for o in observations:
                info = meta[o['image']]; candidates = pred[o['image']]
                require(o['actual_forward'] is True and o['candidate_count']==len(candidates)
                        and o['kind']==info['kind'] and o['frame_id']==info['frame_id'],'Observation/prediction mismatch')
                inputs.bind(info['image_path'],info['image_sha256'])
                if info['kind']=='negative': negatives.append(candidates); continue
                top = max(candidates,key=lambda x:x['score']) if candidates else {}
                shown = frames[info['frame_id']]['runs'][label]
                require(shown['points']==top.get('keypoints_xy') and shown['box']==top.get('box_xyxy')
                        and shown['score']==top.get('score') and shown['n_candidates']==len(candidates),'HTML source coordinates differ')
            neg = inputs.marker(folder/'NEGATIVE_OUTCOMES.json')
            require(neg['n_frames']==neg['actual_forward_count']==len(negatives)==2689,'Negative denominator differs')
            require(neg['candidate_count']==sum(map(len,negatives)),'Negative candidate count differs')
            for threshold in D.THRESHOLDS:
                n = sum(any(c['score']>=threshold for c in cs) for cs in negatives)
                k = sum(c['score']>=threshold for cs in negatives for c in cs)
                row = neg['by_threshold'][str(threshold)]
                require(row['n_frames_with_detection']==n and row['n_candidates']==k,'Negative threshold counts differ')
                close(n/2689,row['fraction_frames_with_detection'],'negative fraction')
                require(result['negative']['by_threshold'][str(threshold)]==row,'RESULTS negative counts differ')
            points,poses,computed,matched,n_points = D.load_rows(inputs,folder,label)
            require(matched==result['two_d']['keypoint_matched_frame_count_iou50'],'2D matching denominator differs')
            require(len(poses)==result['main_6d']['n'],'Pose denominator differs')
            close(len(poses)/319,result['main_6d_coverage'],'Pose coverage')
            delta = {}
            for metric in METRICS:
                section = 'two_d' if metric in KEYPOINT else 'main_6d'
                delta[metric]=close(computed[metric],result[section][metric],label+'/'+metric,5.1e-7 if metric in KEYPOINT else 1e-9)
                close(summary_rows[arm,seed][section][metric],result[section][metric],'SUMMARY per-seed/'+metric)
            grad = gradient_audit(inputs,cell/'GRADIENT_AUDIT.json',arm,protocol['training']['incidence_weight']) if arm in NEW else None
            point_stores[arm].append(points);pose_stores[arm].append(poses);results[arm,seed]=result
            checks.append(dict(arm=arm,seed=seed,reused=arm in REFERENCE,actual_updates=6998,
                new_v2_accuracy_forwards=0 if arm in REFERENCE else 3008,metric_absolute_differences=delta,
                observed_point_errors=n_points,matched_frames=matched,pose_frames=len(poses),gradient_audit=grad,EMA_tensor_sha256=ema))
    # An intervention can legitimately make no updates different from its
    # control (e.g. no gradient conflict). Report equality rather than inventing
    # a scientific requirement that all architecture arms must differ.
    require(all(len(set(v))==1 for v in traces.values()) and len({v[0] for v in traces.values()})==3,'Matched seed/data traces differ')
    require(all(b==budgets[0] for b in budgets),'Training recipes/budgets differ')
    families={}
    for arm in ARMS:
        families[arm]={}
        for metric in METRICS:
            section='two_d' if metric in KEYPOINT else 'main_6d'; values=[results[arm,s][section][metric] for s in SEEDS]
            mean=sum(values)/3 if all(v is not None for v in values) else None
            sd=math.sqrt(sum((v-mean)**2 for v in values)/2) if mean is not None else None
            expected=summary['arms'][arm]['metrics'][metric]
            require(expected['seeds']==values,'Seed order/value differs')
            close(mean,expected['mean'],'Seed mean');close(sd,expected['std'],'Sample SD ddof1')
            families[arm][metric]=dict(mean=mean,sample_std_ddof1=sd)
    representative={}
    for arm,metric in [('incidence','keypoint_location_median_px'),('balanced_pcgrad','translation_median_cm')]:
        stores=point_stores if metric in KEYPOINT else pose_stores
        print('Independent direct 100k session bootstrap:',arm,metric,flush=True)
        actual=direct_session_bootstrap(stores[arm]+stores['hough_joint'],metric)
        expected=summary['comparisons'][arm+'_vs_hough_joint']['metrics'][metric]
        require(actual['paired_frames']==expected['paired_frames'] and actual['paired_sessions']==expected['paired_sessions'],'Representative CI denominator differs')
        close(actual['difference'],expected['difference'],'Representative difference')
        for interval in ('session_cluster','session_simultaneous'):
            for key in ('low','high'):close(actual[interval][key],expected[interval][key],'Independent 100k/'+interval+'/'+key)
        representative[arm+'/'+metric]=actual
    decisions={}
    for arm in NEW:
        decisions[arm]={}
        for reference in ('hough_joint','point_only'):
            criteria={}
            for metric in METRICS:
                a,b=families[arm][metric]['mean'],families[reference][metric]['mean']
                lower=metric in KEYPOINT or metric in ('rotation_median_deg','translation_median_cm')
                change=a is not None and b is not None and (a<b if lower else a>b)
                ci=summary['comparisons'][arm+'_vs_'+reference]['metrics'][metric]['session_simultaneous']
                if ci['low'] is not None or ci['high'] is not None:
                    require(ci['resamples']==100000 and ci['family_size']==48 and ci['adjusted_alpha']==.05/48,'Multiplicity protocol differs')
                benefit=ci['low'] is not None and ci['high'] is not None and (ci['high']<0 if lower else ci['low']>0)
                criteria[metric]=bool(change and benefit)
                require(verdict['candidates'][arm]['references'][reference]['metric_criteria'][metric]['confirmed'] is criteria[metric],'Metric decision differs')
            kp=all(results[arm,s]['two_d']['keypoint_matched_frame_count_iou50']>=results[reference,s]['two_d']['keypoint_matched_frame_count_iou50'] for s in SEEDS)
            pose=all(results[arm,s]['main_6d_coverage']>=results[reference,s]['main_6d_coverage'] for s in SEEDS)
            point_gain=kp and all(criteria[m] for m in KEYPOINT);pose_gain=pose and all(criteria[m] for m in POSE_FIELDS)
            out=verdict['candidates'][arm]['references'][reference]
            require(out['keypoint_gain_confirmed'] is point_gain and out['pose_gain_confirmed'] is pose_gain
                    and out['overall_accuracy_improved'] is (point_gain and pose_gain),'Coverage/overall decision differs')
            decisions[arm][reference]=bool(point_gain and pose_gain)
        require(verdict['candidates'][arm]['meets_both_reference_criteria'] is all(decisions[arm].values()),'Both-reference decision differs')
    overall=any(all(v.values()) for v in decisions.values())
    require(verdict['overall_accuracy_improved'] is overall,'Overall verdict differs')
    require(all(sha(p)==digest for p,digest in inputs.hashes.items()),'Inputs changed during audit')
    result=dict(schema='pallet_dht_coupling_independent_final_audit_v2',complete=True,PASS=True,
        PASS_semantics='Saved-output arithmetic/provenance integrity, separate from accuracy improvement and runtime parity.',
        model_forwards=0,device='cpu',new_cells=12,reused_reference_cells=9,cells=checks,
        recomputed_seed_statistics=families,representative_independent_session_bootstrap=representative,
        independently_checked_decisions=decisions,overall_accuracy_improved=overall,runtime_audit=runtime,
        total_actual_new_positive_forwards=12*319,total_actual_new_negative_forwards=12*2689,
        reused_reference_new_accuracy_forwards=0,HTML_saved_prediction_rows_exact=21*319,
        distinct_actual_EMA_states=len(set(ema_values)),same_seed_augmented_trace_exact=True,
        limitations=['CPU tests/preflight and gradient-statistic arithmetic are not accuracy success.',
            'Canonical solver/GT/AP are not rerun; six metrics and four negative FP thresholds are independently recomputed.',
            'Only two predeclared representative 100k session CIs are independently recomputed; other intervals remain bound aggregate outputs.',
            'Stored training gradients are checked for masks/schedules/arithmetic, not replayed with a new forward/backward.',
            'HTML numeric coordinates are checked; visual/browser QA is separate.',
            'Reused DEV and reconstructed 6D GT limitations remain unchanged.'],
        source_sha256={str(Path(__file__).resolve()):sha(__file__),str(Path(D.__file__).resolve()):sha(D.__file__)},
        input_sha256=inputs.hashes,output_sha256={})
    target=root/'INDEPENDENT_FINAL_AUDIT.json';temporary=target.with_suffix('.pending.json')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(target)
    print(json.dumps(dict(PASS=True,output=str(target),sha256=sha(target))))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    audit(parser.parse_args().run_dir)

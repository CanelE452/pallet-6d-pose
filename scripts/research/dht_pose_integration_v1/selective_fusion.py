"""Select small prediction-only corner gates on synthetic validation only.

Frozen live baseline/DHT outputs are reused. Lambda1 proposals are computed once;
a boolean gate either accepts that proposal or copies the baseline corner.
No neural model is trained and no real/test error selects a rule or threshold.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np

import evaluate as E
import fusion as F


MODELS = ('DOPE_exact_resize', 'YOLO')
FAMILIES = ('confidence_only', 'confidence_line_move')
ARMS = ('baseline', 'unconditional_lambda1', *FAMILIES)
QUANTILES = (.25, .5, .75, 1.)
ADJUSTED_EPS_PX = 1e-9


def sources(root):
    return E.observed_sources(root) | {str(Path(__file__).resolve()): E.sha(Path(__file__))}


def predicted_signals(points, valid, confidence, lines, peak_probability, width, height):
    """Image dimensions and predictions only; no record, GT, or support input."""
    points, valid = np.asarray(points, float), np.asarray(valid, bool)
    lines, peaks = np.asarray(lines, float), np.asarray(peak_probability, float)
    confidence = np.asarray(confidence, float)[:8]
    proposal, effective_valid = F.fuse_corners(points, valid, lines, 1., width, height)
    if not np.array_equal(effective_valid, valid):
        raise ValueError('Unexpected nonfinite baseline validity in frozen input')
    delta = lines[:, 1] - lines[:, 0]
    lengths = np.linalg.norm(delta, axis=-1)
    good = np.isfinite(lines).all((1, 2)) & np.isfinite(peaks) & (lengths > 1e-12)
    quality, sine = np.full(8, np.nan), np.full(8, np.nan)
    for corner, roles in enumerate(F.INCIDENT_ROLES):
        if len(roles) != 2:
            raise ValueError('This preregistered gate expects exactly two incident side lines per corner')
        a, b = roles
        if good[a] and good[b]:
            quality[corner] = min(peaks[a], peaks[b])
            sine[corner] = abs(np.linalg.det(np.stack([delta[a]/lengths[a], delta[b]/lengths[b]])))
    move = np.linalg.norm(proposal[:8]-points[:8], axis=-1) / np.hypot(width, height)
    return proposal, dict(point_confidence=confidence, line_quality=quality,
                          proposed_move_normalized=move, sin_incident_angle=sine)


def prepare_predictions(cfg, manifest, variants, dht):
    result = {}
    for model in MODELS:
        baselines = variants[model]
        bundle = dict(points=np.asarray([b['kps'] for b in baselines], float),
                      valid=np.asarray([b['kp_valid'] for b in baselines], bool),
                      confidence=np.asarray([b['kp_conf'][:8] for b in baselines], float),
                      proposals=[], signals={name: [] for name in
                          ('point_confidence', 'line_quality', 'proposed_move_normalized', 'sin_incident_angle')})
        for seed in cfg['dht']['seeds']:
            proposals, signal_rows = [], []
            for record, base, prediction in zip(manifest['records'], baselines, dht):
                current = prediction['seeds'][str(seed)]
                proposal, signal = predicted_signals(base['kps'], base['kp_valid'], base['kp_conf'],
                    current['lines'], current['peak_probability'], record['width'], record['height'])
                proposals.append(proposal)
                signal_rows.append(signal)
            bundle['proposals'].append(proposals)
            for key in bundle['signals']:
                bundle['signals'][key].append([s[key] for s in signal_rows])
        bundle['proposals'] = np.asarray(bundle['proposals'], float)
        bundle['signals'] = {key: np.asarray(value, float) for key, value in bundle['signals'].items()}
        result[model] = bundle
    return result


def finite_quantiles(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError('Synthetic validation has no finite predicted signal for registered thresholds')
    return np.quantile(values, QUANTILES).tolist()


def register(root, destination, cfg, manifest, prepared):
    indices = manifest['populations']['synth_val']
    if len(indices) != 256 or any(manifest['records'][i]['population'] != 'synth_val' for i in indices):
        raise ValueError('Only the frozen256 synthetic validation records may define thresholds')
    config = dict(schema='dht_selective_gate_config_v1', registered_before_gate_accuracy_analysis=True,
                  config_sha256=E.sha(root/'CONFIG.json'), manifest_sha256=E.sha(root/'manifest.json'),
                  input_sha256={name: E.sha(root/name) for name in
                                ('BASELINE_DOPE.json', 'BASELINE_YOLO.json', 'DHT_LINES.json')},
                  models=list(MODELS), seeds=cfg['dht']['seeds'], proposal_lambda=1.,
                  selection_population='synth_val', selection_frames=256, quantiles=list(QUANTILES),
                  threshold_source='Prediction-only signals on baseline-valid top8corners of synth_val; point confidence once per corner, line/move signals pooled over allthree DHT seeds.',
                  objective='Mean per-frame capped diagonal-normalized top8corner error, missing1; average allthree DHT seeds equally.',
                  tie_break=['smaller validation score', 'fewer actually adjusted corner-seed observations', 'lexicographic rule ID; baseline fallback ID00'],
                  adjusted_epsilon_px=ADJUSTED_EPS_PX, candidates_per_model=70,
                  families={'confidence_only': 'Baseline fallback plus4 confidence-upper-bound rules.',
                            'confidence_line_move': 'Baseline fallback plus64 confidence upper / incident-peak lower / proposed-move upper rules.'},
                  line_quality='Minimum of the two incident role-wise Hough softmax peak values; uncalibrated relative confidence proxy.',
                  quantile_interpretation='Line-qualityQ25 excludes the lower quarter of validation predictions; Q100 admits only values at/above the validation maximum. Neither is a probability of geometric correctness.',
                  preserve='Missing corners and centroid remain unchanged. Gatefalse copies the baseline exactly. Both models retain a baseline fallback.',
                  calibration_limit='DOPE belief amplitudes, YOLO keypoint confidence and Hough peak values are uncalibrated geometric-error proxies.',
                  real_scope='Existing already-inspected canonical realDEV52 only; no new held-out generalization claim.',
                  thresholds={}, rules={})
    for model in MODELS:
        data = prepared[model]
        valid = data['valid'][indices, :8]
        c = finite_quantiles(data['confidence'][indices][valid])
        broadcast_valid = np.broadcast_to(valid, (3, *valid.shape))
        quality = finite_quantiles(data['signals']['line_quality'][:, indices][broadcast_valid])
        move = finite_quantiles(data['signals']['proposed_move_normalized'][:, indices][broadcast_valid])
        config['thresholds'][model] = dict(point_confidence_upper=c, line_quality_lower=quality, move_normalized_upper=move)
        fallback = dict(id='00_baseline_fallback', kind='baseline', proposal_lambda=1., gate_enabled=False)
        confidence_rules = [fallback] + [dict(id=f'confidence_q{int(q*100):03d}', kind='confidence_only',
                                              confidence_upper=c[i], confidence_quantile=q) for i,q in enumerate(QUANTILES)]
        joint_rules = [fallback] + [dict(id=f'joint_c{int(QUANTILES[a]*100):03d}_q{int(QUANTILES[b]*100):03d}_m{int(QUANTILES[cidx]*100):03d}',
                       kind='confidence_line_move', confidence_upper=c[a], line_quality_lower=quality[b],
                       move_normalized_upper=move[cidx], confidence_quantile=QUANTILES[a],
                       line_quality_quantile=QUANTILES[b], move_quantile=QUANTILES[cidx])
                       for a,b,cidx in product(range(4), repeat=3)]
        config['rules'][model] = dict(confidence_only=confidence_rules, confidence_line_move=joint_rules)
        if len(confidence_rules)+len(joint_rules) != 70:
            raise ValueError('Registered candidate budget mismatch')
    path = destination/'CONFIG_GATE.json'
    if path.exists() and E.read(path) != config:
        raise ValueError('Frozen gate configuration changed; do not tune after evaluating real data')
    E.write(path, config)
    if E.read(path) != config:
        raise ValueError('Gate configuration read-back mismatch')
    return config


def gate_mask(rule, data, seed_index, indices):
    valid = data['valid'][indices]
    gate = np.zeros_like(valid)
    if rule['kind'] == 'baseline':
        return gate
    passed = valid[:, :8].copy()
    if rule['kind'] != 'unconditional':
        confidence = data['confidence'][indices]
        passed &= np.isfinite(confidence) & (confidence <= rule['confidence_upper'])
    if rule['kind'] == 'confidence_line_move':
        quality = data['signals']['line_quality'][seed_index, indices]
        move = data['signals']['proposed_move_normalized'][seed_index, indices]
        passed &= np.isfinite(quality) & np.isfinite(move)
        passed &= (quality >= rule['line_quality_lower']) & (move <= rule['move_normalized_upper'])
    gate[:, :8] = passed
    return gate


def apply_rule(rule, data, seed_index, indices):
    points = data['points'][indices].copy()
    proposal = data['proposals'][seed_index, indices]
    gate = gate_mask(rule, data, seed_index, indices)
    points[gate] = proposal[gate]
    move = np.linalg.norm(points-data['points'][indices], axis=-1)
    adjusted = np.isfinite(move) & (move > ADJUSTED_EPS_PX)
    return points, gate, adjusted, move


def select(root, destination, cfg, manifest, prepared, gate_config):
    indices = manifest['populations']['synth_val']
    records = [manifest['records'][i] for i in indices]
    selection = dict(schema='dht_selective_gate_selection_v1', config_sha256=E.sha(root/'CONFIG.json'),
                     manifest_sha256=E.sha(root/'manifest.json'), gate_config_sha256=E.sha(destination/'CONFIG_GATE.json'),
                     source_sha256=sources(root), population='synth_val', n_frames=256,
                     seeds=cfg['dht']['seeds'], objective=gate_config['objective'], models={})
    for model in MODELS:
        data = prepared[model]
        # Error arrays are constructed ONLY for synthetic validation here.
        base_loss, proposed_loss = [], []
        for i, record in zip(indices, records):
            gt = np.asarray(record['gt_points'], float)[:8]
            gt_valid = np.asarray(record.get('gt_valid', [True]*8), bool)[:8]
            gt_valid &= np.isfinite(gt).all(-1)
            if not gt_valid.any():
                raise ValueError('Synthetic selection frame has no valid annotated corners')
            observed = data['valid'][i, :8] & gt_valid
            diagonal = np.hypot(record['width'], record['height'])
            baseline = np.ones(8)
            baseline[observed] = np.minimum(np.linalg.norm(data['points'][i,:8][observed]-gt[observed],axis=-1)/diagonal, 1)
            baseline[~gt_valid] = 0
            base_loss.append(baseline/gt_valid.sum())
            per_seed = []
            for si in range(3):
                values = np.ones(8)
                values[observed] = np.minimum(np.linalg.norm(data['proposals'][si,i,:8][observed]-gt[observed],axis=-1)/diagonal, 1)
                values[~gt_valid] = 0
                per_seed.append(values/gt_valid.sum())
            proposed_loss.append(per_seed)
        base_loss, proposed_loss = np.asarray(base_loss), np.asarray(proposed_loss).transpose(1,0,2)
        selection['models'][model] = {'families': {}}
        for family in FAMILIES:
            candidates = []
            for rule in gate_config['rules'][model][family]:
                scores, modified, admitted = [], 0, 0
                for si in range(3):
                    _, gate, adjusted, _ = apply_rule(rule, data, si, indices)
                    scores.append(float(np.where(gate[:,:8],proposed_loss[si],base_loss).sum(-1).mean()))
                    modified += int(adjusted[:,:8].sum())
                    admitted += int(gate[:,:8].sum())
                candidates.append(dict(rule=rule, validation_score=float(np.mean(scores)), per_seed_score=scores,
                                       n_adjusted_corner_seed_observations=modified, n_gate_passed_corner_seed_observations=admitted))
            best = min(candidates, key=lambda r:(r['validation_score'],r['n_adjusted_corner_seed_observations'],r['rule']['id']))
            selection['models'][model]['families'][family] = dict(selected_rule=best['rule'],
                    validation_score=best['validation_score'], n_adjusted_corner_seed_observations=best['n_adjusted_corner_seed_observations'],
                    candidates=candidates)
            print(f"Selected {model}/{family}: {best['rule']['id']} score={best['validation_score']:.8f} adjusted={best['n_adjusted_corner_seed_observations']}", flush=True)
    path = destination/'GATE_SELECTION.json'
    if path.exists() and E.read(path) != selection:
        raise ValueError('Frozen gate selection differs; do not evaluate new real-derived choices')
    E.write(path, selection)
    if E.read(path) != selection:
        raise ValueError('Gate selection read-back mismatch')
    return selection


def summarize_rows(rows, cfg):
    summary = E.summarize(rows, cfg)
    for key in ('n_gate_passed_corners', 'n_adjusted_corners', 'n_baseline_valid_corners'):
        summary[key] = sum(r[key] for r in rows)
    summary['n_adjusted_frames'] = sum(r['n_adjusted_corners'] > 0 for r in rows)
    summary['adjusted_corner_fraction'] = summary['n_adjusted_corners']/max(summary['n_baseline_valid_corners'],1)
    return summary


def aggregate_seeds(summaries):
    keys = ('corner_median_px', 'corner_p90_px', 'corner_mean_px', 'corner_coverage',
            'frame_penalized_mean_px', 'capped_normalized_mean', 'pck_5px', 'pck_10px', 'pck_20px',
            'all8_10px_success', 'line_distance_median_px', 'line_distance_p90_px',
            'line_angle_median_deg', 'line_success_fraction', 'n_gate_passed_corners',
            'n_adjusted_corners', 'n_adjusted_frames', 'adjusted_corner_fraction', 'n_baseline_valid_corners')
    result = []
    for identity in sorted({(r['model'],r['arm'],r['population'],r['group']) for r in summaries}):
        rows = [r for r in summaries if (r['model'],r['arm'],r['population'],r['group']) == identity]
        metrics = {}
        for key in keys:
            values = [r[key] for r in rows if r[key] is not None]
            metrics[key] = dict(mean=float(np.mean(values)),min=min(values),max=max(values)) if values else None
        result.append(dict(zip(('model','arm','population','group'),identity)) |
                      dict(seeds=[r['seed'] for r in rows], metrics=metrics))
    return result


def reliability(manifest, prepared, frame_rows, gate_config):
    from scipy.stats import rankdata, spearmanr

    def auc(scores, outcome):
        positives, negatives = int(outcome.sum()), int((~outcome).sum())
        if not positives or not negatives:
            return None
        ranks = rankdata(scores, method='average')
        return float((ranks[outcome].sum()-positives*(positives+1)/2)/(positives*negatives))

    def correlation(a, b):
        if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
            return None
        return float(spearmanr(a,b).statistic)

    output = []
    for model in MODELS:
        data = prepared[model]
        base = {r['id']:r for r in frame_rows if r['model']==model and r['arm']=='baseline'}
        proposals = {(r['id'],r['seed']):r for r in frame_rows if r['model']==model and r['arm']=='unconditional_lambda1'}
        cuts = gate_config['thresholds'][model]['point_confidence_upper'][:3]
        for population in ('synth_val','synth_test','cross_v4','real_dev'):
            confidence, error, quality, improvement = [], [], [], []
            for i in manifest['populations'][population]:
                record = manifest['records'][i]
                errors = np.asarray(base[record['id']]['corner_errors'],float)
                point_conf = data['confidence'][i]
                good = np.isfinite(errors) & np.isfinite(point_conf)
                confidence.extend(point_conf[good]); error.extend(errors[good])
                for si,seed in enumerate(gate_config['seeds']):
                    proposal_error = np.asarray(proposals[(record['id'],seed)]['corner_errors'],float)
                    q = data['signals']['line_quality'][si,i]
                    good = np.isfinite(errors) & np.isfinite(proposal_error) & np.isfinite(q)
                    quality.extend(q[good]); improvement.extend((errors-proposal_error)[good])
            confidence,error,quality,improvement = [np.asarray(v,float) for v in (confidence,error,quality,improvement)]
            bins = []
            assignments = np.digitize(confidence,cuts,right=True)
            for b in range(4):
                values = error[assignments==b]
                bins.append(dict(bin=b, n_corners=len(values), mean_error_px=float(values.mean()) if len(values) else None,
                                 error_gt10_fraction=float((values>10).mean()) if len(values) else None))
            output.append(dict(model=model,population=population,n_observed_corners=len(error),
                point_confidence_min=float(confidence.min()) if len(confidence) else None,
                point_confidence_max=float(confidence.max()) if len(confidence) else None,
                confidence_error_spearman=correlation(confidence,error),
                low_confidence_error_gt10_auroc=auc(-confidence,error>10) if len(error) else None,
                fixed_validation_confidence_bins=bins, n_line_quality_corner_seed_observations=len(quality),
                quality_improvement_spearman=correlation(quality,improvement),
                high_quality_improvement_auroc=auc(quality,improvement>1e-9) if len(quality) else None))
    return dict(scope='Post-selection descriptive proxy diagnostics only. Observed annotated corners condition on baseline coverage. Line-quality observations pool three dependent DHT seeds; no independence or calibration claim. No metric here adjusts selected gates.', rows=output)


def evaluate_gates(root, destination, cfg, manifest, variants, prepared, gate_config, selection):
    frozen_sources = sources(root)
    selection_sha, config_gate_sha = E.sha(destination/'GATE_SELECTION.json'), E.sha(destination/'CONFIG_GATE.json')
    records, indices = manifest['records'], list(range(len(manifest['records'])))
    predictions = dict(schema='dht_gated_predictions_v1',config_sha256=E.sha(root/'CONFIG.json'),
                       manifest_sha256=E.sha(root/'manifest.json'),gate_config_sha256=config_gate_sha,
                       selection_sha256=selection_sha,models={})
    frame_rows,summaries,paired = [],[],[]
    invariant_count = 0
    for model in MODELS:
        data,baselines = prepared[model],variants[model]
        stored = []
        for i,(record,base) in enumerate(zip(records,baselines)):
            stored.append(dict(id=record['id'],baseline={key:base[key] for key in ('kps','kp_valid','kp_conf')},
                               seeds={str(seed):{} for seed in cfg['dht']['seeds']},
                               signals={str(seed):{key:E.clean(value[si,i]) for key,value in data['signals'].items()}
                                        for si,seed in enumerate(cfg['dht']['seeds'])}))
        predictions['models'][model] = dict(records=stored)
        model_rows = []
        for i,record in enumerate(records):
            model_rows.append(dict(model=model,arm='baseline',seed=0,n_gate_passed_corners=0,
                                   n_adjusted_corners=0,n_baseline_valid_corners=int(data['valid'][i,:8].sum()),
                                   **E.score_frame(record,data['points'][i],data['valid'][i])))
        rules = {'unconditional_lambda1':dict(id='unconditional_lambda1',kind='unconditional')}
        rules.update({family:selection['models'][model]['families'][family]['selected_rule'] for family in FAMILIES})
        for si,seed in enumerate(cfg['dht']['seeds']):
            for arm,rule in rules.items():
                points,gates,adjusted,moves = apply_rule(rule,data,si,indices)
                if not np.array_equal(points[:,8],data['points'][:,8],equal_nan=True):
                    raise ValueError('Gate changed center')
                if not np.array_equal(points[~gates],data['points'][~gates],equal_nan=True):
                    raise ValueError('Gate changed a rejected corner')
                if gates[~data['valid']].any() or (adjusted & ~gates).any():
                    raise ValueError('Missingness/rejected-corner invariant failed')
                if rule['kind']=='baseline' and (gates.any() or adjusted.any()):
                    raise ValueError('Selected baseline fallback changed predictions')
                invariant_count += len(records)
                for i,record in enumerate(records):
                    stored[i]['seeds'][str(seed)][arm] = dict(kps=E.clean(points[i]),kp_valid=data['valid'][i].tolist(),
                        gate=gates[i].tolist(),adjusted=adjusted[i].tolist(),move_px=E.clean(moves[i]),rule_id=rule['id'])
                    model_rows.append(dict(model=model,arm=arm,seed=seed,n_gate_passed_corners=int(gates[i,:8].sum()),
                        n_adjusted_corners=int(adjusted[i,:8].sum()),n_baseline_valid_corners=int(data['valid'][i,:8].sum()),
                        **E.score_frame(record,points[i],data['valid'][i])))
        baseline_rows = {r['id']:r for r in model_rows if r['arm']=='baseline'}
        for arm,seed in [('baseline',0)]+[(a,s) for s in cfg['dht']['seeds'] for a in ARMS[1:]]:
            for population in cfg['evaluation']['populations']:
                subset = [r for r in model_rows if r['arm']==arm and r['seed']==seed and r['population']==population]
                groups = ['all']+sorted({r['group'] for r in subset}) if population=='real_dev' else ['all']
                for group in groups:
                    rows = [r for r in subset if group=='all' or r['group']==group]
                    summaries.append(dict(model=model,arm=arm,seed=seed,population=population,group=group,**summarize_rows(rows,cfg)))
                    if arm!='baseline':
                        delta=np.asarray([r['frame_corner_penalized_px']-baseline_rows[r['id']]['frame_corner_penalized_px'] for r in rows])
                        paired.append(dict(model=model,arm=arm,seed=seed,population=population,group=group,n_frames=len(rows),
                            mean_delta_px=float(delta.mean()),median_delta_px=float(np.median(delta)),
                            improved_frames=int((delta < -1e-9).sum()),worsened_frames=int((delta > 1e-9).sum()),
                            unchanged_frames=int((np.abs(delta)<=1e-9).sum())))
        frame_rows.extend(model_rows)
        print(f'Evaluated frozen gates: {model}',flush=True)
    # Independently verify selected validation scores using the standard scorer.
    score_rechecks=[]
    for model in MODELS:
        for family in FAMILIES:
            per_seed=[]
            for seed in cfg['dht']['seeds']:
                rows=[r for r in frame_rows if r['model']==model and r['arm']==family and r['seed']==seed and r['population']=='synth_val']
                per_seed.append(float(np.mean([r['corner_capped_normalized'] for r in rows])))
            expected=selection['models'][model]['families'][family]['validation_score']
            difference=abs(float(np.mean(per_seed))-expected)
            if difference>1e-12:
                raise ValueError('Vectorized selection score differs from standard per-frame evaluation')
            score_rechecks.append(dict(model=model,family=family,max_abs_score_delta=difference,PASS=True))
    proxy=reliability(manifest,prepared,frame_rows,gate_config)
    if sources(root)!=frozen_sources or E.sha(destination/'GATE_SELECTION.json')!=selection_sha or E.sha(destination/'CONFIG_GATE.json')!=config_gate_sha:
        raise ValueError('Source, gate config, or synthetic selection changed during evaluation')
    E.write(destination/'GATED_PREDICTIONS.json',predictions)
    E.write(destination/'GATE_FRAME_METRICS.json',dict(schema='dht_gate_frame_metrics_v1',records=frame_rows))
    result=dict(schema='dht_gate_results_v1',complete=True,config_sha256=E.sha(root/'CONFIG.json'),
                manifest_sha256=E.sha(root/'manifest.json'),gate_config_sha256=config_gate_sha,selection_sha256=selection_sha,
                source_sha256=frozen_sources,selection=selection['models'],summaries=summaries,
                seed_summary=aggregate_seeds(summaries),paired=paired,proxy_reliability=proxy,
                n_frames=len(records),n_frame_variant_rows=len(frame_rows),
                scope='Prediction-only boolean corner gate on frozenlambda1 proposals; no new model training. Gate thresholds/selection use synthetic validation only. RealDEV52 was already inspected and is not an independent held-out test.',
                metric_notes='Pixel errors condition on observed corners; PCK includes missing failures. Counts refer to valid predicted top8corners, without GT masking. Seed_summary averages per-seed statistics/counts; no new inference-runtime claim.',
                limitations=['Confidence and Hough softmax peak values are uncalibrated geometric-reliability proxies.',
                             'YOLO26 keypoint confidence is supervised for keypoint presence/visibility (GT visibility nonzero), not the probability of pixel error below10. Its near-one values do not establish accurate corners; trained RLE sigma outputs are not exposed or used by this adapter.',
                             'Quantile thresholds depend on the inherited synthetic validation distribution and may not transfer to real images.',
                             'Missing corners remain missing; the gate cannot recover nondetections.',
                             'Every selected joint gate needs both low point-confidence and sufficient predicted line-quality plus bounded movement.',
                             'Projected structural lines include hidden/amodal edges; finite/peak confidence does not establish physical visibility.',
                             'No additional ablation, threshold tuning or model choice uses real results.'])
    E.write(destination/'GATE_RESULTS.json',result)
    outputs=('CONFIG_GATE.json','GATE_SELECTION.json','GATE_RESULTS.json','GATED_PREDICTIONS.json','GATE_FRAME_METRICS.json')
    audit=dict(schema='dht_gate_audit_v1',complete=True,PASS=True,source_sha256=frozen_sources,
               gate_config_sha256=config_gate_sha,selection_sha256=selection_sha,
               output_sha256={name:E.sha(destination/name) for name in outputs},
               threshold_population='synth_val',selection_population='synth_val',selection_frames=256,
               candidates_per_model=70,models=list(MODELS),seeds=cfg['dht']['seeds'],
               n_frames=len(records),n_frame_variant_rows=len(frame_rows),n_frame_arm_invariant_checks=invariant_count,
               missing_corners_preserved=True,centroid_preserved=True,rejected_corners_bitwise_unchanged=True,
               baseline_fallback_identity=True,predicted_signals_take_no_gt_argument=True,
               selected_scores_standard_scorer_recheck=score_rechecks,
               no_real_selection=True,no_new_training=True,no_runtime_claim=True,
               scope='Artifact/protocol PASS only; not an accuracy-improvement verdict.')
    E.write(destination/'GATE_AUDIT.json',audit)
    print(f'Selective fusion complete: {destination}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True,help='Frozen integration live/ result directory')
    parser.add_argument('--phase',choices=('register','all'),default='all')
    args=parser.parse_args()
    root=args.run_dir.resolve();destination=root/'fusion_diagnosis_v1';destination.mkdir(exist_ok=True)
    if not (destination/'PURPOSE.md').is_file():
        raise ValueError('Diagnosis PURPOSE.md must be prepared before execution')
    cfg,manifest,variants,dht=E.inputs(root)
    prepared=prepare_predictions(cfg,manifest,variants,dht)
    gate_config=register(root,destination,cfg,manifest,prepared)
    print(f'Frozen gate configuration: {E.sha(destination/"CONFIG_GATE.json")}',flush=True)
    if args.phase=='register':
        return
    selection=select(root,destination,cfg,manifest,prepared,gate_config)
    evaluate_gates(root,destination,cfg,manifest,variants,prepared,gate_config,selection)


if __name__=='__main__':
    main()

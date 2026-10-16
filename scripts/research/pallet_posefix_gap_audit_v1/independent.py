"""Independent archived-prediction recount and N1 failure attribution.

No predictor, mask provider, optimizer, or PnP is imported or called. The pure
fixed-reference parser and metric equations are reused from the earlier sealed
independent checker, not the production scorer. All input/output roots are CLI
arguments. Existing files are read only; each execution requires a new output.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

from scripts.research.pallet_guarded_bi_final_v1.independent import (
    Checks, fixed_references, metric, projected_cuboid, ray_cost,
)

VIS = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1')
PF95 = VIS / 'posefix_sub_comparison_v1'
PF143 = VIS / 'remainder_after_test49_v1/posefix_baseline_supplement_v4_14'
ARMS = ('PF', 'E0', 'N0', 'N1')
ERRORS = ('translation_cm', 'rotation_deg', 'ADDsym_cm')
POPULATIONS = {'PRIMARY': 192, 'TEST': 49, 'REMAINDER': 143, 'CAL': 46}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def journal(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def population(rows, name):
    return [r for r in rows if (r['partition'] != 'CAL' if name == 'PRIMARY'
                               else r['partition'] == name)]


def stats(values):
    values = list(values)
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite saved metric must not silently disappear')
    return dict(available_n=len(x), full_n=len(values),
                mean=float(x.mean()) if len(x) else None,
                sample_variance_ddof1=float(x.var(ddof=1)) if len(x) > 1 else None,
                sample_SD_ddof1=float(x.std(ddof=1)) if len(x) > 1 else None,
                max=float(x.max()) if len(x) else None, no_trimming=True)


def category(a, b):
    sa, sb = a['success_le5'], b['success_le5']
    return ('unresolved' if sa is None or sb is None else
            'recovery' if sa and not sb else 'damage' if sb and not sa else
            'both_success' if sa else 'both_failure')


def comparison(rows, arm, baseline):
    groups = {k: [] for k in ('recovery', 'damage', 'both_success', 'both_failure', 'unresolved')}
    by_session = {}
    errors = {k: Counter() for k in ERRORS[:2]}
    examples = []
    for row in rows:
        a, b = row['metrics'][arm], row['metrics'][baseline]
        key = category(a, b)
        groups[key].append(row['id'])
        by_session.setdefault(row['session'], Counter())[key] += 1
        delta = {}
        for k in ERRORS[:2]:
            d = None if a[k] is None or b[k] is None else a[k] - b[k]
            delta[k] = d
            errors[k]['unresolved' if d is None else 'better' if d < -1e-9 else
                      'worse' if d > 1e-9 else 'tie'] += 1
        examples.append(dict(id=row['id'], category=key, arm=a, baseline=b, delta=delta))
    finite = [e for e in examples if e['delta']['translation_cm'] is not None]
    return dict(arm=arm, baseline=baseline, counts={k: len(v) for k, v in groups.items()},
                ids=groups, per_session={s: dict(c, net=c['recovery']-c['damage'])
                                        for s, c in sorted(by_session.items())},
                error_sign_counts={k: dict(v) for k, v in errors.items()},
                largest_T_worsen=sorted(finite, key=lambda e: -e['delta']['translation_cm'])[:10],
                largest_T_improve=sorted(finite, key=lambda e: e['delta']['translation_cm'])[:10])


def summarize(rows):
    return dict(full_n=len(rows), arms={arm: dict(
        full_n=len(rows), success_joint_le5=sum(r['metrics'][arm]['success_le5'] is True for r in rows),
        translation_le5=sum(r['metrics'][arm]['translation_cm'] is not None and
                           r['metrics'][arm]['translation_cm'] <= 5 for r in rows),
        no_pose=sum(not r['metrics'][arm]['pose_available'] for r in rows),
        reference_unresolved=sum(r['metrics'][arm]['success_le5'] is None for r in rows),
        errors={k: stats(r['metrics'][arm][k] for r in rows) for k in ERRORS}) for arm in ARMS},
        comparisons={arm: comparison(rows, arm, 'PF') for arm in ('E0', 'N0', 'N1')})


def stage_summary(records, full_n):
    t = [r['delta_T_cm'] for r in records if r['delta_T_cm'] is not None]
    r = [r['delta_R_deg'] for r in records if r['delta_R_deg'] is not None]
    return dict(frames=len(records), available_T=len(t), available_R=len(r),
                delta_T_sum_cm=math.fsum(t), delta_T_mean_cm=math.fsum(t)/len(t) if t else None,
                contribution_to_primary_T_mean_cm=math.fsum(t)/full_n,
                T_better=sum(d < -1e-9 for d in t), T_worse=sum(d > 1e-9 for d in t),
                T_exact_tie=sum(d == 0 for d in t),
                delta_R_sum_deg=math.fsum(r), delta_R_mean_deg=math.fsum(r)/len(r) if r else None,
                R_better=sum(d < -1e-9 for d in r), R_worse=sum(d > 1e-9 for d in r),
                R_exact_tie=sum(d == 0 for d in r),
                both_T_R_worse=sum(x['delta_T_cm'] is not None and x['delta_R_deg'] is not None and
                                   x['delta_T_cm'] > 1e-9 and x['delta_R_deg'] > 1e-9 for x in records))


def stage(rows, predictions):
    records = []
    for row in rows:
        a, b = row['metrics']['N1'], row['metrics']['N0']
        p = predictions[row['id']]['predictions']['N1']
        records.append(dict(id=row['id'], session=row['session'], category=category(a, b),
            N0=b, N1=a, delta_T_cm=None if a['translation_cm'] is None or b['translation_cm'] is None
            else a['translation_cm']-b['translation_cm'],
            delta_R_deg=None if a['rotation_deg'] is None or b['rotation_deg'] is None
            else a['rotation_deg']-b['rotation_deg'],
            accepted=p['accepted'], fallback=p['fallback'], source=p['actual_pose'].get('source')))
    positive = sorted((r for r in records if r['delta_T_cm'] is not None and r['delta_T_cm'] > 1e-9),
                      key=lambda r: -r['delta_T_cm'])
    total = math.fsum(r['delta_T_cm'] for r in records if r['delta_T_cm'] is not None)
    positives = math.fsum(r['delta_T_cm'] for r in positive)
    tops = {}
    for count in (5, 10):
        selected = positive[:count]
        summed = math.fsum(r['delta_T_cm'] for r in selected)
        tops[str(count)] = dict(cases=selected, delta_T_sum_cm=summed,
            contribution_to_primary_T_mean_cm=summed/len(rows),
            share_of_net_worsening=summed/total if total else None,
            share_of_all_positive_deltas=summed/positives if positives else None)
    return dict(schema='N1_VS_N0_FIXED_PRIMARY192_POSTHOC_STAGE_ATTRIBUTION_V2',
        full_n=len(rows), all_rows_and_failures_kept=True, new_fits_inference_or_rules=0,
        all=stage_summary(records, len(rows)),
        success_categories={k: stage_summary([r for r in records if r['category'] == k], len(rows))
                            for k in ('recovery', 'damage', 'both_success', 'both_failure', 'unresolved')},
        top_positive_translation_contributors=tops,
        total_positive_translation_delta_cm=positives,
        total_negative_translation_delta_cm=math.fsum(r['delta_T_cm'] for r in records
                                                     if r['delta_T_cm'] is not None and r['delta_T_cm'] < -1e-9),
        new_rotation_gt45_cases=[r for r in records if r['N0']['rotation_deg'] is not None and
                                r['N1']['rotation_deg'] is not None and
                                r['N0']['rotation_deg'] <= 45 < r['N1']['rotation_deg']],
        by_N1_status={key: stage_summary([r for r in records if r['accepted'] is value], len(rows))
                      for key, value in [('accepted', True), ('fallback', False)]},
        by_N1_pose_source={key: stage_summary([r for r in records if r['source'] == key], len(rows))
                           for key in sorted({r['source'] for r in records})},
        rows=records,
        interpretation='Arithmetic decomposition of saved changes, not causal allocation to state/mask/measurement/solver.')


def rotdeg(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(a).T@np.asarray(b))-1)/2, -1, 1))))


def objective(measurements, projection):
    values = []
    for m in measurements:
        residual = projection[m['native_corner']] - m['xy']
        e = np.linalg.solve(np.linalg.cholesky(m['covariance_px2']), residual)
        values.append(18*(math.sqrt(1+float(e@e)/9)-1))
    return math.fsum(values)


def case_diagnostics(stage_result, frames, minimal_root, checks):
    """Saved top5 T damages and all new45deg cases; never a new routing rule."""
    selected = {r['id'] for r in stage_result['top_positive_translation_contributors']['5']['cases']}
    selected.update(r['id'] for r in stage_result['new_rotation_gt45_cases'])
    selected.update(f['id'] for f in frames if f['id'].endswith(':029428'))
    scores = {r['id']: r for r in stage_result['rows']}
    cases = []
    for i, frame in enumerate(frames):
        if frame['id'] not in selected:
            continue
        op = Path('run_v2/observations')/f'{i:03d}_N3_SUBPIX.json'
        sp = Path('run_v2/poses')/f'{i:03d}_N3_SUBPIX.json'
        obs = read(minimal_root/op)['result']['observation']
        solved = read(minimal_root/sp)['result']
        solution = solved['solution']
        old = frame['arms']['N3_SUBPIX']['coarse_pose']
        new = solved['actual_pose']
        before = projected_cuboid(obs, old, frame['K'])
        after = projected_cuboid(obs, new, frame['K'])
        ms = obs['frozen_visible_measurements']
        raw = np.asarray(obs['original_raw_points'])
        measures = []
        for m in ms:
            k = m['native_corner']
            measures.append(dict(native_corner=k, state=m['state'], source=m['source'],
                movement_px=float(np.linalg.norm(np.asarray(m['xy'])-raw[k])),
                initial_residual_px=float(np.linalg.norm(before[k]-m['xy'])),
                final_residual_px=float(np.linalg.norm(after[k]-m['xy'])),
                covariance_px2=m['covariance_px2']))
        rays = []
        for rec in obs['visible_correction']['records']:
            for ray in rec.get('rays', []):
                if not ray.get('supported'):
                    continue
                k, j = ray['native_corner'], ray['native_neighbor']
                p = np.asarray(ray['observed_points'])
                w = np.asarray(ray['observation_confidence'])
                rays.append(dict(edge=[k, j], n=len(p), source=ray['measurement_source'],
                    baseline_cost=ray_cost(p, w, before, k, j)[0],
                    candidate_cost=ray_cost(p, w, after, k, j)[0]))
        before_obj, after_obj = objective(ms, before), objective(ms, after)
        checks(abs(after_obj-solution['objective']) <= 1e-7, frame['id']+'/saved objective independently reproduced')
        feasible = [dict(objective=v['objective'], gap=v['objective']-solution['objective'],
                        rotation_from_selected_deg=rotdeg(v['R'], new['R_cf']),
                        initialization=v['initialization']) for v in solution['feasible_pose_samples']]
        cases.append(dict(id=frame['id'], index=i, metrics=scores[frame['id']],
            inputs=dict(observation=dict(path=str(op), sha256=sha(minimal_root/op)),
                        solved=dict(path=str(sp), sha256=sha(minimal_root/sp))),
            states=obs['states'], actual_BI_ids=[m['native_corner'] for m in ms],
            own_measurements=measures, supported_rays=rays,
            initial_to_final_rotation_deg=rotdeg(old['R_physical'], new['R_physical']),
            pose_center_shift_cm=float(np.linalg.norm(np.asarray(old['centroid'])-new['centroid'])*100),
            objective_before=before_obj, objective_after=after_obj,
            rank=solution['rank'], ambiguity=solution['ambiguity'], search=solution['search'],
            feasible_summary=feasible,
            region_domain_count=solved['OT_U_region_pose_constraints'],
            all_initial_native_projection=before.tolist(), all_final_native_projection=after.tolist()))
    return dict(schema='N1_TOP_STORED_DAMAGE_CASES_POSTHOC_V1',
        selection='Top5 positive T differences, all N0<=45deg→N1>45deg cases, and the user-discussed029428 frame in fixed PRIMARY192; posthoc only.',
        new_inference_fit_or_routes=0, cases=cases,
        limitations=['Selection uses saved GT scores for explanation only, never prediction routing.',
                     'No independently annotated per-ray physical edge or per-state truth is opened.',
                     'Rank6 certifies local observability under the selected correspondences, not global uniqueness.',
                     'No counterfactual fit is performed, so downstream amplification is observed but per-stage causality is not isolated.'])


def run(*, source_root, minimal_root, single_root, guarded_root, output):
    start = time.monotonic()
    source_root, minimal_root, single_root, guarded_root, output = map(
        Path, (source_root, minimal_root, single_root, guarded_root, output))
    if output.exists():
        raise FileExistsError('Use a fresh output; preserve every previous attempt')
    checks = Checks()
    bindings = {}

    def bound(path, name):
        bindings[name] = sha(path)
        return read(path)

    frames = bound(minimal_root/'inputs_v2/INPUTS.json', 'minimal/inputs_v2/INPUTS.json')['frames']
    score = bound(single_root/'score/SCORED_ROWS.json', 'single/score/SCORED_ROWS.json')['rows']
    e_score = bound(guarded_root/'score/SCORED_ROWS.json', 'guarded/score/SCORED_ROWS.json')['rows']
    current = {r['id']: r for r in score}
    e_current = {r['id']: r for r in e_score}
    single_pred = {r['id']: r for r in journal(single_root/'PREDICTIONS.jsonl.gz')}
    e_pred = {r['id']: r for r in journal(guarded_root/'PREDICTIONS.jsonl.gz')}
    for key, folder in [('single', single_root), ('guarded', guarded_root)]:
        ss = bound(folder/'score/SCORE_SEAL.json', key+'/score/SCORE_SEAL.json')
        checks(ss['artifacts']['SCORED_ROWS.json'] == sha(folder/'score/SCORED_ROWS.json'), key+'/score sealed')
        bindings[key+'/PREDICTIONS.jsonl.gz'] = sha(folder/'PREDICTIONS.jsonl.gz')
        bindings[key+'/PREDICTION_SEAL.json'] = sha(folder/'PREDICTION_SEAL.json')
    ids = [f['id'] for f in frames]
    checks(len(ids) == len(set(ids)) == 238, 'fixed238 unique')
    checks(ids == [r['id'] for r in score] == [r['id'] for r in e_score], 'same238 exact order')
    checks(set(single_pred) == set(e_pred) == set(ids), 'prediction238 exact population')

    a, b = source_root/PF95, source_root/PF143
    pf_pred95 = [r for r in journal(a/'PREDICTIONS.jsonl.gz') if r['method'] == 'POSEFIX_RAW']
    pf_pred143 = journal(b/'PREDICTIONS.jsonl.gz')
    pf = {r['id']: r for r in pf_pred95+pf_pred143}
    checks(len(pf_pred95) == 95 and len(pf_pred143) == 143 and set(pf) == set(ids), 'PoseFix95+143 same238')
    p95 = bound(a/'PROTOCOL.json', str(PF95/'PROTOCOL.json'))
    p143 = bound(b/'PROTOCOL.json', str(PF143/'PROTOCOL.json'))
    checks(p95['PoseFix'] == p143['PoseFix_variant'], 'identical PoseFix model/seed/step95+143')
    old95 = {r['id']: r['metric'] for r in journal(a/'SCORED_ROWS.jsonl.gz') if r['method'] == 'POSEFIX_RAW'}
    old143 = {r['id']: r['metrics']['POSEFIX_RAW'] for r in bound(
        b/'POSTSEAL_SCORE.json', str(PF143/'POSTSEAL_SCORE.json'))['rows']}
    old_scores = old95 | old143
    metadata95 = {r['frame']['id']: r['frame'] for r in bound(a/'BASELINE_INPUTS.json', str(PF95/'BASELINE_INPUTS.json'))['frames']}
    for folder, rel in [(a, PF95), (b, PF143)]:
        bindings[str(rel/'PREDICTIONS.jsonl.gz')] = sha(folder/'PREDICTIONS.jsonl.gz')
    for frame in frames:
        ID = frame['id']
        source = metadata95.get(ID)
        if source is not None:
            for k in ('image_sha256', 'K', 'raw_hw', 'xyz'):
                checks(frame[k] == source[k], ID+'/PoseFix same '+k)
        else:
            source = pf[ID]
            checks(frame['image_sha256'] == source['image_sha256'], ID+'/same image')
            checks(frame['K'] == source['metadata_K'], ID+'/same K')
            checks(frame['xyz'] == source['metadata_registered_size'], ID+'/same physical dimensions')
    references = fixed_references(source_root, minimal_root/'SCORING_REFERENCE_MANIFEST_V2.json', score, checks)
    rows = []
    max_bridge = 0.
    for frame in frames:
        ID, partition = frame['id'], frame['partition']
        poses = dict(PF=pf[ID]['actual_pose'], E0=e_pred[ID]['predictions']['E0']['actual_pose'],
                     N0=single_pred[ID]['predictions']['N0']['actual_pose'],
                     N1=single_pred[ID]['predictions']['N1']['actual_pose'])
        ms = {arm: metric(pose, references[partition][ID]) for arm, pose in poses.items()}
        for arm in ARMS:
            archived = (old_scores[ID] if arm == 'PF' else e_current[ID]['metrics']['E0'] if arm == 'E0'
                        else current[ID]['metrics'][arm])
            for k in ERRORS:
                av, bv = ms[arm][k], archived[k]
                checks(av is None and bv is None or av is not None and bv is not None and abs(av-bv) <= 1e-9,
                       ID+'/'+arm+'/'+k+'/direct metric bridge')
                if av is not None and bv is not None:
                    max_bridge = max(max_bridge, abs(av-bv))
            success_key = 'success_5cm_5deg' if arm == 'PF' else 'success_strict_lt5'
            checks(ms[arm]['success_strict_lt5'] == archived[success_key], ID+'/'+arm+'/success bridge')
        rows.append(dict(id=ID, partition=partition, session=frame['session'],
                         cluster_id=frame['cluster_id'], metrics=ms))
    pops = {name: summarize(population(rows, name)) for name in POPULATIONS}
    for name, count in POPULATIONS.items():
        checks(pops[name]['full_n'] == count, name+'/full denominator')
    # Stage arithmetic uses the exact stored scores; direct metrics above have
    # independently checked every component before this further decomposition.
    stage_rows = population(score, 'PRIMARY')
    attribution = stage(stage_rows, single_pred)
    diagnostics = case_diagnostics(attribution, frames, minimal_root, checks)
    audit = dict(schema='INDEPENDENT_SAVED_POSEFIX_CURRENT_GAP_AUDIT_V2',
        checks=checks.count, failure_count=len(checks.failures), failures=checks.failures,
        PF_model=p95['PoseFix'],
        PF_training_clarification='ImageNet-initialized RGB ResNet152; pallet adaptation uses synthetic data with real-pallet training0. The historical synthetic-only label describes pallet adaptation, not all pretraining.',
        labels=dict(PF='PoseFix RAW first pass + own common F', E0='previously frozen e context snapshot',
                    N0='YOLO→N3→SubPix→common F', N1='N0 own state/mask/BI measurements→BI-only PnP'),
        same238_images_K_physical_size=True, reference_bindings=read(minimal_root/'SCORING_REFERENCE_MANIFEST_V2.json')['reference_bindings'],
        maximum_recomputed_vs_saved_metric_difference=max_bridge,
        populations=pops, rows=rows, input_sha256=bindings,
        N1_vs_N0_stage_attribution=attribution,
        new_training_NN_mask_PnP_or_fit_calls=0, existing_files_modified=False,
        limitations=['Repeated development-exposed data, not an independent holdout.',
                     'Different full image-head pipelines; no PoseFix plus identical minimal BI arm exists.',
                     'E0 context is not an alias for every later V33/SRC_C variant.',
                     'Saved GT metrics select explanatory cases only; prediction routing is unchanged.'],
        wall_seconds=time.monotonic()-start)
    output.mkdir(parents=True)
    write(output/'INDEPENDENT_AUDIT.json', audit)
    write(output/'N1_STAGE_ATTRIBUTION.json', attribution)
    write(output/'N1_TOP_DAMAGE_CASES.json', diagnostics)
    write(output/'RECEIPT.json', dict(checks=checks.count, failure_count=len(checks.failures),
        source_sha256=sha(Path(__file__)), artifacts={p.name: sha(p) for p in sorted(output.iterdir())},
        all_inputs_read_only=True, new_fits=0))
    print(json.dumps(dict(checks=checks.count, failures=len(checks.failures),
                         wall_seconds=audit['wall_seconds'], outputs=[p.name for p in output.iterdir()])))
    if checks.failures:
        raise AssertionError(checks.failures[:10])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'minimal-root', 'single-root', 'guarded-root', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    run(**vars(parser.parse_args()))

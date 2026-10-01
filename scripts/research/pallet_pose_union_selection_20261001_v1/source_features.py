"""Freeze source candidates first, then inspect declared-C2 TRAIN costs only.

No image model, fit, VAL quality metric, or real reference is used here.
Run ``freeze`` and ``train_gate`` as separate processes.
"""
from . import common as C
READS = C.source_guard(allow_source_targets=True)
import argparse
from collections import Counter
import math
from pathlib import Path
import time

import numpy as np
import torch
from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
from scripts.research.pallet_selector_recovery_v1 import features as F, models as G, common as U

FEATURE_LOCK = C.DOC / 'SOURCE_FEATURE_LOCK.json'
GATE_PROTOCOL = C.DOC / 'TRAIN_FEASIBILITY_PROTOCOL.json'
GEO_LOCK = C.ROOT / '_docs/experiments/pallet_selector_recovery_v1/stage2_synth_scorer/SCORER_SELECTION_LOCK.json'
HYP = tuple(U.HYP)
KEYS = ('translation_cm', 'rotation_deg')
S = np.diag([1., -1., -1.])
C2 = (np.eye(3), np.diag([-1., 1., -1.]))


def bindings(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            yield value
        else:
            for item in value.values():
                yield from bindings(item)
    elif isinstance(value, list):
        for item in value:
            yield from bindings(item)


def verify_all(value):
    unique = {b['path']: b for b in bindings(value)}
    for binding in unique.values():
        C.verify(binding)
    return len(unique)


def save_npz(path, **values):
    path = Path(path).resolve()
    assert path.is_relative_to(C.RAW)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **values)


def verified_predictions():
    protocol = C.protocol()
    verify_all(protocol)
    lock = C.read(C.DOC / 'SOURCE_PREDICTIONS_LOCK.json')
    assert lock['complete'] and lock['models'] == list(C.MODELS) and lock['frames'] == 5120
    assert lock['protocol'] == C.bind(C.DOC / 'SOURCE_PROTOCOL.json')
    assert lock['metadata'] == protocol['input'] == C.bind(C.RAW / 'SOURCE_INPUTS.json')
    assert set(lock['receipts']) == set(C.MODELS)
    verify_all(lock)
    data = C.read(C.RAW / 'SOURCE_INPUTS.json')
    assert len(data) == len({r['id'] for r in data}) == 5120
    assert Counter(r['split'] for r in data) == {'TRAIN': 4096, 'VAL': 1024}
    receipts = {}
    for model in C.MODELS:
        receipt = C.read(C.ROOT / lock['receipts'][model]['path'])
        assert receipt['complete'] and receipt['model'] == model and receipt['frames'] == len(data)
        assert receipt['protocol'] == lock['protocol'] and receipt['checkpoint'] == protocol['checkpoints'][model]
        verify_all(receipt)
        expected = [str((C.RAW / 'source_predictions' / model / f'{j:05d}.json').relative_to(C.ROOT)) for j in range(len(data))]
        assert [b['path'] for b in receipt['files']] == expected
        receipts[model] = receipt
    return protocol, lock, data, receipts


def operator_bindings():
    paths = [Path(__file__), Path(C.__file__), Path(O.__file__), Path(O.D.__file__),
             Path(O.D.Pose.__file__), Path(F.__file__), Path(G.__file__),
             Path(U.__file__), Path(O.D.Selector.__file__)]
    # The imported geometry implementation is part of candidate generation.
    paths.append(Path(O.D.Selector.geometry.__file__))
    return [C.bind(p) for p in dict.fromkeys(paths)]


def make_record(pred, row, checkpoint):
    metadata = dict(K=row['K'], xyz=row['dims'], hw=row['hw'])
    record = O.candidate_record(pred, metadata)
    extracted = F.extract(pred, row['K'], row['dims'], row['hw'])
    assert extracted['selection'] == record['selected_name']
    hp = {h['name']: h for h in record['hypotheses']}
    fh = {h['name']: h for h in extracted['hypotheses']}
    assert set(hp) == set(fh) and set(hp) <= set(HYP)
    geo = np.zeros((2, len(F.names())), np.float32)
    valid = np.zeros(2, bool)
    selected = U.selected(pred)
    for j, name in enumerate(HYP):
        if name not in hp:
            continue
        vector = F.vector(fh[name], selected, row['hw'])
        pose = hp[name]['pose']
        valid[j] = vector is not None and bool(pose['available'])
        if vector is not None:
            geo[j] = vector
        if pose['available']:
            other = F.production_pose(fh[name], row['dims'], source=False)
            assert other['available']
            for key in ('R_cf', 'R_physical', 'centroid', 'cf_extents'):
                np.testing.assert_allclose(pose[key], other[key], atol=1e-6, rtol=1e-7)
    name = record['selected_name']
    scores = None
    if extracted['valid']:
        old_geo = np.asarray(extracted['features'], np.float32)
        np.testing.assert_array_equal(old_geo, geo)
        scores = G.scores(checkpoint, old_geo[None])[0]
        assert np.isfinite(scores).all()
        name = HYP[int(G.selection(scores[None], HYP)[0])]
    pose = hp[name]['pose'] if name in hp else record['current']
    return dict(record, GEO_name=name, GEO_pose=pose,
                GEO_scores=None if scores is None else scores.tolist(),
                GEO_fallback=not extracted['valid']), geo, valid


def freeze():
    # Audit hooks cannot be removed: label scoring deliberately uses a new CLI
    # process. This stronger guard enforces the input-only feature stage.
    C.source_guard(allow_source_targets=False)
    start = time.monotonic()
    if FEATURE_LOCK.exists():
        lock = C.read(FEATURE_LOCK)
        assert lock['complete']
        verify_all(lock)
        print('SOURCE_FEATURES_ALREADY_FROZEN', flush=True)
        return
    protocol, prediction_lock, data, receipts = verified_predictions()
    geo_lock = C.read(GEO_LOCK)
    assert geo_lock['winner'] == 'GEO_LINEAR'
    verify_all(geo_lock)
    checkpoint = torch.load(C.ROOT / geo_lock['checkpoint']['path'], map_location='cpu', weights_only=False)
    assert checkpoint['variant'] == 'GEO_LINEAR' and checkpoint['d'] == 94 and len(F.names()) == 94
    arrays = dict(ids=np.asarray([r['id'] for r in data]), split=np.asarray([r['split'] for r in data]),
                  hypothesis_names=np.asarray(HYP))
    records = {}
    counts = {}
    for model in C.MODELS:
        records[model] = {}
        features, valid, indices = [], [], []
        for j, row in enumerate(data):
            saved = C.read(C.ROOT / receipts[model]['files'][j]['path'])
            assert saved['id'] == row['id'] and saved['model'] == model
            assert saved['protocol_sha'] == prediction_lock['protocol']['sha256']
            assert saved['checkpoint_sha'] == protocol['checkpoints'][model]['sha256']
            record, feature, mask = make_record(saved['prediction'], row, checkpoint)
            records[model][row['id']] = record
            features.append(feature)
            valid.append(mask)
            indices.append(HYP.index(record['GEO_name']) if record['GEO_name'] in HYP else -1)
            if (j + 1) % 512 == 0:
                print('SOURCE_FEATURES', model, j + 1, '/', len(data), flush=True)
        arrays[model + '_geo'] = np.asarray(features, np.float32)
        arrays[model + '_valid'] = np.asarray(valid, bool)
        arrays[model + '_GEO_index'] = np.asarray(indices, np.int64)
        counts[model] = dict(frames=len(data), valid_candidate_count=int(np.sum(valid)),
                             zero_candidate_frames=int(np.sum(~np.asarray(valid).any(1))),
                             operational_failed=sum(not r['GEO_pose']['available'] for r in records[model].values()))
    path = C.RAW / 'SOURCE_FEATURES.npz'
    save_npz(path, **arrays)
    C.save(C.RAW / 'SOURCE_POSES.json', dict(ids=arrays['ids'].tolist(), models=list(C.MODELS),
           hypothesis_names=list(HYP), records=records, source_targets_read=False, real_targets_read=False))
    C.save(FEATURE_LOCK, dict(complete=True, created_at=C.now(),
           protocol=prediction_lock['protocol'], predictions=C.bind(C.DOC / 'SOURCE_PREDICTIONS_LOCK.json'),
           metadata=C.bind(C.RAW / 'SOURCE_INPUTS.json'),
           features=C.bind(path), poses=C.bind(C.RAW / 'SOURCE_POSES.json'),
           GEO_selection_lock=C.bind(GEO_LOCK), GEO_checkpoint=geo_lock['checkpoint'],
           operators=operator_bindings(), feature_names=F.names(), hypothesis_names=list(HYP),
           models=list(C.MODELS), counts=counts, source_targets_read=False, real_targets_read=False,
           VAL_quality_scored=False, fits=0, image_forwards=0,
           invalid_feature_placeholder='Placeholder or ignored finite vector whenever candidate valid mask is false; never a scored candidate.',
           read_paths=sorted(set(READS)), wall_seconds=time.monotonic() - start))
    print('SOURCE_FEATURE_LOCK_COMPLETE', counts, flush=True)


def error_pair(pose, reference_R, reference_t):
    if not pose['available']:
        return np.array([np.inf, np.inf])
    R, t = np.asarray(pose['R_physical'], float), np.asarray(pose['centroid'], float)
    assert np.isfinite(R).all() and np.isfinite(t).all()
    rotation = min(float(np.degrees(np.arccos(np.clip((np.trace((reference_R @ q).T @ R) - 1.) / 2., -1., 1.)))) for q in C2)
    result = np.array([float(np.linalg.norm(t - reference_t) * 100.), rotation])
    assert np.isfinite(result).all() and (result >= 0).all()
    # Same C2 geodesic after source/runtime basis conversion on both sides.
    source_rotation = min(float(np.degrees(np.arccos(np.clip((np.trace(((reference_R @ S) @ q).T @ (R @ S)) - 1.) / 2., -1., 1.)))) for q in C2)
    assert abs(rotation - source_rotation) <= 1e-5
    return result


def quantile(values, fraction):
    values = np.sort(np.asarray(values, float))
    assert not np.isnan(values).any() and not np.isneginf(values).any()
    if not len(values):
        return None
    rank = (len(values) - 1) * fraction
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return float(values[lo])
    if np.isposinf(values[hi]):
        return float('inf')
    return float(values[lo] + (rank - lo) * (values[hi] - values[lo]))


def summary(errors):
    errors = np.asarray(errors, float)
    assert errors.ndim == 2 and errors.shape[1] == 2 and not np.isnan(errors).any()
    available = np.isfinite(errors).all(1)
    assert np.all(np.isfinite(errors).any(1) == available)
    out = dict(frames=len(errors), valid_pose=int(available.sum()), failed_pose=int((~available).sum()))
    for kind, mask in [('full_population', np.ones(len(errors), bool)), ('conditional', available)]:
        out[kind] = {}
        for j, key in enumerate(KEYS):
            out[kind][key] = {}
            for name, q in [('median', .5), ('P90', .9)]:
                value = quantile(errors[mask, j], q)
                out[kind][key][name] = value if value is not None and np.isfinite(value) else None
                out[kind][key][name + '_status'] = 'NA_EMPTY' if value is None else 'FINITE' if np.isfinite(value) else 'POSITIVE_INFINITY'
    return out


def choose_cost(errors, scale, names):
    """Choose one whole pose; exact-cost ties only, then Pareto and fixed names."""
    assert len(errors) == len(names)
    usable = np.flatnonzero(np.isfinite(errors).all(1))
    if not len(usable):
        return -1, dict(exact_cost_tie=False, pareto_tie_removed=0)
    cost = np.max(errors / scale, axis=1)
    ties = [int(j) for j in usable if cost[j] == np.min(cost[usable])]
    frontier = [j for j in ties if not any(np.all(errors[k] <= errors[j]) and np.any(errors[k] < errors[j]) for k in ties)]
    chosen = min(frontier, key=lambda j: (names[j].split(':')[0] != 'R0', names[j].split(':')[1]))
    return chosen, dict(exact_cost_tie=len(ties) > 1, pareto_tie_removed=len(ties) - len(frontier))


def gate(before, after, ratio):
    a, b = summary(before), summary(after)
    checks = dict(failure_no_increase=b['failed_pose'] <= a['failed_pose'])
    for j, key in enumerate(KEYS):
        oldmedian, newmedian = quantile(before[:, j], .5), quantile(after[:, j], .5)
        oldtail, newtail = quantile(before[:, j], .9), quantile(after[:, j], .9)
        checks[key + '_median_strict'] = bool(oldmedian is not None and newmedian is not None and newmedian < oldmedian)
        checks[key + '_P90_guard'] = bool(oldtail is not None and newtail is not None and newtail <= ratio * oldtail)
    return dict(PASS=all(checks.values()), checks=checks, before=a, after=b)


def train_gate():
    start = time.monotonic()
    C.verify(C.read(C.DOC / 'TRAIN_FEASIBILITY_PROTOCOL_SHA.json'))
    protocol = C.read(GATE_PROTOCOL)
    verify_all(protocol)
    assert protocol['scope'] == 'C2_TRAIN_ONLY'
    assert protocol['scalar_cost'] == 'max(T/sT,R/sR)'
    assert protocol['scale'] == 'median_R0_GEO_C2_TRAIN'
    assert protocol['median_strict'] and protocol['failure_no_increase']
    assert protocol['p90_ratio_max'] == 1.05
    assert protocol['tie_rule'] == 'exact_cost_then_pareto_then_R0_then_hypothesis'
    result_path = C.DOC / 'SOURCE_TRAIN_GATE.json'
    if result_path.exists():
        done = C.read(result_path)
        assert done['complete'] and done['protocol'] == C.bind(GATE_PROTOCOL)
        verify_all(done)
        print('SOURCE_TRAIN_GATE_ALREADY_COMPLETE', done['status'], flush=True)
        return
    locked = C.read(FEATURE_LOCK)
    assert locked['complete'] and not locked['source_targets_read'] and not locked['VAL_quality_scored']
    verify_all(locked)
    assert C.read(C.DOC / 'SOURCE_PREDICTIONS_LOCK.json')['complete']
    contract = C.read(C.DOC / 'SOURCE_CONTRACT.json')
    assert contract['complete'] and contract['status'] == 'PASS'
    assert protocol['source_contract'] == C.bind(C.DOC / 'SOURCE_CONTRACT.json')
    assert protocol['source_protocol'] == C.bind(C.DOC / 'SOURCE_PROTOCOL.json')
    verify_all(contract)
    eligible = contract['fit_eligibility']['eligible_ids']
    counts = contract['fit_eligibility']['eligible_counts']
    assert counts == protocol['eligible_counts']
    assert len(eligible['TRAIN']) == counts['TRAIN'] and len(eligible['VAL']) == counts['VAL']
    assert counts['TRAIN'] > 0 and counts['VAL'] == 1024
    assert set(eligible['TRAIN']).isdisjoint(eligible['VAL'])
    data = C.read(C.RAW / 'SOURCE_INPUTS.json')
    ids = [r['id'] for r in data]
    train_set = set(eligible['TRAIN'])
    mask = np.array([r['split'] == 'TRAIN' and r['id'] in train_set for r in data])
    indices = np.flatnonzero(mask)
    assert len(indices) == counts['TRAIN']
    rows = [data[j] for j in indices]
    assert {r['id'] for r in rows} == train_set
    assert set(contract['fit_eligibility']['diagnostic_C1_ids']['TRAIN']).isdisjoint(train_set)
    # Only after feature lock: source references, restricted immediately to TRAIN.
    reference_start = C.now()
    geometry = C.ROOT / 'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
    reference_paths = [str(geometry.relative_to(C.ROOT)), str((C.SOURCE / 'SYNTH_RECORDS.json').relative_to(C.ROOT))]
    historical = C.read(C.ROOT / '_docs/experiments/pallet_selector_recovery_v1/stage2_synth_scorer/SYNTHETIC_SPLIT_LOCK.json')
    for path in reference_paths:
        match = [b for b in bindings(historical) if b['path'] == path]
        assert len(match) == 1
        C.verify(match[0])
    old_rows = C.read(C.SOURCE / 'SYNTH_RECORDS.json')
    source_rows = {r['id']: r for r in old_rows if r['id'] in train_set}
    assert set(source_rows) == train_set
    rowindex = np.array([source_rows[r['id']]['table_index'] for r in rows], np.int64)
    with np.load(geometry) as z:
        stems = z['stems'][rowindex]
        target_R, target_t = z['R'][rowindex], z['t'][rowindex]
        target_K, target_dims = z['K'][rowindex], z['dims'][rowindex]
    assert stems.tolist() == [r['id'] for r in rows]
    for j, row in enumerate(rows):
        fx, fy, cx, cy = target_K[j]
        np.testing.assert_allclose(row['K'], [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], atol=1e-7, rtol=0)
        np.testing.assert_allclose(row['dims'], target_dims[j], atol=1e-7, rtol=0)
    posefile = C.read(C.RAW / 'SOURCE_POSES.json')
    assert posefile['ids'] == ids and posefile['models'] == list(C.MODELS)
    arrays = dict(ids=np.array([r['id'] for r in rows]), source_index=indices,
                  eligible_train_mask=mask, hypothesis_names=np.asarray(HYP))
    errors, operational = {}, {}
    with np.load(C.RAW / 'SOURCE_FEATURES.npz') as feature:
        assert feature['ids'].tolist() == ids
        for model in C.MODELS:
            errors[model] = np.full((len(rows), 2, 2), np.inf)
            operational[model] = np.full((len(rows), 2), np.inf)
            for j, row in enumerate(rows):
                record = posefile['records'][model][row['id']]
                candidates = {h['name']: h['pose'] for h in record['hypotheses']}
                reference_R = np.asarray(target_R[j]) @ S
                for k, name in enumerate(HYP):
                    if feature[model + '_valid'][indices[j], k]:
                        assert name in candidates and candidates[name]['available']
                        errors[model][j, k] = error_pair(candidates[name], reference_R, target_t[j])
                operational[model][j] = error_pair(record['GEO_pose'], reference_R, target_t[j])
            arrays[model + '_T_cm'] = errors[model][:, :, 0]
            arrays[model + '_R_deg'] = errors[model][:, :, 1]
            arrays[model + '_GEO_T_cm'] = operational[model][:, 0]
            arrays[model + '_GEO_R_deg'] = operational[model][:, 1]
    scale = np.array([quantile(operational['R0'][:, j], .5) for j in range(2)], float)
    scale_ok = bool(np.isfinite(scale).all() and (scale > 0).all())
    arrays.update(sT_cm=scale[0], sR_deg=scale[1])
    save_npz(C.RAW / 'SOURCE_TRAIN_LABELS.npz', **arrays)
    result = dict(complete=True, created_at=C.now(), scope='C2_TRAIN_ONLY', frames=len(rows),
                  VAL_quality_scored=False, C1_quality_scored=False, fits=0, image_forwards=0,
                  source_scale=dict(sT_cm=float(scale[0]) if np.isfinite(scale[0]) else None,
                                    sR_deg=float(scale[1]) if np.isfinite(scale[1]) else None, valid=scale_ok),
                  operational={m: summary(operational[m]) for m in C.MODELS},
                  feature_lock=C.bind(FEATURE_LOCK), protocol=C.bind(GATE_PROTOCOL),
                  source_contract=C.bind(C.DOC / 'SOURCE_CONTRACT.json'),
                  labels=C.bind(C.RAW / 'SOURCE_TRAIN_LABELS.npz'),
                  source_reference_bindings=[C.bind(geometry), C.bind(C.SOURCE / 'SYNTH_RECORDS.json')],
                  source_reference_read_time=reference_start,
                  physical_label='runtime R_reference=renderer.R@diag(1,-1,-1); center t unchanged; whole-pose C2 full rotation',
                  label_container_disclosure='The old geometry NPZ contains broader source rows; only eligible C2 TRAIN indices are retained and scored. No VAL/C1 quality arrays are computed or persisted.',
                  gate_summary='Same scalar-cost whole-pose candidate for T and R; strict TRAIN full-population medians, both P90<=1.05, no failure increase.',
                  inference_use=False, cost_oracle_diagnostic_only=True,
                  failure_scope='Failure excludes this fixed cost/source gate; it does not prove every union selector or objective impossible.')
    if not scale_ok:
        result.update(PASS=False, status='STOP_NONFINITE_OR_NONPOSITIVE_TRAIN_SCALE', seeds={})
    else:
        names0 = ['R0:' + h for h in HYP]
        picks0 = [choose_cost(e, scale, names0)[0] for e in errors['R0']]
        base = np.array([e[j] if j >= 0 else [np.inf, np.inf] for e, j in zip(errors['R0'], picks0)])
        result['R0_ONLY_cost_oracle'] = summary(base)
        result['seeds'] = {}
        perframe = {}
        for model in C.MODELS[1:]:
            names = names0 + [model + ':' + h for h in HYP]
            union = np.concatenate([errors['R0'], errors[model]], axis=1)
            picked, ties, details = [], Counter(), []
            opportunities = Counter()
            for j, candidate_errors in enumerate(union):
                selection, info = choose_cost(candidate_errors, scale, names)
                picked.append(candidate_errors[selection] if selection >= 0 else [np.inf, np.inf])
                ties.update(info)
                pairs = [(a, b) for a in errors['R0'][j] for b in errors[model][j] if np.isfinite(a).all() and np.isfinite(b).all()]
                opportunities['frames_with_refiner_pareto_better_pair'] += any(np.all(b <= a) and np.any(b < a) for a, b in pairs)
                opportunities['frames_with_R0_pareto_better_pair'] += any(np.all(a <= b) and np.any(a < b) for a, b in pairs)
                opportunities['frames_with_tradeoff_pair'] += any((b[0] - a[0]) * (b[1] - a[1]) < 0 for a, b in pairs)
                details.append(dict(id=rows[j]['id'], R0_ONLY_choice=names0[picks0[j]] if picks0[j] >= 0 else None,
                                    UNION_choice=names[selection] if selection >= 0 else None,
                                    exact_cost_tie=info['exact_cost_tie'], pareto_tie_removed=info['pareto_tie_removed']))
            picked = np.asarray(picked)
            evaluated = gate(base, picked, protocol['p90_ratio_max'])
            evaluated.update(choices=dict(Counter(d['UNION_choice'] or 'FAILED' for d in details)),
                             ties=dict(ties), cross_expert_Pareto_opportunities=dict(opportunities))
            result['seeds'][model] = evaluated
            perframe[model] = details
        result.update(PASS=all(r['PASS'] for r in result['seeds'].values()),
                      status='PASS_TRAIN_COST_FEASIBILITY' if all(r['PASS'] for r in result['seeds'].values()) else 'STOP_TRAIN_COST_FEASIBILITY')
        C.save(C.RAW / 'SOURCE_TRAIN_COST_CHOICES.json', dict(diagnostic_only=True, inference_use=False, records=perframe))
        result['cost_choices'] = C.bind(C.RAW / 'SOURCE_TRAIN_COST_CHOICES.json')
    result.update(read_paths=sorted(set(READS)), wall_seconds=time.monotonic() - start)
    C.save(result_path, result)
    print('SOURCE_TRAIN_GATE', result['status'], result['source_scale'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'train_gate'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    O.D.cv2.setNumThreads(1)
    globals()[args.stage]()


if __name__ == '__main__':
    main()

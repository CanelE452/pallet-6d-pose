"""One fixed sequential learned-selector -> per-corner LOO audit, no fitting."""
import argparse
import copy
from collections import Counter

import cv2
import numpy as np

from scripts.research.pallet_posefix_corner_gate_v1 import gates as G
from scripts.research.pallet_posefix_corner_gate_v1.evaluate import clean, paired, top
from . import core as P

DOC = P.DOC / 'posthoc_loo'
RAW = P.RAW / 'posthoc_loo'
ARM = 'LEARNED_THEN_LOO'


def filter_selected(n2, selected, learned_mask, valid, K, dims, confidence, kp_conf):
    """LOO on the actual selected pose; reject only, never revive a pair."""
    learned_mask = np.asarray(learned_mask, bool)
    assert learned_mask.shape == (4,)
    geometry = G.geometry_details(selected, valid, K, dims)
    decision = G.decide_pairs('geometry_only', geometry, valid, confidence, kp_conf)
    accepted = learned_mask & np.asarray(decision['pair_accept'], bool)
    history = []
    for round_index in range(4):
        if not accepted.any():
            break
        mixed = G.apply_pairs(n2, selected, accepted)
        checked = G.geometry_details(
            mixed, valid, K, dims, hypothesis_name=geometry['chosen_hypothesis'],
            normalization_px=geometry['projected_diagonal_px'])
        residual = np.asarray(checked['per_corner_remove'], float)
        reject = np.array([accepted[i] and any(
            not np.isfinite(residual[j]) or residual[j] < 0 or residual[j] > G.GEOMETRY_THRESHOLD
            for j in pair) for i, pair in enumerate(G.PAIRS)], bool)
        history.append(dict(round=round_index+1, pair_accept_before=accepted.tolist(),
                            per_corner_remove=residual.tolist(), pair_rejected=reject.tolist()))
        accepted &= ~reject
        if not reject.any():
            break
    assert not (accepted & ~learned_mask).any()
    return G.apply_pairs(n2, selected, accepted), dict(
        learned_pair_accept=learned_mask.tolist(), pair_accept=accepted.tolist(),
        geometry=geometry, decision=decision, recheck_history=history)


def apply():
    cv2.setNumThreads(1)
    for path in [P.DOC/'OUTPUTS_LOCK.json', P.GATE_DOC/'OUTPUTS_LOCK.json']:
        lock = P.read(path)
        for binding in lock['artifacts']:
            P.verify_binding(binding)
    source_paths = [P.RAW/'PREDICTIONS.json', P.RAW/'DECISIONS.json',
                    P.GATE_RAW/'PREDICTIONS.json',
                    P.ROOT/'_docs/experiments/pallet_large_error_refiner_v1/SPLIT.json',
                    P.ROOT/'_docs/paper/final_dimension_v1/green150_saved_labels_v1/DATASET_SNAPSHOT.json',
                    P.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json']
    protocol = dict(experiment=ARM, model_training=False, GPU_inference_reused=True,
        LOO_device='CPU/OpenCV', threshold=G.GEOMETRY_THRESHOLD,
        policy='Actual learned-selected pose -> corner LOO -> intersect learned pairs -> N2 fallback -> monotone mixed-pose recheck; no flip or heatmap gate',
        normalization='Choose single registry W/D hypothesis and D from learned-selected pose, pin during rechecks',
        no_GT_for_gating=True, no_threshold_search=True, independent_test=False,
        no_image_drops=True, final_model_modified=False, self_training=False,
        source_bindings=[P.bound(p) for p in source_paths],
        code=[P.bound(__file__), P.bound(G.__file__),
              P.bound(P.ROOT/'scripts/self_training_yolo/pseudo_label_filters.py'),
              P.bound(P.ROOT/'scripts/research/pallet_posefix_large_error_v1/evaluate.py')])
    P.freeze(DOC/'PROTOCOL.json', protocol)
    if (DOC/'OUTPUTS_LOCK.json').exists():
        for b in P.read(DOC/'OUTPUTS_LOCK.json')['artifacts']:
            P.verify_binding(b)
        print('POSTHOC_OUTPUTS_ALREADY_FROZEN', flush=True)
        return
    saved, masks = P.read(source_paths[0]), P.read(source_paths[1])
    sources = {'DEV72': P.read(source_paths[3])['evaluation'],
               'GREEN150': P.read(source_paths[4])['records']}
    registry = {r['object_type']: r['physical_dimensions_m'] for r in P.read(source_paths[5])['objects']}
    outputs, diagnostics = {}, {}
    for ds, rows in saved.items():
        meta_by_id = {r['id']: r for r in sources[ds]}
        masks_by_id = {r['id']: r for r in masks[ds]}
        outputs[ds], diagnostics[ds] = [], []
        for row in rows:
            meta, mask = meta_by_id[row['id']], masks_by_id[row['id']]['pair_accept']
            base, selected = row['predictions']['A_N2'], row['predictions']['LEARNED_PAIR']
            b, a = top(base), top(selected)
            n2, q = np.asarray(b['keypoints_xy'], float), np.asarray(a['keypoints_xy'], float)
            conf = np.asarray(a['keypoints_conf'], float)
            valid = (conf >= .5) & np.isfinite(q).all(-1) & ~(q == -1).all(-1)
            K = np.asarray(meta['K'] if ds == 'DEV72' else meta['source_K'], float)
            kind = meta['object_type'] if ds == 'DEV72' else 'plastic_standard_110x110x15'
            assert np.array_equal(G.apply_pairs(n2, np.asarray(top(row['predictions']['REPLAY'])['keypoints_xy']), mask), q)
            points, diag = filter_selected(n2, q, mask, valid, K, registry[kind], a['score'], conf)
            pred = copy.deepcopy(base)
            top(pred)['keypoints_xy'] = points.tolist()
            assert points[8].tolist() == n2[8].tolist()
            assert np.all(np.all(points == n2, axis=1) | np.all(points == q, axis=1))
            # Replacing only selected keypoints must leave all detector metadata unchanged.
            restored = copy.deepcopy(pred)
            top(restored)['keypoints_xy'] = b['keypoints_xy']
            assert restored == base
            outputs[ds].append({**{k:v for k,v in row.items() if k != 'predictions'}, 'prediction': pred})
            diagnostics[ds].append(clean(dict(id=row['id'], **diag)))
        print('POSTHOC_LOO_FROZEN_POPULATION', ds, len(rows), flush=True)
    assert sum(map(len, outputs.values())) == 222
    P.freeze(RAW/'PREDICTIONS.json', outputs)
    P.freeze(RAW/'DECISIONS.json', diagnostics)
    P.freeze(DOC/'OUTPUTS_LOCK.json', dict(complete=True, frames=222, GT_read_for_gating=False,
        protocol=P.bound(DOC/'PROTOCOL.json'), artifacts=[P.bound(RAW/n) for n in ('PREDICTIONS.json','DECISIONS.json')]))


def score():
    lock = P.read(DOC/'OUTPUTS_LOCK.json')
    for b in lock['artifacts']+[lock['protocol']]:
        P.verify_binding(b)
    for b in P.read(DOC/'PROTOCOL.json')['source_bindings']+P.read(DOC/'PROTOCOL.json')['code']:
        P.verify_binding(b)
    from scripts.research.pallet_posefix_large_error_v1 import evaluate as O
    _, _, records, groups, pe, targets, _ = O.evaluation_inputs()
    saved, previous = P.read(P.RAW/'PREDICTIONS.json'), P.read(P.RAW/'PER_FRAME_METRICS.json')
    rows = {ds:[] for ds in O.DATASETS}
    checks = 0
    for ds, outputs in P.read(RAW/'PREDICTIONS.json').items():
        for output, original, record in zip(outputs, saved[ds], records[ds]):
            assert output['id'] == original['id'] == record['id']
            gt, box, modes, kind = O.read_targets(ds, record, output['raw_hw'], pe, targets)
            for mode, valid in modes:
                for arm in ('A_N2', 'LEARNED_PAIR'):
                    before = O.score_prediction(original['predictions'][arm], gt, box, valid,
                        groups[kind]['permutations'], output['raw_hw'], record)
                    expected = next(r for r in previous[mode][arm] if r['id'] == output['id'])
                    O.assert_same(before, expected, mode+'/'+arm+'/'+output['id'])
                    checks += 1
                rows[mode].append(O.score_prediction(output['prediction'], gt, box, valid,
                    groups[kind]['permutations'], output['raw_hw'], record))
    summaries, comparisons = {}, {}
    for ds, rr in rows.items():
        summaries[ds] = O.summary(rr)
        summaries[ds].update(correct10=sum(e <= 10 for r in rr for e in r['errors']),
            matched_mean_px=float(np.mean([e for r in rr for e in r['observed_errors']])))
        comparisons[ds] = dict(vs_N2=paired(previous[ds]['A_N2'], rr, previous[ds]['REPLAY']),
            vs_LEARNED=paired(previous[ds]['LEARNED_PAIR'], rr, previous[ds]['LEARNED_PAIR']))
    acceptance = {}
    for ds, rr in P.read(RAW/'DECISIONS.json').items():
        counts = Counter(sum(r['pair_accept']) for r in rr)
        acceptance[ds] = dict(frames=len(rr), before_pairs=sum(sum(r['learned_pair_accept']) for r in rr),
            accepted_pairs=sum(k*v for k,v in counts.items()), frames_any=len(rr)-counts[0],
            frames_all_four=counts[4], frames_no_correction=counts[0], histogram=dict(sorted(counts.items())))
    P.freeze(RAW/'PER_FRAME_METRICS.json', rows)
    result = dict(complete=True, summary=summaries, comparisons=comparisons, acceptance=acceptance,
        baseline_parity_checks=checks, output_lock=P.bound(DOC/'OUTPUTS_LOCK.json'),
        metrics=P.bound(RAW/'PER_FRAME_METRICS.json'), final_model_modified=False,
        thresholds_retuned=False, self_training=False, evaluation_reused=True)
    assert checks == 744
    P.freeze(DOC/'RESULTS.json', result)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['apply', 'score'])
    args = parser.parse_args()
    apply() if args.stage == 'apply' else score()

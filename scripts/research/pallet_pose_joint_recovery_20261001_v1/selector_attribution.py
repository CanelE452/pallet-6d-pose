"""Exact linear score decomposition, without fitting or changing predictions.

This is an exploratory mechanism audit after prior real results were known.
It does not select a model, feature mask, threshold, seed, or runtime rule.
"""
from pathlib import Path
import csv
import hashlib
import json
import time

import cv2
import numpy as np
import torch

from scripts.research.pallet_selector_recovery_v1 import features as F, models as M
from scripts.research.pallet_selector_recovery_v1.feature_contract import names

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_pose_joint_recovery_20261001_v1'
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
PREV = ROOT / '_docs/experiments/pallet_pose_stable_improvement_20261001_v1'
OLD = ROOT / '_docs/experiments/pallet_selector_recovery_v1/stage2_synth_scorer'


def read(p):
    return json.loads(Path(p).read_text())


def bind(p):
    p = Path(p)
    return dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)


def verify(b):
    assert bind(ROOT / b['path']) == b, b['path']


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def family(name):
    if name.startswith('R_cf_'):
        return 'rotation_matrix'
    if name.startswith('t_'):
        return 'translation'
    if name.startswith('cf_'):
        return 'dimensions'
    if name in ('residual8_px', 'residual8_bboxnorm', 'residual_center_px', 'residual_center_bboxnorm'):
        return 'explicit_center_residual'
    if name.startswith('residual'):
        return 'other_residuals'
    if name.startswith('reprojection'):
        return 'reprojection_rmse'
    if 'conf' in name or name.startswith('bbox_'):
        return 'confidence_and_box'
    return 'shape_and_orientation'


def main():
    start = time.monotonic()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    lock = read(PREV / 'PREDICTIONS_LOCK.json')
    assert lock['complete']
    for b in [lock['metadata'], *lock['predictions'].values()]:
        verify(b)
    selected = read(OLD / 'SCORER_SELECTION_LOCK.json')
    verify(selected['checkpoint'])
    ck = torch.load(ROOT / selected['checkpoint']['path'], map_location='cpu', weights_only=False)
    weight = ck['state']['net.weight'].numpy().reshape(-1).astype(float)
    std = np.asarray(ck['std'], dtype=float)
    feature_names = names()
    assert len(feature_names) == len(weight) == 94
    rows = read(ROOT / lock['metadata']['path'])
    candidate_path = ROOT / 'data/pallet/results/pallet_pose_stable_improvement_20261001_v1/POSE_CANDIDATES.json'
    pose_lock = read(PREV / 'POSE_PREDICTIONS_LOCK.json')
    for b in pose_lock['files']:
        verify(b)
    frozen = read(candidate_path)
    arrays = {}
    info = {}
    max_score_gap = 0.
    max_reconstruction_gap = 0.
    records = []
    for model, b in lock['predictions'].items():
        prediction = read(ROOT / b['path'])
        prediction = prediction.get('predictions', prediction)
        features, valid = [], []
        for row in rows:
            feat = F.extract(prediction[row['id']], row['K'], row['xyz'], row['hw'])
            valid.append(feat['valid'])
            features.append(feat['features'] if feat['valid'] else np.zeros((2, 94)))
        x = np.asarray(features, dtype=np.float32)
        valid = np.asarray(valid)
        score = M.scores(ck, x)
        contribution = (x[:, 0].astype(float) - x[:, 1].astype(float)) / std * weight
        margin = score[:, 0] - score[:, 1]
        arrays[model + '_geo'] = x
        arrays[model + '_valid'] = valid
        arrays[model + '_contribution'] = contribution
        info[model] = dict(valid=int(valid.sum()), invalid=int((~valid).sum()))
        for j, row in enumerate(rows):
            if not valid[j]:
                continue
            previous = frozen[model][row['id']]
            gap = float(np.max(np.abs(score[j] - previous['GEO_scores'])))
            reconstruction = abs(float(contribution[j].sum()) - float(margin[j]))
            max_score_gap = max(max_score_gap, gap)
            max_reconstruction_gap = max(max_reconstruction_gap, reconstruction)
            assert np.allclose(score[j], previous['GEO_scores'], atol=2e-5, rtol=1e-5), (model, row['id'], gap)
            choice = F.C.HYP[int(M.selection(score[j:j+1], F.C.HYP)[0])]
            assert choice == previous['GEO_name'], (model, row['id'])
            assert np.isclose(contribution[j].sum(), margin[j], atol=2e-5, rtol=1e-5), reconstruction
            records.append(dict(model=model, id=row['id'], recording=row['recording'],
                choice=choice, margin_long_minus_short=float(margin[j]),
                explicit_center_contribution=float(sum(contribution[j, k] for k,n in enumerate(feature_names) if family(n)=='explicit_center_residual'))))
        print('FEATURE_PARITY', model, info[model], flush=True)
    # Exact additive score difference. Mean and bias cancel for both hypotheses.
    # Family attribution is descriptive; correlated features are not causal effects.
    populations = read(ROOT / 'data/pallet/results/pallet_pose_stable_improvement_20261001_v1/EVAL_GROUPS.json')
    summaries = {}
    family_names = sorted({family(n) for n in feature_names})
    for pop in ('NATURAL99', 'CLEAN29', 'WOOD45'):
        ix = np.asarray([r['id'] in populations[pop] for r in rows])
        summaries[pop] = {}
        for model in lock['predictions']:
            valid = ix & arrays[model + '_valid'] & arrays['R0_valid']
            delta = arrays[model + '_contribution'][valid] - arrays['R0_contribution'][valid]
            total = delta.sum(1)
            mask = np.array([frozen[model][r['id']]['GEO_name'] != frozen['R0'][r['id']]['GEO_name'] for r in rows])[valid]
            summaries[pop][model] = dict(n=int(valid.sum()), switches=int(mask.sum()),
                abs_margin_change_median=float(np.median(np.abs(total))),
                families={f: dict(mean_abs_contribution_change=float(np.mean(np.abs(delta[:, [family(n)==f for n in feature_names]].sum(1)))),
                    switched_mean_abs_contribution_change=float(np.mean(np.abs(delta[mask][:, [family(n)==f for n in feature_names]].sum(1)))) if mask.any() else None)
                    for f in family_names},
                top_features_by_mean_absolute_change=[dict(name=feature_names[k], value=float(np.abs(delta[:, k]).mean()))
                    for k in np.argsort(-np.abs(delta).mean(0))[:10]])
    # Candidate-common terms cancel. This verifies that preserved bbox/confidence
    # cannot directly change the linear decision (though geometry depends on box).
    common_ix = [k for k,n in enumerate(feature_names) if family(n)=='confidence_and_box']
    candidate_common_max = max(float(np.abs(arrays[m+'_contribution'][:,common_ix]).max()) for m in lock['predictions'])
    assert candidate_common_max == 0.
    RAW.mkdir(parents=True, exist_ok=True)
    array_path = RAW / 'REAL_FEATURE_ATTRIBUTION.npz'
    assert not array_path.exists()
    np.savez_compressed(array_path, ids=np.array([r['id'] for r in rows]), feature_names=np.array(feature_names), **arrays)
    csv_path = DOC / 'SELECTOR_MARGIN_ROWS.csv'
    DOC.mkdir(parents=True, exist_ok=True)
    with csv_path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = dict(type='Exploratory linear attribution; not a prospective efficacy experiment',
        inputs=[bind(PREV/'PREDICTIONS_LOCK.json'),bind(PREV/'POSE_PREDICTIONS_LOCK.json'),selected['checkpoint']],
        codes=[bind(Path(__file__)),bind(Path(F.__file__)),bind(Path(M.__file__))],
        artifacts=[bind(array_path),bind(csv_path)], feature_count=94, models=info,
        score_parity_max_absolute_error=max_score_gap,
        additive_margin_reconstruction_max_absolute_error=max_reconstruction_gap,
        common_feature_decision_contribution_max=candidate_common_max, populations=summaries,
        fits=0,image_forwards=0,real_reference_reads=0,changed_predictions=0,
        caution='All original real outcomes were already known. No selected feature ablation or threshold is evaluated/promoted by this audit. Group attribution is algebraic, not causal.',
        wall_seconds=time.monotonic()-start)
    save(DOC/'SELECTOR_ATTRIBUTION.json',result)
    print('COMPLETE',dict(score_error=max_score_gap,reconstruction_error=max_reconstruction_gap,seconds=result['wall_seconds']),flush=True)


if __name__ == '__main__':
    main()

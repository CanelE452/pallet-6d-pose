"""Bounded RGB PoseFix trainability screen; historical artifacts are read-only."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_large_error_refiner_v1 import run as R
from scripts.research.pallet_sensors_submission_v1.prior_model import PoseFixPallet9, expectation
from scripts.research.pallet_sensors_submission_v1.posefix_contract_math import axis_aligned_crop_matrix, transform_points
E, F = R.E, R.F
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments/pallet_posefix_large_error_v1'
RAW = ROOT / 'data/pallet/results/pallet_posefix_large_error_v1'
OUT = ROOT / 'outputs/pallet_posefix_large_error_v1'
PRIOR_DOC = ROOT / '_docs/experiments/pallet_sensors_submission_v1'
PRIOR_RAW = ROOT / 'data/pallet/results/pallet_sensors_submission_v1'
PRIOR_CK = PRIOR_RAW / 'runs/PRIOR1/last.pt'
MEAN = np.array([123.68,116.78,103.94], np.float32)


def write(path, value):
    path = Path(path).resolve()
    assert any(path.is_relative_to(p) for p in (DOC, RAW, OUT)), path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.pending')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def freeze(path, value):
    if path.exists(): assert E.read(path) == value, ('Immutable artifact differs', path)
    else: write(path, value)


def source_bindings():
    return {str(p.relative_to(ROOT)): E.bound(p) for p in (
        PRIOR_CK, PRIOR_DOC/'PRIOR_SELECTION.json', R.DOC/'SPLIT.json', R.GREEN,
        R.RAW/'PREDICTIONS.json', R.RAW/'PER_FRAME_METRICS.json',
        E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json',
        HERE.parent/'pallet_sensors_submission_v1/prior_model.py',
        HERE.parent/'pallet_sensors_submission_v1/posefix_contract_math.py',
        HERE.parent/'pallet_dim_conditioned_p_v1/eval_math.py',
        HERE.parent/'pallet_large_error_refiner_v1/evaluate.py',
        HERE.parent/'pallet_large_error_refiner_v1/run.py')}


def verify_bindings(bindings):
    for b in bindings.values(): F.verify(b)


def load_model(device='cuda'):
    selected = E.read(PRIOR_DOC/'PRIOR_SELECTION.json')
    assert E.sha(PRIOR_CK) == selected['checkpoints']['1']
    ck = torch.load(PRIOR_CK, map_location='cpu', weights_only=False)
    assert ck['complete'] and ck['step'] == 6000
    model = PoseFixPallet9()
    model.load_state_dict(ck['model_state_dict'])
    return model.to(device).eval().requires_grad_(False)


def load_finetuned(device='cuda'):
    fit = E.read(DOC/'FIT.json'); F.verify(fit['checkpoint'])
    assert fit['complete'] and fit['step'] == 300 and fit['trainability_pass']
    ck = torch.load(ROOT/fit['checkpoint']['path'], map_location='cpu', weights_only=False)
    assert ck['protocol_sha256'] == E.sha(DOC/'PROTOCOL.json')
    model = PoseFixPallet9(); model.load_state_dict(ck['model_state_dict'])
    return model.to(device).eval().requires_grad_(False)


def selected(pred):
    idx = pred['selected_index']
    return None if idx is None else pred['candidates'][idx]


def prepare_input(bgr, pred):
    c = selected(pred)
    if c is None or c.get('keypoints_xy') is None: return None
    points = np.array(c['keypoints_xy'], dtype=np.float64)
    box = np.array(c['box_xyxy'], dtype=np.float64)
    if not np.isfinite(box).all() or not (box[2:] > box[:2]).all(): return None
    matrix = axis_aligned_crop_matrix(box)
    valid = np.isfinite(points).all(-1) & ~(points == -1).all(-1)
    rgb = cv2.warpAffine(bgr, matrix[:2], (288,384), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)[:,:,::-1].astype(np.float32) - MEAN
    return dict(rgb=rgb.transpose(2,0,1),
        points=transform_points(np.where(valid[:,None],points,0),matrix).astype(np.float32),
        valid=valid, matrix=matrix, original_points=points, box=box,
        bbox_diagonal=float(np.linalg.norm(box[2:]-box[:2])))


def cap_prediction(base, raw_pred, cap_fraction, hw):
    result = copy.deepcopy(raw_pred)
    a, b = selected(base), selected(result)
    if a is None or b is None or a.get('keypoints_xy') is None: return result
    p, q = np.array(a['keypoints_xy'],float), np.array(b['keypoints_xy'],float)
    valid = np.isfinite(p).all(-1) & ~(p == -1).all(-1); valid[8] = False
    assert np.isfinite(q[valid]).all()
    delta = q - p
    if cap_fraction is not None:
        assert cap_fraction >= 0
        cap = cap_fraction * math.hypot(*hw)
        delta *= np.minimum(1, cap/np.maximum(np.linalg.norm(delta,axis=-1),1e-12))[:,None]
    corrected = p.copy(); corrected[valid] += delta[valid]
    b['keypoints_xy'] = corrected.tolist()
    return result


@torch.no_grad()
def predict(model, bgr, prediction, cap_fraction=None):
    result = copy.deepcopy(prediction)
    inp = prepare_input(bgr, prediction)
    if inp is None: return result
    device = next(model.parameters()).device
    args = [torch.as_tensor(inp[k],device=device)[None] for k in ('rgb','points','valid')]
    q = expectation(model(*args))[0].cpu().numpy()
    raw = transform_points(q, np.linalg.inv(inp['matrix']))
    raw[~inp['valid']] = inp['original_points'][~inp['valid']]
    raw[8] = inp['original_points'][8]
    selected(result)['keypoints_xy'] = raw.tolist()
    return cap_prediction(prediction,result,cap_fraction,bgr.shape[:2])


def prepare():
    R.verify(); F.checked_lock()
    protocol = dict(experiment='pallet_posefix_large_error_v1',
        purpose='Existing PoseFix reevaluation plus tiny real supervised large-error trainability screen; not a paper contribution claim',
        prior_history='PoseFix-derived official-TF1-parity ResNet152 pallet9; already trained synthetic-only seed1/2/3 x 6000; reuse seed1',
        new_model='Same validated PoseFixPallet9 RGB + initial nine-point Gaussian maps; no detector updates',
        inputs='RGB and frozen R0 points/predicted box ONLY; no depth, CAD, 3D rendering, PnP, dimensions, GT or symmetry label at inference',
        seed=1, steps=300, batch=8, microbatch=2, checkpoint='last300 only; never selected on eval',
        initialization='existing PRIOR1 last6000; all model parameters trainable; BN running mean/variance frozen',
        optimizer=dict(name='TFAdam',lr=1e-4,graph_L2_half_coefficient=0.5e-5,gradient_clip=None),
        loss='unchanged PoseFix heatmap cross entropy + coordinate L1 + graph L2; masks restrict supervision to manually clicked finite in-frame corners within R0 crop',
        training='Only existing 9 real TRAIN images with 38 verified manual corners; all candidate recording groups reserved from evaluation',
        input_corruption='Per image: probability .5 unchanged R0 input; .5 replace each manual-valid supervised corner by GT plus independent uniform angle and .05-.15 predicted-box-diagonal radius in image pixels. Other corners/centroid stay R0. No human swap/inversion prior, no label changes.',
        training_rng=6401, probe_rng=6402, probe_repeats_per_image=8,
        sanity='step0 vs last300 on same training RGB: original R0 inputs plus 8 independently drawn fixed corruption cases each image; NOT unseen-image validation',
        sanity_pass='noisy manual-supported corners mean error <=10 original-image px AND recovery of noisy-input >20px to <=10px >=.50 AND mean error <=.50 * noisy-input mean error',
        followup='If sanity passes, evaluate final on untouched recording-separated DEV72 and GREEN150 manual. If not, stop this bounded screen; no hyperparameter search.',
        evaluation='Same 222 cached R0 detections and approved whole-object symmetry scorer as previous large-error test. Report raw and historical .01 image-diagonal cap without choosing a winner; baseline R0 and N2.',
        existing_before_train_evaluation=True, gt_used_to_filter_eval=False, geometry_filter=False,
        evaluation_reused=True, independent_confirmation=False, causality='Warm-start and supervision change together; not a noise-only causal ablation',
        no_original_weights_labels_paper_changes=True, auto_promote=False, gpu_thermal_limit_C=80,
        system_changes=False, sources=source_bindings(),
        code={p.name:E.bound(p) for p in (HERE/'core.py',HERE/'train.py',HERE/'test_core.py',HERE/'evaluate.py')})
    freeze(DOC/'PROTOCOL.json',protocol)
    print('PREPARED', DOC, flush=True)

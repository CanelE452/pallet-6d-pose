"""Two fixed image-only corrections; labels and pose references are not inputs."""
from collections import Counter
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scripts.research.multiteacher_corner_distill_v1 import gate_b_corner_evidence as G

ROOT = Path(__file__).resolve().parents[3]
METHODS = ('SUBPIX', 'CVRANK')
ARMS = ('SUBPIX_NATIVE', 'SUBPIX_CAP1', 'CVRANK_NATIVE', 'CVRANK_CAP1')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _configuration():
    # These imports contain definitions/constants and sys.path insertion only.
    # Their main() entrypoints, GT loaders and model constructors are never run.
    old_dir = ROOT / 'scripts/research/multiteacher_corner_distill_v1'
    assert Path(G.__file__).resolve() == old_dir / 'gate_b_corner_evidence.py'
    assert Path(G.M.__file__).resolve() == old_dir / 'mtcd_common.py'
    lock_path = Path(G.M.METHOD_LOCK_PATH)
    lock = json.loads(lock_path.read_text()); gb = lock['gate_b']
    assert gb['search_radius_px'] == 12
    calibration_path = G.M.AUDIT / 'SOURCE_CALIBRATION.json'
    calibration = json.loads(calibration_path.read_text())
    cfg = gb['candidate_generators_frozen']
    families = {'shi_tomasi':'shi_tomasi', 'harris':'harris',
                'lsd':'lsd_intersections', 'junction':'gradient_junction'}
    choices = {}
    for family, key in families.items():
        rows = calibration['candidate_grid'][family]
        best = max(rows, key=lambda row:(row['coverage_5px'], -row['candidates_per_patch_median']))
        assert best['config'] == cfg[key], 'Frozen generator differs from recorded source selection'
        choices[family] = dict(selected_config=copy.deepcopy(best['config']),
            grid_size=len(rows), source_patches=best['n_patches'], coverage_5px=best['coverage_5px'])
    assert calibration['radius_px'] == 12
    paths = [old_dir/name for name in ('gate_b_corner_evidence.py','calibrate_on_source.py',
                                     'mtcd_common.py','mtcd_teachers.py')]
    return dict(SUBPIX=dict(winSize=[5,5], zeroZone=[-1,-1],
            criteria=dict(type=int(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT),
                          maxCount=40, epsilon=.001), initial_and_refined_arithmetic='OpenCV float32; returned native array float64',
            minimum_image_extent=15, initial_inside_required=True,
            fallback='unsupported/missing/outside initial, too-small image, OpenCV exception, nonfinite or outside refined output'),
        CVRANK=dict(radius_px=12, patch_geometry='clipped x/y half-width 12 square; no Euclidean radius gate',
            candidate_generators=copy.deepcopy(cfg), selector='unchanged collect/selector_score/cuboid_edge_directions',
            score='equal mean of distance(low better), response/cross-angle/edge-alignment(high better) ranks',
            rank_ties='original np.argsort default ordering; first np.argmax of combined scores',
            finite_outside_selected_candidates='retained as in original selector; no new candidate-bound gate',
            missing_direction_neighbors='prediction-support false, missing sentinel or nonfinite become NaN for edge directions only'),
        CAP1=dict(fraction_raw_image_diagonal=.01, arithmetic='float64; exact zero retained',
            applied_to='same algorithm native result; no additional image/candidate computation'),
        opencv_version=cv2.__version__, method_lock=dict(path=str(lock_path),sha256=_sha(lock_path),
            date=lock['date'],status=lock['status']),
        source_calibration=dict(path=str(calibration_path),sha256=_sha(calibration_path),
            population=calibration['population'],how_chosen=gb['how_chosen'],
            residual_count=calibration['r0_coarse_residual']['n'],jitter_sigma_px=calibration['jitter_sigma_px'],
            frozen_choices=choices,all_frozen_configs_match_source_recorded_optimum=True,
            separate_manual_override_evidence='none in the frozen lock/config and source-grid comparison; radius12 was instructed',
            additional_current_calibration=0,additional_current_grid_search=0),
        reused_code=[dict(path=str(path),sha256=_sha(path)) for path in paths],
        import_effects='legacy sys.path insertion and definitions/constants; no top-level file read/write, GT load, inference or training',
        inference_inputs=['gray','initial_points','prediction_support','method'],new_weight_training=0)


def method_configuration():
    """Return a fresh metadata copy, suitable for sealing before evaluation."""
    return copy.deepcopy(_configuration())


def _inputs(initial_points, prediction_support):
    initial = np.array(initial_points, dtype=np.float64, copy=True)
    support = np.array(prediction_support, dtype=bool, copy=True)
    if initial.shape != (9,2) or support.shape != (9,):
        raise ValueError('Expected initial_points[9,2] and prediction_support[9]')
    usable = support & np.isfinite(initial).all(-1) & ~(initial == -1).all(-1)
    return initial, support, usable


def cap_points(initial, native, width, height, support):
    """Clip native displacement once; preserve center and every missing slot."""
    initial, support, usable = _inputs(initial, support)
    native = np.array(native, dtype=np.float64, copy=True)
    if native.shape != (9,2) or not np.isfinite([width,height]).all() or width<=0 or height<=0:
        raise ValueError('Invalid native shape or raw image dimensions')
    result = initial.copy(); indices = np.flatnonzero(usable[:8])
    if not np.isfinite(native[indices]).all():
        raise ValueError('Nonfinite correction for a supported initial corner')
    delta = native[indices]-initial[indices]
    lengths = np.linalg.norm(delta, axis=-1)
    factor = np.ones(len(indices), dtype=np.float64)
    moving = lengths>0
    factor[moving] = np.minimum(1., .01*np.hypot(width,height)/lengths[moving])
    result[indices] = initial[indices]+delta*factor[:,None]
    result[8] = initial[8]
    return result


def correct(gray, initial_points, prediction_support, method):
    """One native correction. CAP1 is a separate pure operation on this output.

    ``prediction_support`` is an inherited inference mask, never human visibility.
    The caller retains the box, confidence and object selection unchanged.
    """
    if method not in METHODS:
        raise ValueError('Native method must be SUBPIX or CVRANK')
    gray = np.asarray(gray)
    if gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
        raise ValueError('Use nonempty raw-resolution grayscale uint8 image')
    h,w = gray.shape; initial,support,usable = _inputs(initial_points,prediction_support)
    native = initial.copy(); config=_configuration(); rows=[]
    directions = initial.copy();directions[~usable] = np.nan
    calls = 0
    for k in range(8):
        row=dict(corner=k,prediction_support=bool(support[k]),attempted=False,
                 fallback_reason=None,exception=None,candidate_count=None,
                 selected_candidate_index=None,selected_family=None,selector_score=None)
        if not support[k]:reason='unsupported_prediction'
        elif not np.isfinite(initial[k]).all():reason='nonfinite_initial'
        elif (initial[k]==-1).all():reason='missing_initial_sentinel'
        else:reason=None
        if reason is not None:
            row['status']=reason
        elif method=='SUBPIX' and not (0<=initial[k,0]<w and 0<=initial[k,1]<h):
            row['status']='outside_initial';row['fallback_reason']='outside_initial'
        elif method=='SUBPIX' and min(h,w)<config['SUBPIX']['minimum_image_extent']:
            row['status']='image_too_small';row['fallback_reason']='image_too_small'
        else:
            row['attempted']=True;calls+=1
            try:
                if method=='SUBPIX':
                    refined=cv2.cornerSubPix(gray,initial[k].astype(np.float32).reshape(1,1,2).copy(),
                        (5,5),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
                    if refined is None or np.asarray(refined).size != 2:
                        raise ValueError('cornerSubPix returned no coordinate')
                    chosen=np.asarray(refined,dtype=np.float64).reshape(2)
                else:
                    cands,info=G.collect(gray,initial[k],12,config['CVRANK']['candidate_generators'])
                    row['candidate_count']=len(cands)
                    if not cands:
                        row['status']='no_candidate';row['fallback_reason']='no_candidate'
                        rows.append(row);continue
                    selected,score=G.selector_score(cands,info,initial[k],G.cuboid_edge_directions(directions,k))
                    chosen=np.asarray(selected['xy'],dtype=np.float64)
                    row['selected_candidate_index']=next(i for i,c in enumerate(cands) if c is selected)
                    row['selected_family']=selected['family'];row['selector_score']=float(score)
                    row['candidate_family_counts']=info.get('counts',{})
                if chosen.shape!=(2,) or not np.isfinite(chosen).all():
                    row['status']='nonfinite_refined';row['fallback_reason']='nonfinite_refined'
                elif method=='SUBPIX' and not (0<=chosen[0]<w and 0<=chosen[1]<h):
                    row['status']='outside_refined';row['fallback_reason']='outside_refined'
                else:
                    native[k]=chosen;row['status']='refined'
                    row['outside_refined_image']=not (0<=chosen[0]<w and 0<=chosen[1]<h)
            except Exception as error:
                row['status']='function_error';row['fallback_reason']='function_error'
                row['exception']=dict(type=type(error).__name__,message=str(error))
        rows.append(row)
    assert len(rows)==8
    assert np.array_equal(native[8],initial[8],equal_nan=True)
    assert np.array_equal(native[~usable],initial[~usable],equal_nan=True)
    movements=[]
    for row in rows:
        k=row['corner'];row['changed']=bool(not np.array_equal(native[k],initial[k],equal_nan=True))
        row['movement_px']=float(np.linalg.norm(native[k]-initial[k])) if usable[k] else None
        if row['movement_px'] is not None:movements.append(row['movement_px'])
    diag=dict(method=method,input_shape_hw=[h,w],corner_records=rows,
        changed_corners=sum(r['changed'] for r in rows),initial_maintained_corners=sum(not r['changed'] for r in rows),
        attempted_corners=sum(r['attempted'] for r in rows),algorithm_corner_calls=calls,
        fallback_counts=dict(Counter(r['fallback_reason'] for r in rows if r['fallback_reason'] is not None)),
        status_counts=dict(Counter(r['status'] for r in rows)),
        movement_mean_px=float(np.mean(movements)) if movements else None,
        movement_median_px=float(np.median(movements)) if movements else None,
        center_preserved=True,missing_prediction_slots_preserved=True,
        prediction_mask_used=True,GT_inputs=False,new_weight_training=0)
    return native,diag

"""C5 object-type tables from existing rows; size bins remain explicit NA."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_wd_info_fast_20261011 import common as C
from scripts.research.pallet_wd_info_fast_20261011.classifier import ARMS, NAMES, summarize


def subgroup(rows):
    out = summarize(rows)
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[row['id']].append(row)
    out['strict_success_seed_rows'] = int(sum(r['success'] for r in rows))
    out['strict_success_frame_equivalent_count'] = float(sum(np.mean([r['success'] for r in rr]) for rr in by_frame.values()))
    out['legacy_WD_confusion_seed_rows'] = int(sum(r['legacy_confusion'] for r in rows))
    out['legacy_WD_confusion_frame_equivalent_count'] = float(sum(np.mean([r['legacy_confusion'] for r in rr]) for rr in by_frame.values()))
    evaluable = [r for r in rows if r['axis_error'] is not None]
    out['direct_axis_error_seed_rows'] = int(sum(r['axis_error'] for r in evaluable))
    out['direct_axis_evaluable_seed_rows'] = len(evaluable)
    out['direct_axis_error_frame_equivalent_count'] = float(sum(np.mean([r['axis_error'] for r in rr if r['axis_error'] is not None]) for rr in by_frame.values() if any(r['axis_error'] is not None for r in rr)))
    out['UNCLEAR_frames'] = len({r['id'] for r in rows if r['axis_error'] is None})
    matrix = np.zeros((2, 2), dtype=int)
    brier = []
    for row in evaluable:
        guessed = int(row['selected_hypothesis'] == NAMES[0])
        target = guessed if row['axis_error'] == 0 else 1-guessed
        matrix[target, guessed] += 1
        if row['p_long'] is not None:
            brier.append((row['p_long']-target)**2)
    recalls = [matrix[i,i]/matrix[i].sum() for i in (0,1) if matrix[i].sum()]
    out['classification'] = dict(accuracy=1-out['axis_error_rate'] if out['axis_error_rate'] is not None else None,
                                balanced_accuracy=float(np.mean(recalls)) if len(recalls)==2 else None,
                                confusion_matrix_seed_rows_true_short_long=matrix.tolist(),
                                Brier=float(np.mean(brier)) if brier else None,
                                seed_rows_not_independent_frames=True,
                                labels_reused_from_already_scored_axis_errors=True)
    return out


def run():
    begun = time.monotonic(); C.lock_ready()
    contract = C.read(C.DOC/'INPUT_CONTRACT.json')
    allowed = set(contract['allowed_ids']); protected = set(contract['excluded_protected_ids'])
    metadata_path = C.SOURCE/'_docs/experiments/pallet_dim_conditioned_p_v1/DEV_CACHE_COMPLETE.json'
    registry_path = C.SOURCE/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'
    # IDs are checked before the runtime object_type field is accessed. This
    # does not open any per-ID image/cache/checkpoint or reference pose.
    types = {r['id']: r['object_type'] for r in C.read(metadata_path)['records'] if r['id'] in allowed}
    assert set(types)==allowed and not set(types)&protected
    registry = {r['object_type']: r for r in C.read(registry_path)['objects'] if r['object_type'] in set(types.values())}
    assert set(registry)==set(types.values())
    rows_path=C.DOC/'C_FRAME_METRICS.jsonl.gz'; rows=list(C.rows(rows_path))
    assert len(rows)==5190 and set(r['id'] for r in rows)==allowed
    assert set(r['method'] for r in rows)==set((*ARMS,'TRAIN_PRIOR','OFFICIAL'))
    tables={}
    for object_type in sorted(registry):
        selected=[r for r in rows if types[r['id']]==object_type]
        tables[object_type]=dict(frames=len({r['id'] for r in selected}),
                                registered_physical_dimensions_XYZ_m=registry[object_type]['physical_dimensions_m'],
                                methods={name: subgroup([r for r in selected if r['method']==name]) for name in (*ARMS,'TRAIN_PRIOR','OFFICIAL')})
    # No prelocked small/medium/large boundaries exist in the input/PLAN or
    # historical feature recipe. Record the available runtime measurement;
    # do not choose bins after observing C results.
    feature_path=C.PRIVATE/'C_REAL_FEATURES.npz'
    with np.load(feature_path,allow_pickle=False) as features:
        frame_ids=features['ids']; seeds=features['seeds']; geometry=features['G']
    feature_contract=C.read(C.DOC/'C_GEOMETRY_RECEIPT.json')
    column=feature_contract['geometry_names'].index('bbox_area_image_fraction')
    assert geometry.shape==(519,2,94) and set(frame_ids.tolist())==allowed
    area=geometry[:,0,column]
    assert np.array_equal(area,geometry[:,1,column])
    per_frame=defaultdict(list)
    for fid,seed,value in zip(frame_ids,seeds,area):
        per_frame[str(fid)].append(float(value))
    assert all(len(values)==3 and np.ptp(values)==0 for values in per_frame.values())
    values=np.array([values[0] for _,values in sorted(per_frame.items())])
    size=dict(status='NA_NO_PRELOCKED_BIN_CONTRACT',
              reason='Runtime predicted-box size exists, but no fixed size-bin boundaries were declared before results in PLAN_LOCK/INPUT_CONTRACT or the reused feature contract. No post-result bins are introduced.',
              runtime_size_available=True,definition='predicted native bbox area / raw image area; original94feature bbox_area_image_fraction',
              quantity_is_GT_free=True,size_type='apparent predicted image size, not object metric size or reference distance',
              frames=173,finite_frames=int(np.isfinite(values).sum()),
              frame_quantiles_only_descriptive={'min':float(np.min(values)),'P10':float(np.quantile(values,.1)),
                                                'median':float(np.median(values)),'P90':float(np.quantile(values,.9)),
                                                'max':float(np.max(values))},
              unclamped_max_can_exceed_one=True,
              same_runtime_size_all_three_seeds=True,
              source_evidence=[dict(owner='new_worktree',path='scripts/research/pallet_wd_info_fast_20261011/classifier.py',line=543,meaning='Prior C subgroup loop has grade/session/material/margin/elevation, no size boundaries'),
                               dict(owner='published_source',path='scripts/research/pallet_selector_recovery_v1/feature_contract.py',line=14,meaning='Declares continuous bbox_area_image_fraction; no size bins'),
                               dict(owner='new_worktree',path='_docs/experiments/pallet_wd_info_fast_20261011/PLAN_LOCK.json',meaning='No size-bin contract locked')],
              methods={name:dict(status='NA',reason='No prelocked size-bin comparison',size_bins=None,metrics=None) for name in (*ARMS,'TRAIN_PRIOR','OFFICIAL')})
    C.write(C.DOC/'C5_SUBGROUPS_SUPPLEMENT.json',dict(status='OBJECT_TYPE_DONE_SIZE_EXPLICIT_NA',
        code=C.binding(Path(__file__)),prior_C_RESULTS_preserved=C.binding(C.DOC/'C_RESULTS.json'),
        source_rows=C.binding(rows_path),runtime_metadata=C.binding(metadata_path),registry=C.binding(registry_path),
        object_type_definition='Existing externally known object_type -> canonical registry; evaluated only, not a new predictor input',
        object_type_counts=dict(Counter(types.values())),object_type_tables=tables,size_subgroups=size,
        aggregation='Mean fixed three seeds within frame first; continuous SD(ddof=1), variance, median/P90/P95 across frames. Classification seed confusion counts are descriptive, not independent frames.',
        full_allowed_frames=173,fixed_RECT245_intersection_frames=110,
        no_new_fit=True,new_model_forwards=0,new_OpenCV_calls=0,new_raw_image_reads=0,
        new_reference_pose_or_annotation_reads=0,protected_per_ID_data_reads=0,
        thresholds_or_bins_selected_from_REAL_results=False,exploratory_only=True,
        elapsed_seconds=time.monotonic()-begun))
    print('C5_SUBGROUPS_DONE',dict(Counter(types.values())),'SIZE_NA',round(time.monotonic()-begun,2),flush=True)


if __name__=='__main__':
    run()

"""Frozen, single-setting corner-gate diagnostic. No model training/promotion."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DOC=ROOT/'_docs/experiments/pallet_posefix_corner_gate_v1'
RAW=ROOT/'data/pallet/results/pallet_posefix_corner_gate_v1'
OUT=ROOT/'outputs/pallet_posefix_corner_gate_v1'
OLD=ROOT/'data/pallet/results/pallet_posefix_replay_v1'
OLD_DOC=ROOT/'_docs/experiments/pallet_posefix_replay_v1'


def read(path):return json.loads(Path(path).read_text())


def bound(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def verify(binding):
    actual=bound(ROOT/binding['path'])
    assert actual['sha256']==binding['sha256'],binding['path']
    if 'bytes' in binding:assert actual['bytes']==binding['bytes']


def write(path,obj):
    path=Path(path).resolve()
    assert any(path.is_relative_to(p) for p in (DOC,RAW,OUT))
    path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists():assert read(path)==obj,('Never overwrite a completed artifact',str(path))
    else:path.write_text(content)


def prepare():
    from scripts.research.pallet_posefix_replay_v1 import core as N
    N.verify()
    sources=[OLD/'PREDICTIONS.json',OLD/'PER_FRAME_METRICS.json',OLD_DOC/'FIT.json',
        ROOT/'_docs/experiments/pallet_large_error_refiner_v1/SPLIT.json',
        ROOT/'_docs/paper/final_dimension_v1/green150_saved_labels_v1/DATASET_SNAPSHOT.json',
        ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json',
        ROOT/'_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json',
        ROOT/'data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json',
        ROOT/'scripts/self_training_yolo/pseudo_label_filters.py']
    protocol=dict(experiment='pallet_posefix_corner_gate_v1',
        purpose='One bounded no-training comparison: frozen N2 fallback vs frozen Replay, whole-frame gate vs pairwise corner gate',
        requested_by_user=True,prior_failure_cases_informed_design=True,
        independent_confirmation=False,evaluation_reused=True,threshold_tuning_on_evaluation=False,
        thresholds=dict(box_conf=.85,kp_conf=.5,min_valid_corners=6,removal=.05,flip=.05,
                        expectation_to_mode=.05,normal_flipped_box_iou=.5),
        threshold_status='0.05 transferred unchanged from existing normalized filter as an exploratory setting, NOT calibrated per-corner or mean-mode confidence; no sweep or rescue run',
        arms=['A_N2','REPLAY','LEGACY_FRAME','CORNER_GEOMETRY_PAIR','CORNER_FULL_PAIR'],
        primary_new_arm='CORNER_FULL_PAIR',
        pair_indices=[[0,3],[1,2],[4,7],[5,6]],
        whole_frame='Exact old confidence>=.85 and >=6 corners conf>=.5, median removal<=.05 AND median flip<=.05. No reprojection veto; frame pass copies all8 Replay corners, otherwise allN2.',
        geometry_pair='Choose ONE registry W/D hypothesis by smallest finite median removal; both endpoints need valid/conf>=.5 and individual removal/D<=.05; globalconfidence/6valid required.',
        full_pair='geometry_pair AND each endpoint unflipped prediction distance/D<=.05 AND expectation-to-heatmap-mode distance/D<=.05. Flip coordinates/index restored by fixed [1,0,3,2,5,4,7,6,8]. Flipped selected detection requires mapped box IoU>=.5 with normal R0.',
        heatmap_note='Mode is only an uncertainty signal. Output remains the original expectation; no argmax coordinate replacement.',
        normalization='Original Replay chosen-hypothesis projected diagonal D. Pin chosen hypothesis and D for mixed rechecks; no GT dimensions/pose.',
        mixed_recheck='After mixing N2/Replay, recalculate per-corner removal on same hypothesis/D; revoke accepted pairs with either endpoint above.05 or not finite. Monotone rejection <=4 pair removals; rejected pairs never restored. Baseline-only corners not required to become accurate.',
        fallback='Exact frozen N2 corner coordinates, no image drops. Centroid/boxes/scores/confidences/candidate order remain unchanged.',
        populations=dict(DEV72='72 non-green plastic rectangle, legacy unknown-origin target; 555 corners',GREEN150_MANUAL='150 squaregreen, manual-only681corners including fixed unmatched penalties'),
        inference='Frozen Replay original RGB + cachedR0, plus actual horizontal-flipped RGB -> frozenR0 -> sameReplay. Original8corner parity against savedReplay atol.01px. No evaluationannotationread in infer/gate stages.',
        runtime='CPU4threads due unrelated GPU job; preserve foreignjob/RustDesk; no driver/power/reboot changes',
        benchmark='First ordered image only for runtime/parity, no GT metrics/parameter changes; separate from fullcache',
        success_interpretation='Report ALL fixed arms and preserved/blocked benefits, prevented/remaining harms, PCK/mean/median/P90 on full denominators. No automatic best-arm promotion.',
        model_training=False,pseudo_labels_generated=False,final_model_modified=False,
        sources=[bound(p) for p in sources],checkpoint=read(OLD_DOC/'FIT.json')['checkpoint'])
    verify(protocol['checkpoint']);write(DOC/'PROTOCOL.json',protocol)
    print(DOC/'PROTOCOL.json')


def lock_code():
    p=read(DOC/'PROTOCOL.json')
    for b in p['sources']+[p['checkpoint']]:verify(b)
    paths=[HERE/name for name in ('protocol.py','gates.py','test_gates.py','infer.py','evaluate.py')]
    payload=dict(protocol=bound(DOC/'PROTOCOL.json'),code=[bound(p) for p in paths])
    if (DOC/'RUNTIME_OVERRIDE_GPU.json').exists():
        payload['runtime_override']=bound(DOC/'RUNTIME_OVERRIDE_GPU.json')
    write(DOC/'CODE_LOCK.json',payload);return payload


if __name__=='__main__':prepare()

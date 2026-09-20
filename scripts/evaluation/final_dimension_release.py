"""Frozen N2 paper handoff. No training, tuning, or automatic dataset admission."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / '_docs/paper/final_dimension_v1'
DCP = ROOT / '_docs/experiments/pallet_dim_conditioned_p_v1'
REVIEW = ROOT / 'outputs/green_paper_review_20260918'
RAW = ROOT / 'data/pallet/results/final_dimension_v1'
ARM = 'N2_DIM_ONLY'
ARMS = ('N0_BASE_REPLAY', ARM)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def verify(b):
    if sha(ROOT / b['path']) != b['sha256']:
        raise ValueError('Frozen input changed: ' + b['path'])


def freeze(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if read(path) != payload:
            raise ValueError(f'Immutable output already exists and differs: {path}')
    else:
        with path.open('x') as handle:
            handle.write(text)


def setup():
    sys.path.insert(0, str(ROOT / 'scripts/research/pallet_dim_conditioned_p_v1'))
    import dcp_env
    return dcp_env


def lock_model():
    import torch
    E = setup()
    from refiner import model
    fits = read(DCP / 'PAPER_TRAINING_COMPLETE.json')
    entries = []
    for fit in fits['runs']:
        if fit['arm'] not in ARMS:
            continue
        b = fit['checkpoint']; verify(b)
        ck = torch.load(ROOT / b['path'], map_location='cpu', weights_only=False)
        assert ck['complete'] and not ck['smoke'] and ck['step'] == 6000
        assert ck['baseline_checkpoint_sha256'] == E.R0_SHA
        head = model(fit['arm'], ck['config'])
        head.load_state_dict(ck['model_state_dict'], strict=True)
        assert all(torch.isfinite(v).all() for v in head.state_dict().values())
        entries.append(dict(arm=fit['arm'], seed=fit['seed'], checkpoint=b,
                            steps=ck['step'], parameters=sum(p.numel() for p in head.parameters())))
    assert len(entries) == 6 and sha(E.R0) == E.R0_SHA
    calibration = read(DCP / 'CALIBRATION_AND_SELECTION.json')
    source_paths = [DCP / p for p in (
        'DIM_NORMALIZATION_LOCK.json', 'CALIBRATION_AND_SELECTION.json',
        'TRAIN_PROTOCOL_LOCK.json', 'SOURCE_BINDINGS.json', 'PAPER_TRAINING_COMPLETE.json')]
    source_paths += [ROOT / 'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json',
        E.SYM_DOC / 'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json']
    source_paths += list((ROOT / 'scripts/research/pallet_dim_conditioned_p_v1').glob('*.py'))
    source_paths += [
        ROOT / 'scripts/research/pallet_line_pose_v1/features.py',
        ROOT / 'scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py',
        ROOT / 'scripts/research/pallet_final_ml_contribution_test_v1/point_inference.py']
    source_paths.append(Path(__file__))
    lock = dict(schema='pallet_final_dimension_v1', architecture=ARM, representative_seed=1,
        seed_rule='Fixed seed 1; not an accuracy-ranked choice. Historical DEV already seen.',
        decision_date='2026-09-19', deadline='2026-09-21', backbone=binding(E.R0), heads=entries,
        temperatures={f'{a}_seed{s}': calibration['temperatures'][f'{a}_seed{s}']['temperature'] for a in ARMS for s in (1,2,3)},
        decode_rule=calibration['rule'], bindings=[binding(p) for p in sorted(set(source_paths))],
        retraining_allowed=False, result_driven_selection_allowed=False,
        explicit_symmetry_code_input=False, DHT_inference=False,
        metadata='Externally known canonical W,D,H metres; no GT pose/phase or camera-facing swap',
        additional_evaluation_status='WAITING_FOR_USER_ANNOTATION_COMPLETION',
        independent_confirmation_claim=False)
    freeze(DOC / 'MODEL_LOCK.json', lock)
    print('MODEL_LOCK_VERIFIED: six heads, R0, normalization, temperatures, code; seed1 representative')


def checked_lock():
    lock = read(DOC / 'MODEL_LOCK.json')
    for b in [lock['backbone'], *lock['bindings'], *(r['checkpoint'] for r in lock['heads'])]:
        verify(b)
    return lock


def review_rows():
    """No prediction access; validation only. Does not freeze mutable annotations."""
    import numpy as np
    sessions = read(REVIEW / 'full_sessions_manifest.json')['records']
    records, issues = [], []
    for session in sessions:
        name = session['session']
        rgb = ROOT / session['source'] / 'rgb'
        annroot = REVIEW / 'full_session_annotations' / (name + '_manual_gt')
        for path in sorted(annroot.glob('*.json')):
            doc = read(path); obj = doc['objects'][0]
            if obj.get('split') != 'eval':
                continue
            row_id = name + '__' + path.stem
            try:
                assert doc.get('object_type', obj.get('object_type')) == 'plastic_standard_110x110x15'
                entries = obj['keypoint_annotations']; assert len(entries) == 9
                known = [e for e in entries[:8] if e.get('xy') is not None and e.get('visibility', 0) != 0]
                assert len(known) >= 4, 'fewer than four known corners'
                assert all(len(e['xy']) == 2 and np.isfinite(e['xy']).all() for e in known)
                images = [rgb / (path.stem + ext) for ext in ('.png', '.jpg', '.jpeg') if (rgb / (path.stem + ext)).exists()]
                assert len(images) == 1, 'missing/ambiguous image'
                camera = ROOT / session['source'] / 'cam_K.txt'
                k = np.loadtxt(camera).reshape(3,3)
                intr = doc['camera_data']['intrinsics']
                a = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]])
                assert np.allclose(k,a,rtol=0,atol=1e-5), 'camera intrinsics mismatch'
                old = ROOT / 'challenge/data/01_real/live_capture_gt' / (name + '_manual_gt') / path.name
                records.append(dict(id=row_id, session=name, image=binding(images[0]), annotation=binding(path),
                    camera=binding(camera), object_type='plastic_standard_110x110x15',
                    canonical_WDH_m=[1.1,1.1,.15], known_corners=len(known),
                    annotation_sources=dict(Counter(e.get('source','unknown') for e in entries[:8])),
                    in_historical_851=old.exists(),
                    related_session_in_prior_square_experiments=bool(session['annotation_count']),
                    independent_test_eligible='NOT_ESTABLISHED'))
            except (AssertionError, KeyError, ValueError, TypeError) as exc:
                issues.append(dict(id=row_id, issue=str(exc)))
    hashes = [r['image']['sha256'] for r in records]
    if len(set(hashes)) != len(hashes):
        issues.append(dict(issue='Duplicate image SHA; resolve before freeze, no silent exclusion'))
    return dict(records=records, issues=issues, count=len(records), sessions=len({r['session'] for r in records}),
        status='MUTABLE_REVIEW_NOT_A_TEST_FREEZE', predictions_accessed=False)


def freeze_eval(confirmed):
    if not confirmed:
        raise ValueError('Requires explicit --confirm-annotation-complete; never freeze ongoing work')
    checked_lock(); data = review_rows()
    if data['issues'] or not data['records']:
        raise ValueError(json.dumps(data['issues']) or 'No EVAL frames')
    data.update(status='FROZEN_ADDITIONAL_DEV_NOT_INDEPENDENT_CONFIRMATION',
        model_lock=binding(DOC / 'MODEL_LOCK.json'), frozen_at=datetime.now(timezone.utc).isoformat(),
        primary='Mean frame normalized corner8 error E_sym with full missing penalty',
        comparisons=['N2 vs R0','N2 vs paired N0'],
        selection='Human annotation selection before new predictions; no model-error filtering',
        metric_contract='known visibility !=0; whole-object C4; bbox of known in-image nine points; IoU>=0.5 match; image-diagonal missing penalty',
        physical_6D_metrology=False)
    freeze(DOC / 'GREEN_EVAL_LOCK.json', data)


def locked_eval():
    checked_lock(); data=read(DOC / 'GREEN_EVAL_LOCK.json'); verify(data['model_lock'])
    for r in data['records']:
        for key in ('image','annotation','camera'):
            verify(r[key])
    return data


def infer():
    import cv2
    import torch
    torch.set_num_threads(4)
    lock=checked_lock(); dataset=locked_eval(); E=setup()
    from inference import load_head, predict_captured, serial
    E.gpu()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not accessible; check execution permissions, never silently switch device')
    heads={(r['arm'],r['seed']):load_head(r['arm'],r['seed'])[0] for r in lock['heads']}
    extractor=E.old('features').FrozenYoloFeatures(E.R0)
    norm=read(DCP / 'DIM_NORMALIZATION_LOCK.json')
    result={name:[] for name in ['R0',*(f'{a}_seed{s}' for a,s in heads)]}
    try:
        for i,r in enumerate(dataset['records']):
            im=cv2.imread(str(ROOT / r['image']['path']))
            if im is None: raise ValueError(r['id'])
            cap=extractor.predict(im)
            result['R0'].append(dict(id=r['id'],raw_hw=list(im.shape[:2]),candidates=serial(cap['candidates']),selected_index=cap['selected_index']))
            for (arm,seed),head in heads.items():
                name=f'{arm}_seed{seed}'
                # order=4 only validates allowed metadata; N2/N0 have no symmetry-code input.
                pred,_=predict_captured(head,arm,cap,r['canonical_WDH_m'],4,
                    lock['temperatures'][name],lock['decode_rule'],im.shape[:2],norm)
                result[name].append(dict(id=r['id'],raw_hw=list(im.shape[:2]),**pred))
            if i%20==0: E.gpu(); print('INFERENCE',i+1,len(dataset['records']),flush=True)
    finally:
        extractor.close()
    freeze(RAW / 'GREEN_PREDICTIONS.json',dict(complete=True,GT_input=False,
        dataset=binding(DOC / 'GREEN_EVAL_LOCK.json'),model=binding(DOC / 'MODEL_LOCK.json'),predictions=result))


def score():
    import numpy as np
    dataset=locked_eval(); payload=read(RAW / 'GREEN_PREDICTIONS.json')
    verify(payload['dataset']); verify(payload['model']); E=setup()
    from eval_math import measure, summary, damage, contrast
    from dev_evaluate import iou
    groups=read(E.SYM_DOC / 'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']
    perms=next(g['permutations'] for g in groups if g['object_type']=='plastic_standard_110x110x15')
    rows={}
    for name,predictions in payload['predictions'].items():
        scored=[]
        assert len(predictions)==len(dataset['records'])
        for r,p in zip(dataset['records'],predictions):
            assert r['id']==p['id']
            ann=read(ROOT / r['annotation']['path'])['objects'][0]['keypoint_annotations']
            gt=np.array([e['xy'] if e.get('xy') is not None else [np.nan,np.nan] for e in ann],float)
            valid=np.array([e.get('visibility',0)!=0 and e.get('xy') is not None for e in ann])
            h,w=p['raw_hw']; inside=valid & np.isfinite(gt).all(1) & (gt[:,0]>=0)&(gt[:,0]<w)&(gt[:,1]>=0)&(gt[:,1]<h)
            if not inside.any(): raise ValueError('No in-image points for GT box: '+r['id'])
            box=np.r_[gt[inside].min(0),gt[inside].max(0)]
            idx=p['selected_index']; candidate=None if idx is None else p['candidates'][idx]
            matched=candidate is not None and iou(candidate['box_xyxy'],box)>=.5
            points=np.full((9,2),np.nan) if candidate is None else candidate['keypoints_xy']
            m=measure(points,gt,valid,perms,(h,w),matched,idx is not None)
            scored.append(dict(id=r['id'],session=r['session'],**m))
        rows[name]=scored
    differences={}
    clusters=[r['session'] for r in dataset['records']]
    for reference in ('R0','N0_BASE_REPLAY'):
        a=[[r['E_sym'] for r in rows[f'{ARM}_seed{s}']] for s in (1,2,3)]
        b=[[r['E_sym'] for r in rows['R0' if reference=='R0' else f'{reference}_seed{s}']] for s in (1,2,3)]
        differences[reference]=contrast(a,b,clusters) if len(set(clusters))>=2 else dict(status='INSUFFICIENT_SESSIONS_FOR_CLUSTER_CI')
    freeze(RAW / 'GREEN_METRICS.json',dict(rows=rows,summary={k:summary(v) for k,v in rows.items()},
        paired_session_contrasts=differences,damage={str(s):damage(rows['R0'],rows[f'{ARM}_seed{s}']) for s in (1,2,3)},
        predictions=binding(RAW / 'GREEN_PREDICTIONS.json'),independent_confirmation=False,
        six_D_measurement='NOT_RUN_NO_INDEPENDENT_METROLOGY',no_latency_claim=True))
    print('GREEN_2D_EVALUATION_COMPLETE; inspect tails and sources, no automatic positive verdict')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['lock-model','preflight','review-status','freeze-eval','infer','score'])
    p.add_argument('--confirm-annotation-complete',action='store_true')
    a=p.parse_args()
    if a.stage=='lock-model': lock_model()
    elif a.stage=='preflight': print(json.dumps(checked_lock(),ensure_ascii=False,indent=2))
    elif a.stage=='review-status': print(json.dumps(review_rows(),ensure_ascii=False,indent=2))
    elif a.stage=='freeze-eval': freeze_eval(a.confirm_annotation_complete)
    elif a.stage=='infer': infer()
    elif a.stage=='score': score()


if __name__=='__main__': main()

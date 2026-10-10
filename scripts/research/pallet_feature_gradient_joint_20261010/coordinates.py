"""GT-free 16-frame smoke, then all319x3x2 coordinates sealed before pose scoring."""
from collections import Counter
import gzip
import json
import time

import cv2
import numpy as np

from . import common as C
from .fusion import joint_refine, prepare_gradients


def main():
    start = time.monotonic()
    locked = C.read(C.DOC / 'INPUT_LOCK.json')
    parity = C.read(C.DOC / 'POSTERIOR_PARITY.json')
    tests = C.read(C.DOC / 'UNIT_TESTS.json')
    assert parity['status'] == tests['status'] == 'PASS'
    assert not (C.DOC / 'NEW_COORDINATES_SEALED.jsonl.gz').exists(), 'Preserve previous/partial output'
    assert C.sha(C.DOC / 'FUSION_METHOD_LOCK.json') == locked['fusion_method_lock_sha256']
    captures = list(C.rows(C.DOC / 'POSTERIOR_CAPTURE.jsonl.gz'))
    assert len(captures) == 957
    lookup = {(r['seed'], r['id']): r for r in captures}
    ids = locked['population']['frame_ids']
    assert len(ids) == 319 and len(lookup) == 957
    manifest = {r['id']: r for r in locked['input_manifest']}
    paths = {r['id']: r for r in C.read(C.PRIVATE / 'INPUT_PATHS.json')['frames']}
    assert set(paths) == set(ids)
    cv2.setNumThreads(1)
    smoke_ids = [ids[i] for i in np.linspace(0, 318, 16, dtype=int)]
    # Seal the code actually used before opening images; public configs contain
    # hashes, never private image paths or any evaluation targets.
    bindings = C.source_bindings(('common.py', 'fusion.py', 'test_fusion.py', 'capture.py',
                                 'preflight.py', 'coordinates.py', 'evaluate.py'))
    config = dict(status='SEALED_BEFORE_COORDINATES_AND_NEW_DEV_SCORING',
        input_lock_sha256=C.sha(C.DOC / 'INPUT_LOCK.json'),
        fusion_method_lock_sha256=C.sha(C.DOC / 'FUSION_METHOD_LOCK.json'),
        posterior_parity_sha256=C.sha(C.DOC / 'POSTERIOR_PARITY.json'),
        posterior_capture_sha256=C.sha(C.DOC / 'POSTERIOR_CAPTURE.jsonl.gz'),
        unit_tests_sha256=C.sha(C.DOC / 'UNIT_TESTS.json'), source_bindings=bindings,
        smoke_ids=smoke_ids, gray_conversion='cv2.cvtColor original BGR uint8 to gray uint8, then float32 /255',
        GT_inference_inputs=False, evaluation_references_loaded=False,
        new_hyperparameter_search=False, new_training=0)
    C.write(C.DOC / 'CONFIG_LOCK.json', config)
    decoded = {}
    gradients = {}
    def image(fid):
        if fid not in decoded:
            path = C.SOURCE / paths[fid]['image']
            assert C.sha(path) == manifest[fid]['image_sha256']
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            assert bgr is not None and list(bgr.shape[:2]) == manifest[fid]['raw_hw']
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / np.float32(255.)
            decoded[fid] = gray
            gradients[fid] = prepare_gradients(gray)
        return decoded[fid]
    def run(fid, seed, method):
        r = lookup[(seed, fid)]
        assert r['GT_inference_inputs'] is False
        points, diagnostics = joint_refine(image(fid), r['q0'], r['qN'], r['logits'],
            r['candidate_displacements_network'], r['gain'], r['prediction_support'],
            temperature=r['temperature'], method=method, point_support=r['point_support'],
            gradient_cache=gradients[fid])
        assert diagnostics['cornerSubPix_calls'] == 0 and diagnostics['GT_inputs'] is False
        q0 = np.asarray(r['q0'], float)
        support = np.asarray(r['prediction_support'], bool)
        assert np.array_equal(points[8], q0[8], equal_nan=True)
        assert np.array_equal(points[~support], q0[~support], equal_nan=True)
        return points, diagnostics
    smoke_start = time.monotonic()
    smoke = []
    for fid in smoke_ids:
        for seed in (1, 2, 3):
            for method in C.NEW_METHODS:
                points, diag = run(fid, seed, method)
                smoke.append(dict(id=fid, seed=seed, method=method,
                    finite_supported_points=bool(np.isfinite(points[np.asarray(lookup[(seed, fid)]['prediction_support'], bool)]).all()),
                    status_counts=diag['status_counts'], fallback_counts=diag['fallback_counts'],
                    cap_corners=int(np.sum(diag['cap_active']))))
    assert len(smoke) == 96 and all(r['finite_supported_points'] for r in smoke)
    C.write(C.DOC / 'SMOKE.json', dict(status='PASS', images=16, seeds=[1, 2, 3], methods=list(C.NEW_METHODS),
        joint_calls=96, detector_forwards=0, N3_additional_forwards=0, F_calls=0,
        evaluation_references_loaded=False, GT_inference_inputs=False,
        elapsed_seconds=time.monotonic() - smoke_start, rows=smoke))
    count = 0
    status = Counter()
    coordinate_start = time.monotonic()
    with gzip.open(C.DOC / 'NEW_COORDINATES_SEALED.jsonl.gz', 'wt', encoding='utf-8', compresslevel=6) as stream:
        for i, fid in enumerate(ids):
            for seed in (1, 2, 3):
                r = lookup[(seed, fid)]
                for method in C.NEW_METHODS:
                    points, diag = run(fid, seed, method)
                    status.update(diag['status_counts'])
                    coordinate = dict(seed=seed, id=fid, session=r['session'], grade=r['grade'], method=method,
                        q0=r['q0'], qN=r['qN'], qS=diag['q_unconstrained'], qFinal=points,
                        prediction_support=r['prediction_support'], point_support=r['point_support'],
                        raw_hw=r['raw_hw'], selected_index=r['selected_index'],
                        correction=dict(cap_px=diag['cap_px'], total_before_cap_px8=diag['total_before_cap_px'],
                            total_final_px8=diag['total_final_px'], cap_active8=diag['cap_active'], diagnostics=diag),
                        GT_inference_inputs=False, mechanism='one joint posterior/image normal-equation solve, then original q0 cap')
                    stream.write(json.dumps(C.finite(coordinate), separators=(',', ':'), allow_nan=False) + '\n')
                    count += 1
            if i % 80 == 0 or i == 318:
                print('COORDINATES', i + 1, '/319', 'rows', count, flush=True)
    assert count == 1914
    C.verify_bindings(bindings)
    C.write(C.DOC / 'COORDINATES_SEAL.json', dict(status='PASS', rows=count, images=319, seeds=[1, 2, 3],
        methods=list(C.NEW_METHODS), coordinates_sha256=C.sha(C.DOC / 'NEW_COORDINATES_SEALED.jsonl.gz'),
        input_lock_sha256=config['input_lock_sha256'], fusion_method_lock_sha256=config['fusion_method_lock_sha256'],
        config_lock_sha256=C.sha(C.DOC / 'CONFIG_LOCK.json'), source_bindings=bindings,
        smoke_status='PASS', smoke_joint_calls=96, accuracy_joint_calls=1914,
        gray_decodes=len(decoded), Sobel_image_computations=len(gradients),
        status_counts=dict(status), inference_wall_seconds=time.monotonic() - start,
        accuracy_coordinate_seconds=time.monotonic() - coordinate_start,
        GT_inference_inputs=False, evaluation_references_loaded=False, F_calls_before_seal=0,
        frozen_before_any_new_pose_error=True, no_training=True, no_tuning=True))
    print('COORDINATES_SEALED_PASS', count, C.sha(C.DOC / 'NEW_COORDINATES_SEALED.jsonl.gz'), flush=True)


if __name__ == '__main__':
    main()

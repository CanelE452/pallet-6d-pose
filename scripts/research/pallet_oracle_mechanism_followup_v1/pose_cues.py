"""Additional GT-free residuals from the unchanged nine-point D9 selector.

This artifact does not reselect any pose. It preserves the earlier candidate
lock and verifies its two hypotheses before exporting private residual arrays.
"""
from pathlib import Path
import time

import numpy as np
from . import pose_oracle as O


def main():
    start = time.perf_counter(), time.process_time()
    dst = O.RAW / 'D9_RESIDUAL9_CUES.json'
    if dst.exists():
        print('D9_RESIDUAL9_CUES_ALREADY_FROZEN'); return
    lock_path = O.RAW / 'CANDIDATES_LOCK.json'
    lock = O.read(lock_path)
    for binding in lock['files'] + lock['sources']:
        O.verify(binding)
    output = {}
    for material in O.ARMS:
        rows, predictions, _, _ = O.population(material)
        original = O.read(O.RAW / f'{material}_CANDIDATES.json')['arms']
        output[material] = {}
        for arm in O.ARMS[material]:
            result = {}
            for row in rows:
                fid = row['id']; q = O.D.points(predictions[arm][fid])
                selector = O.D.select(q, np.asarray(row['K']), np.asarray(row['xyz']))
                assert selector['selected_hypothesis'] == original[arm][fid]['selected_name']
                hypotheses = {}
                for h in selector['hypotheses']:
                    old = next(x for x in original[arm][fid]['hypotheses'] if x['name'] == h['name'])
                    O.D.close(h['score'], old['selector_score'])
                    O.D.close(h['score_components'], old['score_components'])
                    residual = np.linalg.norm(np.asarray(h['projected_keypoints'])-q, axis=1) if h['success'] else None
                    if residual is not None:
                        assert len(residual) == 9
                        assert np.isclose(np.sqrt(np.mean(residual**2)), h['score_components']['reprojection_rmse_px'], rtol=0, atol=1e-10)
                    hypotheses[h['name']] = dict(residual9_px=None if residual is None else residual.tolist(),
                                                  selector_score=h['score'], score_components=h['score_components'])
                result[fid] = hypotheses
            output[material][arm] = result
    payload = dict(created_at=O.now(), materials=output, GT_input=False, no_reference_coordinates_read=True,
                   private_coordinates=True, original_candidates_lock=O.bind(lock_path), source=O.bind(Path(__file__)),
                   contract='Residuals use original selector 9-point solve, including center8; final stored poses remain original corner8 solutions',
                   no_new_selector_evaluated=True, timing=O.elapsed(start))
    O.save(dst, payload)
    print('D9_RESIDUAL9_CUES_FROZEN', payload['timing'], flush=True)


if __name__ == '__main__':
    main()

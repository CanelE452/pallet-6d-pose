"""Descriptive replay of the previously frozen camera-view groups."""
from __future__ import annotations
import argparse
from pathlib import Path
from .preflight import REPO, read, write, sha
from scripts.research.pallet_dht_decoder_probe_v1.evaluate import summarize, point_difficulty

def main(root):
    root = root.resolve()
    path = REPO / 'data/pallet/results/pallet_dht_joint_v1/VIEW_STRATA.json'
    old = REPO / 'data/pallet/results/pallet_dht_decoder_probe_v1/EVALUATION_COMPLETION.json'
    assert sha(path) == read(old)['input_sha256'][str(path)]
    strata = read(path)
    by_id = {r['frame_id']: r for r in strata['records']}
    rows = [r for r in read(root / 'FRAME_METRICS.json')['records'] if r['population'] == 'real_dev']
    assert {r['id'] for r in rows} == set(by_id) and len(rows) == 319
    groups = []
    for dimension in ('frontness_bin', 'elevation_bin'):
        for group in sorted({r[dimension] for r in by_id.values()}):
            subset = [r for r in rows if by_id[r['id']][dimension] == group]
            groups.append(dict(dimension=dimension, group=group, n_frames=len(subset),
                frame_ids=[r['id'] for r in subset],
                summaries=[dict(arm=a, **summarize(subset,a)) for a in
                           ('baseline','independent','global','global_shared_only')],
                difficulty=[dict(**d) for a in ('independent','global') for d in point_difficulty(subset,a)]))
    write(root / 'VIEW_DIAGNOSTIC.json', dict(
        complete=True, PASS=True, groups=groups,
        selected_after_inference_for_reporting=True, used_for_selection=False,
        grouping_preexisted_current_experiment=True, measured_camera_pose=False,
        warning_ko='정면성은 GT 투영 면적비의 기존 proxy이며 측정 yaw가 아닙니다. 앙각도 재구성 값입니다. 완전 정면만 분리한 검증이나 구도별 일반화 입증으로 해석하지 않습니다.',
        input_sha256={str(p):sha(p) for p in (path,old,root/'FRAME_METRICS.json',root/'CALIBRATION_SELECTION.json')},
        source_sha256={str(Path(__file__).resolve()):sha(__file__)}))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=REPO/'data/pallet/results/pallet_dht_global_layout_v1')
    main(p.parse_args().run_dir)

"""Check published derived rows only; no upstream replay or new pose scoring."""
import argparse
import collections
import gzip
import hashlib
import itertools
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    a = p.parse_args(); root = a.folder
    read = lambda n: json.loads((root/n).read_text())
    with gzip.open(root/'ROWS.jsonl.gz','rb') as f: raw=f.read()
    rows=[json.loads(l) for l in raw.splitlines()]
    selected=[json.loads(l) for l in (root/'SELECTED_BOUNDARY_ROWS.jsonl').read_text().splitlines()]
    summary=read('SUMMARY.json'); plot=read('PLOT_CHECKS.json')
    assert len(rows)==len({r['id'] for r in rows})==59
    assert hashlib.sha256(raw).hexdigest()==summary['rows']['sha256']
    assert len(raw)==summary['rows']['bytes']
    assert sum(r['new_pose_estimated'] for r in rows)==55
    assert sum(r['fallback_used'] for r in rows)==4
    assert all(r['new_pose_estimated'] != r['fallback_used'] for r in rows)
    for r in rows:
        used=r['pools']['used']; good=set(used['proxy_correct_ids'])
        bad=set(used['proxy_bad_ids']); unknown=set(used['proxy_unknown_ids'])
        assert not (good&bad or good&unknown or bad&unknown)
        assert good|bad|unknown==set(used['ids'])
        assert not set(used['ids'])&set(r['hidden_initial'])
        assert good=={c['id'] for c in r['input_corner_errors_px'] if c['input_quality']=='CORRECT_WITHIN8PX'}
        generations=r['all_proxy_good_four_ID_generation']
        if len(good)>=4:
            assert {(g['dimension_index'],tuple(g['generator_ids'])) for g in generations} == {
                (d,ids) for d in (0,1) for ids in itertools.combinations(sorted(good),4)}
        else: assert not generations
        assert sum(g['returned_valid_roots'] for g in generations)==r['all_good_four_ID_valid_root_count']
        proposals={p['proposal_index']:p for p in r['all_proxy_good_four_ID_proposals']}
        for c in r['all_proxy_good_four_ID_refit_candidates']:
            assert c['proposal_index'] in proposals
            assert set(proposals[c['proposal_index']]['point_generator_ids'])<=good
            assert c['fit_proxy_good_ids']==sorted(set(c['fit_input_ids'])&good)
            assert c['inlier_proxy_good_ids']==sorted(set(c['final_inliers'])&good)
        if r['new_pose_estimated']:
            c=r['selected_candidate'];assert c['accepted']
            assert set(c['fit_input_ids'])<=set(c['final_inliers'])<=set(used['ids'])
            assert c['score_count']==len(c['final_inliers'])
    good24=[r for r in rows if r['used_proxy_good_count']>=4]
    assert len(good24)==24 and all(r['all_good_four_ID_valid_root_count']>0 for r in good24)
    assert sum(r['all_good_four_ID_accepted_refit_count']>0 for r in good24)==22
    assert plot['IDs']==sorted(r['id'] for r in good24)
    assert len(selected)==len({(r['frame_id'],r['id']) for r in selected})==133
    assert all(r['accepted'] and r['supply_selected'] and not r['self_hidden_excluded'] for r in selected)
    quality=collections.Counter(r['candidate_proxy_quality'] for r in selected)
    assert quality=={'CORRECT_WITHIN8PX':120,'INCORRECT_OVER8PX':13}
    assert sum(r['candidate_error_px']<r['N3_error_px'] for r in selected)==80
    assert sum(r['candidate_error_px']>r['N3_error_px'] for r in selected)==53
    for name, key in [('ALL24_SAVED_ERRORS.png','PNG_sha256'),('ALL24_SAVED_ERRORS.svg','SVG_sha256')]:
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==plot[key]
    result=dict(passed=True,derived_frames=59,GOOD_at_least4_frames=24,
        selected_boundary_rows=133,original_upstream245_replay_performed=False,
        new_PnP=0,new_projection=0,new_pose_accuracy_evaluations=0,
        physical_ground_truth_created=False,scope='published derived-row invariants, not full source or scientific truth certification')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()

"""Bound held-out pixel error from already saved distances; no projections."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    source=ROOT/'real_loo/ROWS.jsonl.gz';raw=source.read_bytes();sourcehash=hashlib.sha256(raw).hexdigest()
    out=ROOT/'bounds';out.mkdir()
    with (out/'PROTOCOL.json').open('x') as f:json.dump(dict(schema='frozen_saved_pixel_distance_triangle_bounds_v25',
        input_sha256=sourcehash,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        formula='abs(candidate_error_px - candidate_LOO_residual_px) <= LOO_projection_error_px <= their sum',
        scope='Existing annotation pixel proxy only; not translation/rotation score or physical ownership.',
        new_projection=0,new_pose_accuracy=0,new_PnP=0),f,indent=2);f.write('\n')
    counts=collections.Counter();rows=[]
    for s in gzip.decompress(raw).decode().splitlines():
        r=json.loads(s);b=r['selected_point'];e=b['candidate_error_px'];v=b['candidate_LOO_residual_px']
        assert e is not None and v is not None
        row=dict(frame_id=r['frame_id'],corner=r['corner'],candidate_proxy_quality=r['candidate_proxy_quality'],
            candidate_error_px=e,candidate_LOO_residual_px=v,LOO_projection_error_lower_bound_px=abs(e-v),
            LOO_projection_error_upper_bound_px=e+v,
            LOO_fit_all_known_GOOD=bool(r['LOO_fit_ids']) and not r['LOO_fit_quality_ids']['BAD'] and not r['LOO_fit_quality_ids']['UNKNOWN'],
            physical_truth='UNKNOWN',pose_translation_rotation_not_scored=True)
        rows.append(row);counts['points']+=1
        if row['candidate_proxy_quality']=='BAD':
            counts['BAD_candidates']+=1;counts['BAD_and_LOO_projection_lower_bound_gt8px']+=row['LOO_projection_error_lower_bound_px']>8
            if row['LOO_fit_all_known_GOOD']:
                counts['BAD_with_all_GOOD_LOO_fit']+=1
                counts['BAD_all_GOOD_fit_and_LOO_lower_bound_gt8px']+=row['LOO_projection_error_lower_bound_px']>8
    assert len(rows)==133
    with (out/'ROWS.jsonl.gz').open('xb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:
            for r in rows:z.write((json.dumps(r,allow_nan=False)+'\n').encode())
    assert sourcehash==hashlib.sha256(source.read_bytes()).hexdigest()
    with (out/'SUMMARY.json').open('x') as f:json.dump(dict(passed=True,counts=dict(counts),
        all_BAD_cases=[r for r in rows if r['candidate_proxy_quality']=='BAD'],
        new_projection=0,new_pose_accuracy=0,new_PnP=0,input_unchanged=True),f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,counts=dict(counts))))


if __name__=='__main__':
    main()

"""Portable verification of the published derived 573 rows, not upstream replay."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


def main():
    folder=Path(sys.argv[1]); rows=[]
    with gzip.open(folder/'ROWS.jsonl.gz','rt') as f:
        for s in f:rows.append(json.loads(s))
    summary=json.loads((folder/'SUMMARY.json').read_text())
    assert len(rows)==573
    assert len({(r['arm'],r['id'],r['corner'],tuple(r['edges'])) for r in rows})==573
    for arm,expected in (('GEOMETRY_ONLY',125),('IMAGE_NO_ROLE',100),('IMAGE_ROLE',348)):
        rr=[r for r in rows if r['arm']==arm];assert len(rr)==expected
        counts=collections.Counter()
        for r in rr:
            h,w=r['raw_hw'];x,y=r['predicted_xy']
            inside=0<=x<w and 0<=y<h
            assert inside==r['in_image']
            norm=r['recorded_radius_px']/max(r['nominal_query_spacing_px'],1e-6)
            assert norm==r['normalized_uncertainty']
            state=('DEFINITELY_REJECTED_IMAGE' if not inside else
                   'DEFINITELY_REJECTED_RADIUS' if norm>1 else 'NECESSARY_GATES_PASS_FINAL_UNKNOWN')
            assert state==r['necessary_gate_state'] and r['final_adopted_corner']=='UNKNOWN'
            assert r['physical_corner_ownership']=='UNKNOWN'
            assert math.isclose(math.dist(r['predicted_xy'],r['reference_xy']),r['error_px'],abs_tol=1e-8)
            supports=r['support_query_errors'];assert len(supports)>=6
            assert all(abs(s['predicted_bin']-s['target_bin'])==s['error_px'] for s in supports)
            all2=all(s['error_px']<=2 for s in supports);assert all2==r['all_support_queries_within2px']
            counts['rows']+=1;counts[state]+=1
            if r['error_px']>8:
                counts['corner_error_gt8px']+=1;counts['corner_error_gt8px:'+state]+=1
                if all2:
                    counts['all_support_within2px_but_corner_gt8px']+=1
                    counts['all_support_within2px_but_corner_gt8px:'+state]+=1
        assert dict(counts)==summary['arm_summaries'][arm]['counts']
    assert summary['final_admission_not_claimed'] is True
    assert all(summary[k]==0 for k in ('new_detector','new_head','new_line_fit','new_PnP','new_pose_accuracy','new_training','new_RGB'))
    plot=json.loads((folder/'PLOT_CHECKS.json').read_text())
    assert plot['all_rows_plotted']==573
    image=folder/'ALL573_SAVED_CANDIDATES.png'
    assert hashlib.sha256(image.read_bytes()).hexdigest()==plot['image_sha256']
    result=dict(passed=True,rows=573,all_three_counts_recomputed=True,
        necessary_gates_recomputed=True,final_admission_claimed=False,
        source_inputs_replayed=False,scope='Published derived rows, not original source supervision or physical ownership.',
        new_inference=0,new_PnP=0,new_pose_scoring=0)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()

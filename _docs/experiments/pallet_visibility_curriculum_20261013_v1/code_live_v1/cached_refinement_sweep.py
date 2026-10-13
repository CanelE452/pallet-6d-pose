"""CAL-only finite robustness ablation on sealed image evidence."""
from pathlib import Path
from collections import Counter
import numpy as np

from . import common as C
from .curriculum import solve_curriculum
from .run import no_annotations


def main():
    output = C.SCRATCH / 'cal_robustness_v2'
    output.mkdir(exist_ok=False)
    variants = {
        'BASELINE': dict(robust_px=8., enable_unknown=True),
        'ROBUST4': dict(robust_px=4., enable_unknown=True),
        'NO_UNKNOWN': dict(robust_px=8., enable_unknown=False),
        'ROBUST4_NO_UNKNOWN': dict(robust_px=4., enable_unknown=False),
        'EXACT_SELF_VISIBILITY': dict(robust_px=8., enable_unknown=True),
    }
    # These variants are written before inference or reference access.
    C.write(output / 'PROTOCOL.json', dict(partition='CAL', variants=variants,
        visibility_threshold=.35, self_hidden_margin_deg={'BASELINE':-2., 'EXACT_SELF_VISIBILITY':0.},
        evidence_sha256=C.sha(C.SCRATCH/'cal_segmentation_v2/PREDICTIONS.jsonl.gz'),
        prediction_inputs_sha256=C.sha(C.SCRATCH/'CAL_INPUTS.json'),
        code_sha256={n:C.sha(Path(__file__).parent/n) for n in ('curriculum.py','common.py','cached_refinement_sweep.py')},
        TEST_accessed=False, FINAL_accessed=False, selection_metrics='all-frame pose plus accepted visible observation counts'))
    rows=[]
    with no_annotations():
        frames=C.read(C.SCRATCH/'CAL_INPUTS.json')['frames']
        cached={r['id']:r for r in C.rows(C.SCRATCH/'cal_segmentation_v2/PREDICTIONS.jsonl.gz') if r['method']=='SEGMENTED_RGB_CURRICULUM'}
        for index,frame in enumerate(frames):
            p=frame['initial_poses']['N3_SUBPIX'];q=np.asarray(frame['points']['N3_SUBPIX'],float)[:8]
            R=np.asarray(p['R_cf']);t=np.asarray(p['centroid']);K=np.asarray(frame['K']);xyz=C.cuboid(*p['cf_extents'])
            hidden=C.self_hidden(xyz,R,t)
            assert np.array_equal(hidden,cached[frame['id']]['self_hidden'])
            Q=R.T@np.asarray(p['R_physical'])
            for name,config in variants.items():
                H=hidden.copy()
                if name=='EXACT_SELF_VISIBILITY':
                    rays=(-R.T@t)[None]-xyz
                    H=(np.sign(xyz)*rays/np.linalg.norm(rays,axis=1)[:,None]).max(1)<0
                solution=solve_curriculum(q,xyz,K,R,t,H,cached[frame['id']]['appearance'],dict(config,visible_threshold=.35))
                if name=='BASELINE':
                    np.testing.assert_allclose(solution['R'],cached[frame['id']]['solution']['R'],atol=1e-12,rtol=0)
                    np.testing.assert_allclose(solution['t'],cached[frame['id']]['solution']['t'],atol=1e-12,rtol=0)
                rows.append(dict(id=frame['id'],session=frame['session'],method=name,solution=solution,self_hidden=H.tolist(),
                    actual_pose=dict(R_cf=solution['R'],R_physical=(np.asarray(solution['R'])@Q).tolist(),centroid=solution['t'])))
            if (index+1)%10==0: print('INFER',index+1,len(frames),flush=True)
    C.save_rows(output/'PREDICTIONS.jsonl.gz',rows)
    C.write(output/'PREDICTION_SEAL.json',dict(before_scoring=True,sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),rows=len(rows),GT_read_canary='PASS'))
    from .scoring import pose_metric,distribution,paired
    oracle={f['id']:f['target'] for f in C.read(C.SCRATCH/'ORACLE_CAL_INPUTS.json')['frames']}
    for row in rows:row['metric']=pose_metric(row['actual_pose'],oracle[row['id']]['truth'])
    initial={r['id']:r for r in C.rows(C.SCRATCH/'cal_segmentation_v2/SCORED_ROWS.jsonl.gz') if r['method']=='INITIAL'}
    baseline={r['id']:r for r in rows if r['method']=='BASELINE'}
    summary={}
    for name in variants:
        current=[r for r in rows if r['method']==name]
        summary[name]=dict(frames=len(current),accepted=sum(r['solution']['accepted'] for r in current),
            metrics={metric:distribution([r['metric'][metric] for r in current]) for metric in ('translation_cm','rotation_deg','ADDsym_cm')},
            success=sum(r['metric']['success_5cm_5deg'] for r in current),
            reasons=dict(Counter(r['solution']['reason'] for r in current)),
            minus_baseline=paired(current,baseline),minus_initial=paired(current,initial))
        print(name,summary[name]['metrics']['translation_cm']['mean'],summary[name]['metrics']['rotation_deg']['mean'],summary[name]['success'],flush=True)
    C.save_rows(output/'SCORED_ROWS.jsonl.gz',rows)
    C.write(output/'RESULTS.json',summary)


if __name__=='__main__':
    main()

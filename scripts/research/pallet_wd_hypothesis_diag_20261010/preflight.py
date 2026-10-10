"""Input/schema/cardinality audit only, before any new solver call."""
from collections import Counter
import platform
import cv2
import numpy as np
from . import common as C
from . import inputs as I

def run():
    assert not (C.DOC/'INPUT_AUDIT.json').exists(), 'Preserve existing audit'
    C.pin_core()
    paths=[C.REAL,C.SYNTH,C.SOURCE/C.GT,C.SOURCE/C.GEOMETRY,
           C.SOURCE/I.REVIEW,C.SOURCE/I.RECHECK,
           C.SOURCE/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json',
           C.SOURCE/'data/pallet/raw_data/real_data/README.md']
    bindings=[C.binding(p) for p in paths]
    assert C.sha(C.REAL).startswith('bfb002e063c4b114'), 'Attachment-bound frozen predictions differ'
    populations={'REAL':I.real_inputs(),'SYNTH':I.synth_inputs()}
    populations['AUX']=I.aux_inputs(populations['REAL'])
    summary={}
    for name,rows in populations.items():
        keys=[(r['backbone'],r['method'],r['seed'],r['id']) for r in rows]
        assert len(keys)==len(set(keys)),name
        for r in rows:
            assert r['seed'] in C.SEEDS and r['method'] in C.METHODS
            q=np.asarray(r['qFinal'],float);K=np.asarray(r['fixed_metadata']['K'],float);xyz=np.asarray(r['fixed_metadata']['dimensions_pnp_WH_D_m'],float)
            assert q.shape==(9,2) and K.shape==(3,3) and xyz.shape==(3,)
            assert np.isfinite(K).all() and np.isfinite(xyz).all() and (xyz>0).all()
            assert len(r['prediction_support'])==9
            if name!='AUX':assert np.isfinite(q).all()
        summary[name]=dict(rows=len(rows),frames=len({r['id'] for r in rows}),
            methods=sorted({r['method'] for r in rows}),backbones=sorted({r['backbone'] for r in rows}),
            count_by_route={f'{b}/{m}/seed{s}':n for (b,m,s),n in sorted(Counter((r['backbone'],r['method'],r['seed']) for r in rows).items())},
            source_fields='qFinal, prediction_support, fixed_metadata.K, dimensions_pnp_WH_D_m, id, seed, method')
    gt=C.read(C.SOURCE/C.GT);required=('R_gt_representative','t_gt','physical_dimensions_m','physical_long_axis','elevation_deg','resolution_margin_px')
    assert len(gt['frames'])==319 and all(all(k in g for k in required) for g in gt['frames'].values())
    assert {r['id'].replace(':','__',1) for r in populations['REAL']}==set(gt['frames'])
    legacy=C.read(C.SOURCE/I.REVIEW)['frames'];recheck=C.read(C.SOURCE/I.RECHECK)['frames']
    assert len(legacy)==319 and Counter(r['status'] for r in legacy.values())=={'CONFIRMED':319}
    assert len(recheck)==40 and Counter(r['status'] for r in recheck.values())=={'CONFIRMED':36,'UNCLEAR':4}
    for rel in I.AUX.values():
        if (C.SOURCE/rel).exists():bindings.append(C.binding(C.SOURCE/rel,C.SOURCE))
    sessions={r['id']:r['session'] for r in populations['REAL']}
    result=dict(status='PASS',bindings=bindings,populations=summary,
        reference_required_fields=list(required),reference_schema_checked_only=True,
        reference_pose_values_consumed_for_selection=False,
        reference_limitations='Geometry-reconstructed manual2D reference; not independent physical metrology.',
        human_axis_review=dict(legacy=dict(rows=319,status_counts={'CONFIRMED':319}),
            recheck=dict(rows=40,status_counts={'CONFIRMED':36,'UNCLEAR':4},unclear_ids=[k.replace('__',':',1) for k,v in recheck.items() if v['status']=='UNCLEAR']),
            note='Do not equate historical319 confirmed with all-current human confirmations; frozen reference is preserved.'),
        camera_fixed_record=dict(status='NOT_CONFIRMED',protocol='data/pallet/raw_data/real_data/README.md',
            searched='capture protocol, per-session dataset cards, history/audits; no session-specific world-camera extrinsics-fixed evidence',
            note='Rig height or K fixed is not evidence of fixed world-camera extrinsics. S3 eligibility is planner post-hoc unless additional records surface.'),
        final_test4=dict(sessions=C.FINAL4,excluded_frames=sum(s in C.FINAL4 for s in sessions.values()),remaining_DEV_frames=sum(s not in C.FINAL4 for s in sessions.values()),
            historical_role='development since2026-08-20; no resealing; excluded from threshold/model selection'),
        environment=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__),
        execution=dict(model_forwards=0,training_updates=0,new_synthetic_images=0,solver_calls_before_audit=0),
        public_privacy='No raw RGB, review actor or personal absolute paths are copied.')
    C.write(C.DOC/'INPUT_AUDIT.json',result)
    print('INPUT_AUDIT PASS',summary,flush=True)
    return result

if __name__=='__main__':run()

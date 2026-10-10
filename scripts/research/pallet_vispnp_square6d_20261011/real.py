"""Single locked REAL319 run, permitted only after the synthetic harm gate passes."""
import copy
from collections import Counter
import time
import numpy as np
import cv2
from . import common as C

def main():
    from .adapter import correspondence_mask
    assert C.read(C.DOC/'A0.json')['status']=='PASS'
    source_lock=C.read(C.DOC/'SOURCE_LOCK.json')
    for b in source_lock['new_core']:
        assert C.sha(C.ROOT/b['path'])==b['sha256'],b['path']
    gate=C.read(C.DOC/'SYNTH_VERDICT.json')
    assert gate['continue_to_A2'], 'Synthetic preregistered harm gate forbids REAL execution'
    seal=C.read(C.DOC/'COORDINATES_SEAL.json')
    assert seal['status']=='PASS' and seal['sha256']==C.sha(C.DOC/'REAL_COORDINATES_MASKS.jsonl.gz')
    assert not (C.DOC/'REAL_STARTED.json').exists(),'No repeat of actual REAL execution'
    assert not (C.DOC/'PREDICTIONS.jsonl.gz').exists()
    inputs=list(C.rows(C.DOC/'REAL_COORDINATES_MASKS.jsonl.gz'))
    assert len(inputs)==319*4*3
    C.write(C.DOC/'REAL_STARTED.json',dict(status='STARTED',mask_seal_verified_before_reference_load=True,
        mask_sha256=seal['sha256'],synthetic_verdict_sha256=C.sha(C.DOC/'SYNTH_VERDICT.json')))
    # Only now load the references; all masks/coordinates already exist on disk.
    from scripts.research.pallet_n3_subpix_final_20261010.evaluate import score,same
    load_real,_,_=C.existing()
    E,frames,targets,_,_=load_real();frames={f['id']:f for f in frames}
    old={(r['seed'],r['method'],r['id']):r for r in C.historical_rows()}
    started=time.monotonic();counts=Counter();out=[];no_pose=[]
    cv2.setNumThreads(1)
    for i,r in enumerate(inputs):
        f=frames[r['id']];q=np.array(r['qFinal'],float);before=q.copy()
        oldrow=old[(r['seed'],r['method'],r['id'])]
        assert C.digest(f['K'])==C.digest(r['fixed_metadata']['K'])
        assert C.digest(f['xyz'])==C.digest(r['fixed_metadata']['dimensions_pnp_WH_D_m'])
        with correspondence_mask(q,f['K'],r['visibility']['effective_mask']) as audit:
            new=score(E,f,q,targets[r['id']],r['method']+'_VIS',r['seed'])
        assert np.array_equal(q,before,equal_nan=True)
        same(oldrow['corner'],C.finite(new['corner']),r['id']+'/coordinate-only-corner-invariance',0)
        assert oldrow['canonical_observed']==C.finite(new['canonical_observed'])
        new.update(grade=r['grade'],qFinal=oldrow['qFinal'],prediction_support=oldrow['prediction_support'],
            raw_hw=oldrow['raw_hw'],fixed_metadata=copy.deepcopy(oldrow['fixed_metadata']),visibility=r['visibility'],
            solver_mask_audit=audit,baseline_method=r['method'],coordinate_changes=0,
            old_row_binding=dict(file='_docs/experiments/pallet_feature_gradient_joint_20261010/PREDICTIONS.jsonl.gz',
                sha256=seal['source_sha256'],seed=r['seed'],method=r['method'],id=r['id']),
            hypothesis_switched=new['final_hypothesis']!=oldrow['final_hypothesis'])
        for key in ['q0','qN','qS']:new[key]=oldrow[key]
        if not new['pose']['available']:no_pose.append(dict(id=r['id'],seed=r['seed'],method=new['method']))
        out.append(new);counts.update(new['PnP_counts'])
        if i%200==0:print('REAL_VIS',i+1,len(inputs),'seconds',round(time.monotonic()-started,1),flush=True)
    C.write_rows(C.DOC/'PREDICTIONS.jsonl.gz',out)
    C.write(C.DOC/'REAL_EXECUTION.json',dict(status='COMPLETE',actual_F_calls=len(out),rows=len(out),images=319,
        seeds=[1,2,3],methods=4,PnP_counts=dict(counts),no_pose=no_pose,coordinates_exact=True,corner_metrics_exact=True,
        mask_seal_verified_before_reference_load=True,new_training_updates=0,new_annotations=0,new_RGB=0,
        detector_calls=0,N3_forwards=0,elapsed_seconds=time.monotonic()-started,
        predictions_sha256=C.sha(C.DOC/'PREDICTIONS.jsonl.gz')))
    print('REAL_COMPLETE',len(out),flush=True)

if __name__=='__main__':main()

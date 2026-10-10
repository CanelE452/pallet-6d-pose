"""Post-seal scoring recovery; never recompute completed REAL/SYNTH F choices.

The original source and selection seals remain immutable. NPZ arrays are loaded
once, so per-ID views share one backing array rather than retaining one newly
decompressed60000-row array per ID. Metric definitions remain unchanged.
"""
from collections import Counter
from pathlib import Path
import time
import numpy as np
from . import common as C
from . import inputs as I
from . import solver
from . import stage1 as S

def minimal_truth(population,with_margin=False):
    if population!='SYNTH':return I.truth('REAL')
    pose=C.pose_api();solver.configure(pose)
    inputs={r['id']:r for r in I.synth_inputs()};result={};diag={}
    with np.load(C.SOURCE/C.GEOMETRY,allow_pickle=False) as archive:
        g={k:archive[k] for k in ('stems','dims','R','t','Xcf')}
    ix={str(s):i for i,s in enumerate(g['stems'])}
    for fid,r in inputs.items():
        i=ix[fid];xyz=g['dims'][i];R=g['R'][i];t=g['t'][i];X=g['Xcf'][i]
        width=np.linalg.norm(X[1]-X[0]);depth=np.linalg.norm(X[4]-X[0]);height=np.linalg.norm(X[3]-X[0])
        assert np.allclose(sorted([width,depth]),sorted([xyz[0],xyz[2]]),rtol=0,atol=1e-6)
        assert abs(height-xyz[1])<1e-6 and abs(width-depth)>1e-6
        name='long-face-front' if width>depth else 'short-face-front'
        down=(X[[2,3,6,7]].mean(0)-X[[0,1,4,5]].mean(0))@R.T;down/=np.linalg.norm(down)
        ray=t/max(np.linalg.norm(t),1e-9)
        elevation=float(abs(90-np.degrees(np.arccos(np.clip(abs(down@ray),-1,1)))))
        margin=None
        if with_margin:
            K=np.array(r['fixed_metadata']['K']);cam=np.vstack([X,np.zeros((1,3))])@R.T+t
            projected=cam@K.T;q=projected[:,:2]/projected[:,2:]
            fits={h:solver.fit(q,K,xyz,h,list(range(8)),True) for h in solver.NAMES}
            residual={h:c.get('reprojection_mean_8_px') for h,c in fits.items()}
            other=next(h for h in solver.NAMES if h!=name)
            margin=None if residual[name] is None or residual[other] is None else residual[other]-residual[name]
        result[fid]=dict(R=R,t=t,xyz=xyz,body_R=R,body_xyz=xyz,order=r['pose_symmetry_order'])
        diag[fid]=dict(hypOracle=name,elevation_deg=elevation,reference_margin_px=margin,
            distance_m=float(np.linalg.norm(t)),material='NOT_APPLICABLE',source=r['source'],
            reference_axis_source='exact renderer Xcf edge01 vs edge04; GT parity supplied',
            reference_margin_source='same GT-builder8-corner mean reprojection residual alternative minus GT-parity; diagnostic only',
            reference_margin_actual_solver_calls=2 if with_margin else 0,gt_cf_width_m=width,gt_cf_depth_m=depth,
            reference_bottom_center=X[[2,3,6,7]].mean(0)@R.T+t,reference_down=down)
    return result,diag

def run():
    C.verify_core();S._verify_fixed_inputs()
    assert C.read(C.DOC/'STAGE0_PARITY.json')['status']=='PASS'
    assert not (C.DOC/'STAGE1_ROWS_SYNTH.jsonl.gz').exists()
    assert not (C.DOC/'STAGE1_SELECTIONS_AUX.jsonl.gz').exists()
    assert not (C.DOC/'STAGE1_POSTSEAL_AMENDMENT.json').exists()
    paths={p:C.DOC/f'STAGE1_SELECTIONS_{p}.jsonl.gz' for p in ('REAL','SYNTH')}
    for p,path in paths.items():
        seal=C.read(C.DOC/f'STAGE1_SELECTION_SEAL_{p}.json')
        assert C.sha(path)==seal['selection_sha256']
    binding=C.binding(Path(__file__),C.ROOT)
    C.write(C.DOC/'STAGE1_POSTSEAL_AMENDMENT.json',dict(
        status='POSTSEAL_SCORING_RECOVERY_BEFORE_AUX_SELECTION',source=binding,
        previous_lock_path=C.source_lock_path().name,previous_lock_sha256=C.sha(C.source_lock_path()),
        original_frozen_core=C.read(C.source_lock_path())['new_core'],
        cause='Lazy NPZ member reads inside each ID loop repeatedly decompressed arrays and retained full array bases through per-ID views; process safely interrupted during reference preparation.',
        correction='Load stems/dims/R/t/Xcf once; use shared views; keep all arithmetic, axes, solver and metric definitions identical.',
        completed_original_F_calls={'REAL':3828,'SYNTH':23820},repeated_original_F_calls=0,
        scoring_files_before_recovery={'REAL':'complete','SYNTH':'not generated','AUX':'not started'},
        interrupted_reference_only_solver_calls='NOT_MEASURED; partial reference-margin fits during interrupted preparation; no published results or new F selection',
        recovery_reference_only_margin_solver_calls=3970,AUX_original_F_calls_planned=3828,
        preserved_selection_bindings={p:C.binding(path,C.ROOT) for p,path in paths.items()}))
    started=time.monotonic();solver.configure(C.pose_api())
    truth,diagnostics=minimal_truth('SYNTH',with_margin=True)
    print('RECOVERY_SYNTH_REFERENCES_READY',len(truth),'seconds',round(time.monotonic()-started,1),flush=True)
    def scored():
        seal_sha=C.sha(C.DOC/'STAGE1_SELECTION_SEAL_SYNTH.json')
        for i,r in enumerate(C.rows(paths['SYNTH'])):
            yield S._scored(r,truth[r['id']],diagnostics[r['id']],seal_sha)
            if (i+1)%2000==0:print('RECOVERY_SYNTH_SCORED',i+1,23820,'seconds',round(time.monotonic()-started,1),flush=True)
    C.write_rows(C.DOC/'STAGE1_ROWS_SYNTH.jsonl.gz',scored())
    paths['AUX'],aux_seal,naux=S._select_population('AUX',I.aux_inputs,started)
    truth,diagnostics=I.truth('AUX');seal_sha=C.sha(aux_seal)
    C.write_rows(C.DOC/'STAGE1_ROWS_AUX.jsonl.gz',
        (S._scored(r,truth[r['id']],diagnostics[r['id']],seal_sha) for r in C.rows(paths['AUX'])))
    assert C.sha(C.ROOT/binding['path'])==binding['sha256']
    C.verify_core();S._verify_fixed_inputs()
    counts={'REAL':3828,'SYNTH':23820,'AUX':naux}
    C.write(C.DOC/'STAGE1_SELECTION_SEAL.json',dict(status='ALL_POPULATIONS_SEALED_BEFORE_THEIR_REFERENCE_SCORING',
        populations={p:dict(rows=counts[p],selection_path=paths[p].name,selection_sha256=C.sha(paths[p]),
            seal_path=f'STAGE1_SELECTION_SEAL_{p}.json',seal_sha256=C.sha(C.DOC/f'STAGE1_SELECTION_SEAL_{p}.json')) for p in counts},
        execution_order='REAL selection/seal -> REAL parity PASS -> SYNTH selection/seal -> interrupted GT-only prep -> postseal scoring recovery -> AUX initial selection/seal/scoring',
        source_lock_path=C.source_lock_path().name,source_lock_sha256=C.sha(C.source_lock_path()),
        postseal_amendment_sha256=C.sha(C.DOC/'STAGE1_POSTSEAL_AMENDMENT.json'),
        stage0_parity_sha256=C.sha(C.DOC/'STAGE0_PARITY.json')))
    C.write(C.DOC/'STAGE1_EXECUTION.json',dict(status='COMPLETE_STAGE1_DIAGNOSTIC_ONLY_NO_VERDICT',
        actual_original_F_calls=counts,actual_original_F_calls_total=sum(counts.values()),repeated_original_F_calls=0,
        source_lock_path=C.source_lock_path().name,source_lock_sha256=C.sha(C.source_lock_path()),
        postseal_amendment_sha256=C.sha(C.DOC/'STAGE1_POSTSEAL_AMENDMENT.json'),
        selection_seal_sha256=C.sha(C.DOC/'STAGE1_SELECTION_SEAL.json'),reference_only_SYNTH_margin_solver_calls=3970,
        interrupted_reference_only_solver_calls='NOT_MEASURED; disclosed in postseal amendment',
        outputs={p:C.binding(C.DOC/f'STAGE1_ROWS_{p}.jsonl.gz',C.ROOT) for p in counts},
        stage2_rules_executed=0,network_inference=0,training_updates=0,
        recovery_elapsed_seconds=time.monotonic()-started,all_fixed_inputs_preserved=True))
    print('STAGE1 RECOVERY COMPLETE',counts,'seconds',round(time.monotonic()-started,1),flush=True)

if __name__=='__main__':run()

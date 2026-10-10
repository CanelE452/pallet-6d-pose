"""Whitelisted frozen predictions before seal; references only through truth()."""
from collections import Counter
import copy
import numpy as np
from . import common as C

AUX={b:f'data/pallet/results/pallet_n3_completion_v3/predictions/{b}_DEV319.json' for b in ('dope','resnet18')}
REVIEW='data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_LABELS.json'
RECHECK='data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_LABELS_RECHECK.json'

def real_inputs():
    result=[]
    for r in C.rows(C.REAL):
        if r['method'] not in C.METHODS:continue
        result.append(dict(backbone='YOLO',method=r['method'],seed=r['seed'],id=r['id'],session=r['session'],
                           grade=r['grade'],qFinal=r['qFinal'],prediction_support=r['prediction_support'],
                           raw_hw=r['raw_hw'],fixed_metadata=copy.deepcopy(r['fixed_metadata']),source_flag=False))
    assert len(result)==3828 and len({r['id'] for r in result})==319
    return result

def synth_inputs():
    result=[]
    for r in C.rows(C.SYNTH):
        assert r['method'] in C.METHODS and r['variant']=='ALL'
        result.append(dict(backbone='YOLO',method=r['method'],seed=r['seed'],id=r['id'],session=r['session'],
                           source=r['source'],grade='NOT_APPLICABLE',qFinal=r['qFinal'],
                           prediction_support=r['prediction_support'],raw_hw=r['raw_hw'],
                           fixed_metadata=copy.deepcopy(r['fixed_metadata']),source_flag=True,
                           pose_symmetry_order=r['canonical_symmetry_order']))
    assert len(result)==23820 and len({r['id'] for r in result})==1985
    return result

def aux_inputs(real=None):
    real=real or real_inputs();by={r['id']:r for r in real};out=[]
    for backbone,rel in AUX.items():
        p=C.SOURCE/rel
        if not p.exists():continue
        packet=C.read(p)
        assert packet['complete'] and packet['dataset']=='DEV319' and len(packet['frames'])==319
        assert set(f['id'] for f in packet['frames'])==set(by)
        for f in packet['frames']:
            base=by[f['id']]
            assert f['raw_shape_hw']==base['raw_hw']
            assert np.allclose(np.array(f['dimensions_wdh_m'])[[0,2,1]],base['fixed_metadata']['dimensions_pnp_WH_D_m'],rtol=0,atol=1e-8)
            for seed in C.SEEDS:
                for method,key in [('BASE','base'),('N3_DIM_SYM',f'N3_seed{seed}')]:
                    pred=f['predictions'][key]
                    q=np.full((9,2),np.nan) if pred['points'] is None else np.array(pred['points'],float)
                    assert q.shape==(9,2) and len(pred['valid'])==9
                    out.append(dict(backbone='DOPE' if backbone=='dope' else 'ResNet-18',method=method,seed=seed,
                        id=f['id'],session=base['session'],grade=base['grade'],qFinal=q,
                        prediction_support=pred['valid'],raw_hw=base['raw_hw'],source_flag=False,
                        fixed_metadata=dict(K=base['fixed_metadata']['K'],
                            dimensions_pnp_WH_D_m=base['fixed_metadata']['dimensions_pnp_WH_D_m'],
                            canonical_symmetry_order=base['fixed_metadata']['canonical_symmetry_order'],object_type=f['object_type'],
                            candidate_metadata={k:pred.get(k) for k in ('bbox','score','confidence')},
                            native_coordinate_system=packet['coordinate_system']),
                        detected=pred['detected'],native_prediction_status=pred['status'],
                        auxiliary_semantics='Existing native Base/N3 only; no SubPix, no new network inference'))
    return out

def truth(population):
    """Return per-ID original proper-group truth and reference-only diagnostics."""
    name='SYNTH' if population in ('SYNTH','SYNTH_HELDOUT') else 'AUX' if population=='AUX' else 'REAL'
    assert (C.DOC/f'STAGE1_SELECTION_SEAL_{name}.json').exists(), 'Seal this population before its reference values'
    pose=C.pose_api();result={};diag={}
    if population in ('REAL','AUX','REAL_DEV'):
        by={r['id']:r for r in real_inputs()}
        gt=C.read(C.SOURCE/C.GT)['frames']
        review=C.read(C.SOURCE/REVIEW)['frames'];recheck=C.read(C.SOURCE/RECHECK)['frames']
        for fid,r in by.items():
            gid=fid.replace(':','__',1);g=gt[gid]
            xyz=np.array(r['fixed_metadata']['dimensions_pnp_WH_D_m'],float)
            d=g['physical_dimensions_m'];cf=np.array([d['across'],d['height'],d['along']]);Rcf=np.array(g['R_gt_representative'])
            Q=np.eye(3) if abs(cf[0]-xyz[0])<1e-6 else pose.rotations(4)[1]
            order=int(r['fixed_metadata']['canonical_symmetry_order'])
            result[fid]=dict(R=Rcf@Q,t=np.array(g['t_gt']),xyz=xyz,body_R=Rcf,body_xyz=cf,order=order)
            assert g['physical_long_axis'] in ('CF_WIDTH','CF_DEPTH')
            name='long-face-front' if g['physical_long_axis']=='CF_WIDTH' else 'short-face-front'
            assert abs(cf[0]-(max(xyz[0],xyz[2]) if name=='long-face-front' else min(xyz[0],xyz[2])))<1e-6
            diag[fid]=dict(hypOracle=name,elevation_deg=g['elevation_deg'],reference_margin_px=g['resolution_margin_px'],
                distance_m=float(np.linalg.norm(g['t_gt'])),material='plastic' if xyz[0]==1.1 else 'wood',
                reference_axis_source='geometry-resolved manual 2D plus known dimensions; GT parity supplied',
                legacy_axis_review_status=review[gid]['status'],recheck_axis_review_status=None if gid not in recheck else recheck[gid]['status'],
                gt_physical_long_axis=g['physical_long_axis'],reference_bottom_center=np.array(g['t_gt'])+Rcf@np.array([0,cf[1]/2,0]),
                reference_down=Rcf[:,1],reference_R_cf=Rcf,reference_cf_extents=cf)
    elif population in ('SYNTH','SYNTH_HELDOUT'):
        from . import solver
        solver.configure(pose)
        inputs={r['id']:r for r in synth_inputs()}
        with np.load(C.SOURCE/C.GEOMETRY,allow_pickle=False) as g:
            ix={str(s):i for i,s in enumerate(g['stems'])}
            for fid,r in inputs.items():
                i=ix[fid];xyz=g['dims'][i];R=g['R'][i];t=g['t'][i];X=g['Xcf'][i]
                width=np.linalg.norm(X[1]-X[0]);depth=np.linalg.norm(X[4]-X[0]);height=np.linalg.norm(X[3]-X[0])
                assert np.allclose(sorted([width,depth]),sorted([xyz[0],xyz[2]]),rtol=0,atol=1e-6)
                assert abs(height-xyz[1])<1e-6 and abs(width-depth)>1e-6, 'Square/invalid parity must remain undefined'
                name='long-face-front' if width>depth else 'short-face-front'
                K=np.array(r['fixed_metadata']['K']);cam=np.vstack([X,np.zeros((1,3))])@R.T+t
                projected=cam@K.T;q=projected[:,:2]/projected[:,2:]
                fits={h:solver.fit(q,K,xyz,h,list(range(8)),True) for h in solver.NAMES}
                residual={h:c.get('reprojection_mean_8_px') for h,c in fits.items()}
                other=next(h for h in solver.NAMES if h!=name)
                margin=None if residual[name] is None or residual[other] is None else residual[other]-residual[name]
                down=(X[[2,3,6,7]].mean(0)-X[[0,1,4,5]].mean(0))@R.T;down/=np.linalg.norm(down)
                ray=t/max(np.linalg.norm(t),1e-9)
                elevation=float(abs(90-np.degrees(np.arccos(np.clip(abs(down@ray),-1,1)))))
                result[fid]=dict(R=R,t=t,xyz=xyz,body_R=R,body_xyz=xyz,order=r['pose_symmetry_order'])
                diag[fid]=dict(hypOracle=name,elevation_deg=elevation,reference_margin_px=margin,
                    distance_m=float(np.linalg.norm(t)),material='NOT_APPLICABLE',source=r['source'],
                    reference_axis_source='exact renderer Xcf edge01 vs edge04; GT parity supplied',
                    reference_margin_source='same GT-builder8-corner mean reprojection residual alternative minus GT-parity; diagnostic only',
                    reference_margin_actual_solver_calls=2,gt_cf_width_m=width,gt_cf_depth_m=depth,
                    reference_bottom_center=X[[2,3,6,7]].mean(0)@R.T+t,reference_down=down)
    else:raise ValueError(population)
    return result,diag

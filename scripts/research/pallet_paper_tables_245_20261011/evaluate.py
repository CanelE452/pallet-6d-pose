"""Actual fixed F replay and fixed image subpixel corrections; no training/search."""
from collections import Counter
import copy
import time
import cv2
import numpy as np
from . import common as C

ALIASES = {'N0':'N0_BASE_REPLAY','N1':'N1_SYM_ONLY','N2':'N2_DIM_ONLY','P':'OLD_P'}

def key(r): return r['backbone'], r['method'], r['seed'], r['id']

def seal(name, records, note):
    C.write_rows(name+'.jsonl.gz',records)
    C.write(name+'_SEAL.json',dict(status='SEALED_BEFORE_THIS_STAGE_REFERENCE_SCORING',rows=len(records),
        path=name+'.jsonl.gz',sha256=C.sha(C.DOC/(name+'.jsonl.gz')),method_lock_sha256=C.sha(C.DOC/'METHOD_LOCK.json'),
        correction_reference_inputs=False,F_reference_inputs=False,note=note))

def corner_score(record, target, matched, math_api):
    q=np.asarray(record['qFinal'],float);gt=np.asarray(target['gt'],float);valid=np.asarray(target['valid'],bool)
    detected=bool(record.get('detected',True))
    value=math_api.measure(q,gt,valid,target['permutations'],record['raw_hw'],bool(matched and detected),detected)
    mask=np.isfinite(gt[:8]).all(1)&~(gt[:8]==-1).all(1)
    diagonal=float(np.linalg.norm(np.ptp(gt[:8][mask],axis=0))) if mask.sum()>=2 else None
    count=int((valid[:8]&mask).sum());errors=value.get('errors',[])
    assert not value.get('evaluable') or len(errors)==count
    value.update(id=record['id'],session=record['session'],gt_count=count,pck10_denominator=count,
        pck10_hits=int(np.sum(np.asarray(errors,float)<=10)) if value.get('evaluable') else 0,
        gt_corner_valid8=(valid[:8]&mask).tolist(),pallet_diagonal_px=diagonal,
        diagonal_definition='ALL finite reference corners0..7 axis-aligned bbox diagonal; native pixels, unclipped; includes existing extrapolated coordinates; diagnostic only')
    return value

def scored(record, actual, target, matched, math_api, pose_api):
    truth={k:np.asarray(v,float) if k!='order' else int(v) for k,v in target['truth'].items()}
    corner=corner_score(record,target,matched,math_api)
    output=dict(record,actual_pose=actual,pose=pose_api.metric((record['id'],actual,truth)),corner=corner,
        final_hypothesis=actual.get('selected_hypothesis'),inference_reference_inputs=False,
        evaluation_reference_points=target['gt'],evaluation_reference_valid=target['valid'],
        evaluation_permutations=target['permutations'],material='wood' if target.get('object','').startswith('wood') else 'plastic')
    return output

class ActualF:
    def __init__(self):
        self.pose=C.pose_api();self.calls=Counter();self.pnp=Counter();self.original={name:getattr(cv2,name) for name in ('solvePnP','solvePnPGeneric','solvePnPRefineLM')}
    def infer(self,record,phase):
        before=self.pnp.copy()
        def wrapper(name):
            def call(*a,**kw):
                self.pnp[name]+=1
                return self.original[name](*a,**kw)
            return call
        for name in self.original:setattr(cv2,name,wrapper(name))
        try:
            q=np.asarray(record['qFinal'],float) if record.get('detected',True) else None
            # Complete production F receives only frozen prediction/native K/registry dimensions.
            actual=self.pose.infer(q,np.asarray(record['fixed_metadata']['K'],float),
                np.asarray(record['fixed_metadata']['dimensions_pnp_WH_D_m'],float),False)
        finally:
            for name,fn in self.original.items():setattr(cv2,name,fn)
        self.calls[phase]+=1
        return actual,{name:self.pnp[name]-before[name] for name in self.original}

def stage0(engine):
    assert C.read(C.DOC/'TABLE_DISPLAY_PARITY.json')['status']=='PASS'
    ids=set(C.read(C.DOC/'RECT_245_IDS.json')['ids'])
    previous={(r['method'],r['id']):r for r in C.rows(C.JOINT) if r['seed']==1 and r['method'] in C.CORE_METHODS and r['id'] in ids}
    coords=[C.prediction_only(r) for r in previous.values()]
    assert len(coords)==980
    seal('STAGE0_COORDINATES',coords,'Existing seed1 four-method coordinates for fixed245 IDs, before actual-F parity.')
    actual=[]
    for r in coords:
        p,counts=engine.infer(r,'STAGE0');actual.append(dict(r,actual_pose=p,PnP_counts=counts))
    seal('STAGE0_POSES',actual,'Actual prediction-only F completed and fixed before any current pose-reference value scoring.')
    targets={fid:t for fid,t in C.read(C.SOURCE/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json').items() if t['split']=='REAL_DEV'}
    checked=[];cache={};math_api=C.math_api()
    for r in actual:
        out=scored(r,r['actual_pose'],targets[r['id']],targets[r['id']]['matched'],math_api,engine.pose)
        old=previous[(r['method'],r['id'])]
        a=out['pose'];b=old['pose'];available=a['available']==b['available']
        dt=abs(a['translation_cm']-b['translation_cm']) if a['available'] and b['available'] else None
        dr=abs(a['rotation_deg']-b['rotation_deg']) if a['available'] and b['available'] else None
        hyp=out['final_hypothesis']==old['final_hypothesis']
        ok=available and hyp and (not a['available'] or (dt<=.01 and dr<=.01))
        checked.append(dict(id=r['id'],method=r['method'],seed=1,availability_exact=available,hypothesis_exact=hyp,
            delta_translation_cm=dt,delta_rotation_deg=dr,PASS=ok))
        cache[key(r)]=r
    status='PASS' if all(r['PASS'] for r in checked) else 'FAIL_STOP'
    C.write('STAGE0_F_PARITY.json',dict(status=status,actual_F_calls=980,frames=245,methods=list(C.CORE_METHODS),seed=1,
        translation_cm_tolerance=.01,rotation_deg_tolerance=.01,
        max_translation_cm_delta=max(r['delta_translation_cm'] or 0 for r in checked),
        max_rotation_deg_delta=max(r['delta_rotation_deg'] or 0 for r in checked),
        failures=[r for r in checked if not r['PASS']],per_row=checked,
        coordinate_seal_sha256=C.sha(C.DOC/'STAGE0_COORDINATES_SEAL.json'),pose_seal_sha256=C.sha(C.DOC/'STAGE0_POSES_SEAL.json')))
    assert status=='PASS','Strict F parity failed: STOP'
    print('STAGE0_ACTUAL_F_PARITY_PASS',len(checked),flush=True)
    return cache

def raw_coordinate_inputs():
    """Only inference fields are retained; evaluation fields are not used."""
    old=[C.prediction_only(r) for r in C.rows(C.JOINT)]
    base={r['id']:r for r in old if r['method']=='BASE' and r['seed']==1}
    assert len(old)==5742 and len(base)==319
    result=list(old);reuse=C.read(C.REUSE);manifest={r['id']:r for r in C.original_manifest()}
    for method,arm in ALIASES.items():
        for seed in C.SEEDS:
            rel=reuse['source_bindings']['core']['dcp_predictions'][f'{arm}_seed{seed}']['prediction']['path']
            for r in C.read(C.SOURCE/rel)['records']:
                b=copy.deepcopy(base[r['id']]);idx=r['selected_index']
                q=np.full((9,2),np.nan) if idx is None else np.asarray(r['candidates'][idx]['keypoints_xy'],float)
                b.update(method=method,seed=seed,qFinal=q,detected=idx is not None,
                    coordinate_provenance='Existing SHA-bound '+arm+' seed-specific final candidate; no model forward')
                assert np.array_equal(q[8],np.asarray(b['q0'])[8],equal_nan=True)
                result.append(b)
    for method in ('D','L'):
        for seed in C.SEEDS:
            rel=reuse['source_bindings']['comparators'][f'H_{method}_seed{seed}']['path']
            frames=C.read(C.SOURCE/rel)['frames']
            for fid,b0 in base.items():
                candidates=frames[manifest[fid]['image']['path']]
                candidate=max(candidates,key=lambda c:float(c['score'])) if candidates else None
                b=copy.deepcopy(b0);q=np.full((9,2),np.nan) if candidate is None else np.asarray(candidate['keypoints_xy'],float)
                b.update(method=method,seed=seed,qFinal=q,detected=candidate is not None,
                    coordinate_provenance='Existing SHA-bound '+method+' seed-specific highest-score candidate; no GT choice')
                if candidate is not None:
                    assert candidate['score']==b['fixed_metadata']['candidate_metadata']['score']
                    assert np.array_equal(candidate['box_xyxy'],b['fixed_metadata']['candidate_metadata']['box_xyxy'])
                    assert np.array_equal(q[8],np.asarray(b['q0'])[8],equal_nan=True)
                result.append(b)
    for seed in C.SEEDS:
        rel=reuse['source_bindings']['comparators'][f'H_PoseFix_seed{seed}']['path']
        with np.load(C.SOURCE/rel,allow_pickle=False) as z:
            points={str(fid):np.array(q,copy=True) for fid,q in zip(z['ids'],z['points'][:,0,1])}
        for fid,b0 in base.items():
            b=copy.deepcopy(b0);q=points[fid]
            assert np.array_equal(q[8],np.asarray(b['q0'])[8],equal_nan=True)
            b.update(method='POSEFIX',seed=seed,qFinal=q,coordinate_provenance='Existing PoseFix-style RAW chain0/pass1 points[:,0,1]; uncapped; no current model forward')
            result.append(b)
    for name in ('dope','resnet18'):
        packet=C.read(C.SOURCE/f'data/pallet/results/pallet_n3_completion_v3/predictions/{name}_DEV319.json')
        for r in packet['frames']:
            for seed in C.SEEDS:
                nativebase=r['predictions']['base'];q0=np.full((9,2),np.nan) if nativebase['points'] is None else np.asarray(nativebase['points'],float)
                for method,pkey in [('BASE','base'),('N3_DIM_SYM',f'N3_seed{seed}')]:
                    p=r['predictions'][pkey];q=np.full((9,2),np.nan) if p['points'] is None else np.asarray(p['points'],float)
                    b=copy.deepcopy(base[r['id']]);b.update(backbone='DOPE' if name=='dope' else 'ResNet-18',method=method,seed=seed,
                        q0=q0,qFinal=q,prediction_support=p['valid'],detected=p['detected'],native_prediction_status=p['status'],
                        coordinate_provenance='Existing native '+name+' Base/N3 predictions; original pixels; no current model forward')
                    b['fixed_metadata']['candidate_metadata']={k:p.get(k) for k in ('bbox','score','confidence','status')}
                    assert np.array_equal(q[8],q0[8],equal_nan=True)
                    assert np.array_equal(p['valid'],nativebase['valid'])
                    result.append(b)
    assert len(result)==17*319*3, len(result)  # YOLO13 + two native paths each for DOPE/ResNet.
    assert len({key(r) for r in result})==len(result)
    return result

def posefix_parity(engine, coordinates):
    pf=[r for r in coordinates if r['method']=='POSEFIX']
    assert len(pf)==319*3
    seal('POSEFIX_COORDINATES',pf,'Existing first uncapped PoseFix RAW pass, fixed before 319-frame replay-parity.')
    actual=[]
    for r in pf:
        p,counts=engine.infer(r,'POSEFIX_PARITY');actual.append(dict(r,actual_pose=p,PnP_counts=counts))
    seal('POSEFIX_POSES',actual,'Production F only; reference values enter after all PoseFix poses have been sealed.')
    old=C.read(C.OLD_RAW)['comparators'];reuse=C.read(C.REUSE)
    targets=C.read(C.SOURCE/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json');math_api=C.math_api()
    checked=[];cache={};byseed={}
    for seed in C.SEEDS:
        previous={r['id']:r for r in old[f'H_PoseFix_seed{seed}']['pose_scores']};sc=[]
        for r in [r for r in actual if r['seed']==seed]:
            metric=engine.pose.metric((r['id'],r['actual_pose'],{k:np.asarray(v,float) if k!='order' else int(v) for k,v in targets[r['id']]['truth'].items()}))
            a=metric;b=previous[r['id']];availability=a['available']==b['available'];deltas={}
            if a['available'] and b['available']:
                deltas={k:abs(a[k]-b[k]) for k in ('translation_cm','rotation_deg','yaw_deg','IoU3D','ADDsym_m','ADDsym_normalized')}
            ok=availability and (not a['available'] or max(deltas.values(),default=0)<=1e-7)
            checked.append(dict(id=r['id'],seed=seed,availability_exact=availability,max_metric_delta=max(deltas.values(),default=0),PASS=ok))
            cache[key(r)]=r;sc.append(a)
        headline=reuse['contracts']['H']['methods']['PoseFix']['per_seed'][str(seed)]['headline']
        comparisons=[]
        for k,name in [('translation_cm','pose_translation_cm_median'),('rotation_deg','pose_rotation_deg_median'),('yaw_deg','pose_yaw_deg_median'),('IoU3D','pose_IoU3D_median')]:
            value=float(np.median([r[k] for r in sc if r['available']]))
            comparisons.append(dict(metric=name,actual=value,full_precision_H=headline[name],delta=abs(value-headline[name]),PASS=abs(value-headline[name])<=1e-7))
        pckmath=math_api.summary([corner_score(r,targets[r['id']],targets[r['id']]['matched'],math_api) for r in actual if r['seed']==seed])
        comparisons.append(dict(metric='full_PCK10_fraction',actual=pckmath['PCK']['10'],full_precision_H=headline['full_PCK10_fraction'],
            delta=abs(pckmath['PCK']['10']-headline['full_PCK10_fraction']),PASS=abs(pckmath['PCK']['10']-headline['full_PCK10_fraction'])<=1e-7))
        byseed[str(seed)]=comparisons
    ok=all(r['PASS'] for r in checked) and all(r['PASS'] for rows in byseed.values() for r in rows)
    C.write('POSEFIX_319_PARITY.json',dict(status='PASS' if ok else 'FAIL_OMIT_POSEFIX_ROWS',frames=319,seeds=list(C.SEEDS),
        tensor='points[:,0,1]',uncapped=True,per_ID_pose_comparisons=checked,full_precision_H_aggregate_comparisons=byseed,
        maximum_per_ID_metric_delta=max(r['max_metric_delta'] for r in checked),absolute_tolerance=1e-7,
        source_per_frame_sha256=C.sha(C.OLD_RAW),source_H_summary_sha256=C.sha(C.REUSE),
        actual_F_calls=len(pf),reference_inputs_in_F=False))
    print('POSEFIX_319_FULL_PRECISION_PARITY', 'PASS' if ok else 'FAIL_OMIT',flush=True)
    return ok,cache

def generate_subpix(coordinates, posefix_ok):
    from scripts.research.pallet_training_free_compare_20261007_v1.methods import correct,cap_points
    result=list(coordinates);images={r['id']:r for r in C.read(C.DOC/'INPUT_AUDIT.json')['images']};counts=Counter();decodes=0
    grouped={}
    for r in coordinates:
        if (r['backbone']!='YOLO' and r['method'] in ('BASE','N3_DIM_SYM')) or (r['method']=='POSEFIX' and posefix_ok):
            grouped.setdefault(r['id'],[]).append(r)
    for index,(fid,rr) in enumerate(grouped.items()):
        image=images[fid]['image'];path=C.SOURCE/image['path'];assert C.sha(path)==image['sha256']
        bgr=cv2.imread(str(path),cv2.IMREAD_COLOR);assert bgr is not None and list(bgr.shape[:2])==images[fid]['raw_hw']
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);decodes+=1;h,w=gray.shape
        for r in rr:
            q=np.asarray(r['qFinal'],float);q0=np.asarray(r['q0'],float);support=np.asarray(r['prediction_support'],bool)
            native,diag=correct(gray,q,support,'SUBPIX');final=cap_points(q0,native,w,h,support)
            out=copy.deepcopy(r);method='POSEFIX_SUBPIX' if r['method']=='POSEFIX' else 'SUBPIX' if r['method']=='BASE' else 'N3_THEN_SUBPIX'
            out.update(method=method,qFinal=final,qS=native,correction=diag,
                coordinate_provenance='Fixed cornerSubPix (5,5), (-1,-1), EPS|COUNT40,.001; original Base IMAGE-diagonal1% cap applied once at end',
                correction_reference_inputs=False,cap_reference='original Base q0',cap_px=.01*np.hypot(h,w))
            assert np.array_equal(final[8],q0[8],equal_nan=True)
            assert np.array_equal(final[~support],q0[~support],equal_nan=True)
            usable=support[:8]&np.isfinite(q0[:8]).all(1)&~(q0[:8]==-1).all(1)
            assert np.max(np.linalg.norm(final[:8][usable]-q0[:8][usable],axis=1),initial=0)<=out['cap_px']+1e-10
            counts[f"{r['backbone']}/{method}"]+=diag['algorithm_corner_calls'];result.append(out)
        if (index+1)%60==0:print('FIXED_SUBPIX_COORDINATES',index+1,319,flush=True)
    if not posefix_ok:result=[r for r in result if r['method'] not in ('POSEFIX','POSEFIX_SUBPIX')]
    assert len({key(r) for r in result})==len(result)
    expected=(22 if posefix_ok else 20)*319*3
    assert len(result)==expected,(len(result),expected)
    seal('RECT_COORDINATES',result,'All319 frozen outputs incl. new DOPE/ResNet and (if parityPASS) PoseFix subpixels; no reference field passed to coordinate generator.')
    C.write('COORDINATE_EXECUTION.json',dict(status='COMPLETE_SEALED',rows=len(result),images=319,gray_decodes=decodes,
        actual_subpix_corner_calls=dict(counts),model_forwards=0,new_training_updates=0,new_synthetic_images=0,
        cap_fraction_raw_IMAGE_diagonal=.01,cap_reference='original native Base',cap_applications_at_end=1,
        correction_inputs=['gray','initial_prediction','original_Base','prediction_support'],reference_inputs=False,
        coordinate_sha256=C.sha(C.DOC/'RECT_COORDINATES.jsonl.gz'),PoseFix_included=posefix_ok))
    return result

def target_context():
    """Evaluation-only references; called only after all prediction coordinates are sealed."""
    assert (C.DOC/'RECT_COORDINATES_SEAL.json').exists()
    targets={fid:t for fid,t in C.read(C.SOURCE/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json').items() if t['split']=='REAL_DEV'}
    assert len(targets)==319
    from scripts.research.pallet_n3_completion_v3 import evaluation as E
    native=E.load_dev319_truth(include_pose=False)
    native={r['id']:r for r in native};assert set(native)==set(targets)
    for fid,t in targets.items():
        assert np.array_equal(np.asarray(native[fid]['gt'],float),np.asarray(t['gt'],float),equal_nan=True)
        assert np.array_equal(native[fid]['valid'],t['valid'])
        assert np.array_equal(native[fid]['permutations'],t['permutations'])
    return targets,native,E

def square_rows():
    audit=C.read(C.ROOT/'_docs/experiments/pallet_square6d_manualpnp_20261010/INPUT_AUDIT.json')
    by={r['id']:r for r in audit['inputs'] if r['eligible']}
    symmetry=C.read(C.ROOT/'_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')
    group=next(r for r in symmetry['objects'] if r['object_type']=='plastic_standard_110x110x15')
    from scripts.evaluation.green_saved_labels_v1 import annotation_arrays
    from scripts.research.pallet_n3_completion_v3 import evaluation as E
    references={}
    for fid,a in by.items():
        annotation=C.read(C.SOURCE/a['annotation']['path'])
        gt,known=annotation_arrays(annotation)
        assert np.array_equal(np.asarray(gt[:8]),np.asarray(a['manual_points8']),equal_nan=True)
        h,w=a['raw_hw'];inside=np.isfinite(gt).all(1)&(gt[:,0]>=0)&(gt[:,0]<w)&(gt[:,1]>=0)&(gt[:,1]<h)
        selected=np.asarray(gt)[np.asarray(known)&inside]
        assert len(selected)>0
        box=np.r_[selected.min(0),selected.max(0)]
        valid=np.zeros(9,bool);valid[a['manual_in_frame_indices']]=True
        references[fid]=dict(gt=gt,valid=valid,permutations=group['permutations'],box=box)
    result=[];math_api=C.math_api()
    for r0 in C.rows(C.SQUARE):
        r=copy.deepcopy(r0);a=by[r['id']];r.update(population='SQUARE',grade='unclassified',
            backbone={'yolo':'YOLO','dope':'DOPE','resnet18':'ResNet-18'}[r['backbone']],raw_hw=a['raw_hw'],material='square',
            current_experiment_F_attempt=False,current_experiment_pose_reused=True)
        r['fixed_metadata']['canonical_symmetry_order']=4
        target=references[r['id']];metadata=r['fixed_metadata']['base']
        if r['backbone']=='YOLO':
            idx=metadata['selected_index'];box=None if idx is None else metadata['candidates'][idx]['box_xyxy']
        else:box=metadata['bbox']
        matched=bool(r['detected'] and E._iou(box,target['box'])>=E.MATCH_IOU)
        r['corner']=corner_score(r,target,matched,math_api)
        r.update(evaluation_reference_points=target['gt'],evaluation_reference_valid=target['valid'],evaluation_permutations=group['permutations'])
        result.append(r)
    assert len(result)==4248
    C.write_rows('SQUARE_ROWS.jsonl.gz',result)
    C.write('SQUARE_REUSE_RECEIPT.json',dict(status='COMPLETE',rows=4248,frames=118,seed_count=3,
        source_sha256=C.sha(C.SQUARE),new_F_calls=0,new_model_forwards=0,reference='Existing manual-only118 reference; four-point allowance and C4; single-session feasibility only',
        denominator='Existing manual-in-frame valid reference corners;597 across118; missing predictions diagonal penalty',
        output_sha256=C.sha(C.DOC/'SQUARE_ROWS.jsonl.gz')))
    return len(result)

def run():
    started=time.monotonic();C.method_lock();C.verify_inputs();cv2.setNumThreads(1)
    assert not (C.DOC/'STAGE0_F_PARITY.json').exists() and not (C.DOC/'RECT_ROWS.jsonl.gz').exists()
    C.write('EVALUATION_SOURCE_LOCK.json',dict(status='LOCKED_BEFORE_ANY_ACTUAL_F_OR_NEW_COORDINATE',
        source=[C.binding(p) for p in (C.ROOT/'scripts/research/pallet_paper_tables_245_20261011/common.py',
            C.ROOT/'scripts/research/pallet_paper_tables_245_20261011/preflight.py',C.ROOT/'scripts/research/pallet_paper_tables_245_20261011/evaluate.py')],
        input_audit_sha256=C.sha(C.DOC/'INPUT_AUDIT.json'),method_lock_sha256=C.sha(C.DOC/'METHOD_LOCK.json')))
    engine=ActualF();cache=stage0(engine)
    coordinates=raw_coordinate_inputs();pf_ok,pfcache=posefix_parity(engine,coordinates);cache.update(pfcache)
    coordinates=generate_subpix(coordinates,pf_ok)
    inferred=[]
    for index,r in enumerate(coordinates):
        if key(r) in cache:
            saved=cache[key(r)];assert C.digest(saved['qFinal'])==C.digest(r['qFinal'])
            p=saved['actual_pose'];counts=saved['PnP_counts'];reused=True
        else:p,counts=engine.infer(r,'FULL319');reused=False
        inferred.append(dict(r,actual_pose=p,PnP_counts=counts,current_experiment_F_attempt=not reused,
            actual_F_result_reused_from_current_parity=reused,reference_inputs_in_F=False))
        if (index+1)%1000==0:print('RECT_ACTUAL_F',index+1,len(coordinates),'newcalls',sum(engine.calls.values()),flush=True)
    seal('RECT_POSES',inferred,'All production-F outputs fixed before final scoring. Every infer signature uses only coordinates/K/registered dimensions/source flag.')
    targets,native,E=target_context();math_api=C.math_api();rows=[]
    for r in inferred:
        fid=r['id'];matched=targets[fid]['matched'] if r['backbone']=='YOLO' else E._iou(r['fixed_metadata']['candidate_metadata']['bbox'],native[fid]['box'])>=E.MATCH_IOU
        rows.append(scored(r,r['actual_pose'],targets[fid],matched,math_api,engine.pose))
    C.write_rows('RECT_ROWS.jsonl.gz',rows)
    sq_count=square_rows()
    C.verify_inputs()
    for b in C.read(C.DOC/'EVALUATION_SOURCE_LOCK.json')['source']:C.verify_binding(b)
    C.write('EVALUATION_EXECUTION.json',dict(status='COMPLETE',rectangle_rows=len(rows),rectangle_frames=319,square_rows=sq_count,square_frames=118,
        actual_F_calls=dict(engine.calls),actual_F_calls_total=sum(engine.calls.values()),actual_PnP_calls=dict(engine.pnp),
        expected_actual_F_unique_rows=len(rows),current_parity_F_results_reused=len(cache),
        network_forwards=0,new_training_updates=0,new_synthetic_images=0,all_existing_inputs_SHA_unchanged=True,
        correction_reference_inputs=False,F_reference_inputs=False,GT_entered_only_for_postsealed_scoring=True,
        PoseFix_319_parity='PASS' if pf_ok else 'FAIL_OMIT_POSEFIX_ROWS',
        outputs={'RECT_ROWS.jsonl.gz':C.sha(C.DOC/'RECT_ROWS.jsonl.gz'),'SQUARE_ROWS.jsonl.gz':C.sha(C.DOC/'SQUARE_ROWS.jsonl.gz')},
        elapsed_seconds=time.monotonic()-started))
    print('PAPER_FIXED_EVALUATION_COMPLETE',len(rows),sq_count,dict(engine.calls),round(time.monotonic()-started,2),flush=True)

if __name__=='__main__':run()

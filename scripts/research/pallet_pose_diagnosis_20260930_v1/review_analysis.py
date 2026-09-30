"""Repair review fields and complete cached analyses; no model inference/fitting.

Run source, repair, statistics separately. Original artifacts are archived before
replacement so historical manifests remain auditable.
"""
import argparse
import copy
import csv
import itertools
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from . import run as C
from . import scoring as S
from .close import scalar_check

ARCHIVE = C.RAW / 'before_github_review'

def write(path, value):
    path = Path(path)
    assert path.is_relative_to(C.DOC) or path.is_relative_to(C.RAW)
    if path.exists():
        old = ARCHIVE / path.relative_to(C.ROOT)
        old.parent.mkdir(parents=True, exist_ok=True)
        if not old.exists():
            shutil.copy2(path, old)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(C.M.clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.write_text(text)

def csvwrite(path, rows):
    import io
    stream = io.StringIO()
    keys = list(dict.fromkeys(k for r in rows for k in r))
    writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
    writer.writeheader()
    for r in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    write(path, stream.getvalue())

def source():
    from challenge.yolo_pose_one_model.spatial_concat_scratch.build_probe_metadata import fixed_dimensions
    path = C.ROOT / '_docs/experiments/pallet_clean19_pose_sensitive_diag_v1/SOURCE_GEOMETRY_BINDING.json'
    binding = C.read(path)
    C.verify(binding['table'])
    table = dict(np.load(C.ROOT / binding['table']['path']))
    index = {str(s): i for i, s in enumerate(table['stems'])}
    meta, truth = C.O.D.Pose.metadata('SYNTH_HELDOUT')
    _, real_truth = C.O.D.Pose.metadata('REAL_DEV')
    real_orders = sorted({real_truth[r['id']]['order'] for r in C.metadata()})
    assert real_orders == [2]
    records = []
    for b in binding['records']:
        for key in ('image', 'label', 'renderer'):
            C.verify(b[key])
        fid = b['id']; j = index[fid]
        ann = C.read(C.ROOT / b['renderer']['path']); obj = ann['objects'][0]
        intr = ann['camera_data']['intrinsics']; pad = table['pad'][j]
        k4 = np.array([intr['fx'], intr['fy'], intr['cx'] + pad, intr['cy'] + pad])
        np.testing.assert_allclose(k4, table['K'][j], atol=1e-10)
        K = np.array([[k4[0], 0, k4[2]], [0, k4[1], k4[3]], [0, 0, 1.]])
        T = np.array(obj['pose_transform'])
        np.testing.assert_allclose(T[:3, :3], table['R'][j], atol=1e-10)
        np.testing.assert_allclose(T[:3, 3], table['t'][j], atol=1e-10)
        _, dims, perm, case = fixed_dimensions(obj, fid)
        np.testing.assert_allclose(dims, table['dims'][j], atol=1e-10)
        width, height = Image.open(C.ROOT / b['image']['path']).size
        assert (width, height) == (ann['camera_data']['width'] + 2 * pad, ann['camera_data']['height'] + 2 * pad)
        X = table['Xcf'][j]
        corners = np.array(list(itertools.product(*[[-d/2, d/2] for d in dims])))
        assignment = np.linalg.norm(X[:, None] - corners[None], axis=-1).argmin(axis=1)
        assert len(set(assignment)) == 8
        np.testing.assert_allclose(X, corners[assignment], atol=1e-10)
        camera = X @ T[:3, :3].T + T[:3, 3]
        projected = camera @ K.T; projected = projected[:, :2] / projected[:, 2:]
        renderer_error = float(np.linalg.norm(projected - (np.array(obj['projected_cuboid'])[:8] + pad), axis=1).max())
        label = np.asarray((C.ROOT / b['label']['path']).read_text().split(), float)[5:].reshape(9, 3)
        valid = label[:8, 2] > 0
        label_error = float(np.linalg.norm(projected[valid] - label[:8, :2][valid] * [width, height], axis=1).max()) if valid.any() else None
        assert renderer_error <= .05 and (label_error is None or label_error <= .05)
        nativeK, current_dims, source_flag = meta[fid]
        preparedK = nativeK.copy(); preparedK[:2, 2] += pad
        np.testing.assert_allclose(preparedK, K, atol=1e-10)
        np.testing.assert_allclose(current_dims, dims, atol=1e-10)
        for key, expected in [('R', T[:3, :3]), ('t', T[:3, 3])]:
            np.testing.assert_allclose(truth[fid][key], expected, atol=1e-10)
        assert source_flag is True
        # An exact renderer-coordinate control, not a network prediction.
        solved = C.O.D.Pose.solve(X, projected, K, np.ones(8, bool))
        assert solved is not None
        rr, tt, _ = solved
        translation_error = float(np.linalg.norm(tt - T[:3, 3]))
        rotation_error = float(np.linalg.norm(rr - T[:3, :3]))
        assert translation_error < 1e-5 and rotation_error < 1e-5
        order = int(truth[fid]['order'])
        group = C.O.D.Pose.rotations(order)
        for Q in group:
            np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=1e-10)
            assert abs(np.linalg.det(Q) - 1) < 1e-10
        records.append(dict(id=fid,table_index=j,pad_px=float(pad),image=b['image'],label=b['label'],renderer=b['renderer'],
            renderer_projection_max_px=renderer_error,label_projection_max_px=label_error,index_bijection=assignment.tolist(),
            dimensions_m=list(dims),current_source_symmetry_order=order,current_natural_symmetry_order=2,
            renderer_control_translation_m=translation_error,renderer_control_rotation_matrix_error=rotation_error,
            K_native_plus_pad_matches=True,pose_matches=True,origin='renderer cuboid center'))
    out = dict(status='VERIFIED_WITH_EXPLICIT_DOMAIN_CONTRACTS',N=len(records),source_binding=C.bind(path),sidetable=binding['table'],
        evaluator=C.bind(Path(C.O.D.Pose.__file__)),index_builder=C.bind(C.ROOT/'scripts/research/pallet_translation_loss_v1/build_geometry_sidetable.py'),
        source_symmetry_orders=dict(Counter(r['current_source_symmetry_order'] for r in records)),natural_symmetry_orders=real_orders,
        source_frame_mapping='Current source infer converts camera-facing R by the W/D correspondence then diag(1,-1,-1) to renderer physical frame; natural infer has no final renderer flip.',
        source_and_natural_not_identical_contract=True,source_operational_output_status='SKIPPED_NO_FULL125_OPERATIONAL_XY_CACHE',
        no_old_source_rotation_reused=True,no_model_forward=True,renderer_PnP_is_geometry_check_not_model_accuracy=True,records=records)
    write(C.DOC/'SOURCE_CONTRACT_REVIEW.json', out)
    print('SOURCE_REVIEW',len(records),out['source_symmetry_orders'],flush=True)

def repair():
    rows = C.metadata(); meta = {r['id']:r for r in rows}
    base = C.read(C.RAW/'INPUT_PREDICTIONS.json')['identity']
    refs = S.references(); support = C.read(C.RAW/'E4_VISIBILITY_SUPPORT.json')
    twod = {}; predictions = {}
    for kind in ('baseline','occluded','visible','both'):
        twod[kind] = {}; predictions[kind] = {}
        for r in rows:
            fid = r['id']; p = copy.deepcopy(base[fid]); candidate = S.CORE.selected(p)
            q = np.array(candidate['keypoints_xy'])
            sets = support[fid]['replacement_indices']
            replace = sets['occluded'] + sets['visible'] if kind == 'both' else sets.get(kind, [])
            for predicted, canonical in replace:
                q[predicted] = refs[fid]['gt'][canonical]
            candidate['keypoints_xy'] = q.tolist()
            predictions[kind][fid] = p
            twod[kind][fid] = S.two_metric(fid,p,refs[fid],r['hw'])
    write(C.RAW/'E4_REVIEW_PREDICTIONS.json',predictions)
    write(C.RAW/'E4_REVIEW_2D_METRICS.json',twod)
    archived_frame = ARCHIVE/(C.DOC/'FRAME_RESULTS.csv').relative_to(C.ROOT)
    frame = list(csv.DictReader((archived_frame if archived_frame.exists() else C.DOC/'FRAME_RESULTS.csv').open()))
    candidates = C.read(C.RAW/'E1_CANDIDATES.json')
    c4 = C.read(C.RAW/'E4_REFERENCE_INTERVENTION_CANDIDATES.json')
    _,gt = C.O.D.Pose.metadata('REAL_DEV')
    corrected_names = corrected2d = validated = 0
    e4metrics = C.read(C.RAW/'E4_METRICS.json')
    for row in frame:
        fid = row['id']; row['twoD_value_scope'] = 'condition_output'
        row['GEO_free_name'] = row.get('GEO_name','')
        row['selected_hypothesis_for_pose'] = row.get('GEO_name','')
        if row['stage']=='E1' and row['condition']=='held_identity':
            held = candidates['identity'][fid]['GEO_name']
            corrected_names += row['GEO_name'] != held
            row['selected_hypothesis_for_pose'] = row['GEO_name'] = held
        if row['stage']=='E4':
            kind,mode=row['condition'].rsplit('_',1); m=twod[kind][fid]
            for column,key in [('twoD_frame_mean_px','frame_mean_px'),('twoD_corner_count','corners'),('twoD_canonical_errors','canonical_errors')]:
                value=m.get(key); row[column]=json.dumps(value) if isinstance(value,list) else value
            rec=c4[kind][fid]; free=rec['GEO_name']
            chosen=free if mode=='GEO' else candidates['identity'][fid]['GEO_name']
            row.update(GEO_free_name=free,GEO_name=chosen,selected_hypothesis_for_pose=chosen,D9_name=rec['selected_name'],GEO_fallback=rec['GEO_fallback'])
            pose=next(h['pose'] for h in rec['hypotheses'] if h['name']==chosen)
            scalar_check(pose,e4metrics[row['condition']][fid],gt[fid]);validated+=1
            corrected2d+=kind!='baseline'
        row['twoD_NA_reason']='UNMATCHED_OBJECT_OR_NO_VALID_SUPERVISED_CORNERS' if row.get('twoD_frame_mean_px') in ('',None) else ''
    assert len(frame)==2964 and corrected_names==33 and corrected2d==768 and validated==1024
    csvwrite(C.DOC/'FRAME_RESULTS.csv',frame)
    # Preserve the pose values bit-for-bit; independently check all changed semantics.
    old=list(csv.DictReader((ARCHIVE/(C.DOC/'FRAME_RESULTS.csv').relative_to(C.ROOT)).open()))
    for a,b in zip(old,frame):
        for key in ('translation_cm','rotation_deg','delta_T_cm','delta_R_deg','valid'):
            assert a[key]==b[key]
        if b['condition']=='held_identity':
            assert b['selected_hypothesis_for_pose']==candidates['identity'][b['id']]['GEO_name']
    write(C.DOC/'CSV_FIELD_REPAIR.json',dict(status='PASSED',rows=2964,held_names_corrected=corrected_names,E4_intervention_rows=corrected2d,
        E4_pose_rechecks=validated,all_existing_T_R_unchanged=True,fields=dict(GEO_name='Hypothesis actually used for this row pose',GEO_free_name='Hypothesis selected by free GEO',selected_hypothesis_for_pose='Explicit actual pose hypothesis',twoD_value_scope='condition_output'),
        historical_csv=C.bind(ARCHIVE/(C.DOC/'FRAME_RESULTS.csv').relative_to(C.ROOT))))
    print('CSV_REPAIRED',corrected_names,corrected2d,flush=True)

def comparisons():
    e1=C.read(C.RAW/'E1_POSE_METRICS.json'); comp={}
    for a in e1:
        if a!='identity':comp['E1/'+a]=(e1['identity'],e1[a])
    comp['E1/FULL125_vs_PRIOR1']=(e1['PRIOR1'],e1['FULL125'])
    cm=C.read(C.RAW/'E1_CANDIDATE_METRICS.json'); choices=C.read(C.RAW/'E2_ORACLE_CHOICES.json')
    for a in choices:
        for kind in ('T_best','R_best'):
            values={i:next((h['metric'] for h in cm[a][i] if h['name']==v[kind]),dict(id=i,available=False)) for i,v in choices[a].items()}
            comp['E2/'+a+'/'+kind]=(e1[a],values)
    for a,v in C.read(C.RAW/'E2_BOX_ORACLE_METRICS.json').items():comp['E2/'+a+'/best_box']=(e1[a],v)
    for kind,arms in C.read(C.RAW/'E3_POSE_METRICS.json').items():
        for a,conds in arms.items():
            if a=='identity':continue
            for cond,v in conds.items():comp[f'E3/{kind}/{a}/{cond}_vs_identity']=(arms['identity'][cond],v)
            for before,after in [('CO','OO'),('OO','nativeOO'),('CC','OC'),('CC','OO')]:
                comp[f'E3/{kind}/{a}/{after}_vs_{before}']=(conds[before],conds[after])
    e4=C.read(C.RAW/'E4_METRICS.json')
    for a,v in e4.items():
        if not a.startswith('baseline'):comp['E4/'+a]=(e4['baseline_'+a.rsplit('_',1)[1]],v)
    stress=C.read(C.RAW/'E5_STRESS_POSE_METRICS.json')
    for a,kinds in stress.items():
        if a!='identity':
            for kind,v in kinds.items():comp['E5stress/'+a+'/'+kind]=(stress['identity'][kind],v)
        comp['E5stress/'+a+'/correlated_vs_independent']=(kinds['independent'],kinds['correlated'])
    ft=C.read(C.RAW/'E6_POSE_METRICS.json')
    for a in ('identity','FULL125','OLD_REF217'):comp['E6/REALFT_A_vs_'+a]=(e1[a],ft)
    return comp

def statistics():
    rows=C.metadata();provenance=C.read(C.RAW/'E7_REFERENCE_PROVENANCE.json')
    candidates=C.read(C.RAW/'E1_CANDIDATES.json')['identity'];two=C.read(C.RAW/'E1_2D_METRICS.json')['identity']
    result={};recordrows=[];stratarows=[];frames=[]
    for label,(before,after) in comparisons().items():
        assert set(before)==set(after)
        result[label]={}
        for pop in ('NATURAL99','CLEAN29'):
            rr=[r for r in rows if r['id'] in before and (r['severity']=='CLEAN')==(pop=='CLEAN29')]
            if not rr:continue
            paired=C.paired(before,after,rr,True)
            result[label][pop]=dict(N=len(rr),recordings=dict(Counter(r['recording'] for r in rr)),before=C.summarize(before[r['id']] for r in rr),after=C.summarize(after[r['id']] for r in rr),paired=paired)
            groupings={('recording',rec):[r for r in rr if r['recording']==rec] for rec in sorted({r['recording'] for r in rr})}
            for field in ('occlusion','elevation_bin','distance_bin','size_bin','view_bin','baseline_WD','baseline_matching'):
                def val(r):
                    i=r['id']
                    if field=='baseline_WD':return candidates[i]['GEO_name']
                    if field=='baseline_matching':return str(two[i]['matched'])
                    return (provenance[i]['condition_metadata'] or {}).get(field,'UNKNOWN')
                for value in sorted({val(r) for r in rr}):groupings[(field,value)]=[r for r in rr if val(r)==value]
            for (field,value),sub in groupings.items():
                ids=[r['id'] for r in sub]; bs=C.summarize(before[i] for i in ids);ass=C.summarize(after[i] for i in ids);pp=C.paired(before,after,sub,False)
                flat=dict(comparison=label,population=pop,field=field,value=value,N=len(ids),recordings=dict(Counter(r['recording'] for r in sub)),before_valid=bs['valid_pose'],after_valid=ass['valid_pose'])
                for key in ('translation_cm','rotation_deg'):
                    flat['before_'+key]=bs['conditional'][key]['median'];flat['after_'+key]=ass['conditional'][key]['median']
                    flat['difference_of_medians_'+key]=pp['difference_of_conditional_medians'][key]
                    flat['median_of_frame_differences_'+key]=pp['median_of_common_frame_differences'][key]
                    flat['before_P90_'+key]=bs['conditional'][key]['P90'];flat['after_P90_'+key]=ass['conditional'][key]['P90']
                flat['common_valid_frames']=pp['common_valid_frames']
                flat['direction_counts']=pp['direction_counts_tolerance_1e_7']
                (recordrows if field=='recording' else stratarows).append(flat)
            if label.startswith('E2/'):
                for r in rr:
                    i=r['id'];b=before[i];a=after[i]
                    frames.append(dict(comparison=label,id=i,recording=r['recording'],population=pop,before=b,after=a))
        print('E7',label,flush=True)
    write(C.DOC/'E7_ALL_COMPARISONS.json',dict(repeats=2000,seed=20260929,interpretation='Exploratory reused DEV; paired whole-recording sampling; natural and clean separate; undefined draws retained; metadata strata descriptive only.',comparisons=result,
        E5_train=dict(E5_A='Historical TRAIN253, one recording REC_001. Target following only; no independent T/R reference or multi-recording uncertainty.',E5_B='TRAIN29 nativeOO, same REC_001. E5B_SUMMARY and E5B_FRAME_REVIEW.csv preserve paired 2D errors. A one-recording bootstrap is degenerate; cross-recording interval not identifiable.')))
    csvwrite(C.DOC/'E7_BY_RECORDING_ALL.csv',recordrows)
    csvwrite(C.DOC/'E7_METADATA_STRATA_ALL.csv',stratarows)
    csvwrite(C.DOC/'E2_FRAME_REVIEW.csv',frames)
    train=[]
    for fid,kinds in C.read(C.RAW/'E5B_ERRORS.json').items():
        for kind,arms in kinds.items():
            for arm,v in arms.items():
                train.append(dict(id=fid,recording='REC_001',condition=kind,model=arm,**v))
    csvwrite(C.DOC/'E5B_FRAME_REVIEW.csv',train)
    print('STATISTICS_DONE',len(result),len(recordrows),len(stratarows),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['source','repair','statistics']);args=parser.parse_args()
    start,cpu=time.monotonic(),time.process_time();C.torch.set_num_threads(2);C.cv2.setNumThreads(1)
    globals()[args.stage]()
    import resource
    write(C.DOC/('REVIEW_COST_'+args.stage+'.json'),dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,image_forwards=0,new_fits=0,GPU_seconds=0,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

if __name__=='__main__':main()

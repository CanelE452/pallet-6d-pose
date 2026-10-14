"""Independent scalar CAL max/fold audit; no model fitting or pose solves."""
from pathlib import Path
import gzip
import hashlib
import json
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'scripts/research/pallet_visibility_curriculum_20261013_v1').is_dir())
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1.truncation_interval_v4_2 import load_truncation_interval_model

F=HERE.parent;M=F/'truncation_interval_model_CAL_v4_2';V=F/'truncation_pose_validation_v4_2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    full=load_truncation_interval_model(M/'MODEL.json');logo=read(M/'CAL_LOGO_REPORT.json');seal=read(M/'MODEL_SEAL.json')
    assert sha(M/'MODEL.json')==seal['MODEL_sha256'] and sha(M/'CAL_LOGO_REPORT.json')==seal['LOGO_sha256']
    assert seal['TEST_files_opened'] is False and seal['fit_partition']=='CAL'
    assert sha(F/'TRUNCATION_INTERVAL_FIX_PLAN_V1.json')==seal['recipeplan_sha256']
    assert sha(F/'TRUNCATION_INTERVAL_CAL_ACCEPTANCE_V1.json')==seal['CAL_acceptance_sha256']
    for p,h in seal['source_sha256'].items():assert sha(ROOT/p)==h,(p,h)
    for key in ['CAL_cache','CAL_annotations','CAL_validation_protocol']:
        assert sha(ROOT/seal[key]['path'])==seal[key]['sha256']
    with gzip.open(ROOT/seal['CAL_cache']['path'],'rt') as stream:rows=list(map(json.loads,stream))
    annotations=read(ROOT/seal['CAL_annotations']['path']);assert annotations['TEST_rows']==0 and annotations['FINAL_rows']==0
    assert len(rows)==46 and set(annotations['rows'])=={r['id'] for r in rows}
    points=[];compatible=[];phases={};initial_S_skips=0
    for row in rows:
        assert row['partition']=='CAL';label=annotations['rows'][row['id']];assert label['metadata']['partition']=='CAL'
        geom=row['runtime']['initial_geometry'];truth=label['target']['truth'];mapping=np.asarray(label['metadata']['native_to_reference_corner_ids'],int)
        source=np.asarray(geom['native_xyz'])@np.asarray(geom['physical_registration_Q'])
        reference=C.cuboid(*truth['body_xyz'])[mapping]@(np.asarray(truth['body_R']).T@np.asarray(truth['R']))
        pairs=[]
        for k in range(truth['order']):
            a=2*np.pi*k/truth['order'];co,si=np.cos(a),np.sin(a);sym=np.array([[co,0,si],[0,1,0],[-si,0,co]])
            pairs.append((float(np.linalg.norm(source@sym.T-reference,axis=1).max()),k))
        gap,phase=min(pairs);phases[row['id']]=phase
        if gap>=1e-6:continue
        compatible.append(row['id'])
        scale=row['runtime']['ot_domains']['nonself_prediction_scale_px']
        for k in range(8):
            if geom['initial_self_hidden'][k]:initial_S_skips+=1;continue
            human=label['metadata']['human_states_native'][k]
            if human not in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED') or not label['target']['valid'][int(mapping[k])]:continue
            # Fixed native mapping, not any nearest error/pose/corner assignment.
            raw=np.asarray(row['runtime']['original_raw_points'][k]);target=np.asarray(label['target']['gt'][int(mapping[k])])
            error=float(np.linalg.norm(raw-target));points.append(dict(id=row['id'],session=row['session'],native_corner=k,human_state=human,raw_error_px=error,nonself_prediction_scale_px=scale,normalized_error=error/scale))
    assert len(compatible)==44 and len(points)==264
    maximum=max(points,key=lambda p:p['normalized_error'])
    assert maximum==full['max_error_point'] and maximum['id']=='wood_184309:000372' and maximum['native_corner']==0
    assert maximum['normalized_error']==full['radius_ratio']==.544750754221928 and phases[maximum['id']]==1
    assert [r['id'] for r in rows]==full['fit_ids'] and compatible==full['compatible_ids']
    folds={}
    for s,m in logo['fold_models'].items():
        m=load_truncation_interval_model(m);train=[r for r in rows if r['session']!=s];held=[r for r in rows if r['session']==s]
        trainpoints=[p for p in points if p['session']!=s];heldpoints=[p for p in points if p['session']==s];mx=max(trainpoints,key=lambda p:p['normalized_error'])
        assert m['heldout_session']==s and m['fit_ids']==[r['id'] for r in train]
        assert not set(m['fit_ids']) & {r['id'] for r in held}
        assert m['eligible_point_count']==len(trainpoints) and m['radius_ratio']==mx['normalized_error'] and m['max_error_point']==mx
        covered=sum(p['normalized_error']<=m['radius_ratio']+1e-12 for p in heldpoints)
        report=logo['coverage']['folds'][s]
        assert report['eligible_points']==len(heldpoints) and report['covered_points']==covered
        assert report['frame_ids']==[r['id'] for r in held]
        folds[s]=dict(train_frames=len(train),train_eligible_points=len(trainpoints),radius_ratio=m['radius_ratio'],heldout_frames=len(held),heldout_eligible_points=len(heldpoints),heldout_covered_points=covered,heldout_missed_points=len(heldpoints)-covered,model_sha256=m['model_sha256'])
    assert logo['coverage']['full']['covered_points']==264
    protocol=read(V/'PROTOCOL.json');inputseal=read(V/'INPUT_SEAL.json')
    assert sha(V/'PROTOCOL.json')==inputseal['protocol_sha256'] and sha(V/'SOLVER_INPUTS.jsonl.gz')==inputseal['inputs_sha256']
    for p,h in protocol['protected_user_sha256'].items():assert sha(ROOT/p)==h
    for p,h in seal['source_sha256'].items():assert sha(V/'code_snapshot'/p)==h
    result={'schema':'INDEPENDENT_CAL_INTERVAL_EXPORT_AUDIT_V4_2','status':'PASS','model_sha256':seal['MODEL_sha256'],'model_seal_sha256':sha(M/'MODEL_SEAL.json'),'independent_source':{'path':str(Path(__file__).relative_to(ROOT)),'sha256':sha(__file__)},'CAL_only':True,'CAL_rows':46,'compatible44':True,'eligible_points':264,'full_radius_ratio':full['radius_ratio'],'max_error_point':maximum,'fixed_geometry_native_phase_of_max':1,'initial_S_coordinates_skipped_before_raw_GT_index':True,'initial_S_skips':initial_S_skips,'source_human_counts':{h:sum(p['human_state']==h for p in points) for h in ['DIRECT_VISIBLE','EXTERNAL_OCCLUDED']},'folds':folds,'all_full_fold_maxima_untrimmed_and_exact':True,'fit_scope_and_model_descriptors_bound':True,'sources34_current_and_snapshot_match':True,'protected_user6_match':True,'new_fits_or_pose_solves_or_RGB_reads':0,'TEST_files_opened':False,'future_or_physical_coverage_claim':False}
    out=HERE/'MODEL_EXPORT_AUDIT_V1.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

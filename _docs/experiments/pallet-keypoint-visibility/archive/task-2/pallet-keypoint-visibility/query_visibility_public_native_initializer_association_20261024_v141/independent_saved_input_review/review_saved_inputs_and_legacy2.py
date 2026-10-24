"""Independent saved source-output arithmetic only; no RGB/GT/model access."""
from pathlib import Path
import ast,hashlib,json,struct,sys
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
OUT=V/'actual_public_initializer_canonical_transport_v2'
SEAL=OUT/'PREDICTION_SEAL.json'
OLD=P/'query_visibility_matched_query_scalar_20261023_v94/pallox_actual_initializer/FIXED100'
ROI=P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
LEGACY={'0.55m_lab:0017':'PALLOX50:0.55m_lab:0016_0017:after','1.23m_hall:0012':'PALLOX50:1.23m_hall:0012_0013:before'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return[json.loads(l)for l in Path(p).read_text().splitlines()if l]
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def equal(a,b):
    if type(a)is not type(b):return False
    if isinstance(a,float):return struct.pack('!d',a)==struct.pack('!d',b)
    if isinstance(a,dict):return a.keys()==b.keys()and all(equal(a[k],b[k])for k in a)
    if isinstance(a,list):return len(a)==len(b)and all(equal(x,y)for x,y in zip(a,b))
    return a==b
def nullable(a):
    if isinstance(a,np.ndarray):return nullable(a.tolist())
    if isinstance(a,np.generic):return nullable(a.item())
    if isinstance(a,float):return a if np.isfinite(a)else None
    if isinstance(a,list):return[nullable(x)for x in a]
    return a
def bytes_equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def legacy_rows(path):
    # Only the two predeclared frame rows/their16 query rows are deserialized.
    found=[]
    with Path(path).open()as f:
        for line in f:
            if any(('"'+ident+'"')in line for ident in LEGACY.values()):found.append(json.loads(line))
    return found
def main():
    assert not(H/'REVIEW.json').exists()
    seal=read(SEAL);assert seal['completed']and seal['PASS']and seal['frames']==16 and seal['rows']==128 and seal['GT_read']is False and seal['historical_predictions_read']is False
    for b in seal['outputs']:check(b)
    inp=lines(check(seal['inputs']));proposals=lines(OUT/'FROZEN_PROPOSALS_ALL16_INPUT_ONLY.jsonl');queries=lines(OUT/'ORIGINAL_NATIVE_QUERY128_INPUT_ONLY.jsonl')
    assert len(inp)==len(proposals)==16 and len(queries)==128
    with np.load(OUT/'ORIGINAL_NATIVE_QUERY128_AND_VALID9X9_INPUT_ONLY.npz',allow_pickle=False)as f:q={k:f[k].copy()for k in f.files}
    with np.load(OUT/'ALL_CANDIDATES_RAW9_AND_BOXES_INPUT_ONLY.npz',allow_pickle=False)as f:c={k:f[k].copy()for k in f.files}
    shapes=dict(query_xy=((128,2),np.float64),L=((128,),np.float64),native_grid_valid=((128,9,9),bool),global_index=((128,),np.int32),frame_index=((128,),np.int32),native_slot=((128,),np.int8))
    for key,(shape,dtype)in shapes.items():assert q[key].shape==shape and q[key].dtype==dtype,key
    for key in('finite_non_sentinel','continuous_valid','nearest_valid','RGBcentre_valid','no_detection'):assert q[key].shape==(128,)and q[key].dtype==bool,key
    assert set(q)==set(shapes)|{'finite_non_sentinel','continuous_valid','nearest_valid','RGBcentre_valid','no_detection','query_id','frame_id','split'}
    assert q['query_id'].dtype.kind==q['frame_id'].dtype.kind==q['split'].dtype.kind=='U'
    assert np.array_equal(q['global_index'],np.arange(128))and np.array_equal(q['frame_index'],np.repeat(np.arange(16),8))and np.array_equal(q['native_slot'],np.tile(np.arange(8),16))
    assert c['frame_offsets'].shape==(17,)and c['selected_index'].shape==(16,)and c['frame_offsets'].dtype==c['selected_index'].dtype==np.int32
    assert c['frame_offsets'][0]==0 and c['frame_offsets'][-1]==120 and(np.diff(c['frame_offsets'])>=0).all()
    for key,shape in dict(box_xyxy=(120,4),keypoints_xy=(120,9,2),keypoints_conf=(120,9),score=(120,)).items():assert c[key].shape==shape and c[key].dtype==np.float64
    nodes=[n for n in ast.parse(ROI.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='native_grid'];ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROI),'exec'),ns)
    all_flags=[];all_masks=[];nodet=[];frame_checks=[]
    inputkeys=set(inp[0]);assert inputkeys=={'frame_index','id','source','source_id','split','image','image_sha256','decoded_RGB_sha256','raw_hw','decoder'}
    extra={'candidates','selected_index','input_shape','canvas_shape','added_border','affine_gain','affine_offset','actual_original_initializer','source_emulation'}
    for fi,(source,proposal)in enumerate(zip(inp,proposals)):
        assert set(proposal)==inputkeys|extra and all(equal(source[k],proposal[k])for k in inputkeys)
        assert proposal['frame_index']==fi and proposal['actual_original_initializer']is True and proposal['source_emulation']is False
        lo,hi=map(int,c['frame_offsets'][fi:fi+2]);assert hi-lo==len(proposal['candidates'])
        selected=None if lo==hi else int(np.argmax(c['score'][lo:hi]));assert selected==proposal['selected_index']and c['selected_index'][fi]==(-1 if selected is None else selected)
        for j,candidate in enumerate(proposal['candidates']):
            assert candidate['candidate_index']==j
            for key in('box_xyxy','keypoints_xy','keypoints_conf','score'):assert equal(candidate[key],nullable(c[key][lo+j])),(fi,j,key)
        section=slice(fi*8,(fi+1)*8);Hh,Ww=source['raw_hw'];missing=selected is None
        if missing:
            points=np.full((8,2),np.nan,np.float64);L=np.nan;valid=np.zeros((8,9,9),bool)
        else:
            raw9=c['keypoints_xy'][lo+selected];points=raw9[:8];_,valid,L=ns['native_grid'](raw9,source['raw_hw'])
            # The centroid never changes L or native-grid validity.
            other=raw9.copy();other[8]=[1e8,-1e8];_,ov,ol=ns['native_grid'](other,source['raw_hw']);assert np.array_equal(ov,valid)and ol==L
        assert bytes_equal(q['query_xy'][section],points)and bytes_equal(q['L'][section],np.full(8,L,np.float64))
        assert np.array_equal(q['native_grid_valid'][section],valid)
        finite=np.isfinite(points).all(1)&~(points==[-1,-1]).all(1)
        cont=finite&(points[:,0]>=0)&(points[:,0]<Ww)&(points[:,1]>=0)&(points[:,1]<Hh)
        nearest=finite&(np.floor(points[:,0]+.5)>=0)&(np.floor(points[:,0]+.5)<Ww)&(np.floor(points[:,1]+.5)>=0)&(np.floor(points[:,1]+.5)<Hh)
        centre=finite&(points[:,0]>=-.5)&(points[:,0]<Ww-.5)&(points[:,1]>=-.5)&(points[:,1]<Hh-.5)
        for key,expected in(('finite_non_sentinel',finite),('continuous_valid',cont),('nearest_valid',nearest),('RGBcentre_valid',centre)):
            assert np.array_equal(q[key][section],expected),key
        assert np.array_equal(nearest,centre)and np.array_equal(q['no_detection'][section],np.repeat(missing,8))
        for k in range(8):
            i=fi*8+k;r=queries[i];assert r['frame_index']==fi and r['frame_id']==source['id']and r['source_id']==source['source_id']and r['split']==source['split']and r['native_slot']==k
            assert r['query_id']==source['id']+':native'+str(k)==q['query_id'][i]and q['frame_id'][i]==source['id']and q['split'][i]==source['split']
            assert equal(r['query_xy'],nullable(points[k]))and equal(r['L'],None if missing else float(L))
            for key,value in(('no_detection',missing),('finite_non_sentinel',finite[k]),('native_continuous_valid',cont[k]),('query_nearest_pixel_valid',nearest[k]),('query_RGB96_centre_domain_valid',centre[k]),('native_grid_any_valid',valid[k].any())):assert r[key]==bool(value),key
            assert r['association_status']=='UNASSIGNED_BY_INPUT_PRODUCER'and r['actual_original_initializer']is True and r['source_emulation']is False
        frame_checks.append(dict(frame_index=fi,id=source['id'],source_id=source['source_id'],split=source['split'],candidates=hi-lo,selected_index=selected,
            no_detection=missing,finite_slots=int(finite.sum()),continuous_valid=int(cont.sum()),finite_OOB=int((finite&~cont).sum()),all8_preserved=True))
    oldseal=read(OLD/'INTAKE_SEAL.json');assert oldseal['completed']and oldseal['PASS']
    old_proposals=next(b for b in oldseal['files']if Path(b['path']).name=='FROZEN_PROPOSALS_ALL100.jsonl');old_queries=next(b for b in oldseal['files']if Path(b['path']).name=='ORIGINAL_NATIVE_QUERY800_INPUT_ONLY.jsonl')
    check(old_proposals);check(old_queries);oldp=legacy_rows(old_proposals['path']);oldq=legacy_rows(old_queries['path']);assert len(oldp)==2 and len(oldq)==16
    legacy=[]
    for source_id,old_id in LEGACY.items():
        new=next(r for r in proposals if r['source_id']==source_id);old=next(r for r in oldp if r['frame_input_id']==old_id)
        fields=('candidates','selected_index','input_shape','canvas_shape','added_border','affine_gain','affine_offset','image','image_sha256','raw_hw','split')
        comparisons={key:equal(new[key],old[key])for key in fields};assert all(comparisons.values()),comparisons
        slots=[]
        for k in range(8):
            nr=next(r for r in queries if r['source_id']==source_id and r['native_slot']==k);orr=next(r for r in oldq if r['frame_input_id']==old_id and r['native_slot']==k)
            fields=dict(query_xy=equal(nr['query_xy'],orr['query_xy']),L=equal(nr['L'],orr['roi_side']),continuous=nr['native_continuous_valid']==orr['original_query_continuous_valid'],nearest=nr['query_nearest_pixel_valid']==orr['query_nearest_pixel_valid'],RGBcentre=nr['query_RGB96_centre_domain_valid']==orr['query_rgb96_centre_domain_valid'])
            assert all(fields.values()),(source_id,k,fields);slots.append(dict(native_slot=k,all_coordinate_L_and_boundary_fields_exact=True,fields=fields))
        legacy.append(dict(source_id=source_id,new_frame_id=new['id'],old_frame_input_id=old_id,source_split=new['split'],candidate_count=len(new['candidates']),
            all_candidate_raw9_scores_conf_boxes_structure_bitexact=True,frame_fields=comparisons,all8_queries_L_flags_exact=True,slots=slots))
    result=read(OUT/'RESULT.json');counts=dict(frames=16,slots=128,candidates=int(c['frame_offsets'][-1]),no_detection_frames=sum(r['no_detection']for r in frame_checks),finite_non_sentinel=int(q['finite_non_sentinel'].sum()),continuous_valid=int(q['continuous_valid'].sum()),nearest_valid=int(q['nearest_valid'].sum()),finite_OOB=int((q['finite_non_sentinel']&~q['continuous_valid']).sum()))
    assert counts==dict(frames=16,slots=128,candidates=120,no_detection_frames=1,finite_non_sentinel=120,continuous_valid=107,nearest_valid=107,finite_OOB=13)
    assert result['candidate_count']==counts['candidates']and result['no_detection_frames']==counts['no_detection_frames']and result['continuous_valid_slots']==counts['continuous_valid']and result['finite_OOB_slots']==counts['finite_OOB']
    receipt=dict(completed=True,PASS=True,code=bind(__file__),producer_seal=bind(SEAL),producer_result=bind(OUT/'RESULT.json'),inputs=seal['inputs'],counts=counts,
        all128_raw_NPZ_vs_JSON_ID_q_L_dtype_and_flags_exact=True,all120_candidates_raw9_boxes_conf_scores_vs_JSON_exact=True,all16_firstargmax_exact=True,
        all128_V60_native_grid_L_and_valid9x9_masks_exact=True,no_detection8_and_finiteOOB13_retained=True,centroid_exclusion_replayed_all_detected_frames=True,
        input_proposal_schema_whitelist_no_GT_fields=True,frames=frame_checks,legacy2=legacy,legacy2_all_original_prediction_query_fields_exact=True,
        legacy_original_proposals=old_proposals,legacy_original_queries=old_queries,legacy_rows_deserialized=2,legacy_query_rows_deserialized=16,
        source_RGB_newreads=0,GT_pose_K_semantic_reference_bodies_read=0,model_checkpoint_forward_fit_private_V112_PnP=0,
        limitation='This verifies input preservation and repeatability. It does not establish selected target/native role/semantic visibility correctness.')
    path=H/'REVIEW.json';path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(completed=True,PASS=True,review=bind(path),counts=counts,legacy2_exact=True)))
if __name__=='__main__':main()

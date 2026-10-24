"""SOURCE nominal-output replay with independent scalar algebra; no producer import."""
import collections
import csv
import hashlib
import io
import json
import math
import os
import stat
import traceback

D='/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_fixed_public_train_instance_part_GT_correspondence_20261025_v217'
BASE_O=D+'/independent_actual_nominal_query_review_c'
O=BASE_O+'/v2'
reads=[];max_abs={};comparisons=0
def raw(p):
    assert 'v112' not in p.casefold() and os.path.normpath(p)==p and p.endswith(('.json','.csv','.py'))
    assert '/provided_metadata/' not in p and '/original/' not in p
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for s in p.strip('/').split('/')[:-1]:
            z=os.open(s,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=z
        f=os.open(p.rsplit('/',1)[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        try:
            assert stat.S_ISREG(os.fstat(f).st_mode)
            with os.fdopen(f,'rb',closefd=False) as x:b=x.read(4000000)
        finally:os.close(f)
    finally:os.close(fd)
    reads.append({'path':p,'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)})
    return b
def data(b):
    v=raw(b['path']);assert hashlib.sha256(v).hexdigest()==b['sha256']
    if b.get('size_bytes') is not None:assert len(v)==b['size_bytes']
    return v
def doc(b):return json.loads(data(b))
def save(n,d):
    b=(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with open(O+'/'+n,'xb') as f:f.write(b)
    return {'path':O+'/'+n,'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
def near(a,b,path='root'):
    global comparisons
    comparisons+=1
    if isinstance(a,bool) or isinstance(b,bool):assert type(a) is type(b) and a==b,path;return
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        assert math.isfinite(a) and math.isfinite(b),path
        err=abs(a-b);key=path.split('/')[0];max_abs[key]=max(max_abs.get(key,0),err)
        assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-8),(path,a,b,err)
    elif isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),(path,set(a),set(b))
        for k in a:near(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):near(x,y,path+'/'+str(i))
    else:assert a==b,(path,a,b)
def mat(x):
    if len(x)==9 and not isinstance(x[0],list):return [list(map(float,x[i:i+3])) for i in (0,3,6)]
    assert len(x)==3 and all(len(r)==3 for r in x)
    return [list(map(float,r)) for r in x]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def det(a):
    return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])
def solve(a,b):
    denominator=det(a)
    if denominator==0:raise ValueError('Singular independent solve')
    out=[]
    for j in range(3):
        replaced=[list(r) for r in a]
        for i in range(3):replaced[i][j]=b[i]
        out.append(det(replaced)/denominator)
    return out
def mapping(gt,camera,fixed,faces):
    R=mat(gt['cam_R_m2c']);t=list(map(float,gt['cam_t_m2c']));K=mat(camera['cam_K'])
    cam=solve(R,[-x for x in t]);length=math.sqrt(dot(cam,cam))
    if not length:raise ValueError('Camera at model center')
    scores=[-cam[1]/length,cam[1]/length,-cam[0]/length,cam[0]/length]
    choice=max(range(4),key=lambda j:scores[j]);ordered=sorted(scores)
    projected=[]
    for i,p in enumerate(fixed):
        xyz=[dot(row,p)+u for row,u in zip(R,t)]
        xy=[dot(K[j],xyz)/xyz[2] for j in range(2)] if xyz[2]>0 and all(math.isfinite(v) for v in xyz) else None
        projected.append({'canonical_index':i,'fixed_XYZ_mm':p,'camera_Z_mm':xyz[2],'reference_xy':xy,'projection_status':'POSITIVE_Z_FINITE' if xy is not None else 'UNKNOWN_NONPOSITIVE_Z_OR_NONFINITE'})
    top=[i for i in faces[choice] if fixed[i][2]==72]
    assert all(projected[i]['reference_xy'] is not None for i in top)
    left,right=sorted(top,key=lambda i:projected[i]['reference_xy'][0])
    lookup={tuple(p):i for i,p in enumerate(fixed)}
    def bottom(i):return lookup[(fixed[i][0],fixed[i][1],-72)]
    front=[left,right,bottom(right),bottom(left)]
    axis=1 if choice in (0,1) else 0
    rear=[]
    for i in front:
        p=list(fixed[i]);p[axis]=-p[axis];rear.append(lookup[tuple(p)])
    margin=abs(projected[right]['reference_xy'][0]-projected[left]['reference_xy'][0])
    orth=max(abs(dot(R[i],R[j])-(1 if i==j else 0)) for i in range(3) for j in range(3))
    return {'status':'NOMINAL_MAPPING_AVAILABLE_WITH_UNCERTAINTY','reason':'DECLARED_NOMINAL_ONLY_TIES_AND_MARGINS_NOT_SEMANTIC_GATES','perm_native_to_fixed':front+rear,'camera_object_mm':cam,'front_face_index':choice,'front_scores':scores,'front_score_margin':ordered[-1]-ordered[-2],'front_exact_tie':scores.count(scores[choice])>1,'top_left_right_X_margin':margin,'top_left_right_exact_tie':margin==0,'fixed_projection8':projected,'R_determinant':det(R),'R_orthogonality_maxabs_error':orth}
def query(original,ref):
    out={'status':'ORIGINAL_INPUT_U' if not original['original_query_available'] else 'UNKNOWN_REFERENCE','delta_reference_minus_q_px':None,'delta_over_original_L':None,'RAW96_reference_address':None,'image_inbounds':'UNKNOWN_NO_PIXEL_DIMENSIONS','coordinate_error_gold':False}
    if not original['original_query_available'] or ref is None:return out
    q=original['query_xy'];L=original['L']
    if q is None or L is None or len(q)!=2 or not all(math.isfinite(v) for v in q) or not math.isfinite(L) or L<=0:
        out['status']='UNKNOWN_ORIGINAL_SCALAR';return out
    delta=[ref[j]-q[j] for j in (0,1)]
    out.update(status='NOMINAL_REFERENCE_DIAGNOSTIC_ONLY',delta_reference_minus_q_px=delta,delta_over_original_L=[x/L for x in delta],RAW96_reference_address=[x*96/L+47.5 for x in delta])
    return out
def strata(rows):
    groups={}
    for r in rows:
        keys=['ALL','role:'+r['role'],'native:'+str(r['native']),'role_native:'+r['role']+':'+str(r['native'])]
        for key in keys:groups.setdefault(key,[]).append(r)
    result={}
    for key,rr in groups.items():
        ds=sorted(r['nominal_displacement_px'] for r in rr if r['nominal_displacement_px'] is not None)
        result[key]={'candidate_rows':len(rr),'original_input_U_rows':sum(r['original_U'] for r in rr),'finite_available_q_reference_pairs':len(ds),'nominal_displacement_le_3px':sum(x<=3 for x in ds),'nominal_displacement_gt_3px':sum(x>3 for x in ds),'minimum_px':min(ds) if ds else None,'median_px':(ds[(len(ds)-1)//2]+ds[len(ds)//2])/2 if ds else None,'maximum_px':max(ds) if ds else None,'reference_in_saved_aligned_canvas':sum(r['reference_in_saved_aligned_canvas'] is True for r in rr),'reference_outside_saved_aligned_canvas':sum(r['reference_in_saved_aligned_canvas'] is False for r in rr)}
    return result

def main():
    ownproto={'path':BASE_O+'/PROTOCOL_BEFORE_INDEPENDENT_NOMINAL_RECOMPUTE.json','sha256':'4bbbf5f913928ad5b0e3e76ace8193488a664992ef16169d0ecccad081df5dea'}
    p=doc(ownproto);scope=doc(p['scope']);complete=doc(p['source_projection_complete']);entry=doc(complete['entry'])
    assert all(entry['root_actual_scope'][k]==p['scope'][k] for k in ['path','sha256']) and scope['approved_actual_nominal_candidate_projection'] is True
    source={k:doc(b) for k,b in p['inputs'].items()}
    contract=source['contract']['exact_coordinate_and_order_contract'];fixed=contract['fixed_canonical_XYZ_mm'];faces=contract['front_candidates_fixed_indices']
    assert contract==source['definition']['exact_coordinate_and_order_contract']
    assert faces==[[0,1,2,3],[4,5,6,7],[0,4,7,3],[1,5,6,2]]
    # Invented independent scalar checks before processing any SOURCE geometry.
    I=[[1.,0,0],[0,1.,0],[0,0,1.]];K=[[700.,0,320.],[0,800.,240.],[0,0,1.]]
    for cam,face in [([0,-3000,-4000],0),([0,3000,-4000],1),([-3000,0,-4000],2),([3000,0,-4000],3)]:
        t=[-v for v in cam];m=mapping({'cam_R_m2c':I,'cam_t_m2c':t},{'cam_K':K},fixed,faces)
        assert m['camera_object_mm']==cam and m['front_face_index']==face and sorted(m['perm_native_to_fixed'])==list(range(8))
        xy=m['fixed_projection8'][0]['reference_xy'];x,y,z=[fixed[0][j]+t[j] for j in range(3)]
        near(xy,[700*x/z+320,800*y/z+240],'invented')
    assert query({'original_query_available':True,'query_xy':[10,20],'L':192},[106,-76])['RAW96_reference_address']==[95.5,-.5]
    try:solve([[0,0,0]]*3,[1,2,3])
    except ValueError:pass
    else:raise AssertionError('Singular fixture accepted')
    rows=source['original96']['rows96'];frames=source['selected_GT']['frames12']
    assert len(rows)==96 and sum(not r['original_query_available'] for r in rows)==11
    assert collections.Counter(r['internal_source_role'] for r in rows)=={'SOURCE_FIT':64,'SOURCE_DEV':32}
    saved_rows=doc(p['artifacts']['ALL120_CANDIDATE_NATIVE_QUERY_DIAGNOSTICS.json'])['rows120']
    saved_maps=doc(p['artifacts']['ALL15_NOMINAL_CANDIDATE_MAPPINGS.json'])['candidates']
    summary=doc(p['artifacts']['SUMMARY.json']);csv_rows=list(csv.DictReader(io.StringIO(data(p['artifacts']['ALL120_NOMINAL_CANDIDATE_QUERY_DIAGNOSTICS.csv']).decode())))
    assert len(saved_rows)==len(csv_rows)==120 and len(saved_maps)==15
    pixel_complete=doc(p['pixel_scalar_complete']);assert pixel_complete['all_rows']==p['pixel_scalar_rows']
    pixel=doc(p['pixel_scalar_rows'])['candidate_rows15']
    canvases={(r['frame_position'],r['candidate_gt_list_index']):(r['height'],r['width']) for r in pixel if r['diagnostic_status']=='DIAGNOSTIC_COMPLETE_NO_LABEL'}
    expected_rows=[];expected_maps=[];readout=[]
    for f in frames:
        rr=[r for r in rows if r['frame_position']==f['frame_position']]
        assert [r['native_slot'] for r in rr]==list(range(8)) and f['selected_gt_instance_index'] is None
        for candidate in f['obj1_candidates']:
            m=mapping(candidate['GT'],f['camera'],fixed,faces)
            expected_maps.append({'frame_position':f['frame_position'],'frame_key':f['frame_key'],'gt_list_index':candidate['gt_list_index'],'mapping':m,'selected_instance':False,'training_admission':False})
            for o in rr:
                k=o['native_slot'];ci=m['perm_native_to_fixed'][k];proj=m['fixed_projection8'][ci];ref=proj['reference_xy'];qd=query(o,ref)
                expected_rows.append({'original_input':o,'diagnostic_candidate_gt_list_index':candidate['gt_list_index'],'obj_id':candidate['GT']['obj_id'],'selected_gt_instance_index':None,'native_slot':k,'canonical_index':ci,'nominal_fixed_XYZ_mm':fixed[ci],'nominal_camera_Z_mm':proj['camera_Z_mm'],'nominal_reference_xy':ref,'nominal_mapping_status':m['status'],'query_diagnostic':qd,'nominal_not_physical_native_gold':True,'new_native_visibility_label':None,'training_admission':False})
                dims=canvases.get((f['frame_position'],candidate['gt_list_index']));inside=None if ref is None or dims is None else (-.5<=ref[0]<dims[1]-.5 and -.5<=ref[1]<dims[0]-.5)
                d=qd['delta_reference_minus_q_px'];distance=math.hypot(*d) if d is not None else None
                readout.append({'row_id':o['row_id'],'frame_position':f['frame_position'],'gt_list_index':candidate['gt_list_index'],'role':o['internal_source_role'],'native':k,'original_U':not o['original_query_available'],'nominal_displacement_px':distance,'reference_in_saved_aligned_canvas':inside,'reference_inside_original_crop_canvas':None if qd['RAW96_reference_address'] is None else all(-.5<=x<95.5 for x in qd['RAW96_reference_address']),'reference_is_physical_gold':False,'training_admission':False})
    near(expected_maps,saved_maps,'mapping');near(expected_rows,saved_rows,'row')
    assert len({(r['original_input']['row_id'],r['diagnostic_candidate_gt_list_index']) for r in saved_rows})==120
    assert {r['original_input']['row_id'] for r in saved_rows}=={r['row_id'] for r in rows}
    assert sum(not r['original_input']['original_query_available'] for r in saved_rows)==17
    expected_summary={'completed':True,'PASS':True,'source_frames':12,'original_unique_rows':96,'original_inputU':11,'original_SOURCE_FIT_rows':64,'original_SOURCE_DEV_rows':32,'candidate_instances':15,'candidate_native_rows':120,'expanded_inputU_rows':17,'mapping_status_counts':dict(collections.Counter(m['mapping']['status'] for m in expected_maps)),'face_exact_tie_candidates':sum(m['mapping']['front_exact_tie'] for m in expected_maps),'top_LR_exact_tie_candidates':sum(m['mapping']['top_left_right_exact_tie'] for m in expected_maps),'image_inbounds':'UNKNOWN_NO_DIMENSIONS_OPENED','candidate_selection':None,'native_labels_created':0,'training_admission':False,'nominal_point_not_physical_surface_or_native_gold':True,'source_input_qL_native_order_or_roles_changed':0,'original_payload_or_model_score_reads':0}
    near(expected_summary,summary,'summary');near(expected_summary,complete['summary'],'complete_summary')
    for expected,record in zip(expected_rows,csv_rows):
        o=expected['original_input'];q=expected['query_diagnostic']
        scalar={'row_id':o['row_id'],'frame_position':str(o['frame_position']),'frame_key':o['frame_key'],'component_id':o['component_id'],'original_role':o['original_role'],'internal_source_role':o['internal_source_role'],'native_slot':str(o['native_slot']),'original_query_available':str(o['original_query_available']),'diagnostic_gt_list_index':str(expected['diagnostic_candidate_gt_list_index']),'selected_gt_instance_index':'','canonical_index':str(expected['canonical_index']),'mapping_status':expected['nominal_mapping_status'],'query_diagnostic_status':q['status'],'image_inbounds':q['image_inbounds'],'native_label':'','training_admission':'False'}
        for k,v in scalar.items():assert record[k]==v,(k,record[k],v)
        for k,v in [('original_query_xy',o['query_xy']),('fixed_XYZ_mm',expected['nominal_fixed_XYZ_mm']),('reference_xy',expected['nominal_reference_xy']),('delta_ref_minus_q_px',q['delta_reference_minus_q_px']),('delta_over_original_L',q['delta_over_original_L']),('RAW96_reference_address',q['RAW96_reference_address'])]:near(v,json.loads(record[k]),'CSV')
        if o['L'] is None:assert record['original_L']==''
        else:near(o['L'],float(record['original_L']),'CSV')
        near(expected['nominal_camera_Z_mm'],float(record['camera_Z_mm']),'CSV')
    stats=strata(readout)
    result=save('INDEPENDENT_RECOMPUTATION_AND_STRATA.json',{'completed':True,'PASS':True,'scalar_comparisons':comparisons,'max_abs_arithmetic_difference':max_abs,'strata':stats,'candidate_native_readout120':readout,'saved_aligned_canvas_dimension_source':p['pixel_scalar_rows'],'nominal_reference_is_not_physical_GT_coordinate_error':True,'no_candidate_selection':True,'native_labels_created':0,'training_admission':False})
    out=save('REVIEW.json',{'completed':True,'PASS':True,'source_complete':p['source_projection_complete'],'scope':p['scope'],'entry':complete['entry'],'protocol':ownproto,'code':{'path':__file__,'sha256':hashlib.sha256(raw(__file__)).hexdigest()},'independent_recomputation':result,'counts':{'source_frames':12,'original_unique_rows':96,'original_U':11,'SOURCE_FIT':64,'SOURCE_DEV':32,'candidate_instances':15,'candidate_rows':120,'candidate_U':17,'all_mappings_recomputed':15,'all_JSON_CSV_rows_compared':120},'max_abs_arithmetic_difference':max_abs,'all_candidate_nominal_displacement_readout':stats['ALL'],'findings_KO':['생산자를 import하지 않고 scalar determinant/Cramer camera inverse·명시적4법선·pinhole·frontLR/rear partner를 독립 계산해 전체15 mapping/120 JSON+CSV/summary와 비교했다.','원96행의 전체 original_input·q/L/native/U/role가 유지됐다. 원U11은 후보중복U17로 보존하며 q relation을 UNKNOWN으로 둔다. instance는 모든 후보 미선택이고 새 D/E가 없다.','3px는 요청된 명목 변위 기술 구간일 뿐 physical-GT 오차/모델 calibration/채택 기준이 아니다. 모든 후보를 반복 보존하며 가장 가까운 instance나 permutation을 고르지 않는다.','producer의 image_inbounds UNKNOWN 원문은 유지했다. 별도 in-bounds 표만 이미 저장된 정합 mask/depth/context scalar 차원을 사용했으며 실제 PNG를 열지 않았다.'],'limits_KO':['기하식 재현은 nominal convention의 정합 검산이다. 실제 physical part/instance·가시성·카메라 보정 정확도 또는 모델 성능을 인증하지 않는다.','SOURCE mask/depth 진단은 다른 observer/producer의 저장 scalar 권한을 인용했다. 본인은 원GT·RGB·mask/depth/CAD·모델 payload를 재열람하지 않았다.'],'prior_U14_assumption_failure_and_message_correction_preserved':{'path':D+'/independent_nominal_projection_code_preflight_c/PRIOR_PIXEL_MESSAGE_U_COUNT_CLARIFICATION.json','sha256':'af47a08a8a6012fb1d784d47126b37d2ae5e8ce245592769bd6e7cb3b93a715f'},'original_GT_camera_PNG_CAD_NPZ_model_private_payload_reads':0,'producer_rerun_or_import':0,'new_labels_admission_q_changes_PnP_fit_selection_Git_network':0,'metadata_read_bindings':reads,'own_folder_frozen_no_further_edits':True})
    with open(O+'/KO.md','x') as f:
        a=stats['ALL'];f.write('# V217 nominal 후보 projection 독립 QA\n\nPASS. scalar Cramer inverse·4법선 cosine·pinhole·front/rear 순서를 독립 구현해 15 mapping과 120 JSON/CSV 전량을 대조했다. 원96행/U11/FIT64·DEV32와 후보 중복U17을 보존했다. 원 instance는 모두 미선택이며 새 라벨·학습 채택은 없다.\n\n명목 reference와 available q의 변위는 '+str(a['finite_available_q_reference_pairs'])+'개 후보행 중 3px 이하 '+str(a['nominal_displacement_le_3px'])+'개다. 이는 후보가 반복된 기술 통계이며 물리 좌표 정답·최적 instance 선택·모델 오류율이 아니다. 원 q를 이동하거나 입력을 recrop하지 않았다.\n\n저장 정합 canvas scalar로 별도 in-bounds 기술만 추가했고 source PNG/원 GT/배열/모델은 열지 않았다. 실제 instance·part·visibility 및 nominal reference 불확실성은 별도 근거가 필요하다.\n')
    print(json.dumps({'completed':True,'PASS':True,'REVIEW':out,'ALL':stats['ALL'],'max_abs_difference':max_abs},ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save('FAILURE.json',{'completed':False,'PASS':False,'error_type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'metadata_read_bindings':reads})
        raise

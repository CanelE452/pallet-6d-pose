"""Independent nominal projection preflight: code, selected metadata, invented poses only."""
import ast
import copy
import hashlib
import json
import math
import os
import stat
from unittest.mock import patch
import numpy as np

D='/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_fixed_public_train_instance_part_GT_correspondence_20261025_v217'
H=D+'/nominal_candidate_query_projection_preparation_b'
O=D+'/independent_nominal_projection_code_preflight_c'
reads=[]
def raw(p):
    assert 'v112' not in p.casefold() and os.path.normpath(p)==p and p.endswith(('.json','.py'))
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
def checked(b):
    v=raw(b['path']);assert hashlib.sha256(v).hexdigest()==b['sha256']
    if b.get('size_bytes') is not None:assert len(v)==b['size_bytes']
    return v
def doc(b):return json.loads(checked(b))
def save(n,d):
    b=(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with open(O+'/'+n,'xb') as f:f.write(b)
    return {'path':O+'/'+n,'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}

ready_b={'path':H+'/FINAL_READY.json','sha256':'bf974d72b01a348ab5d08e73e6735fae235968b761cdf28cfe436cbc3a3a803b'}
r=doc(ready_b);code=checked(r['code']);proto=doc(r['protocol']);checks=doc(r['checks']);false=doc(r['false_entry'])
inputs={k:doc(v) for k,v in r['inputs'].items()}
assert false['approved_actual_nominal_candidate_projection'] is False and false['training_admission'] is False
assert false['inputs']==r['inputs'] and false['original_payloads_or_models_or_scores_allowed'] is False
done=inputs['selected_complete'];frames=inputs['selected_GT']['frames12'];rows=inputs['original96']['rows96']
assert done['completed'] is done['PASS'] is True and done['selected_GT_metadata']['sha256']==r['inputs']['selected_GT']['sha256']
c=inputs['contract']['exact_coordinate_and_order_contract']
assert c==inputs['definition']['exact_coordinate_and_order_contract']
assert c['permutation_direction']=='dynamic[k] = fixed[perm[k]]; never reapply to already dynamic stored arrays.'
assert len(frames)==12 and len(rows)==96 and len({x['row_id'] for x in rows})==96
assert sum(not x['original_query_available'] for x in rows)==11
assert sum(x['internal_source_role']=='SOURCE_FIT' for x in rows)==64 and sum(x['internal_source_role']=='SOURCE_DEV' for x in rows)==32
assert sum(len(f['obj1_candidates']) for f in frames)==15
expanded_U=0
for pos,f in enumerate(frames):
    rr=[x for x in rows if x['frame_position']==pos]
    assert f['frame_position']==pos and [x['native_slot'] for x in rr]==list(range(8))
    assert f['selected_gt_instance_index'] is None and f['native_target_admission'] is False
    assert [x['gt_list_index'] for x in f['obj1_candidates']]==rr[0]['obj1_candidate_gt_list_indices']
    for x in rr:
        assert x['original_role']==f['original_role']=='TRAIN' and x['internal_source_role']==f['internal_source_role']
        assert x['frame_key']==f['frame_key'] and x['training_admission'] is False
    expanded_U+=sum(not x['original_query_available'] for x in rr)*len(f['obj1_candidates'])
assert expanded_U==14
ns={'__name__':'invented_nominal_module','__file__':r['code']['path']}
exec(compile(code,r['code']['path'],'exec'),ns)
assert np.array_equal(ns['FIXED'],np.array(c['fixed_canonical_XYZ_mm'])) and ns['FACES']==c['front_candidates_fixed_indices']
own=[]
with patch.object(os,'open',side_effect=AssertionError('No filesystem calls in pure fixtures')):
    reused=ns['invented_tests']();assert reused['completed'] is reused['PASS'] is True and reused['count']==checks['count']==10
    I=np.eye(3);K=np.array([[700.,0,320.],[0,800.,240.],[0,0,1.]])
    normals=[(0,-1,0),(0,1,0),(-1,0,0),(1,0,0)]
    cases=[((0,-3000,-4000),[0,1,2,3,4,5,6,7]),((0,3000,-4000),[4,5,6,7,0,1,2,3]),((-3000,0,-4000),[0,4,7,3,1,5,6,2]),((3000,0,-4000),[1,5,6,2,0,4,7,3])]
    for cam,perm in cases:
        t=-np.array(cam,dtype=np.float64);m=ns['nominal_mapping'](I,t,K)
        scores=[sum(a*b for a,b in zip(n,cam))/math.sqrt(sum(v*v for v in cam)) for n in normals]
        assert m['front_face_index']==scores.index(max(scores)) and m['perm_native_to_fixed']==perm
        assert np.allclose(m['front_scores'],scores,rtol=0,atol=1e-15)
        for item,point in zip(m['fixed_projection8'],c['fixed_canonical_XYZ_mm']):
            x,y,z=[point[k]+t[k] for k in range(3)]
            expected=[700*x/z+320,800*y/z+240]
            assert np.allclose(item['reference_xy'],expected,rtol=0,atol=1e-12)
    own.append('four_face_independent_scalar_cosines_pinhole_and_opposite_depth_permutations')
    q={'original_query_available':True,'query_xy':[10.,30.],'L':192.}
    assert ns['query_diagnostic'](q,[10.,30.])['RAW96_reference_address']==[47.5,47.5]
    assert ns['query_diagnostic'](q,[106.,-66.])['RAW96_reference_address']==[95.5,-.5]
    assert ns['query_diagnostic'](q,[106.,-66.])['image_inbounds']=='UNKNOWN_NO_PIXEL_DIMENSIONS'
    own.append('RAW96_inverse_sample_center_formula_not_recenter_or_image_bounds')
    original=[dict(row_id='toy'+str(k),native_slot=k,query_xy=[k+1.,5.],L=192.,original_query_available=k!=7,original_role='TRAIN',internal_source_role='SOURCE_DEV') for k in range(8)]
    before=copy.deepcopy(original)
    cand={'gt_list_index':3,'GT':{'obj_id':1,'cam_R_m2c':I.tolist(),'cam_t_m2c':[0.,3000.,4000.]}}
    out,ma=ns['candidate_rows'](original,cand,{'cam_K':K.tolist()})
    shifted=copy.deepcopy(original)
    for x in shifted:x['query_xy']=[99999.,-88888.]
    _,mb=ns['candidate_rows'](shifted,cand,{'cam_K':K.tolist()})
    assert ma==mb and original==before and all(x['original_input']==y for x,y in zip(out,original))
    assert out[7]['query_diagnostic']['status']=='ORIGINAL_INPUT_U' and out[7]['query_diagnostic']['delta_reference_minus_q_px'] is None
    assert all(x['selected_gt_instance_index'] is None and x['new_native_visibility_label'] is None and x['training_admission'] is False for x in out)
    own.append('q_independent_geometry_original_full_metadata_U_DEV_and_no_label_mutation')
    for R,t,reason in [(I,[0,0,0],'CAMERA_AT_MODEL_CENTER_OR_NONFINITE'),(np.zeros((3,3)),[0,0,3000],'SINGULAR_R'),(I,[0,0,-3000],'FRONT_TOP_PROJECTION_UNKNOWN')]:
        m=ns['nominal_mapping'](R,t,K);assert m['perm_native_to_fixed'] is None and m['reason']==reason
    own.append('degenerate_zero_center_singular_and_nonpositive_Z_unknown_no_fallback')
    tied=ns['nominal_mapping'](I,[0.,0.,3000.],K)
    assert tied['front_exact_tie'] is True and tied['front_face_index']==0
    assert 'WITH_UNCERTAINTY' in tied['status']
    own.append('exact_tie_declared_first_nominal_only_not_semantic_identity')
    try:ns['safe']('/never/open/v112/input.json')
    except AssertionError:pass
    else:raise AssertionError('Lexical forbidden guard failed')
    with patch.dict(ns,rb=lambda _:json.dumps(false).encode()):
        try:ns['run'](str(ns['V']/'ROOT_INVENTED_FALSE.json'))
        except AssertionError:pass
        else:raise AssertionError('False root gate failed')
    own.append('lexical_V112_and_false_actual_gate_before_FS')
fx=save('INDEPENDENT_INVENTED_AND_METADATA_CHECKS.json',{'completed':True,'PASS':True,'reused_B_pure_checks':10,'own_checks':own,'own_count':len(own),'source_projection_calls':0,'source_payload_FS_calls':0,'original96_and_expanded120_metadata_verified':True})
out=save('REVIEW.json',{'completed':True,'PASS':True,'prepared_only':True,'FINAL_READY':ready_b,'code':r['code'],'protocol':r['protocol'],'inputs':r['inputs'],'own_checker':{'path':__file__,'sha256':hashlib.sha256(raw(__file__)).hexdigest()},'checks':fx,'counts':{'original_frames':12,'original_rows':96,'original_U':11,'SOURCE_FIT':64,'SOURCE_DEV':32,'candidate_instances':15,'future_candidate_native_rows':120,'candidate_U_rows':14},'findings_KO':['기존 V159 고정 canonical/네 vertical face/방향 규약과 계약 값이 같다. solve(R,-t)·outward-normal cosine·front projected-X·vertical bottom/opposite-depth rear 순서이며 q를 이용한 선택·최적화가 없다.','dynamic[k]=fixed[perm[k]]는 nominal reference만 생성한다. 원 native/q/L/U/role·저장 입력을 permute하지 않는다.','각 obj1 후보를 모두 보존한다. source96/15×8=120, 원U11/후보중복U14 및 FIT64/DEV32를 metadata로 검산했다. 실제 SOURCE pose에 projection 함수는 호출하지 않았다.','음수/0 Z·singular pose·zero camera center는 UNKNOWN/reference null을 보존한다. exact face/LR tie는 기존 declared-first deterministic convention을 사용하면서 tie/margin과 nominal uncertainty를 기록한다. 이 tie 규칙은 historical writer exact 복구나 part identity 인증이 아니다.','RAW96 좌표는 원 q/L에 대한 inverse sampler 대수 주소만 기술한다. 이미지 bounds는 실제 차원을 읽기 전 UNKNOWN이며 recrop·입력수정·물리 coordinate-error gold가 아니다.'],'limits_KO':['selected source GT metadata를 확인했으나 original GT/camera/mask/depth/CAD/RGB/NPZ payload는 열지 않았다. 실제 SOURCE projection은 별도 root actual entry 뒤에만 허용된다.','nominal AABB 역할은 공식 MR native8 gold나 실제 표면/가시성을 뜻하지 않는다. instance/native label 선택·학습 채택은 없다.'],'actual_SOURCE_projection_executed':False,'original_GT_camera_mask_depth_CAD_RGB_array_weight_protected_payload_FS_calls':0,'new_labels_admission_model_forward_fit_Git_network':0,'metadata_read_bindings':reads,'own_folder_frozen_no_further_edits':True})
print(json.dumps({'completed':True,'PASS':True,'REVIEW':out,'own_checks':len(own),'reused_checks':10,'actual_source_projection_calls':0},ensure_ascii=False))

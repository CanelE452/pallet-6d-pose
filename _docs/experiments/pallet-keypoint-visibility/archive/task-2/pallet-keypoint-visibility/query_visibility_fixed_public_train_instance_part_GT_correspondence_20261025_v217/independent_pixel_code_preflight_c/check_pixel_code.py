"""Pixel V2 CODE/metadata preflight; source PNG and original GT access forbidden."""
import ast
import collections
import hashlib
import json
import os
import stat
from unittest.mock import patch
import numpy as np

D='/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_fixed_public_train_instance_part_GT_correspondence_20261025_v217'
H=D+'/fixed12_mask_depth_query_diagnostic_preparation_a_v2'
O=D+'/independent_pixel_code_preflight_c'
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

ready_b={'path':H+'/FINAL_READY.json','sha256':'c97b98aec45c51da6cc675aa1aed69c2aa7a94f20e2808bed6e84a6fa7ea3b60'}
r=doc(ready_b);code=checked(r['code']);proto=doc(r['protocol']);prior=doc(r['V1_provenance_and_AST'])
v1=checked(prior['V1_code']);checked(prior['V1_READY']);checked(prior['V1_protocol'])
checks=doc(r['checks']);assert checks['invented_functional_checks']==20 and checks['PASS'] is True
checked(r['test_code']);false=doc(r['false_entry'])
assert false['approved_actual_fixed12_mask_depth_diagnostic'] is false['approved_exact12_derived_context_PNG'] is False
assert false['training_admission'] is False
assert false['selected_GT_complete']['sha256']=='ef3b92c0563955504e8a6d8a5caad594f05b8b9f5c7a7e78d18334cc7324e10e'
review_GT=doc({'path':D+'/independent_actual_selected_GT_review_c/REVIEW.json','sha256':'8c62b84665bd8a9a817de3b0fbf78faa0642712d594781816c7a08b375542a4c'})
assert review_GT['completed'] is review_GT['PASS'] is True
frames=doc(r['inputs']['fixed12_frames'])['frames12'];rows=doc(r['inputs']['original96_inputs'])['rows96']
joined=doc(r['inputs']['V216_joined96'])['rows96'];contexts=doc(r['context12_selection'])['frames12']
wl=doc(r['inputs']['payload_whitelist'])['payloads'];pixels=doc(r['exact42_pixel_whitelist'])['payloads']
expected=[p for p in wl if p['kind'] in {'FULL_INSTANCE_MASK_PNG','VISIBLE_INSTANCE_MASK_PNG','DEPTH_PNG'}]
assert {p['path']:p for p in pixels}=={p['path']:p for p in expected}
assert len(pixels)==42 and len({p['path'] for p in pixels})==42
assert collections.Counter(p['kind'] for p in pixels)=={'FULL_INSTANCE_MASK_PNG':15,'VISIBLE_INSTANCE_MASK_PNG':15,'DEPTH_PNG':12}
assert [c['frame_position'] for c in contexts]==list(range(12)) and len({c['binding']['path'] for c in contexts})==12
assert len(rows)==len(joined)==96 and sum(not x['original_query_available'] for x in rows)==11
assert collections.Counter(x['internal_source_role'] for x in rows)=={'SOURCE_FIT':64,'SOURCE_DEV':32}
for x,j in zip(rows,joined):
    i=j['input'];c=contexts[x['frame_position']]
    assert i['position']==x['frame_position']==c['frame_position'] and i['role']==x['original_role']==c['outer_role']=='TRAIN'
    for k in ['row_id','native_slot','query_xy','L','original_query_available','original_flags','component_id','internal_source_role']:
        assert x[k]==i[k]
    a=[b for b in j['display_bindings']['full_context'] if b['path'].endswith('/ORIGINAL_CANONICAL_RGB.png')]
    assert len(a)==1 and a[0]['path']==c['binding']['path'] and a[0]['sha256']==c['binding']['sha256']
    assert c['frame_key']==x['frame_key'] and c['internal_source_role']==x['internal_source_role']
known={p['path'] for p in pixels}
for f in frames:
    assert f['depth']['path'] in known
    for p in f['candidate_masks'].values():assert all(v['path'] in known for v in p.values())
a1={x.name:x for x in ast.parse(v1).body if isinstance(x,ast.FunctionDef)}
a2={x.name:x for x in ast.parse(code).body if isinstance(x,ast.FunctionDef)}
for n in prior['existing_functions_AST_exact']:
    assert ast.dump(a1[n],include_attributes=False)==ast.dump(a2[n],include_attributes=False),n
ns={'__name__':'independent_invented_pixel_module','__file__':r['code']['path']}
exec(compile(code,r['code']['path'],'exec'),ns)
own=[]
def bad(n,fn):
    try:fn()
    except ValueError:own.append(n);return
    raise AssertionError('Expected rejection '+n)
f=np.array([[0,1,1],[0,1,0],[0,0,0]],np.uint8)
v=np.array([[1,0,1],[0,1,0],[0,0,0]],np.uint8)
z=np.array([[0,10,20],[30,40,50],[60,0,80]],np.uint16)
with patch.object(os,'open',side_effect=AssertionError('No source filesystem from invented test')):
    a=ns['candidate_summary'](f,v,z,None)
    assert a['full_area']==a['visible_area']==3 and a['visible_outside_full_area']==1 and a['depth_positive_finite_pixels']==7
    own.append('mask_subset_depth_counts_are_descriptive')
    bad('mask_dimension_mismatch_no_numeric_resize',lambda:ns['candidate_summary'](f,v[:2],z,None))
    q=ns['query_diagnostic']([.5,1],False,f,v,z,None)
    assert q['nearest_x']==1 and q['nearest_y']==1 and q['original_inputU'] and q['new_native_visibility_label']=='UNKNOWN' and q['training_admission'] is False
    own.append('nearest_half_up_preserves_original_U_and_unknown_label')
    q=ns['query_diagnostic']([-.5,-.5],True,f,v,z,2)
    assert q['nearest_x']==q['nearest_y']==0 and q['window3x3_in_frame_samples']==4 and q['scaled_depth_mm_at_nearest'] is None
    own.append('clipped_3x3_and_zero_depth_no_metric')
    q=ns['query_diagnostic']([2.5,1],True,f,v,z,2)
    assert not q['nearest_pixel_in_frame'] and q['window3x3_in_frame_samples']==0
    own.append('right_half_open_boundary_no_clamp')
    assert ns['GT_info_context']({'bbox_obj':[-1,-1,-1,-1],'px_count_valid':0,'px_count_visib':0},3,3)['GT_bbox_obj_oob_status']=='NEGATIVE_BBOX_SENTINEL_NO_GEOMETRIC_SUPPORT'
    assert ns['GT_info_context']({'bbox_obj':[-2,0,4,3],'px_count_valid':2,'px_count_visib':10},3,3)['GT_bbox_obj_oob_status']=='BBOX_PARTLY_OUTSIDE'
    own.append('sentinel_and_OOB_not_native_E_or_pxcount_failure')
    ns['gate_pixels'](pixels)
    bad('unknown_SHA_before_FS',lambda:ns['gate_pixels']([{**pixels[0],'sha256':None}]+pixels[1:]))
    bad('V112_before_FS',lambda:ns['opened']('/forbidden/v112/data.png'))
    bad('context_mismatch_overlay_no_resize',lambda:ns['panel'](f,v,z,[],'INVENTED',np.zeros((2,3,3),np.uint8)))
    with patch.dict(ns,raw=lambda _:json.dumps({'approved_actual_fixed12_mask_depth_diagnostic':False}).encode()):
        bad('first_actual_flag_before_payload',lambda:ns['run'](D+'/INVENTED_FALSE.json'))
    with patch.dict(ns,raw=lambda _:json.dumps({'approved_actual_fixed12_mask_depth_diagnostic':True,'approved_exact12_derived_context_PNG':False}).encode()):
        bad('second_context_actual_flag_before_payload',lambda:ns['run'](D+'/INVENTED_FALSE.json'))
fx=save('INDEPENDENT_INVENTED_AND_METADATA_CHECKS.json',{'completed':True,'PASS':True,'own_checks':own,'checks':len(own),'V1_AST_exact_functions':prior['existing_functions_AST_exact'],'context_original96_membership_exact':True,'42_original_whitelist_exact':True,'actual_source_PNG_GT_binary_FS_calls':0})
out=save('REVIEW.json',{'completed':True,'PASS':True,'prepared_only':True,'FINAL_READY':ready_b,'code':r['code'],'protocol':r['protocol'],'selected_GT_complete':r['inputs']['selected_GT_complete'],'independent_selected_GT_QA':{'path':D+'/independent_actual_selected_GT_review_c/REVIEW.json','sha256':'8c62b84665bd8a9a817de3b0fbf78faa0642712d594781816c7a08b375542a4c'},'V1_provenance_and_AST':r['V1_provenance_and_AST'],'own_checker':{'path':__file__,'sha256':hashlib.sha256(raw(__file__)).hexdigest()},'checks':fx,'metadata_counts':{'original_rows':96,'original_U':11,'SOURCE_FIT':64,'SOURCE_DEV':32,'candidates':15,'candidate_native_rows':120,'candidate_U_rows':sum(sum(not r['original_query_available'] for r in rows if r['frame_position']==f['frame_position'])*len(f['obj1_candidate_gt_list_indices']) for f in frames),'mask_depth_PNG':42,'derived_context_PNG':12},'findings_KO':['42개 mask/depth와 12개 context를 정확한 저장 경로·SHA로 결합하고 전부 opaque 검증한 뒤 PNG decode한다. 두 root actual flag와 fresh output이 필요하다.','V1의 기존 핵심 수치/경로 함수15개는 AST 동일하다. 추가 context는 원96행의 같은 full-context binding과 모두 일치하며 입력 대체나 recrop에 쓰지 않는다.','full/visible/depth 차원 불일치는 후보+8행 UNKNOWN 보존이다. context만 불일치하면 overlay만 보류하고 정합한 mask/depth 수치는 유지한다. 실제 source alignment를 resize하지 않는다.','floor(q+.5) 최근접 및 잘린3x3 진단이며 원 RAW96 sampler·ROI support 검사가 아니다. 원U/qL/native/weakstate/ROI는 유지하며 mask 긍정도 U를 고치지 않는다.','bbox sentinel/OOB/global visible fraction 및 visible-count>valid-depth-count는 기술 축이다. instance 선택·native E·새 학습 채택으로 자동 변환하지 않는다.'],'limits_KO':['코드·저장 메타데이터와 가상 배열만 검사했다. 실제42 mask/depth/12 context 파일 존재·bytes·pixels·정합 및 calibration 정확도는 future actual 결과로 확인해야 한다.','root actual entry는 이 exact frozen code/protocol/selection/GT-complete 및 independent QA를 명시적으로 바인딩해야 한다.'],'actual_source_PNG_or_original_GT_payload_stat_hash_header_decode':0,'model_projection_training_private_Git_network':0,'training_admission':False,'metadata_read_bindings':reads,'own_folder_frozen_no_further_edits':True})
print(json.dumps({'completed':True,'PASS':True,'REVIEW':out,'own_checks':len(own),'source_payload_FS_calls':0},ensure_ascii=False))

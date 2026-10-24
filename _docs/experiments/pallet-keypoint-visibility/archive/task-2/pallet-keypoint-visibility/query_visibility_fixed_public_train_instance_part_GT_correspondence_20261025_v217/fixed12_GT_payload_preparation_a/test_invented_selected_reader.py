#!/usr/bin/env python3
"""Pure invented JSON/group fixtures; only own temporary files for path guards."""
import copy
import hashlib
import importlib.util
import json
import os
import tempfile
from unittest.mock import patch

H=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location('selected_reader',H+'/read_fixed12_selected_GT_metadata.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
def expect_bad(name,fn):
    try:fn()
    except (ValueError,OSError):checks.append(name);return
    raise AssertionError(name)

text=' {"skip":[{"x":"comma, } \\\" \\u1234", "n":1e9999}], "51":[{"obj_id":1,"cam_R_m2c":[1,0]}], "64":{"depth_scale":1}, "other":{"hidden":[1,2]}} '
calls=[]
def decoder(s):calls.append(s);return json.loads(s)
v,stats=m.select_top_level_values(text,{'51','64'},decoder)
assert list(v)==['51','64'] and len(calls)==2 and not any('hidden' in x or '1e9999' in x for x in calls)
assert stats['skipped_value_count']==2 and stats['skipped_values_JSON_decoded']==0
checks.append('only_selected_value_slices_reach_decoder_nested_escaped_skipped_values_opaque')
v,_=m.select_top_level_values('{"x":{"this_is_opaque":UNINTERPRETED_TOKEN},"51":[1]}',{'51'})
assert v=={'51':[1]};checks.append('skipped_values_not_semantically_decoded_or_validated_explicitly')
expect_bad('missing_selected_key_rejected',lambda:m.select_top_level_values('{"1":[]}',{'51'}))
expect_bad('duplicate_selected_key_rejected',lambda:m.select_top_level_values('{"51":[],"51":[]}',{'51'}))
expect_bad('duplicate_skipped_key_rejected',lambda:m.select_top_level_values('{"x":0,"x":1,"51":[]}',{'51'}))
for name,text2 in [('top_list','[]'),('unbalanced','{"x":[1},"51":[]}'),('trailing_comma','{"51":[],}'),('trailing_text','{"51":[]} 1')]:
    expect_bad(name,lambda t=text2:m.select_top_level_values(t,{'51'}))
rows=[]
for g in range(12):
 for n in range(8):
  u=(g*8+n)<11
  rows.append({'row_id':str(g)+':native'+str(n),'frame_position':g,'frame_key':'scene:'+str(g),'component_id':'c'+str(g),'original_role':'TRAIN','internal_source_role':'SOURCE_FIT' if g<8 else 'SOURCE_DEV','native_slot':n,'query_xy':None if u else [12.0,21.0],'L':None if u else 32.0,'original_query_available':not u})
before=copy.deepcopy(rows);assert len(m.validate_all8_inputs(rows))==12 and rows==before
checks.append('all96_all8_original_null_U11_FIT64_DEV32_unchanged')
bad=copy.deepcopy(rows);bad[1]['native_slot']=0
expect_bad('duplicate_group_native_rejected',lambda:m.validate_all8_inputs(bad))
bad=copy.deepcopy(rows);bad[1]['original_role']='HOLDOUT'
expect_bad('non_TRAIN_role_rejected_before_GTparse',lambda:m.validate_all8_inputs(bad))
bad=copy.deepcopy(rows);bad[1]['internal_source_role']='SOURCE_DEV'
expect_bad('mixed_internal_role_group_rejected',lambda:m.validate_all8_inputs(bad))
record={'path':'/declared/scene_camera.json','sha256':None,'kind':'SCENE_CAMERA_JSON'}
expect_bad('missing_prior_SHA_needs_explicit_first_intake',lambda:m.gate_payload(record,{record['path']:record},set()))
m.gate_payload(record,{record['path']:record},{record['path']});checks.append('exact_missing_SHA_first_opaque_permission_allowed_without_FS')
wrong={**record,'sha256':'a'*64}
with patch.object(m.os,'open',side_effect=AssertionError('FS must not happen')):
 expect_bad('changed_SHA_identity_rejected_before_FS',lambda:m.gate_payload(wrong,{record['path']:record},set()))
 expect_bad('undeclared_path_rejected_before_FS',lambda:m.gate_payload({**record,'path':'/other/file'}, {record['path']:record},set()))
 expect_bad('V112_lexical_rejected_before_FS',lambda:m.opened('/no/v112/file.json'))
record2={**record,'kind':'DEPTH_PNG','sha256':'b'*64}
expect_bad('binary_payload_not_opened_by_JSONreader',lambda:m.gate_payload(record2,{record2['path']:record2},set()))
declared=[{'kind':kind,'path':'/'+str(scene)+'/'+kind} for scene in range(4) for kind in sorted(m.JSON_KINDS)]
assert len(m.requested_records(declared,['SCENE_GT_JSON']))==4
assert len(m.requested_records(declared,sorted(m.JSON_KINDS)))==12
checks.append('known_scene_GT_only_branch_does_not_require_info_camera_SHA_or_values')
expect_bad('unrequested_binary_kind_cannot_enter_branch',lambda:m.requested_records(declared,['SCENE_GT_JSON','DEPTH_PNG']))
with tempfile.TemporaryDirectory(prefix='invented_path_',dir=H) as tmp:
 with open(tmp+'/regular','w') as f:f.write('{}')
 os.symlink(tmp+'/regular',tmp+'/link')
 expect_bad('symlink_file_rejected',lambda:m.opened(tmp+'/link'))
 os.symlink(tmp,tmp+'/dirlink')
 expect_bad('symlink_component_rejected',lambda:m.opened(tmp+'/dirlink/regular'))
result={'completed':True,'PASS':True,'invented_functional_checks':len(checks),'checks':checks,'actual_GT_source_payload_reads':0,'model_or_projection_execution':0}
b=(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode()
with os.fdopen(m.opened(H+'/INVENTED_FUNCTIONAL_CHECKS_V2.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL),'wb') as f:f.write(b)
print(json.dumps({'completed':True,'PASS':True,'checks':len(checks),'sha256':hashlib.sha256(b).hexdigest()}))

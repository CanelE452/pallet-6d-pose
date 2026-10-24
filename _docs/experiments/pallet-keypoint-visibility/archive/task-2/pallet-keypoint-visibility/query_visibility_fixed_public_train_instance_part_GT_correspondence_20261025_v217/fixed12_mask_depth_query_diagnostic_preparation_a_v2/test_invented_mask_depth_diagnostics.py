#!/usr/bin/env python3
import importlib.util
import io
import json
import os
import tempfile
from unittest.mock import patch
import numpy as np
from PIL import Image

H=os.path.dirname(os.path.abspath(__file__))
s=importlib.util.spec_from_file_location('diag',H+'/diagnose_fixed12_masks_depth_queries.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
checks=[]
def bad(name,fn):
    try:fn()
    except (ValueError,OSError):checks.append(name);return
    raise AssertionError(name)
f=np.array([[0,255,0],[255,255,0],[0,0,0]],np.uint8)
v=np.array([[0,255,255],[0,255,0],[0,0,0]],np.uint8)
d=np.array([[0,10,20],[30,40,50],[60,0,80]],np.uint16)
a=m.candidate_summary(f,v,d,None)
assert a['full_area']==a['visible_area']==3 and a['visible_outside_full_area']==1 and not a['visible_subset_full']
assert a['depth_positive_finite_pixels']==7 and a['full_mask_positive_depth_pixels']==3 and a['visible_full_area_ratio']==1
assert a['metric_depth_status'].startswith('UNKNOWN')
checks.append('full_visible_subset_failure_depth_support_is_numeric_not_label')
a0=m.candidate_summary(f*0,v*0,d,2.)
assert a0['visible_full_area_ratio'] is None and a0['full_mask_positive_depth_fraction'] is None
checks.append('zero_mask_missing_denominator_no_E_or_fabricated_ratio')
bad('dimension_mismatch_no_resize',lambda:m.candidate_summary(f,v[:2],d,None))
q=m.query_diagnostic([.5,1.0],False,f,v,d,None)
assert q['nearest_x']==1 and q['nearest_y']==1 and q['full_mask_at_nearest'] and q['original_inputU'] and q['new_native_visibility_label']=='UNKNOWN'
assert q['scaled_depth_mm_at_nearest'] is None and q['window3x3_in_frame_samples']==9
checks.append('nearest_half_up_query_U_preserved_despite_positive_mask')
q=m.query_diagnostic([-.5,-.5],True,f,v,d,2.)
assert q['nearest_x']==q['nearest_y']==0 and q['window3x3_in_frame_samples']==4 and q['raw_depth_at_nearest']==0 and q['scaled_depth_mm_at_nearest'] is None
checks.append('border_3x3_clipped_and_invalid_zero_depth_not_metric')
q=m.query_diagnostic([2.5,1.],True,f,v,d,2.)
assert not q['nearest_pixel_in_frame'] and not q['original_q_in_continuous_image_canvas'] and q['full_mask_at_nearest'] is None
checks.append('right_canvas_boundary_no_clamp_or_fallback')
q=m.query_diagnostic([1.,1.],True,f,v,d,2.)
assert q['raw_depth_at_nearest']==40 and q['scaled_depth_mm_at_nearest']==80
checks.append('metric_depth_only_explicit_positive_scale')
for q in [None,[float('nan'),0]]:
    assert not m.query_diagnostic(q,False,f,v,d,None)['finite_original_q']
checks.append('null_nan_query_uncomputed_unknown')
assert m.scale_from_camera(None) is None and m.scale_from_camera({'depth_scale':'2'}) is None
assert m.scale_from_camera({'depth_scale':0}) is None and m.scale_from_camera({'depth_scale':float('nan')}) is None
assert m.scale_from_camera({'depth_scale':.1})==.1
checks.append('missing_invalid_camera_scale_not_invented')
buf=io.BytesIO();Image.fromarray(d).save(buf,format='PNG');dd=m.decode_png(buf.getvalue())
assert np.array_equal(dd,d)
checks.append('invented_depthPNG_integer_values_lossless')
rows=[{'native_slot':i,'query_xy':[float(i%3),float(i//3)],'original_query_available':i!=0} for i in range(8)]
png=m.panel(f,v,d,rows,'INVENTED ONLY',np.repeat(f[:,:,None],3,axis=2))
with Image.open(io.BytesIO(png)) as im:assert im.size==(1920,525)
checks.append('invented_mask_only_panel_no_source_RGB_or_q_mutation')
assert rows[0]['query_xy']==[0.,0.] and not rows[0]['original_query_available']
records=[]
for kind,n in [('FULL_INSTANCE_MASK_PNG',15),('VISIBLE_INSTANCE_MASK_PNG',15),('DEPTH_PNG',12)]:
    records += [{'kind':kind,'path':'/declared/'+kind+'/'+str(i),'sha256':'a'*64} for i in range(n)]
m.gate_pixels(records);checks.append('exact42_kind_census_gate')
bad('RGB_or_CAD_record_rejected',lambda:m.gate_pixels([{**records[0],'kind':'NOMINAL_OBJ1_CAD'}]+records[1:]))
bad('unknown_pixel_SHA_rejected',lambda:m.gate_pixels([{**records[0],'sha256':None}]+records[1:]))
with patch.object(m.os,'open',side_effect=AssertionError('FS must not happen')):
    bad('V112_before_FS',lambda:m.opened('/forbidden/V112/data.png'))
with patch.object(m,'raw',return_value=b'{"approved_actual_fixed12_mask_depth_diagnostic":false}'):
    bad('approved_false_before_any_payload_read',lambda:m.run(m.D+'/OWN_FALSE_ENTRY.json'))
with tempfile.TemporaryDirectory(prefix='invented_paths_',dir=H) as p:
    with open(p+'/regular','w') as h:h.write('x')
    os.symlink(p+'/regular',p+'/link')
    bad('symlink_file_rejected',lambda:m.opened(p+'/link'))
assert m.GT_info_context({'bbox_obj':[-2,0,4,3],'px_count_visib':10,'px_count_valid':2},3,3)['GT_bbox_obj_oob_status']=='BBOX_PARTLY_OUTSIDE'
assert m.GT_info_context({'bbox_obj':[0,0,3,3]},3,3)['GT_bbox_obj_clipped_area_fraction']==1
assert m.GT_info_context(None,3,3)['GT_bbox_obj_oob_status'].startswith('UNKNOWN')
checks.append('saved_global_bbox_OOB_and_pxvis_gt_pxvalid_not_native_label_or_invalid_assumption')
bad('context_dimension_mismatch_no_alignment_resize',lambda:m.panel(f,v,d,rows,'INVENTED',np.zeros((2,3,3),np.uint8)))
assert m.GT_info_context({'bbox_obj':[-1,-1,-1,-1],'px_count_valid':0,'px_count_visib':0,'visib_fract':0},3,3)['GT_bbox_obj_oob_status']=='NEGATIVE_BBOX_SENTINEL_NO_GEOMETRIC_SUPPORT'
checks.append('negative_bbox_sentinel_retained_no_OOBfraction_or_nativeE')
result={'completed':True,'PASS':True,'invented_functional_checks':len(checks),'checks':checks,'actual_source_payload_stat_hash_read_decode':0,'sourceGT_or_model_or_projection_execution':0}
b=(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode()
with os.fdopen(m.opened(H+'/INVENTED_FUNCTIONAL_CHECKS_BOUNDARIES_FINAL.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL),'wb') as h:h.write(b)
print(json.dumps({'completed':True,'PASS':True,'checks':len(checks),'sha256':m.sha(b)}))

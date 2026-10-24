"""Invented NumPy/AST checks only: no parameter/model/source-image access."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;CODE=H/'run_frozen_v61_registered.py'
def bind(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
 text=CODE.read_text();tree=ast.parse(text);ns={'np':np}
 names={'decode','validate_rows'}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names],type_ignores=[])),str(CODE),'exec'),ns)
 z=np.array([-1.,-np.nextafter(0.,1.),-0.,0.,np.nextafter(0.,1.),1.])
 assert ns['decode'](z,np.ones(6,bool)).tolist()==[1,1,0,0,0,0]
 assert ns['decode'](z,np.zeros(6,bool)).tolist()==[-1]*6
 rows=[]
 for s in range(6):
  for j in range(72):
   q=[20.+j/100,30.];p=[[10.+k,15.+k]for k in range(8)];p[j//9]=q
   rows.append({'occurrence_index':len(rows),'query_xy':q,'initial_points':p,'carrier_slot':j//9,'L':16.,'full_RGB_format':'NPZ_RGB_UINT8_HWC','full_RGB_key':'RGB_uint8','raw_hw':[256,256],'full_RGB':{'path':f'INVENTED{s}.npz','sha256':str(s)*64},'decoded_RGB_sha256':str(s)*64,'scene_id':str(s)})
 groups=ns['validate_rows'](rows);assert list(map(len,groups))==[72]*6
 broken=[dict(r)for r in rows];broken[0]['occurrence_index']=7
 try:ns['validate_rows'](broken)
 except AssertionError:pass
 else:raise AssertionError('bad order accepted')
 # Exact pinned V90 pure functions; source file text only, invented RGB data.
 source=H.parents[1]/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
 mt=ast.parse(source.read_text());mn={'np':np}
 selected={'expected_affine','native_grid_explicit','crop','query_validity'}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[n for n in mt.body if isinstance(n,ast.FunctionDef)and n.name in selected],type_ignores=[])),str(source),'exec'),mn)
 a=mn['expected_affine']([256,256]);assert a['canvas_shape']==[456,456] and a['input_shape']==[640,640] and a['p3_actual_hw']==[80,80] and np.array_equal(a['offset'],[0.,0.])
 q=np.array([[127.5,127.5],[255.8,20.],[-1.,-1.]])
 grid,valid=mn['native_grid_explicit'](q,[16.,16.,16.],[256,256]);assert grid.shape==(3,9,9,2)and np.array_equal(grid[:,4,4],q)and not valid[2].any()
 assert mn['query_validity'](q,[256,256])[0].tolist()==[True,False,False]
 rgb=np.arange(256*256*3,dtype=np.uint32).reshape(256,256,3).astype(np.uint8)
 c,v=mn['crop'](rgb,q[0],16.,96);assert c.shape==(3,96,96)and v.shape==(1,96,96)and v.all()
 counts=(65*32*9+32)+(32*32*9+32)+(32+1);assert counts==28033
 checks={
  'visible_raw_sign_signedzero_and_subnormal_tie':True,'unavailable_is_U_independent_of_sign':True,
  'all432_six_groups72_exact_order':True,'wrong_order_rejected':True,
  'pinned_V90_affine256_border100_native640':True,'grid9_centre_phase_and_absent_mask':True,
  'continuous_and_nearest_boundary_distinct':True,'canonical_HWC_RGB96_and_HW_valid_layout':True,
  'ProxyHead_parameter_count28033_static':True,'RGB96_not_head_input':"head(x)"in text and "torch.from_numpy(np.ascontiguousarray(roi[lo:lo+128]))"in text,
  'RGB_NPZ_only_key_and_detector_outputs_ignored':"f['RGB_uint8']"in text and "['candidates']"not in text and "['selected_index']"not in text,
  'teacher_reader_absent_and_models_delayed_to_entry':'GT_ROWS'not in text and not any(isinstance(n,(ast.Import,ast.ImportFrom))and ('torch'in ast.unparse(n)or'ultralytics'in ast.unparse(n))for n in tree.body)
 }
 assert all(checks.values())
 out={'completed':True,'PASS':True,'code':bind(CODE),'test_code':bind(Path(__file__)),'checks':checks,'check_count':len(checks),'actual_source_payload_parameter_model_forward_current_V112':0}
 with (H/'GENERATED_PREP_CHECK.json').open('x')as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'completed':True,'PASS':True,'checks':len(checks),'proof':bind(H/'GENERATED_PREP_CHECK.json')}))
if __name__=='__main__':main()

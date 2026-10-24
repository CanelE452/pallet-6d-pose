"""V195 authored scalar/metadata fixtures only. No actual evaluation bodies."""
from pathlib import Path
import ast,hashlib,importlib.util,json,tempfile
import numpy as np
H=Path(__file__).absolute().parent

def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def check():
 e=module(H/'evaluation_preparation_b/evaluate_saved_grouped_direct_end_real.py','invented_v195_evaluator')
 old=e.OLD191/'evaluation_preparation_a/evaluate_saved_direct_end_real.py'
 def fs(p):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
 a,b=fs(old),fs(H/'evaluation_preparation_b/evaluate_saved_grouped_direct_end_real.py')
 unchanged=['safe','opened','sha','bind','check','read','lines','write','write_lines','equal','extract','kernels','decode','direct_decode','joint','qarray','packet_schema','current_associations','validate_prediction_seal']
 tests={'inherited_scalar_path_decode_schema_and_metric_kernels_AST_exact':all(a[k]==b[k]for k in unchanged)}
 binding=dict(path='/invented/meta.json',sha256='a'*64)
 protocol=dict(prior_direct_prediction_code=binding,prior_direct_prediction_protocol=binding,input_complete=binding,original_raw_complete=binding)
 seal=dict(completed=True,PASS=True,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,code=binding,protocol=binding,input_complete=binding,original_raw_complete=binding,arms={k:dict(outputs={'D':binding})for k in e.PRIOR_DIRECT_ARMS})
 tests['old_direct_saved_seal_maps_C54_C55_only']=set(e.prior_direct_outputs(seal,protocol))=={'C54','C55'}
 rejected=0
 for changes in [dict(code=dict(binding,sha256='b'*64)),dict(input_complete=dict(binding,sha256='b'*64)),dict(arms={e.PRIOR_DIRECT_ARMS[0]:seal['arms'][e.PRIOR_DIRECT_ARMS[0]]}),dict(all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=False)]:
  try:e.prior_direct_outputs(dict(seal,**changes),protocol)
  except ValueError:rejected+=1
 tests['old_direct_wrong_code_input_missing_arm_or_open_seal_rejected']=rejected==4
 with tempfile.TemporaryDirectory(prefix='invented_V195_packet_',dir=H)as d:
  d=Path(d);n=8;av=np.array([True]*7+[False]);q=np.ones((n,2),np.float64);q[-1]=np.nan;L=np.full(n,96.,np.float64);L[-1]=np.nan;s=np.zeros((n,2),np.float32);o=np.zeros((n,3),np.float32);s[-1]=np.nan;o[-1]=np.nan;dr=np.full(n,-1.,np.float32);dr[-1]=np.nan
  sp,op,proxy=e.decode(s,o,av);dp=e.direct_decode(dr,av)
  raw=dict(row_id=np.array(['TOY:native'+str(i)for i in range(n)]),original_global_index=np.arange(n,dtype=np.int64),native_slot=np.arange(n,dtype=np.int8),query_xy=q,L=L,original_query_available=av,raw_support_logits=s,raw_owner_logits=o,support_prediction=sp,owner_prediction=op,regional_support_proxy_D0_E1_Uminus1=proxy,support_order=np.array(e.SUPPORT_ORDER),owner_order=np.array(e.OWNER_ORDER))
  direct=dict(raw,raw_direct_end_external_logit=dr,direct_end_prediction_D0_E1_Uminus1=dp,direct_end_order=np.array(['VISIBLE_END0','EXTERNAL_BLOCKED_END1','UNKNOWNminus1']))
  rows=[]
  for i in range(n):
   qq=None if i==7 else q[i].tolist();ll=None if i==7 else float(L[i]);rows.append(dict(domain='TOY8',domain_index=i,original_global_index=i,id='TOY',row_id='TOY:native'+str(i),native_slot=i,original_query_available=bool(av[i]),query_xy=qq,L=ll,actual_crop_grid_query_xy=qq,actual_crop_grid_L=ll))
  def save(name,v):
   f=d/name
   with f.open('xb')as h:np.savez_compressed(h,**v)
   return e.bind(f)
  e.DOMAINS['TOY8']=(8,0,1)
  db=save('DIRECT16.npz',direct);rb=save('REGIONAL13.npz',raw)
  old_direct=e.load_packet(db,rows,'TOY8',tag='C54');new_direct=e.load_packet(db,rows,'TOY8',old_direct,tag='C56');regional=e.load_packet(rb,rows,'TOY8',old_direct,tag='C52')
  tests['actual_packet_adapter_old_and_new_direct_keep_direct_route']=old_direct['primary_prediction'].tolist()==new_direct['primary_prediction'].tolist()==[0]*7+[-1]
  tests['regional_NONE_stays_U_while_direct_visible_and_inputU_first']=regional['primary_prediction'].tolist()==[-1]*8 and new_direct['primary_prediction'][-1]==-1
  tests['all_three_saved_routes_keep_META6_byteexact']=all(e.equal(old_direct[k],new_direct[k])and e.equal(old_direct[k],regional[k])for k in e.META)
  del e.DOMAINS['TOY8']
 tests['all8_current_and7_public_model_routes_fixed']=e.MODELS==('v61','legacy','C52','C53','C54','C55','C56','C57')and e.DIRECT_TAGS==('C54','C55','C56','C57')
 assert all(tests.values()),tests
 return dict(completed=True,PASS=True,invented_only=True,tests={k:bool(v)for k,v in tests.items()},actual_score_label_RGB_weight_header_or_output_reads=0)

if __name__=='__main__':
 r=check();f=H/'GENERATED_ADAPTATION_CHECKS.json'
 with f.open('x')as h:json.dump(r,h,indent=2);h.write('\n')
 print(json.dumps(r))

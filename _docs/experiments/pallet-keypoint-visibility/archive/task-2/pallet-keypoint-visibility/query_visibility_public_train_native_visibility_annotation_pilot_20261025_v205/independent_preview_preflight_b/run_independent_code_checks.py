"""Independent CODE/selected-metadata preflight. No operational source body access."""
from pathlib import Path
from collections import Counter,defaultdict
import ast,copy,hashlib,itertools,json,os,stat
import numpy as np
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).parent;V=O.parent;P=V.parent;H=V/'preview_preparation_a'
def rb(p):
 p=Path(p)
 if 'v112' in str(p).casefold():raise ValueError('forbidden before FS')
 for q in reversed([p,*p.parents]):
  if stat.S_ISLNK(q.lstat().st_mode):raise ValueError('symlink')
 return p.read_bytes()
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(rb(p)).hexdigest()}
def get(b):
 data=rb(b['path']);assert hashlib.sha256(data).hexdigest()==b['sha256'];return json.loads(data)
def save(n,d):
 with (O/n).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 readyb=bind(H/'FINAL_READY_V2_LAYOUT.json');assert readyb['sha256']=='e6356e497071dfa7eacf3ce66ee33310ae1f734a5e065757c3cdbb1d2e98d6fa'
 ready=get(readyb);proto=get(ready['protocol']);plan=get(ready['selection']);entry=get(ready['false_entry']);ownchecks=get(ready['checks'])
 assert ready['code']==proto['code'] and ready['selection']==proto['selection'] and ownchecks['PASS'] and ownchecks['count']==8
 src=rb(ready['code']['path']);assert hashlib.sha256(src).hexdigest()==ready['code']['sha256'];tree=ast.parse(src)
 names={'need','safe','sameq','prefix','match_possible','select_six','input_rows','marker','contact','observation_rows'}
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names];assert {x.name for x in nodes}==names
 ns={'Path':Path,'P':P,'np':np,'Counter':Counter,'defaultdict':defaultdict,'hashlib':hashlib,'os':os,'stat':stat,'Image':Image,'ImageDraw':ImageDraw,'ImageFont':ImageFont,'FONT':'/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','PREFIXES':('val/000000','val/000002','val/000005','val/000006'),'QUOTA':(2,2,1,1)}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'exact-reviewed-pure-functions','exec'),ns)
 checks=[]
 def check(n,v):assert v,n;checks.append({'name':n,'PASS':True})
 def rejects(n,fn):
  try:fn()
  except (AssertionError,KeyError,ValueError):check(n,True)
  else:raise AssertionError(n+' did not reject')
 fake=[]
 for p in ns['PREFIXES']:
  for i in range(3):
   key=p+':'+str(i);fake.append(dict(frame_key=key,prefix=p,component_id=key,stable_SHA256=hashlib.sha256(key.encode()).hexdigest()))
 actual,m=ns['select_six'](fake)
 expected=[]
 for p,q in zip(ns['PREFIXES'],ns['QUOTA']):expected+=sorted((r for r in fake if r['prefix']==p),key=lambda r:(hashlib.sha256(r['frame_key'].encode()).hexdigest(),r['frame_key']))[:q]
 check('Independent_SHA256_scalar_prefix2_2_1_1_order',actual==expected and m['selected_distinct_components']==6)
 check('Selection_invariant_under_reversed_input',ns['select_six'](fake[::-1])[0]==actual)
 shared=[dict(r,component_id='shared')for r in fake];a,m=ns['select_six'](shared);check('Component_shortfall_disclosed_no_data_fabrication',len(a)==6 and m['selected_distinct_components']==1 and not m['all6_distinct_components_possible'])
 # Full invented185x8 metadata exercises actual join, including None/finite U and no GT bbox fallback.
 rows=[];roles=[];frames=[]
 pp=[p for p,n in zip(ns['PREFIXES'],[65,38,24,28])for _ in range(n)]+['val/000000']*30
 for fi in range(185):
  sid=pp[fi]+':'+str(fi);fid='INVENTED_'+str(fi);packet={'path':'/NEVER_OPEN/'+str(fi)+'.npz','sha256':'a'*64};qs=[];ls=[];gids=[];ids=[]
  for k in range(8):
   idx=fi*8+k;av=not(fi<155 and idx<161);q=None if not av and k%2 else [100.125+k,200.5];L=None if q is None else 220.75;rid=fid+':native'+str(k);qs.append(q);ls.append(L);gids.append(idx);ids.append(rid)
   r=dict(row_id=rid,original_global_index=idx,native_slot=k,query_xy=q,L=L,original_query_available=av,raw_packet=packet,raw_row=k,actual_raw_RGB96_sha256='b'*64,actual_raw_valid96_sha256='c'*64,original=dict(split='SOURCE_DEVELOPMENT',native_slot=k,query_id=rid,frame_index=fi,source_id=sid,frame_id=fid,feature_constructed=av),SAM_output={'path':'/NEVER_OPEN/SAM.npz'},owner_target=2,teacher='MUST_NOT_ESCAPE')
   rows.append(r);roles.append({**{x:r[x]for x in ('row_id','original_global_index','native_slot','query_xy','L','original_query_available')},'role':'TRAIN'if fi<155 else'HOLDOUT','component_id':'C'+str(fi%103)})
  frames.append(dict(original_frame_index=fi,split='SOURCE_DEVELOPMENT',source_id=sid,frame_id=fid,saved_rgb96_packet=packet,original_initial_points8=qs,original_L8=ls,original_global_indices=gids,original_query_ids=ids,image={'binding':{'path':'/NEVER_OPEN/full.npy','sha256':'d'*64}},original_RGB={'path':'/NEVER_OPEN/original.jpg','sha256':'e'*64},original_RGB_decoder='ORIGINAL',context_box_xyxy=[10,20,30,40],GT_bbox=[1,2,3,4]))
 out=ns['input_rows'](rows,roles,frames)
 check('OriginalTRAIN_only_155x8_U161_META_join',len(out)==155 and sum(not r['original_query_available']for f in out for r in f['native8'])==161)
 check('Null_L_and_q_U_preserved_without_imputation',out[0]['native8'][1]['L']is None and out[0]['native8'][1]['query_xy']is None and not out[0]['native8'][1]['original_query_available'])
 check('No_GT_context_bbox_or_owner_SAM_teacher_export',all(f['selected_detection_bbox_xyxy']is None and all(not any(k in r for k in ('SAM_output','owner_target','teacher','GT_bbox'))for r in f['native8'])for f in out))
 bad=copy.deepcopy(roles);bad[0]['query_xy']=[999.,999.];rejects('Query_META_drift_rejected',lambda:ns['input_rows'](rows,bad,frames))
 bad=copy.deepcopy(roles);bad[0]['native_slot']=1;rejects('Native_META_drift_rejected',lambda:ns['input_rows'](rows,bad,frames))
 check('Finite_U_marker_kept_false_not_recentered',ns['marker']([2.25,3.5],False,10,10)==(2.25,3.5,False))
 check('None_and_outside_U_no_fabricated_marker',ns['marker'](None,False,10,10)is None and ns['marker']([11,3],False,10,10)is None)
 raw={'rgb96':np.arange(8*96*96*3,dtype=np.uint32).reshape(8,96,96,3).astype(np.uint8),'valid96':np.ones((8,96,96),bool),'crop_constructed':np.ones(8,bool)};before=raw['rgb96'].copy();im=np.asarray(ns['contact'](out[0],raw))
 check('Actual_contact_clean_ROIs_exact_native_order_no_input_mutation',np.array_equal(before,raw['rgb96'])and all(np.array_equal(im[86+450*(k//4)+78:86+450*(k//4)+174,400*(k%4)+300:400*(k%4)+396],raw['rgb96'][k])for k in range(8)))
 obs=ns['observation_rows']([out[0]]);check('Only_UNREVIEWED_template_no_generated_labels_or_admission',len(obs)==8 and all(r['part_visibility_D_E_S_UNKNOWN']=='UNREVIEWED' and r['training_admission']is False for r in obs))
 rejects('V112_lexical_before_FS',lambda:ns['safe'](P/'forbidden_V112'/'must_not_stat'))
 # Selected metadata, no original data/roles/arrays are reopened.
 selected=plan['frames'];expected_whitelist=[b for f in selected for b in (f['image']['binding'],f['native8'][0]['raw_packet'])]
 check('Exact12_selected_only_bindings_false_entry',len(expected_whitelist)==len({b['path']for b in expected_whitelist})==12 and expected_whitelist==proto['binary_input_whitelist']==entry['binary_input_whitelist'] and entry['approved_selected6_TRAIN_input_preview']is False)
 check('Fixed6_indices_6components_48native_U8_no_direct12_overlap',[f['original_frame_index']for f in selected]==[17,67,150,155,187,205]and len({f['component_id']for f in selected})==6 and sum(len(f['native8'])for f in selected)==48 and sum(not r['original_query_available']for f in selected for r in f['native8'])==8 and not({f['original_frame_index']for f in selected}&set(plan['census']['excluded_existing_direct12_frame_indices'])))
 check('Stable_hash_categories_only_no_outcome_selection',all(f['stable_SHA256']==hashlib.sha256(f['frame_key'].encode()).hexdigest() and f['role']=='TRAIN' and [r['native_slot']for r in f['native8']]==list(range(8))for f in selected)and plan['old_weak_labels_or_predictions_used']is False)
 check('No_deployed_selected_bbox_no_fallback',all(f['selected_detection_bbox_xyxy']is None for f in selected))
 run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run');loads=[x for x in ast.walk(run)if isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)and isinstance(x.func.value,ast.Name)and x.func.value.id=='np'and x.func.attr=='load']
 check('Only_two_np_load_calls_fullRGB_and_selectedframe_inputpacket',len(loads)==2 and all(any(k.arg=='allow_pickle'and isinstance(k.value,ast.Constant)and k.value.value is False for k in x.keywords)for x in loads))
 save('GENERATED_INDEPENDENT_FUNCTIONAL_CHECKS.json',{'completed':True,'PASS':True,'checks':checks,'count':len(checks),'invented_only':True,'selected_metadata_only':True,'production_source_array_header_image_GT_model_score_reads':0})
 print(json.dumps({'completed':True,'PASS':True,'checks':len(checks),'check_binding':bind(O/'GENERATED_INDEPENDENT_FUNCTIONAL_CHECKS.json')}))
if __name__=='__main__':main()

"""V216 sealed two-reader join and display. Preparation reads no actual observations/PNGs."""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import html
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
import stat
import traceback

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
V215=P/'query_visibility_public_train_visible_part_coverage_expansion_20261025_v215'
OUT=V/'joined_part_query_support_v2'
FONT=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
PROTO=H/'PROTOCOL_BEFORE_SEALED_OBSERVATIONS_OR_GENERATED_PNG_BODIES.json'
INPUTS={
 'ready':{'path':str(V/'contract_preparation_a/FINAL_READY.json'),'sha256':'b2130d4a3dc018d304556af28fa8b08556e813cdb1738c0004ea0b5582fccb99'},
 'helper':{'path':str(V/'contract_preparation_a/part_query_support_contract.py'),'sha256':'72d4286ffc085472719a334cee969ee11b5231dd98001792a6817ef2b63ed551'},
 'contract':{'path':str(V/'contract_preparation_a/ANNOTATION_AND_SUPERVISION_CANDIDATE_CONTRACT.json'),'sha256':'054a04e6e0890d94bd5dfcf7c69195290554a2ce39c55f53ad1d5c66b845e8cc'},
 'template':{'path':str(V/'contract_preparation_a/INDEPENDENT_ANNOTATION_TEMPLATE96.json'),'sha256':'483e3da86a8ab064092228a5691be0ba848621287e32d5a5df778c37bd1de5a3'},
 'split':{'path':str(V/'component_split_preparation_b/INTERNAL_SOURCE_FIT_DEV_COMPONENT_SPLIT.json'),'sha256':'1ee8271189e82c72a89461f880a5c4464ce474035ba12189ddaf175fce407ce9'},
 'exact_OR_addendum':{'path':str(V/'ROOT_EXACT_TWO_READER_PIXEL_UNION_ADDENDUM_BEFORE_ANNOTATION.json'),'sha256':'3a1b2cb910c67bc65df460de743dc458fd739583fc5789d3797d34b687962905'},
 'declared_PNGs':{'path':str(V/'ROOT_DECLARED228_DERIVED_PUBLIC_PNG_BINDINGS_BEFORE_NEW_ANNOTATION.json'),'sha256':'ef3c338700490bef0c2d57f7da426aa53160468a93b3c2ad5fe879ad03da3fd9'},
}
AXES=('instance','part','witness','addressability','query_support','visibility')

def need(v,m):
 if not bool(v):raise AssertionError(m)

def safe(path):
 s=str(path);need('v112'not in s.casefold(),'V112 lexical reject before filesystem')
 p=Path(s);need(p.is_absolute()and'..'not in p.parts and(P in p.parents or p==FONT),'Exact safe owned path')
 return p

def parentfd(path):
 p=safe(path);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for q in p.parts[1:-1]:
   n=os.open(q,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  return p,fd
 except BaseException:os.close(fd);raise

def opened(path,write=False):
 p,fd=parentfd(path)
 try:n=os.open(p.name,(os.O_WRONLY|os.O_CREAT|os.O_EXCL if write else os.O_RDONLY)|os.O_NOFOLLOW,0o644,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'Regular file/no symlink')
 return os.fdopen(n,'wb'if write else'rb')

def mkdir_new(path):
 p,fd=parentfd(path)
 try:os.mkdir(p.name,dir_fd=fd)
 finally:os.close(fd)

def rb(path):
 with opened(path)as f:return f.read()

def sha(data):return hashlib.sha256(data).hexdigest()
def bind(path):return dict(path=str(safe(path)),sha256=sha(rb(path)))
def canonical_binding_identity(b):
 need(isinstance(b,dict)and set(b)in({'path','sha256'},{'path','sha256','size_bytes'}),'Only path/SHA and optional size metadata keys')
 need(type(b['path'])is str and type(b['sha256'])is str and re.fullmatch('[0-9a-f]{64}',b['sha256']),'Canonical path/SHA binding types')
 safe(b['path'])
 if 'size_bytes'in b:need(type(b['size_bytes'])is int and b['size_bytes']>=0,'Optional size is a nonnegative integer, not bool')
 return(b['path'],b['sha256'])
def same_binding(a,b):return canonical_binding_identity(a)==canonical_binding_identity(b)
def same_binding_map(a,b):return isinstance(a,dict)and isinstance(b,dict)and set(a)==set(b)and all(same_binding(a[k],b[k])for k in a)
def validate_bound_bytes(b,data):
 canonical_binding_identity(b);need(sha(data)==b['sha256'],'Exact SHA binding')
 if 'size_bytes'in b:need(len(data)==b['size_bytes'],'Recorded optional byte size matches actual read')
 return data
def checked_bytes(b):
 canonical_binding_identity(b)
 return validate_bound_bytes(b,rb(b['path']))
def checked(b):
 checked_bytes(b)
 return b['path']
def read(b):
 need(Path(b['path']).suffix=='.json','Only bound JSON metadata here')
 return json.loads(checked_bytes(b))
def text(path,value):
 with opened(path,True)as f:f.write(value.encode())
def write(path,value):text(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def bkey(b):return canonical_binding_identity(b)
def helper():
 s=checked_bytes(INPUTS['helper']);ns={'__name__':'v216_frozen_pure_contract'};exec(compile(s,INPUTS['helper']['path'],'exec'),ns);return ns

def metadata(h):
 ready=read(INPUTS['ready']);contract=read(INPUTS['contract']);template=read(INPUTS['template']);split=read(INPUTS['split']);add=read(INPUTS['exact_OR_addendum']);declared=read(INPUTS['declared_PNGs'])
 need(ready['completed']and ready['PASS']and same_binding(ready['code'],INPUTS['helper'])and same_binding(ready['template'],INPUTS['template'])and same_binding(ready['SOURCE_split'],INPUTS['split']),'Frozen A contract closure')
 rr=template['rows'];ss=split['rows96']
 need(len(rr)==len(ss)==96 and len({r['input']['row_id']for r in rr})==96,'Exact96 unreviewed row identities')
 for expected,s in zip(rr,ss):
  h['validate_observation'](expected,expected,allow_template=True)
  m=expected['input']
  for k in('row_id','component_id','source_group_key','role','internal_source_role','native_slot','query_xy','L','original_query_available','original_flags','position','frame_key','original_frame_index','original_global_index'):
   need(m[k]==s[k],'Original input/split metadata parity: '+k)
 need(Counter(r['input']['internal_source_role']for r in rr)=={'SOURCE_FIT':64,'SOURCE_DEV':32}and sum(not r['input']['original_query_available']for r in rr)==11,'OriginalFIT64/DEV32/U11')
 images=declared['images'];need(len(images)==228 and len({bkey(i['binding'])for i in images})==228,'Exact predeclared228 derived PNGs')
 imap={(x['kind'],x['position'],x.get('native_slot')):x['binding']for x in images}
 for r,s in zip(rr,ss):
  m=r['input'];p=m['position'];k=m['native_slot']
  need(same_binding(r['display_bindings']['clean96'],imap['clean96',p,k])and same_binding(s['clean96'],imap['clean96',p,k])and same_binding(s['valid96'],imap['valid96',p,k])and same_binding(r['display_bindings']['contact'],imap['all8contact',p,None]),'Original derived PNG bindings exact')
 return rr,ss,imap,images,add,contract

def exact_mask(a,c):
 mask=bytearray(96*96)
 for r in(a,c):
  if r is None:continue
  need(type(r)is list and len(r)==4 and all(type(x)is int and 0<=x<=95 for x in r)and r[0]<=r[2]and r[1]<=r[3],'Original inclusive ROI only')
  for y in range(r[1],r[3]+1):mask[y*96+r[0]:y*96+r[2]+1]=b'\1'*(r[2]-r[0]+1)
 return bytes(mask)

def field_differences(a,c):
 return{axis:{'categorical_difference':any(a[axis].get(k)!=c[axis].get(k)for k in('status','state','kind')if k in a[axis]),'any_field_difference':a[axis]!=c[axis],'different_fields':[k for k in a[axis]if a[axis][k]!=c[axis][k]]}for axis in AXES}

def mask_all_valid(mask,valid):
 need(len(mask)==len(valid)==9216,'Fixed96 validity canvas')
 return None if not any(mask)else all(not x or valid[i]==255 for i,x in enumerate(mask))

def join_rows(h,a,c,expected,labels):
 need(len(a)==len(c)==len(expected)==96,'All96 rows for both readers')
 joined=[];sidecars=[]
 for ar,cr,ex in zip(a,c,expected):
  need(ar['reader_id']=='A'and cr['reader_id']=='C','Exact independent reader identities and original order')
  raw=h['join_two_readers'](ar,cr,ex);m=ex['input'];wa=ar['witness']['roi'];wc=cr['witness']['roi'];mask=exact_mask(wa,wc)
  roi_list=[r for r in(wa,wc)if r is not None]
  bbox=None if not roi_list else[min(r[0]for r in roi_list),min(r[1]for r in roi_list),max(r[2]for r in roi_list),max(r[3]for r in roi_list)]
  joined.append(dict(input=m,display_bindings=ex['display_bindings'],reader_A_axes={k:ar[k]for k in AXES+('ambiguity_note',)},reader_C_axes={k:cr[k]for k in AXES+('ambiguity_note',)},raw_helper_candidate=raw,all_axis_differences=field_differences(ar,cr),root_semantic_same_feature_and_addressability_review='PENDING',training_admission=False))
  sidecars.append(dict(row_id=m['row_id'],input=m,original_clean96=ex['display_bindings']['clean96'],original_A_witness_ROI=wa,original_C_witness_ROI=wc,original_A_visibility_evidence_ROI=ar['visibility']['evidence_roi'],original_C_visibility_evidence_ROI=cr['visibility']['evidence_roi'],whole_A_observations=labels['A'],whole_C_observations=labels['C'],original_A_C_ROIs_individually_preserved=True,exact_OR_rule='Pixelwise OR of original inclusive witness rectangles on96x96; rowmajor uint8 values0/1',exact_OR_mask_sha256=sha(mask),exact_OR_area=sum(mask),both_original_witness_ROIs_present=wa is not None and wc is not None,legacy_helper_witness_union_bbox_DISPLAY_ONLY=raw['witness_ROI_union'],all_available_ROIs_bbox_DISPLAY_ONLY=bbox,bbox_is_never_alpha_or_proof_mask=True,candidate_union_all_valid96=None,valid96_check_status='NOT_REQUESTED_PENDING_ROOT_PNG_SCOPE',pixel_proof_failures=[],root_semantic_review_pending=True,training_admission=False))
 return joined,sidecars

def aggregate(joined):
 table=[];summary={}
 for role in('ALL','SOURCE_FIT','SOURCE_DEV'):
  rr=[r for r in joined if role=='ALL'or r['input']['internal_source_role']==role]
  item=dict(N=len(rr),frames=len({r['input']['position']for r in rr}),components=len({r['input']['component_id']for r in rr}),original_inputU=sum(not r['input']['original_query_available']for r in rr),readers={},join={},native8={},local_witness_candidates=sum(r['raw_helper_candidate']['spatial_witness_agreement_candidate']for r in rr))
  for who,key in(('A','reader_A_axes'),('C','reader_C_axes')):
   item['readers'][who]={axis:dict(Counter(r[key][axis]['state']for r in rr))for axis in('visibility','query_support')}
   for axis,counts in item['readers'][who].items():
    for state,n in counts.items():table.append(dict(internal_source_role=role,native_slot='ALL',reader=who,axis=axis,state=state,N=n))
  item['join']={'visibility_candidate':dict(Counter(r['raw_helper_candidate']['native_weak_visibility_agreement_candidate']for r in rr)),'query_support_agreement':dict(Counter(r['raw_helper_candidate']['query_support_agreement']for r in rr)),'categorical_axis_disagreements':{axis:sum(r['all_axis_differences'][axis]['categorical_difference']for r in rr)for axis in AXES}}
  for axis,counts in item['join'].items():
   if axis=='categorical_axis_disagreements':continue
   for state,n in counts.items():table.append(dict(internal_source_role=role,native_slot='ALL',reader='JOIN_RAW_CANDIDATE',axis=axis,state=state,N=n))
  summary[role]=item
  for k in range(8):
   nr=[r for r in rr if r['input']['native_slot']==k]
   ni=dict(N=len(nr),original_inputU=sum(not r['input']['original_query_available']for r in nr),local_witness_candidates=sum(r['raw_helper_candidate']['spatial_witness_agreement_candidate']for r in nr),join_visibility_candidate=dict(Counter(r['raw_helper_candidate']['native_weak_visibility_agreement_candidate']for r in nr)),join_query_support_agreement=dict(Counter(r['raw_helper_candidate']['query_support_agreement']for r in nr)))
   item['native8'][str(k)]=ni
   for state,n in(('NATIVE_ROWS',ni['N']),('ORIGINAL_INPUT_U',ni['original_inputU'])):table.append(dict(internal_source_role=role,native_slot=k,reader='INPUT_METADATA',axis='population',state=state,N=n))
   table.append(dict(internal_source_role=role,native_slot=k,reader='JOIN_RAW_CANDIDATE',axis='local_witness',state='SPATIAL_CANDIDATE',N=ni['local_witness_candidates']))
   for axis,counts in(('visibility_candidate',ni['join_visibility_candidate']),('query_support_agreement',ni['join_query_support_agreement'])):
    for state,n in counts.items():table.append(dict(internal_source_role=role,native_slot=k,reader='JOIN_RAW_CANDIDATE',axis=axis,state=state,N=n))
   for who,key in(('A','reader_A_axes'),('C','reader_C_axes')):
    for axis in('visibility','query_support'):
     for state,n in Counter(r[key][axis]['state']for r in nr).items():table.append(dict(internal_source_role=role,native_slot=k,reader=who,axis=axis,state=state,N=n))
 return summary,table

def invented_tests(h,add):
 b={'path':str(H/'INVENTED_UNOPENED.png'),'sha256':'0'*64}
 ex=dict(schema_version='V216_SOURCE_PART_QUERY_WEAK_V1',input=dict(row_id='INVENTED',component_id='i',source_group_key='i',role='TRAIN',internal_source_role='SOURCE_DEV',native_slot=0,original_query_available=True,query_xy=[12.25,20.5],L=96.),display_bindings=dict(contact=b,clean96=b,full_context=[b]),reader_id=None,observation_complete=False,instance=dict(status='UNREVIEWED',evidence=''),part=dict(status='UNREVIEWED',description='',evidence=''),witness=dict(status='UNREVIEWED',kind='UNKNOWN',roi=None,description='',evidence=''),addressability=dict(status='UNREVIEWED',evidence=''),query_support=dict(state='UNREVIEWED',evidence=''),visibility=dict(state='UNREVIEWED',evidence='',occlusion_evidence='',evidence_roi=None),ambiguity_note='',actually_viewed_image_bindings=[],training_admission=False)
 def actual(who,roi):
  r=copy.deepcopy(ex);r.update(reader_id=who,observation_complete=True,actually_viewed_image_bindings=[b]);r['instance']=dict(status='YES',evidence='invented instance');r['part']=dict(status='YES',description='invented connected end',evidence='invented topology');r['witness']=dict(status='VISIBLE',kind='END_CORNER',roi=roi,description='invented feature',evidence='invented only');r['addressability']=dict(status='QUERY_NATIVE_SUPPORTED',evidence='invented query relation');r['query_support']=dict(state='ON',evidence='invented');r['visibility']=dict(state='D',evidence='invented visible',occlusion_evidence='',evidence_roi=None);return r
 a=actual('A',[10,10,19,14]);c=actual('C',[15,12,24,19]);raw=h['join_two_readers'](a,c,ex);mask=exact_mask(a['witness']['roi'],c['witness']['roi']);f=add['invented_L_shape_fixture']
 need(sum(mask)==115 and sha(mask)==f['exact_union_bool96_rowmajor_uint8_sha256']and raw['witness_ROI_union']==[10,10,24,19]and sum(mask)!=150 and raw['training_admission']is False,'Exact OR115 not displaybbox150; rawcandidate never admission')
 need(not raw['candidate_may_enter_future_FIT_review'],'DEV excluded from futureFIT review')
 cc=copy.deepcopy(c);cc['query_support']['state']='OFF';d=field_differences(a,cc);need(d['query_support']['categorical_difference']and d['witness']['any_field_difference'],'Allaxis disagreement retained separately from ROI geometry')
 try:h['join_two_readers'](a,a,ex);ok=False
 except ValueError:ok=True
 need(ok,'Same reader rejected')
 wrong=copy.deepcopy(c);wrong['input']['query_xy']=[0.,0.]
 try:h['join_two_readers'](a,wrong,ex);ok=False
 except ValueError:ok=True
 need(ok,'Original q drift rejected')
 wrong=copy.deepcopy(c);wrong['actually_viewed_image_bindings']=[dict(path=str(H/'INVENTED_OTHER_CASE.png'),sha256='1'*64)]
 try:h['join_two_readers'](a,wrong,ex);ok=False
 except ValueError:ok=True
 need(ok,'Othercase actualview rejected')
 need(sum(exact_mask(None,None))==0 and sum(exact_mask([0,0,0,0],None))==1,'UNKNOWN none/one inclusive pixel preserved')
 valid=bytes([255])*9216;need(mask_all_valid(exact_mask(None,None),valid)is None and mask_all_valid(mask,valid)is True and mask_all_valid(exact_mask([0,0,0,0],None),bytes(9216))is False,'Optional validity preserves noROI UNKNOWN and invalid failure')
 ux=copy.deepcopy(ex);ux['input'].update(original_query_available=False,query_xy=None,L=None)
 uu=[]
 for rid in('A','C'):
  r=copy.deepcopy(ux);r.update(reader_id=rid,observation_complete=True,actually_viewed_image_bindings=[b])
  for axis in AXES:r[axis]['state'if axis in('query_support','visibility')else'status']='UNKNOWN'
  uu.append(r)
 ur=h['join_two_readers'](uu[0],uu[1],ux);need(ur['original_inputU']and ur['native_weak_visibility_agreement_candidate']=='UNKNOWN'and ur['witness_ROI_union']is None and ur['training_admission']is False,'Original U stays UNKNOWN without replacement ROI')
 return dict(completed=True,PASS=True,invented_only=True,count=9,checks=['L_shape_exact_OR115_SHA_not_bbox150','DEV_not_futureFIT','Allaxis_differences_retained','Same_reader_rejected','Original_q_drift_rejected','Othercase_view_rejected','UNKNOWN_and_inclusive_onepixel','Valid96_none_valid_invalid_noROIrepair','Original_null_U_UNKNOWN_no_ROI_or_admission'],actual_annotation_or_PNG_reads=0)

def invented_binding_shape_tests():
 payload=b'{"completed":true,"PASS":true,"invented_only":true}'
 base=dict(path=str(H/'INVENTED_UNOPENED_METADATA.json'),sha256=sha(payload));sized=dict(base,size_bytes=len(payload))
 need(same_binding(base,sized)and validate_bound_bytes(base,payload)==validate_bound_bytes(sized,payload),'Two/three-key binding identity and actual byte size agree')
 graph2={k:dict(base,path=str(H/('INVENTED_'+k+'.json')))for k in('observations','view_receipt','schema_QA','entry','contract','template','own_observations_binding')}
 graph3={k:dict(v,size_bytes=len(payload))for k,v in graph2.items()}
 need(all(same_binding(graph2[k],graph3[k])for k in graph2)and same_binding_map(graph2,graph3),'Actual-shaped graph/entry/contract/template/independence references accept optional size')
 wrong=dict(base,sha256='0'*64);need(not same_binding(base,wrong),'Wrong SHA is a different identity')
 def rejected(b):
  try:validate_bound_bytes(b,payload);return False
  except AssertionError:return True
 need(rejected(wrong),'Wrong SHA body rejected')
 wrongsize=dict(base,size_bytes=len(payload)+1);need(same_binding(base,wrongsize)and rejected(wrongsize),'Size separate from identity but checked on actual byte read')
 need(all(rejected(dict(base,size_bytes=x))for x in(-1,True,1.5,None)),'Optional size must be exact nonnegative integer')
 need(rejected(dict(base,undeclared='no'))and rejected(dict(sized,undeclared='no')),'Undeclared binding metadata keys rejected')
 return dict(completed=True,PASS=True,invented_only=True,count=6,checks=['Two_and_three_key_exact_identity_plus_byte_validation','Actual_shape_all_graph_references_same_path_SHA','Wrong_SHA_identity_and_payload_rejected','Wrong_size_payload_rejected_without_changing_identity','Negative_bool_float_null_size_rejected','Undeclared_extra_keys_rejected'],actual_label_metadata_or_PNG_body_reads=0)

def prepare():
 h=helper();expected,split,imap,images,add,contract=metadata(h);checks=invented_tests(h,add)
 binding_checks=invented_binding_shape_tests();checks['binding_shape_checks']=binding_checks['checks'];checks['count']+=binding_checks['count']
 write(H/'INVENTED_JOIN_CHECKS.json',checks)
 p=dict(completed=True,prepared_only=True,code=bind(__file__),inputs=INPUTS,checks=bind(H/'INVENTED_JOIN_CHECKS.json'),v1_preservation_and_minimal_repair=bind(H/'V1_PRESERVATION_AND_MINIMAL_BINDING_REPAIR.json'),output=str(OUT),binding_identity='Canonical(path,sha256), optional size_bytes integer>=0 checked against payload whenever read; no arbitrary extra keys. Original metadata retained.',reader_output_schema='Standard sealed observations.rows96, COMPLETE.observations/view_receipt/schema_QA/entry, view_receipt.views[{binding,position,actually_viewed}], separate independence read0 receipt.',mandatory_independence_fields=['other_reader_new_observation_files_read=0','other_reader_new_values_received=0','own_observations_binding exact','historical_V215_appearance_exposure_disclosed=true'],root_actual_scope_required=True,two_COMPLETE_validator_view_independence_seals_checked_before_any_observation_JSON_read=True,exact_OR_addendum=INPUTS['exact_OR_addendum'],legacy_bbox_display_only=True,root_semantic_review_always_pending=True,training_admission=False,DEV_excluded_from_every_future_loss_pool_normalization_and_derived_variant=True,only_future_PNGs='Declared228 generated V215PNG subset only; clean96+displayq8 render12, valid96 only when explicitly requested. No raw source or caches.',maximum_new_overlay_PNGs=12,root_report_illustrations_fixed_positions=[0,6,10],actual_new_observation_PNG_array_GT_model_score_reads=0,automatic_retry=False)
 write(PROTO,p)
 template=dict(approved_actual_two_reader_join_and_overlay=False,code=p['code'],protocol=bind(PROTO),inputs=INPUTS,root_actual_scope=None,root_verified_both_sealed_readers_before_join=False,readers={rid:dict(observations=None,complete=None,validator=None,view_receipt=None,independence=None,entry=None)for rid in('A','C')},observation_array_key='rows96',approved_generated_PNG_decode=False,png_input_whitelist=None,check_exact_union_against_valid96=False,output_directory=str(OUT),models_GT_scores_source_arrays_allowed=False,training_admission=False,automatic_retry=False)
 write(H/'ROOT_ACTUAL_JOIN_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
 write(H/'FINAL_READY.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTO),checks=p['checks'],v1_preservation_and_minimal_repair=p['v1_preservation_and_minimal_repair'],false_entry=bind(H/'ROOT_ACTUAL_JOIN_ENTRY_TEMPLATE_APPROVED_FALSE.json'),input_bindings=INPUTS,expected_rows=96,FIT64_DEV32_U11=True,actual_labels_or_PNG_bodies_read=0,frozen_no_further_edits=True))
 print(json.dumps(dict(completed=True,PASS=True,ready=bind(H/'FINAL_READY.json'),code=p['code'],protocol=bind(PROTO)),ensure_ascii=False))

def sealed_readers(e,declared_positions):
 need(set(e['readers'])=={'A','C'},'Exactly two reader seal records')
 seals={}
 for rid,b in e['readers'].items():
  need(set(b)=={'observations','complete','validator','view_receipt','independence','entry'},'All actual seal binding roles required')
  for x in b.values():canonical_binding_identity(x)  # Schema only; observations stay unopened until BOTH seals.
  for v in b.values():need(V in safe(v['path']).parents,'Own V216 sealed reader files only')
  c=read(b['complete']);qa=read(b['validator']);vr=read(b['view_receipt']);ind=read(b['independence'])
  need(all(x['completed']is True and x['PASS']is True for x in(c,qa,vr,ind)),'Complete/validator/view/independence actual PASS')
  need(c['reader_id']==vr['reader_id']==ind['reader_id']==rid and same_binding(c['observations'],b['observations'])and same_binding(c['view_receipt'],b['view_receipt'])and same_binding(c['schema_QA'],b['validator'])and all(same_binding(x,b['entry'])for x in(c['entry'],vr['entry'],ind['entry'])),'Sealed observations/views/validator/entry graph canonical identity exact')
  checked(b['entry'])
  need(c['training_admission']is False and ind['other_reader_new_observation_files_read']==0 and ind['other_reader_new_values_received']==0 and same_binding(ind['own_observations_binding'],b['observations'])and ind['historical_V215_appearance_exposure_disclosed']is True,'Independence before peer read, not blind, no training admission')
  views=set()
  for v in vr['views']:
   need(v['actually_viewed']is True and type(v['position'])is int and 0<=v['position']<12 and isinstance(v['tool'],str)and v['tool'],'Actual named image tool view')
   need(bkey(v['binding'])in declared_positions and declared_positions[bkey(v['binding'])]==v['position'],'View receipt only exact declared same-frame images')
   views.add((v['position'],bkey(v['binding'])))
  seals[rid]=dict(bindings=b,viewed=views)
 return seals

def png_bytes(b,allowed,declared):
 matches=[x for x in allowed if same_binding(b,x)]
 need(len(matches)==1 and bkey(b)in declared and V215 in safe(b['path']).parents and Path(b['path']).suffix=='.png','Exact root-approved generated PNG before stat/read/decode')
 data=checked_bytes(b);validate_bound_bytes(matches[0],data);return data

def png_image(b,allowed,declared):
 from PIL import Image
 with Image.open(io.BytesIO(png_bytes(b,allowed,declared)))as im:im.load();return im.copy()

def render_cards(joined,side,imap,allowed,declared):
 from PIL import Image,ImageDraw,ImageFont
 def font(n):
  with opened(FONT)as f:return ImageFont.truetype(f,n)
 cards=[]
 for pos in range(12):
  rr=[r for r in joined if r['input']['position']==pos];need(len(rr)==8 and[r['input']['native_slot']for r in rr]==list(range(8)),'Fixed12/all8 overlay order')
  full=png_image(imap['display_original_q8',pos,None],allowed,declared).convert('RGB')
  canvas=Image.new('RGB',(2304,1360),'white');d=ImageDraw.Draw(canvas);d.text((16,10),'V216 weak SOURCE candidates | '+rr[0]['input']['frame_key'],font=font(27),fill='black');d.text((16,52),'A red / C blue. q47.5 cyan DISPLAY ONLY. Exact OR is separate from display bbox. No training admission.',font=font(21),fill='black')
  scale=min(2304/full.width,680/full.height);full=full.resize((int(full.width*scale),int(full.height*scale)),Image.Resampling.LANCZOS);canvas.paste(full,((2304-full.width)//2,95+(680-full.height)//2))
  for k,r in enumerate(rr):
   x=288*k;m=r['input'];a=r['reader_A_axes'];c=r['reader_C_axes'];y=820
   d.text((x+7,y),'native%d %s'%(k,'U'if not m['original_query_available']else'available'),font=font(21),fill='black');d.text((x+7,y+29),m['internal_source_role'],font=font(16),fill='black')
   d.text((x+7,y+56),'A '+a['visibility']['state']+' / C '+c['visibility']['state'],font=font(19),fill='black');d.text((x+7,y+82),'q A:'+a['query_support']['state']+' C:'+c['query_support']['state'],font=font(17),fill='black')
   clean=png_image(imap['clean96',pos,k],allowed,declared);need(clean.mode=='RGB'and clean.size==(96,96),'Original generated clean96 RGB')
   draw=ImageDraw.Draw(clean)
   for roi,color in((a['witness']['roi'],'red'),(c['witness']['roi'],'blue')):
    if roi is not None:draw.rectangle(roi,outline=color,width=1)
   draw.line((41,47.5,54,47.5),fill='cyan',width=1);draw.line((47.5,41,47.5,54),fill='cyan',width=1)
   canvas.paste(clean.resize((272,272),Image.Resampling.NEAREST),(x+7,y+118))
   d.text((x+7,y+404),'join candidate '+r['raw_helper_candidate']['native_weak_visibility_agreement_candidate'],font=font(18),fill='black')
  d.text((16,1305),'ROI is weak visible-feature evidence, never corrected native keypoint/physical gold. DEV excluded from all future fit branches.',font=font(21),fill='black')
  path=OUT/('FIXED_FRAME_%02d_TWO_READER_OVERLAY.png'%pos)
  with opened(path,True)as f:canvas.save(f,format='PNG')
  cards.append(dict(position=pos,frame_key=rr[0]['input']['frame_key'],binding=bind(path)))
 return cards

def relative(path,base=OUT):return os.path.relpath(safe(path),base).replace(os.sep,'/')

def reports(summary,joined,side,cards,e):
 rootresults=dict(completed=True,PASS=True,source_only=True,counts=summary,original96_inputU11=True,raw_candidates_not_semantic_gold=True,root_semantic_samepart_and_addressability_review='PENDING',training_admission=False,DEV_all_future_loss_pool_normalization_exclusion=True,sameframe_two_sealed_reader_weak_observations_not_human_gold=True,own_overlay_PNG_count=12,exact_OR_validity_status_counts=dict(Counter(s['valid96_check_status']for s in side)),exact_OR_invalid_proof_failures=sum(bool(s['pixel_proof_failures'])for s in side),legacy_bbox_DISPLAY_ONLY_exact_OR_mask_separate=True,fully_solved=False,reader_observation_bindings={k:v['observations']for k,v in e['readers'].items()})
 write(OUT/'ROOT_RESULTS_KO.json',rootresults)
 lead='고정 SOURCE TRAIN12프레임/native96의 두 독립 봉인 약관찰을 원ID/qL/U11 그대로 조인했다. 아래 D/E/SELF와 witness 수는 원관찰/기계적 합의후보이며 모델 성능이나 최종 semantic admission이 아니다. 같은 부위·query addressability의 root 검토는 pending이고 학습 admission은 false다.'
 md=['# V216 SOURCE 부위·query-support 두 관찰 조인','',lead,'','원role TRAIN과 internal SOURCE_FIT64/SOURCE_DEV32를 분리한다. DEV는 미래 모든 loss/pool/normalization 및 sibling/pairedvariant 학습에서 제외한다. 이전 appearance·oldaux 노출이 있어 freshblind/OOD 또는 물리적 독립 검증이 아니다.','', '원 U11과 의미 UNKNOWN은 다른 축이다. 배경·다른 목재/물체·qOFF를 native E/D로 복사하지 않는다. ROI는 교정 keypoint/physical gold가 아니다. 두 원 ROI와 sealed wholeJSON을 보존하고 exactOR를 rowmajor0/1 mask SHA/area로 기록한다. legacy bbox는 표시용일 뿐 alpha/proof mask가 아니다.','', '|cohort|N|원U|local witness 합의후보|join visibility 후보|','|---|---:|---:|---:|---|']
 for role,s in summary.items():md.append('|'+role+'|'+str(s['N'])+'|'+str(s['original_inputU'])+'|'+str(s['local_witness_candidates'])+'|'+json.dumps(s['join']['visibility_candidate'],ensure_ascii=False)+'|')
 countrows=[]
 md+=['','|cohort|reader / raw후보|D / E / SELF / UNKNOWN|ON / NEAR / OFF / UNKNOWN|','|---|---|---|---|']
 for role,s in summary.items():
  for who in('A','C','JOIN_RAW_CANDIDATE'):
   vv=s['join']['visibility_candidate']if who=='JOIN_RAW_CANDIDATE'else s['readers'][who]['visibility'];qq=s['join']['query_support_agreement']if who=='JOIN_RAW_CANDIDATE'else s['readers'][who]['query_support']
   line=[role,who,' / '.join(str(vv.get(k,0))for k in('D','E','SELF','UNKNOWN')),' / '.join(str(qq.get(k,0))for k in('ON','NEAR','OFF','UNKNOWN'))];countrows.append(line);md.append('|'+'|'.join(line)+'|')
 md+=['','|native|FIT N / 원U|DEV N / 원U|ALL N / 원U|','|---:|---|---|---|']
 for k in range(8):md.append('|'+str(k)+'|'+'|'.join(str(summary[role]['native8'][str(k)]['N'])+' / '+str(summary[role]['native8'][str(k)]['original_inputU'])for role in('SOURCE_FIT','SOURCE_DEV','ALL'))+'|')
 md+=['','[전체 조인96](ALL96_JOINED_CANDIDATES.json) · [두ROI exactOR sidecar](ALL96_EXACT_WITNESS_MASK_SIDECAR.json) · [reader/role/native/축 count CSV](SOURCE_COUNTS_BY_ROLE_NATIVE_READER.csv) · [모든 축 차이 CSV](ALL96_AXIS_DIFFERENCES.csv) · [전체12 overlay gallery](PORTABLE_GALLERY_KO.md)','', '원 관찰 각축/증거와 차이는 삭제하지 않았다. ROI overlap/같은 category는 같은 semantic feature 증명이 아니며, helper proposal을 승인된 학습 target으로 승격하지 않는다. valid96를 확인하지 않았거나 exactOR가 invalid를 건드리면 proof 미완료/실패를 그대로 기록하고 ROI를 줄이거나 이동하지 않는다.','', '## 고정 설명 위치0/6/10','', '다음3장은 사전 지정 설명 카드이며 대표성·prevalence나 점수 기반 실패선별이 아니다.','']
 for i in(0,6,10):md+=['![고정 위치'+str(i)+']('+relative(cards[i]['binding']['path'])+')','']
 for rid,b in e['readers'].items():md+=['['+rid+' sealed96 observations]('+relative(b['observations']['path'])+') · ['+rid+' 실제view receipt]('+relative(b['view_receipt']['path'])+')','']
 md+=['[ROOT JSON](ROOT_RESULTS_KO.json) · [HTML](ROOT_READOUT_KO.html) · [SUMMARY](SUMMARY.json) · [조인 코드]('+relative(__file__)+') · [조인 protocol]('+relative(PROTO)+')','', '실제 entry와 양쪽 wholeJSON SHA는 COMPLETE/sidecar에 봉인했다.','']
 doc='\n'.join(md);text(OUT/'ROOT_REPORT_KO.md',doc);text(OUT/'PORTABLE_REPORT_KO.md',doc)
 gallery=['# V216 고정12 두-reader ROI 표시','','A red/C blue와 originalq cyan은 표시만 바꾼 사본이다. 원V215 PNG는 변경하거나 복제하지 않았다. 의미/admission 검토는 pending이다.','']
 for c in cards:gallery+=['## '+str(c['position'])+' '+c['frame_key'],'','!['+c['frame_key']+']('+relative(c['binding']['path'])+')','']
 text(OUT/'PORTABLE_GALLERY_KO.md','\n'.join(gallery))
 body='<h1>V216 SOURCE 약관찰 조인</h1><p>'+html.escape(lead)+'</p><p>원U11 / SOURCE_FIT64 / SOURCE_DEV32. Root semantic review pending; no training admission.</p><p><a href="PORTABLE_REPORT_KO.md">보고서</a> · <a href="ROOT_RESULTS_KO.json">JSON</a> · <a href="SOURCE_COUNTS_BY_ROLE_NATIVE_READER.csv">축별 count CSV</a> · <a href="PORTABLE_GALLERY_KO.md">전체12 gallery</a></p>'
 body+='<h2>약관찰/기계적 합의후보 수 — 모델 성능 아님</h2><table border="1"><tr><th>cohort</th><th>reader/raw후보</th><th>D/E/SELF/UNKNOWN</th><th>ON/NEAR/OFF/UNKNOWN</th></tr>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>'for x in row)+'</tr>'for row in countrows)+'</table>'
 for c in cards:body+='<h2>'+str(c['position'])+' '+html.escape(c['frame_key'])+'</h2><img style="max-width:100%;height:auto" src="'+relative(c['binding']['path'])+'">'
 page='<!doctype html><meta charset="utf-8"><style>body{font:18px/1.6 sans-serif;max-width:1800px;margin:auto;padding:20px}img{width:100%;height:auto}</style>'+body
 text(OUT/'ROOT_READOUT_KO.html',page)

class PageLinks(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in('href','src')and v)

def link_QA():
 result={}
 for name in('ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','PORTABLE_GALLERY_KO.md','ROOT_READOUT_KO.html'):
  doc=OUT/name;value=rb(doc).decode()
  if doc.suffix=='.html':p=PageLinks();p.feed(value);links=p.links
  else:links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',value)
  for url in links:
   need(not url.startswith(('/','file:','http:','https:'))and'://'not in url,'Relative portable link only')
   target=safe(Path(os.path.normpath(doc.parent/url)));need((V in target.parents or V215 in target.parents)and target.suffix not in('.npz','.npy','.pt','.pth','.bin'),'Only generated/owned phase links, never source binary')
   with opened(target)as f:need(stat.S_ISREG(os.fstat(f.fileno()).st_mode),'Portable target regular file exists')
  result[name]=len(links)
 return result

def run(entry_path):
 e=json.loads(rb(entry_path))
 need(e['approved_actual_two_reader_join_and_overlay']is True and e['root_verified_both_sealed_readers_before_join']is True and e['models_GT_scores_source_arrays_allowed']is False and e['training_admission']is False and e['automatic_retry']is False,'Root sealed actual join authority before new labels/PNGs')
 p=read(e['protocol']);need(same_binding(e['protocol'],bind(PROTO))and same_binding(e['code'],p['code'])and same_binding(p['code'],bind(__file__))and same_binding_map(e['inputs'],p['inputs'])and same_binding_map(p['inputs'],INPUTS),'Exact frozen code/protocol/input canonical closure')
 actualscope=read(e['root_actual_scope']);need(actualscope['approved_actual_two_reader_join_and_overlay']is True,'Separate actual root scope')
 need(e['output_directory']==str(OUT)and e['observation_array_key']=='rows96'and e['approved_generated_PNG_decode']is True,'Fresh output / rows96 / explicit generated PNG authority')
 h=helper();expected,split,imap,images,add,contract=metadata(h);declared={bkey(i['binding'])for i in images};allowed=e['png_input_whitelist']
 need(isinstance(allowed,list)and len({bkey(b)for b in allowed})==len(allowed)and all(bkey(b)in declared for b in allowed),'Root exact declared-derived PNG subset only')
 mkdir_new(OUT);write(OUT/'STARTED.json',dict(completed=False,entry=bind(entry_path),automatic_retry=False))
 try:
  seals=sealed_readers(e,{bkey(i['binding']):i['position']for i in images})  # BOTH seals before observations.
  observations={}
  for rid in('A','C'):
   b=seals[rid]['bindings'];doc=read(b['observations'])
   need(doc['completed']and doc['PASS']and doc['reader_id']==rid and same_binding(doc['entry'],b['entry'])and same_binding(doc['contract'],INPUTS['contract'])and same_binding(doc['template'],INPUTS['template'])and doc['training_admission']is False,'Actual reader sealed standard output')
   rr=doc['rows96'];need([r['input']['row_id']for r in rr]==[r['input']['row_id']for r in expected],'All96 original row order')
   for r,x in zip(rr,expected):
    h['validate_observation'](r,x)
    need(all((r['input']['position'],bkey(v))in seals[rid]['viewed']for v in r['actually_viewed_image_bindings']),'Perrow actualview binds own exact case receipt')
   observations[rid]=rr
  labels={rid:e['readers'][rid]['observations']for rid in('A','C')};joined,side=join_rows(h,observations['A'],observations['C'],expected,labels)
  for s,original in zip(side,split):
   s['original_valid96']=original['valid96']
   if e['check_exact_union_against_valid96']:
    mask=exact_mask(s['original_A_witness_ROI'],s['original_C_witness_ROI'])
    if not any(mask):s['valid96_check_status']='NO_WITNESS_ROI_TO_CHECK';continue
    im=png_image(original['valid96'],allowed,declared);need(im.mode=='L'and im.size==(96,96),'Original valid96 PNG shape/mode')
    data=im.tobytes();need(set(data)<={0,255},'Exact original binary valid PNG')
    valid=mask_all_valid(mask,data)
    s['candidate_union_all_valid96']=valid;s['valid96_check_status']='CHECKED_EXACT_OR_NO_ROI_REPAIR'
    if not valid:s['pixel_proof_failures'].append('EXACT_OR_TOUCHES_ORIGINAL_INVALID96')
  summary,counts=aggregate(joined);write(OUT/'ALL96_JOINED_CANDIDATES.json',dict(completed=True,rows96=joined,whole_observation_bindings=labels,root_semantic_review_pending=True,training_admission=False));write(OUT/'ALL96_EXACT_WITNESS_MASK_SIDECAR.json',dict(completed=True,rows96=side,exact_OR_addendum=INPUTS['exact_OR_addendum'],legacy_bbox_display_only=True,training_admission=False));write(OUT/'SUMMARY.json',dict(completed=True,counts=summary,not_model_performance=True,training_admission=False))
  for name,rows in [('SOURCE_COUNTS_BY_ROLE_NATIVE_READER.csv',counts),('ALL96_AXIS_DIFFERENCES.csv',[dict(row_id=r['input']['row_id'],internal_source_role=r['input']['internal_source_role'],native_slot=r['input']['native_slot'],**{k:json.dumps(v,ensure_ascii=False)for k,v in r['all_axis_differences'].items()})for r in joined])]:
   out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);text(OUT/name,out.getvalue())
  cards=render_cards(joined,side,imap,allowed,declared);e['_entry_binding']=bind(entry_path);reports(summary,joined,side,cards,e)
  links=link_QA();need(rb(OUT/'ROOT_REPORT_KO.md')==rb(OUT/'PORTABLE_REPORT_KO.md'),'Portable/ROOT MD byte identity')
  artifacts={name:bind(OUT/name)for name in('ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html','ALL96_JOINED_CANDIDATES.json','ALL96_EXACT_WITNESS_MASK_SIDECAR.json','SUMMARY.json','SOURCE_COUNTS_BY_ROLE_NATIVE_READER.csv','ALL96_AXIS_DIFFERENCES.csv','PORTABLE_GALLERY_KO.md')}
  write(OUT/'SAVED_JOIN_QA.json',dict(completed=True,PASS=True,all96_original_qL_U_roles_exact=True,both_seals_checked_before_rows=True,exact_OR_addendum_bound=True,all_raw_candidates_disagreements_and_ROIs_preserved=True,portable_link_counts=links,all_links_resolve=True,original_PNG_mutations=0,training_admission=False))
  write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=e['protocol'],entry=e['_entry_binding'],inputs=INPUTS,sealed_observations=labels,artifacts=artifacts,QA=bind(OUT/'SAVED_JOIN_QA.json'),all96_original_input_qL_native_U_roles_preserved=True,FIT64_DEV32_inputU11=True,all_axis_disagreements_and_raw_helper_candidates_preserved=True,both_original_ROIs_and_exact_OR_mask_sha_area_preserved=True,legacy_bbox_alpha_use=False,new_overlay_PNGs=cards,maximum12_new_PNGs=True,root_semantic_review='PENDING',training_admission=False,DEV_future_allbranch_fit_exclusion=True,model_GT_sourcearray_score_reads=0,source_PNG_mutations=0,HTML_browser_rendered=False,automatic_retry=False))
  print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json')),ensure_ascii=False))
 except BaseException:
  write(OUT/'FAILURE_PRESERVED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry');a=ap.parse_args();prepare()if a.prepare else run(a.entry)

"""V197 TRAIN-only target-contract viewer. Preparation never opens source metadata/arrays.
Runtime freezes metadata/hash selection before any target NPZ/header/pixel body.
No score reader, model/SAM/initializer/recrop/PnP path exists.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,html,io,json,math,os,stat,traceback
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_PROSPECTIVE_TRAIN_ONLY_TARGET_CONTRACT_DIAGNOSIS_SCOPE.json'
PROTO=H/'PROTOCOL_BEFORE_TRAIN_METADATA_SELECTION_OR_PIXEL_BODIES.json'
OUT=V/'fixed_train_target_contract_viewer_v1'
VARS=('BEFORE','COVER','SHAM');META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
SALT='V197_TRAIN_CONTRACT_FIXED12_V1|'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
OLD_VIEWER=P/'query_visibility_grouped_source_holdout_target_readout_20261024_v196/viewer_preparation_b/build_all_source_holdout_viewer.py'
SOURCE_BUILDER=P/'query_visibility_source_group_split_direct_end_training_20261024_v194/direct_cache_preparation_a/build_expanded_direct_control_cache.py'
def need(x,m):
 if not bool(x):raise AssertionError(m)
def safe(value):
 p=Path(value);need(p.is_absolute()and '..'not in p.parts and all('v112'not in q.casefold()for q in p.parts),'Lexical safe path before any access')
 need(p.is_relative_to(P)and not any(q in str(p).casefold()for q in('current223','_v169/','_v177/','_v178/','_v184/','_v191/','_v195/','_v182/','synthetic_val')),'Exact source-only paths')
 return p
def opened(path):
 p=safe(path);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for q in p.parts[1:-1]:
   nxt=os.open(q,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
  last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(last).st_mode),'Regular nofollow source file');return os.fdopen(last,'rb')
def sha(path):
 h=hashlib.sha256()
 with opened(path)as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(path):return dict(path=str(safe(path)),sha256=sha(path))
def check(b):need(bind(b['path'])=={k:b[k]for k in('path','sha256')},'Exact saved binding');return safe(b['path'])
def read(path):
 with opened(path)as f:return json.load(f)
def jd(b):return read(check(b))
def lines(b):
 with opened(check(b))as f:return [json.loads(q)for q in f if q.strip()]
def write(path,d):
 safe(path)
 with Path(path).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def text(path,s):
 safe(path)
 with Path(path).open('x')as f:f.write(s)
def arrays(b,keys,allowed):
 need(allowed.get(b['path'])==b,'Exact root-declared or selected-TRAIN-derived packet before header/hash')
 with opened(b['path'])as f:
  h=hashlib.sha256()
  for q in iter(lambda:f.read(1<<20),b''):h.update(q)
  need(h.hexdigest()==b['sha256'],'Saved source NPZ immutable');f.seek(0)
  with np.load(f,allow_pickle=False)as z:return {k:z[k].copy()for k in keys}
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def eq(a,b):
 a,b=np.asarray(a),np.asarray(b);return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def qarray(q):return np.asarray([np.nan,np.nan]if q is None else q,np.float64)
def finite(x):return float(x)if math.isfinite(float(x))else None


def metadata_population(rows,plan,expected_known_triplets=109):
 need(len(rows)==len(plan)and len({tuple(r['control_join_key'])for r in rows})==len(rows),'Unique full source occurrence metadata')
 lookup={tuple(r['control_join_key']):r for r in plan};need(len(lookup)==len(plan),'Unique split keys')
 joined=[];groups=defaultdict(list)
 for i,r in enumerate(rows):
  k=tuple(r['control_join_key']);need(k in lookup and r['cache_index']==i,'Exact full cache index and split key')
  s=lookup[k];need(s['role']in ('TRAIN','HOLDOUT'),'Unchanged component role')
  for field in META:
   if field=='query_xy':ok=np.array_equal(qarray(r[field]),qarray(s[field]),equal_nan=True)
   elif field=='L':ok=np.array_equal(np.asarray(np.nan if r[field]is None else r[field]),np.asarray(np.nan if s[field]is None else s[field]),equal_nan=True)
   else:ok=r[field]==s[field]
   need(ok,'Exact source split/cache META '+field)
  need(r['source_packet']==s['source_packet'] and r['authority']==s['authority'],'Exact source packet/target authority')
  t=int(s['stored_weak_direct_target']);need(t in(-1,0,1),'Existing weak target enum')
  a=dict(r,role=s['role'],component_id=s['component_id'],direct_weak_target=t,selected_native_slot=s['selected_native_slot'],layout=s['layout'],variant_index=s['variant_index'],training_admission=False)
  joined.append(a)
  if s['role']=='TRAIN':groups[(r['domain'],r['attempt_id'])].append(a)
 known=[]
 for (domain,attempt),rr in groups.items():
  positive=[r for r in rr if r['direct_weak_target']>=0]
  if not positive:continue
  need(len(rr)==24 and {(r['variant'],r['native_slot'])for r in rr}=={(v,n)for v in VARS for n in range(8)},'Known TRAIN triplet all3/all8 preserved')
  slots={r['native_slot']for r in positive};need(len(positive)==3 and len(slots)==1,'Only selected native weak target known')
  slot=next(iter(slots));by={r['variant']:r for r in positive}
  need({v:by[v]['direct_weak_target']for v in VARS}==dict(BEFORE=0,COVER=1,SHAM=0),'Frozen weak D/E/D, no native label alias')
  need(all(r['selected_native_slot']==slot for r in rr)and all(r['root_admitted']is True for r in positive),'Stored root direct admission only')
  need(len({r['component_id']for r in rr})==1 and all(r['role']=='TRAIN'for r in rr),'No split leakage within direct triplet')
  rank=hashlib.sha256((SALT+attempt).encode('utf-8')).hexdigest()
  known.append(dict(domain=domain,attempt_id=attempt,selected_native_slot=slot,rank_sha256=rank,selected_query_available=all(by[v]['original_query_available']is True for v in VARS),rows=rr))
 known.sort(key=lambda t:(t['rank_sha256'],t['attempt_id'],t['domain'],t['selected_native_slot']))
 need(len(known)==expected_known_triplets,'All109 TRAIN known triplets before selection')
 need(sum(r['direct_weak_target']>=0 for r in joined if r['role']=='TRAIN')==3*expected_known_triplets,'All327TRAIN labels, not pixels/scores')
 return joined,known

def select_fixed(known,first=12):
 ordered=sorted(known,key=lambda t:(t['rank_sha256'],t['attempt_id'],t['domain'],t['selected_native_slot']))
 chosen={(t['domain'],t['attempt_id'],t['selected_native_slot'])for t in ordered[:first]};by_native={}
 for n in range(8):
  match=next((t for t in ordered if t['selected_native_slot']==n and t['selected_query_available']),None)
  by_native[str(n)]=None if match is None else dict(domain=match['domain'],attempt_id=match['attempt_id'],selected_native_slot=n,rank_sha256=match['rank_sha256'])
  if match:chosen.add((match['domain'],match['attempt_id'],n))
 selected=[t for t in ordered if(t['domain'],t['attempt_id'],t['selected_native_slot'])in chosen]
 need(len(selected)<=first+8 and len({(t['domain'],t['attempt_id'],t['selected_native_slot'])for t in selected})==len(selected),'Union<=20, no replacement/duplicate')
 return selected,by_native

def source_association(t,source):
 if t['domain']=='SYNTH_FINITE_END':
  ep=source.get('geometry_plan',{}).get('endpoint_observation',{});proof=source.get('closed_proof',{});canonical=ep.get('evidence_binding',{}).get('canonical')
 else:
  ep=source.get('endpoint_observation',{});proof=source.get('closed_end_contract',{});canonical=None
 rr=t['rows'];base=rr[0];slot=t['selected_native_slot'];q=qarray(next(r['query_xy']for r in rr if r['variant']=='BEFORE'and r['native_slot']==slot));L=next(r['L']for r in rr if r['variant']=='BEFORE'and r['native_slot']==slot)
 boxes=ep.get('candidate_enclosures_xyxy');projected=None
 if boxes is not None and np.isfinite(q).all()and L is not None and math.isfinite(L)and L>0:projected=[(96*(np.asarray(bb,np.float64).reshape(2,2)-q)/L+47.5).reshape(-1).tolist()for bb in boxes]
 return dict(domain=t['domain'],attempt_id=t['attempt_id'],source_base_index=base.get('source_base_index'),source_frame_index=base.get('source_frame_index'),source_case_id=base.get('source_case_id'),source_donor_id=base.get('source_donor_id'),selected_native_slot=slot,canonical_index_metadata=canonical,canonical_authority='existing constructive evidence_binding.canonical'if canonical is not None else'UNKNOWN_NOT_PROVIDED_NO_PERMUTATION_FALLBACK',weak_authority=base['authority'],selected_instance_supported=ep.get('selected_instance_supported'),intended_part_supported=ep.get('intended_part_supported'),queried_end_localization_supported=ep.get('queried_end_localization_supported'),end_locality_id=ep.get('end_locality_id','UNKNOWN'),candidate_enclosures_fullRGB=boxes,candidate_enclosures_original96_diagnostic=projected,surrounding_material_witnesses=ep.get('surrounding_material_witnesses',[]),evidence_binding=ep.get('evidence_binding'),original_selected_query_xy=q.tolist(),original_selected_L=L,proof_status=proof.get('status','UNKNOWN'),source_proof_binding=base['source_proof_binding'],association_only_not_nativephysicalgold=True,all_other_native_targets_UNKNOWN=True,training_admission=False)

def csvfile(path,rows):
 s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);text(path,s.getvalue())
def png(path,im):
 safe(path)
 with Path(path).open('xb')as f:im.save(f,format='PNG')

def mark96(rgb,mask,boxes):
 a=rgb.copy()
 if mask is not None:a[mask]=(a[mask].astype(np.uint16)+np.asarray([255,0,255],np.uint16))//2
 im=Image.fromarray(a).resize((384,384),Image.Resampling.NEAREST);d=ImageDraw.Draw(im)
 for x0,y0,x1,y1 in boxes or []:d.rectangle([(x0+.5)*4-.5,(y0+.5)*4-.5,(x1+.5)*4-.5,(y1+.5)*4-.5],outline=(255,220,0),width=2)
 d.line([(182,191.5),(201,191.5)],fill=(0,255,255));d.line([(191.5,182),(191.5,201)],fill=(0,255,255));return im

def contact(t,rgb,valid,mask,assoc):
 im=Image.new('RGB',(1488,1040),'white');d=ImageDraw.Draw(im);font=lambda n:ImageFont.truetype(FONT,n)
 def label(x,y,s,n=17):d.text((x,y),s,font=font(n),fill='black')
 label(18,10,'V197 TRAIN CONTRACT | '+t['attempt_id'],23)
 label(18,42,'native %d -> source canonical %s | %s'%(t['selected_native_slot'],str(assoc['canonical_index_metadata']),t['domain']),17)
 label(18,68,'Original q='+str(assoc['original_selected_query_xy'])+' L='+str(assoc['original_selected_L']),16)
 label(18,92,'No predictions. Colors/display only; weak association is NOT physical/native gold.',17)
 for vi,var in enumerate(VARS):
  x=18+vi*486;label(x,122,var+' | stored weak '+('E'if var=='COVER'else'D'),21)
  im.paste(Image.fromarray(rgb[vi]).resize((384,384),Image.Resampling.NEAREST),(x,158));im.paste(Image.fromarray(valid[vi].astype(np.uint8)*255).convert('RGB'),(x+390,158));label(x+390,263,'valid96',13)
  label(x,550,'Clean96 x4; unchanged PNG saved separately',16);im.paste(mark96(rgb[vi],mask,assoc['candidate_enclosures_original96_diagnostic']),(x,585))
  label(x,979,'Magenta: saved end tapmask'if mask is not None else'Saved end tapmask MISSING; no fallback',15)
 label(18,1008,'Yellow: saved enclosure. Cyan: original q at(47.5,47.5). All other native targets remain UNKNOWN.',16)
 return im

def all8_contact(rr,raw,valid):
 im=Image.new('RGB',(1230,8*155+70),'white');d=ImageDraw.Draw(im);f=ImageFont.truetype(FONT,15)
 d.text((12,10),'Original all8 queries | no scores / no relabel / no recrop',font=f,fill='black')
 for vi,var in enumerate(VARS):
  x=vi*410+12;d.text((x,36),var,font=f,fill='black')
  for k in range(8):
   r=next(r for r in rr if r['variant']==var and r['native_slot']==k);y=70+k*155
   im.paste(Image.fromarray(raw[vi,k]),(x,y));im.paste(Image.fromarray(valid[vi,k].astype(np.uint8)*255).convert('RGB'),(x+102,y))
   d.text((x+207,y),'native %d target%d'%(k,r['direct_weak_target']),font=f,fill='black');d.text((x+207,y+22),'avail '+str(r['original_query_available']),font=f,fill='black')
   d.text((x,y+100),'q='+str(r['query_xy']),font=f,fill='black');d.text((x,y+122),'L='+str(r['L']),font=f,fill='black')
 return im

def validate_packet(a,rr,var,layout):
 vi=VARS.index(var);rgb=a['rgb96'][vi]if layout=='SYNTH_TRIPLET'else a['rgb96'];valid=a['valid96'][vi]if layout=='SYNTH_TRIPLET'else a['valid96']
 need(rgb.dtype==np.uint8 and rgb.shape==(8,96,96,3)and valid.dtype==bool and valid.shape==(8,96,96),'Exact original raw/valid all8')
 need(a['query_xy'].dtype==np.float64 and a['L'].dtype==np.float64 and a['native_slot'].tolist()==list(range(8)),'Original F64/native8')
 avail=a['original_query_available']if layout=='SYNTH_TRIPLET'else a['query_available'];ids=a['original_occurrence_id']if layout=='SYNTH_TRIPLET'else a['query_id'];targets=a['direct_end_external_target'][vi]if layout=='SYNTH_TRIPLET'else a.get('direct_end_external_target')
 for k in range(8):
  r=next(r for r in rr if r['variant']==var and r['native_slot']==k)
  need(np.array_equal(a['query_xy'][k],qarray(r['query_xy']),equal_nan=True)and np.array_equal(np.asarray(a['L'][k]),np.asarray(np.nan if r['L']is None else r['L']),equal_nan=True)and int(a['original_global_index'][k])==r['original_global_index']and bool(avail[k])==r['original_query_available'],'Packet original q/L/availability/ID')
  need(str(ids[k])==r['original_occurrence_id']and ash(rgb[k])==r['raw_RGB96_sha256']and ash(valid[k])==r['raw_valid96_sha256'],'Original occurrence/raw/valid bytehash')
  if targets is not None:need(int(targets[k])==r['direct_weak_target'],'Stored direct target remains unchanged')
 full=a.get('full_RGB');full=(full[vi]if layout=='SYNTH_TRIPLET'else full)if full is not None else None
 if full is not None:need(full.dtype==np.uint8 and full.ndim==3 and full.shape[-1]==3,'Exact declared full_RGB member only')
 return rgb,valid,full

def read_packet(b,layout,allowed):
 need(allowed.get(b['path'])==b,'Only selected TRAIN exact packet, before hash/header')
 keys=['rgb96','valid96','query_xy','L','native_slot','original_global_index']+(['original_occurrence_id','original_query_available','direct_end_external_target']if layout=='SYNTH_TRIPLET'else['query_id','query_available'])
 with opened(check(b))as f:
  with np.load(f,allow_pickle=False)as z:
   out={k:z[k].copy()for k in keys}
   for k in('full_RGB','end_related96_mask'):
    if k in z.files:out[k]=z[k].copy()
 return out

def full_display(rgb,rows,assoc):
 im=Image.fromarray(rgb.copy());d=ImageDraw.Draw(im);font=ImageFont.truetype(FONT,16)
 for r in rows:
  q=qarray(r['query_xy'])
  if np.isfinite(q).all():
   x,y=q;d.ellipse((x-3,y-3,x+3,y+3),outline=(0,255,255),width=2);d.text((x+5,y+2),str(r['native_slot']),font=font,fill=(0,255,255))
 for box in assoc['candidate_enclosures_fullRGB']or[]:d.rectangle(box,outline=(255,220,0),width=2)
 return im

def invented():
 rows=[];plan=[]
 for gi in range(18):
  slot=gi%6
  for vi,var in enumerate(VARS):
   for k in range(8):
    i=len(rows);r=dict(cache_index=i,domain='SYNTH_FINITE_END',attempt_id='TOY%03d'%gi,variant=var,native_slot=k,row_id=str(i),original_global_index=k,query_xy=[float(k),0.],L=96.,original_query_available=not(gi==0 and k==slot),control_join_key=['SYNTH_FINITE_END',str(gi),var,k],source_packet=dict(path='/invented/TRAIN.npz',sha256='a'*64),authority='CONSTRUCTIVE_WEAK',root_admitted=True)
    rows.append(r);plan.append(dict(r,role='TRAIN',stored_weak_direct_target=(1 if var=='COVER'else 0)if k==slot else-1,selected_native_slot=slot,component_id=str(gi),layout='SYNTH_TRIPLET',variant_index=vi))
 _,known=metadata_population(rows,plan,18);chosen,per=select_fixed(known);tests={'all18TRAIN_metadata_first':len(known)==18,'fixedhash12_plus_native_union_max20':12<=len(chosen)<=20 and len(chosen)<=18,'missingnative6_7_remainnull':per['6']is None and per['7']is None,'U_metadata_not_promoted':next(t for t in known if t['attempt_id']=='TOY000')['selected_query_available']is False,'reversed_inventory_order_same_selection':[(t['attempt_id'],t['selected_native_slot'])for t in select_fixed(list(reversed(known)))[0]]==[(t['attempt_id'],t['selected_native_slot'])for t in chosen]}
 for name,change in [('duplicate_split_key',lambda p:p[1].update(control_join_key=p[0]['control_join_key'])),('mixed_component_role',lambda p:p[0].update(role='HOLDOUT')),('wrong_q',lambda p:p[0].update(query_xy=[99.,1.])),('wrong_target',lambda p:p[2].update(stored_weak_direct_target=1))]:
  import copy
  bad=copy.deepcopy(plan);change(bad)
  try:metadata_population(rows,bad,18);ok=False
  except(AssertionError,KeyError):ok=True
  tests[name+'_rejected']=ok
 # Independent hash ordering of metadata only.
 expected=sorted(known,key=lambda t:(hashlib.sha256(('V197_TRAIN_CONTRACT_FIXED12_V1|'+t['attempt_id']).encode()).hexdigest(),t['attempt_id'],t['domain'],t['selected_native_slot']))[:12]
 tests['literal_scope_salt_first12_exact']=set(t['attempt_id']for t in expected)<={t['attempt_id']for t in chosen}
 rgb=np.arange(96*96*3,dtype=np.uint32).reshape(96,96,3).astype(np.uint8);copy=rgb.copy();mask=np.zeros((96,96),bool);mask[20:22,30:32]=True;mark96(rgb,mask,[[30.,20.,31.,21.]])
 tests['diagnostic_copy_never_changes_original96']=eq(rgb,copy)
 t=chosen[0];assoc=dict(canonical_index_metadata=None,domain='SYNTH_FINITE_END',original_selected_query_xy=[0.,0.],original_selected_L=96.,candidate_enclosures_original96_diagnostic=[[30.,20.,31.,21.]])
 im=contact(t,np.stack([rgb]*3),np.ones((3,96,96),bool),None,assoc)
 tests['clean_contact_ROI_byteexact_no_mask_fallback']=np.array_equal(np.asarray(im)[158:542,18:402],np.repeat(np.repeat(rgb,4,0),4,1))and im.size==(1488,1040)
 tests['missingcanonical_not_filled_by_native']=assoc['canonical_index_metadata']is None
 # Exercise the actual production packet validator with all8 mixed-null U metadata.
 fixture_rows=[];fixture_rgb=np.zeros((3,8,96,96,3),np.uint8);fixture_valid=np.ones((3,8,96,96),bool)
 for vi,var in enumerate(VARS):
  for k in range(8):
   fixture_rows.append(dict(variant=var,native_slot=k,query_xy=[float(k),0.],L=96.,original_global_index=k,original_query_available=True,original_occurrence_id='TOY_ID%d'%k,raw_RGB96_sha256=ash(fixture_rgb[vi,k]),raw_valid96_sha256=ash(fixture_valid[vi,k]),direct_weak_target=(1 if var=='COVER'else 0)if k==2 else-1))
 fixture=dict(rgb96=fixture_rgb,valid96=fixture_valid,query_xy=np.asarray([[float(k),0.]for k in range(8)],np.float64),L=np.full(8,96.,np.float64),native_slot=np.arange(8,dtype=np.int8),original_global_index=np.arange(8,dtype=np.int64),original_occurrence_id=np.asarray(['TOY_ID%d'%k for k in range(8)]),original_query_available=np.ones(8,bool),direct_end_external_target=np.asarray([[r['direct_weak_target']for r in fixture_rows if r['variant']==var]for var in VARS],np.int8))
 validate_packet(fixture,fixture_rows,'BEFORE','SYNTH_TRIPLET');tests['actual_validator_all8_finite_fixture']=True
 for r in fixture_rows:
  if r['native_slot']==7:r.update(query_xy=None,L=None,original_query_available=False)
 fixture['query_xy'][7]=[np.nan,np.nan];fixture['L'][7]=np.nan;fixture['original_query_available'][7]=False
 validate_packet(fixture,fixture_rows,'BEFORE','SYNTH_TRIPLET');tests['actual_validator_mixed_all8_original_None_saved_NaN_U_retained']=True
 for name,index,badvalue in [('finite_L_mismatch',0,97.),('None_metadata_finite_saved_mismatch',7,96.)]:
  bad={k:a.copy()for k,a in fixture.items()};bad['L'][index]=badvalue
  try:validate_packet(bad,fixture_rows,'BEFORE','SYNTH_TRIPLET');ok=False
  except AssertionError:ok=True
  tests['actual_validator_'+name+'_rejected']=ok
 need(all(tests.values()),'Invented contract/readability tests');return dict(completed=True,PASS=True,invented_only=True,tests=tests,actual_source_metadata_inventory_array_header_pixel_score_or_model_reads=0)

def prepare():
 s=read(SCOPE);need(s['all_training_known_triplets_metadata_first']and s['expected_TRAIN_known_triplets']==109 and s['expected_TRAIN_known_rows']==327,'Prospective TRAIN-only contract')
 checks=H/'GENERATED_INVENTED_CHECKS.json';write(checks,invented())
 p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),checks=bind(checks),inherited_viewer_code=bind(OLD_VIEWER),source_layout_code=bind(SOURCE_BUILDER),source_contracts={k:s[k]for k in('source_component_split','source_cache_complete')},future_inputs='Discover exact bound cache metadata/targets and DIRECT_CONTROL split rows only after rootentry; no COMPLETE or actualsource inventory read at preparation.',selection=dict(salt=SALT,first12=True,add_first_hash_available_per_native=list(range(8)),max_union20=True,no_missing_native_replacement=True),source_packet_policy='Only selected TRAIN sourcepacket bindings derived from exact metadata before any header/pixels; optional full_RGB only same exactpacket member; no canonical image fallback. Fullsource targetsidecar equality only is disclosed; no heldout pixels or any predictions.',metadata_and_target_authority='Original q/L/META/flags/all8 unchanged; canonical from existing constructive metadata only, otherwiseUNKNOWN; oldunknown/native/quality/owner notpromoted; D/E/D directweak remains distinct.',output=str(OUT),no_score_or_model_input_or_new_target_reader=True,requires_root_actual_entry=True,automatic_retry=False)
 write(PROTO,p);t=H/'ROOT_ACTUAL_TRAIN_CONTRACT_VIEWER_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(t,dict(approved_train_contract_viewer=False,source_TRAIN_only=True,prediction_or_model_or_new_sampling_allowed=False,full_source_target_container_equality_only=True,code=p['code'],protocol=bind(PROTO),scope=p['scope'],source_contracts=p['source_contracts'],output=str(OUT)))
 ready=dict(completed=True,PASS=True,FINAL_READY_FROZEN=True,prepared_only=True,code=p['code'],protocol=bind(PROTO),checks=p['checks'],false_entry=bind(t),actual_source_metadata_inventory_array_header_pixel_score_model_or_fit=0)
 write(H/'FINAL_READY.json',ready);print(json.dumps(ready))

def run(entry):
 e=read(entry);need(e['approved_train_contract_viewer']is True and e['source_TRAIN_only']is True and e['prediction_or_model_or_new_sampling_allowed']is False and e['full_source_target_container_equality_only']is True,'Separate root actual TRAIN-only viewer entry')
 p=jd(e['protocol']);need(e['code']==bind(__file__)==p['code']and e['protocol']==bind(PROTO)and e['scope']==bind(SCOPE)==p['scope']and e['source_contracts']==p['source_contracts']and e['output']==p['output']==str(OUT),'Frozen contract/viewer code')
 cache=jd(p['source_contracts']['source_cache_complete']);split=jd(p['source_contracts']['source_component_split']);need(cache['completed']and cache['PASS']and cache['source_only'],'Closed unchanged sourcecache')
 need(not OUT.exists(),'Exclusive actual output/no auto retry');OUT.mkdir()
 try:
  rows=lines(cache['metadata']);plan=lines(split['outputs']['DIRECT_CONTROL_SPLIT_ROWS.jsonl']);need(len(rows)==len(plan)==3144,'Full immutable direct cache metadata')
  joined,known=metadata_population(rows,plan);selected,per_native=select_fixed(known)
  census_cache={};associations={}
  for t in known:
   base=t['rows'][0];pb=base['source_proof_binding'];cb=pb['source_census']
   if cb['path']not in census_cache:census_cache[cb['path']]=lines(cb)
   source=census_cache[cb['path']][pb['source_census_row_index']];need(source.get('attempt_id',source.get('pair_id'))==t['attempt_id'],'Exact sourceproof metadata attempt')
   for r in t['rows']:
    bound=source['outputs']if t['domain']=='SYNTH_FINITE_END'else source['variants'][r['source_variant']]
    need(bound==r['source_packet'],'Exact source packet from closed proof metadata')
   associations[(t['domain'],t['attempt_id'])]=source_association(t,source)
  meta=[]
  for t in known:
   a=associations[(t['domain'],t['attempt_id'])]
   meta.append(dict(**{k:t[k]for k in('domain','attempt_id','selected_native_slot','rank_sha256','selected_query_available')},component_id=t['rows'][0]['component_id'],cache_indices24=[r['cache_index']for r in t['rows']],source_packet_bindings=list({r['source_packet']['path']:r['source_packet']for r in t['rows']}.values()),association=a))
  write(OUT/'ALL109_TRAIN_TRIPLET_CONTRACTS_METADATA_BEFORE_PIXELS.json',dict(completed=True,triplets109=meta,metadata_only=True,prediction_or_pixel_header_reads_so_far=0))
  chosen_ids=[dict(domain=t['domain'],attempt_id=t['attempt_id'],native=t['selected_native_slot'],rank_sha256=t['rank_sha256'])for t in selected];packets={r['source_packet']['path']:r['source_packet']for t in selected for r in t['rows']}
  write(OUT/'SELECTION_BEFORE_ANY_TARGET_NPZ_HEADER_OR_PIXEL_READ.json',dict(completed=True,PASS=True,scope=p['scope'],all109_metadata=bind(OUT/'ALL109_TRAIN_TRIPLET_CONTRACTS_METADATA_BEFORE_PIXELS.json'),rule=p['selection'],selected=chosen_ids,per_native_available_first=per_native,source_packet_whitelist=list(packets.values()),original_q_L_native_all8_unchanged=True,prediction_or_pixel_or_target_header_reads_so_far=0))
  allowed={cache['targets']['path']:cache['targets']};y=arrays(cache['targets'],META+('direct_end_external_target','authority'),allowed)
  need(len(y['row_id'])==3144,'Equality-only fullsource targetcontainer')
  for r,s in zip(joined,rows):
   i=r['cache_index'];need(int(y['direct_end_external_target'][i])==r['direct_weak_target']and str(y['authority'][i])==r['authority'],'All source stored targets unchanged')
   for k in META:
    value=qarray(r[k])if k=='query_xy'else(np.nan if r[k]is None else r[k]);need(np.array_equal(np.asarray(value),y[k][i],equal_nan=True)if k in ('query_xy','L')else value==y[k][i],'Fullsource metadata/target exact '+k)
  mapping=[];contacts=[];manifest=[];contexts=[]
  for gi,t in enumerate(selected):
   rr=t['rows'];slot=t['selected_native_slot'];a=associations[(t['domain'],t['attempt_id'])];folder=OUT/('TRIPLET_%02d'%gi);folder.mkdir();raw=[];valid=[];full=[];packet_cache={};masks=[]
   for var in VARS:
    rs=[r for r in rr if r['variant']==var];base=rs[0];b=base['source_packet']
    if b['path']not in packet_cache:packet_cache[b['path']]=read_packet(b,base['layout'],packets)
    z=packet_cache[b['path']];rgb,v,ctx=validate_packet(z,rr,var,base['layout']);raw.append(rgb);valid.append(v);full.append(ctx);mask=z.get('end_related96_mask')
    if mask is not None:need(mask.dtype==bool and mask.shape==(96,96),'Saved diagnostic mask only')
    masks.append(mask)
    for n in range(8):
     r=next(q for q in rs if q['native_slot']==n);cp=folder/(var+'_N%d_CLEAN96.png'%n);vp=folder/(var+'_N%d_VALID96.png'%n);png(cp,Image.fromarray(rgb[n]));png(vp,Image.fromarray(v[n].astype(np.uint8)*255))
     with Image.open(cp)as saved:need(np.array_equal(np.asarray(saved),rgb[n]),'Generated clean96 PNG exact sourcepixels')
     with Image.open(vp)as saved:need(np.array_equal(np.asarray(saved),v[n].astype(np.uint8)*255),'Generated valid96 PNG exact sourcepixels')
     mapping.append(dict(cache_index=r['cache_index'],row_id=r['row_id'],attempt_id=t['attempt_id'],domain=t['domain'],variant=var,native_slot=n,role='TRAIN',direct_weak_target=r['direct_weak_target'],authority=r['authority'],original_global_index=r['original_global_index'],original_query_xy=json.dumps(r['query_xy']),original_L=r['L'],original_query_available=r['original_query_available'],clean_path=cp.relative_to(OUT).as_posix(),valid_path=vp.relative_to(OUT).as_posix(),training_admission=False));manifest.append(dict(cache_index=r['cache_index'],clean=bind(cp),valid=bind(vp),source_raw96_sha256=ash(rgb[n]),source_valid96_sha256=ash(v[n])))
    if ctx is None:contexts.append(dict(attempt_id=t['attempt_id'],variant=var,available=False,reason='full_RGB_MEMBER_NOT_DECLARED_OR_MISSING_NO_FALLBACK',source_packet=b))
    else:
     c=folder/(var+'_FULL_CONTEXT_ORIGINAL.png');d=folder/(var+'_FULL_CONTEXT_DISPLAY.png');png(c,Image.fromarray(ctx));png(d,full_display(ctx,rs,a));contexts.append(dict(attempt_id=t['attempt_id'],variant=var,available=True,source_packet=b,exact_member='full_RGB',original=bind(c),display=bind(d),original_RGB_array_sha256=ash(ctx),display_only=True))
   raw=np.stack(raw);valid=np.stack(valid);savedmask=masks[0]
   if savedmask is not None:need(all(m is not None and eq(m,savedmask)for m in masks),'Same saved selected-end mask acrossvariants')
   card=folder/'CONTACT_SELECTED_WEAK_END.png';all8=folder/'ALL8_QUERY_CONTACT.png';png(card,contact(t,raw[:,slot],valid[:,slot],savedmask,a));png(all8,all8_contact(rr,raw,valid));contacts.append(dict(attempt_id=t['attempt_id'],contact=bind(card),all8=bind(all8)));write(folder/'TARGET_ASSOCIATION.json',a)
  csvfile(OUT/'SELECTED_TRAIN_ALL8_TARGET_MAPPING.csv',mapping);write(OUT/'SELECTED_TRAIN_ALL8_TARGET_MAPPING.json',dict(completed=True,rows=mapping,no_predictions=True));write(OUT/'IMAGE_MANIFEST.json',dict(completed=True,images=manifest,contacts=contacts,fullcontexts=contexts))
  intro='V197 TRAIN-only固定hash12+各native firstavailable union; all109knowntriplet metadata beforepixels. Scores/predictions are not read. Existing weakD/E/D and UNKNOWN remain unchanged, q/L/native8 preserved. This is target-contract readability, not physical/native gold, new labels or training selection.'
  md=['# V197 TRAIN 부위-감독 대응 고정 뷰어','',intro,'','[전체109 metadata](ALL109_TRAIN_TRIPLET_CONTRACTS_METADATA_BEFORE_PIXELS.json) · [픽셀전 고정selection](SELECTION_BEFORE_ANY_TARGET_NPZ_HEADER_OR_PIXEL_READ.json) · [선택all8표](SELECTED_TRAIN_ALL8_TARGET_MAPPING.csv)','','clean원본과 display표시를 구분한다. fullcontext는 같은 exact TRAINpacket의 full_RGB가 있는 경우만 별도 표시하며 없으면 MISSING/no fallback이다.',''];body='<h1>V197 TRAIN contract viewer</h1><p>'+html.escape(intro)+'</p>'
  for gi,t in enumerate(selected):
   md+=['## '+t['attempt_id'],'',f'![selected end](TRIPLET_{gi:02d}/CONTACT_SELECTED_WEAK_END.png)','',f'![all8 original queries](TRIPLET_{gi:02d}/ALL8_QUERY_CONTACT.png)','',f'[기존 weak association](TRIPLET_{gi:02d}/TARGET_ASSOCIATION.json)',''];body+='<h2>'+html.escape(t['attempt_id'])+'</h2><img src="TRIPLET_%02d/CONTACT_SELECTED_WEAK_END.png"><img src="TRIPLET_%02d/ALL8_QUERY_CONTACT.png">'%(gi,gi)
   for ctx in [c for c in contexts if c['attempt_id']==t['attempt_id']]:
    if ctx['available']:
     rel=Path(ctx['display']['path']).relative_to(OUT).as_posix();clean=Path(ctx['original']['path']).relative_to(OUT).as_posix();md += [f"### {ctx['variant']} full context (display copy)",'',f'![full context]({rel})','',f'[unmarked original]({clean})',''];body+='<img src="'+rel+'">'
    else:md += ['Full context '+ctx['variant']+': MISSING; no fallback.','']
  text(OUT/'PORTABLE_GALLERY_KO.md','\n'.join(md));text(OUT/'index.html','<!doctype html><meta charset="utf-8"><style>body{font:18px sans-serif;margin:24px}img{max-width:100%;height:auto}</style>'+body)
  write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTO),entry=bind(entry),source_contracts=p['source_contracts'],all109_TRAIN_known_triplets_metadata_first=True,all327_TRAIN_known_labels_preserved=True,selected_triplets=len(selected),selected_all8_rows=len(mapping),selection=bind(OUT/'SELECTION_BEFORE_ANY_TARGET_NPZ_HEADER_OR_PIXEL_READ.json'),associations=bind(OUT/'ALL109_TRAIN_TRIPLET_CONTRACTS_METADATA_BEFORE_PIXELS.json'),mapping=bind(OUT/'SELECTED_TRAIN_ALL8_TARGET_MAPPING.csv'),manifest=bind(OUT/'IMAGE_MANIFEST.json'),contacts=contacts,fullcontext_available=sum(c['available']for c in contexts),fullcontext_missing=sum(not c['available']for c in contexts),score_bodies_or_model_forward_fit_SAM_initializer_recrop_PnP=0,source_HOLDOUT_pixel_or_score_access=0,actual_image_tool_view='PENDING_SEPARATE_QA',HTML_browser_rendered=False,training_admission=False,selected_model=None,fully_solved=False,automatic_retry=False));print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'))))
 except BaseException:
  write(OUT/'FAILURE_NO_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare()if a.prepare else run(a.entry)

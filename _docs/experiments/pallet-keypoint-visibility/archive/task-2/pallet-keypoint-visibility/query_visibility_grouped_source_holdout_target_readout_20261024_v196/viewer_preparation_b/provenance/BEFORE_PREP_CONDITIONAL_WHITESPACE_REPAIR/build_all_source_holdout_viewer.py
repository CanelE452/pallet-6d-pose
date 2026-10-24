"""All source HOLDOUT360 viewer. Saved source bytes only after root entry.
No model/SAM/initializer/crop regeneration; no case or threshold selection.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,html,io,json,math,os,stat,traceback
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_PROSPECTIVE_ALL_SOURCE_HOLDOUT_TARGET_DIAGNOSTIC_SCOPE.json'
PROTO=H/'PROTOCOL_BEFORE_ALL_SOURCE_HOLDOUT_ARRAY_OR_SCORE_BODIES.json'
OUT=V/'all360_source_holdout_viewer_v1'
ARMS=('C56_RAW96_DIRECT_END_GROUPED','C57_RAW96_PLUS_FROZEN_SAM_DIRECT_END_GROUPED')
VARS=('BEFORE','COVER','SHAM');META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
SCORE_KEYS=META+('raw_direct_end_external_logit','direct_end_prediction_D0_E1_Uminus1')
SOURCE_KEYS=('rgb96','valid96','query_xy','L','native_slot','original_global_index','original_occurrence_id','original_query_available','end_related96_mask','variant','direct_end_external_target')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

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
 need(allowed.get(b['path'])==b,'Exact root-declared or all-HOLDOUT-derived packet before header/hash')
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

def select_all(rows,plan,target,expected_groups=15):
 bykey={tuple(r['control_join_key']):r for r in plan};need(len(bykey)==len(plan)==len(rows)==len(target),'Full unique source split/cache join')
 selected=[];groups=defaultdict(list)
 for i,r in enumerate(rows):
  k=tuple(r['control_join_key']);need(k in bykey and r['cache_index']==i,'Original full cache order/control key')
  s=bykey[k];need(s['role']in('TRAIN','HOLDOUT')and int(target[i])==s['stored_weak_direct_target'],'Original stored weak target/role')
  if s['role']!='HOLDOUT':continue
  need(r['source_packet']==s['source_packet']and r['authority']==s['authority'],'Exact heldout source/authority')
  for key in META:
   if key=='query_xy':need(np.array_equal(qarray(r[key]),qarray(s[key]),equal_nan=True),'Original heldout q')
   elif key=='L':need(np.array_equal(np.asarray(np.nan if r[key]is None else r[key]),np.asarray(np.nan if s[key]is None else s[key]),equal_nan=True),'Original heldout L')
   else:need(r[key]==s[key],'Heldout original identity '+key)
  a=dict(r,role='HOLDOUT',component_id=s['component_id'],direct_weak_target=int(target[i]),selected_native_slot=s['selected_native_slot'],training_admission=False)
  selected.append(a);groups[(r['domain'],r['attempt_id'])].append(a)
 need(len(groups)==expected_groups and len(selected)==expected_groups*24,'All15 triplets/all360 rows; no score/native selection')
 for key,rr in groups.items():
  need({(r['variant'],r['native_slot'])for r in rr}=={(v,k)for v in VARS for k in range(8)},'Full3variants x native8')
  need(len({tuple((r['source_packet']['path'],r['source_packet']['sha256']))for r in rr})==1,'One exact source triplet packet')
  known=[r for r in rr if r['direct_weak_target']>=0];need(len(known)==3 and len({r['native_slot']for r in known})==1,'Selected weak-end only, other7 remain UNKNOWN')
  need({r['variant']:r['direct_weak_target']for r in known}==dict(BEFORE=0,COVER=1,SHAM=0),'Existing direct weak D/E/D unchanged')
 return selected,groups

def target_overlay(rgb,mask,bounds=None):
 need(rgb.dtype==np.uint8 and rgb.shape==(96,96,3)and mask.dtype==bool and mask.shape==(96,96),'Exact source clean96 + selected end sample mask')
 a=rgb.copy();a[mask]=(np.asarray(a[mask],np.uint16)+np.array([255,0,255],np.uint16))//2;im=Image.fromarray(a).resize((384,384),Image.Resampling.NEAREST);d=ImageDraw.Draw(im)
 if bounds is not None:
  for x0,y0,x1,y1 in bounds:d.rectangle([(x0+.5)*4-.5,(y0+.5)*4-.5,(x1+.5)*4-.5,(y1+.5)*4-.5],outline=(255,220,0),width=2)
 q=191.5;d.line([(q-9,q),(q+9,q)],fill=(0,255,255),width=1);d.line([(q,q-9),(q,q+9)],fill=(0,255,255),width=1);return im

def contact(rows,rgb,valid,mask,bounds):
 im=Image.new('RGB',(1470,980),'white');d=ImageDraw.Draw(im);font=lambda n:ImageFont.truetype(FONT,n)
 def label(x,y,s,n=18):d.text((x,y),s,font=font(n),fill='black')
 label(18,12,'V196 SOURCE HOLDOUT | '+rows[0]['attempt_id'],23)
 label(18,48,'Selected weak native %d | all8 status retained | constructive finite-end authority != physical native gold'%rows[0]['native_slot'],18)
 for vi,var in enumerate(VARS):
  r=next(q for q in rows if q['variant']==var);x=18+vi*486;label(x,82,var+' | weak '+{0:'D',1:'E',-1:'UNKNOWN'}[r['direct_weak_target']],22);im.paste(Image.fromarray(rgb[vi]).resize((384,384),Image.Resampling.NEAREST),(x,118));label(x,510,'Clean96 x4 DISPLAY; original PNG separate',17);im.paste(target_overlay(rgb[vi],mask,bounds),(x,545))
  for j,tag in enumerate(('C56','C57')):label(x,938+j*20,tag+' raw='+str(r[tag+'_raw'])+' -> '+r[tag+'_D_E_U'],17)
 return im

def csvfile(path,rows):
 keys=list(rows[0]);s=io.StringIO();w=csv.DictWriter(s,fieldnames=keys);w.writeheader();w.writerows(rows);text(path,s.getvalue())
def png(path,im):
 safe(path)
 with Path(path).open('xb')as f:im.save(f,format='PNG')

def invented():
 rows=[];plan=[];target=[]
 for vi,var in enumerate(VARS):
  for k in range(8):
   i=8*vi+k;t=(1 if var=='COVER'else0)if k==2 else-1;r=dict(cache_index=i,domain='SYNTH_FINITE_END',attempt_id='INVENTED',variant=var,native_slot=k,row_id=var+'/native'+str(k),original_global_index=k,query_xy=[float(k),0.],L=96.,original_query_available=True,control_join_key=['SYNTH_FINITE_END','INVENTED',var,k],source_packet=dict(path='/invented/source.npz',sha256='a'*64),authority='CONSTRUCTIVE_WEAK')
   rows.append(r);plan.append(dict(r,role='HOLDOUT',stored_weak_direct_target=t,selected_native_slot=2,component_id='fixture'));target.append(t)
 ss,gg=select_all(rows,plan,np.asarray(target,np.int8),1);tests=dict(all24_rows_all8_and_unknown21_retained=len(ss)==24 and sum(r['direct_weak_target']<0 for r in ss)==21)
 bad=[dict(r)for r in rows];bad[1]['control_join_key']=bad[0]['control_join_key']
 try:select_all(bad,plan,np.asarray(target,np.int8),1);ok=False
 except AssertionError:ok=True
 tests['duplicate_or_missing_occurrence_fails']=ok
 p=[dict(r)for r in plan];p[0]['role']='TRAIN'
 try:select_all(rows,p,np.asarray(target,np.int8),1);ok=False
 except AssertionError:ok=True
 tests['missing_heldout_row_never_replaced']=ok
 rgb=np.arange(96*96*3,dtype=np.uint32).reshape(96,96,3).astype(np.uint8);before=rgb.copy();mask=np.zeros((96,96),bool);mask[10:12,20:22]=True;target_overlay(rgb,mask,[[19.5,9.5,21.5,11.5]])
 tests['display_overlay_never_mutates_clean_input']=eq(rgb,before)
 tests['F64_raw_to_crop_bounds_uses_original_halfpixel']=np.array_equal(96*(np.asarray([[1.,2.]])-np.asarray([1.,2.]))/96+47.5,np.array([[47.5,47.5]]))
 need(all(tests.values()),'Invented fixed viewer checks');return dict(completed=True,PASS=True,invented_only=True,tests=tests,actual_source_array_header_target_score_image_or_model_reads=0)
def prepare():
 s=read(SCOPE);need(s['code_only_preparation_authorized']is True,'Root code-only scope')
 fit=jd(s['source_fit_complete']);cache=jd(s['source_cache_complete']);split=jd(s['source_component_split'])
 write(H/'GENERATED_INVENTED_CHECKS.json',invented())
 p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),checks=bind(H/'GENERATED_INVENTED_CHECKS.json'),source_contracts={k:s[k]for k in('source_phase_freeze','source_fit_complete','source_cache_complete','source_component_split','source_fit_integrity_review')},input_bindings=dict(cache_metadata=cache['metadata'],cache_targets=cache['targets'],direct_split_rows=split['outputs']['DIRECT_CONTROL_SPLIT_ROWS.jsonl'],heldout_outputs={a:fit['arms'][a]['source_cohorts']['DIRECT_CONTROL']['HOLDOUT']for a in ARMS}),fit_code=fit['code'],fit_protocol=fit['protocol'],cache_code=cache['code'],cache_protocol=cache['protocol'],output=str(OUT),selection='ALL360 immutable DIRECT_CONTROL component HOLDOUT rows,15complete BEFORE/COVER/SHAM attempts x8native;45known/315UNKNOWN. No score/error/native/material selection or replacement.',source_pixel_reads='Only rgb96/valid96 and numeric original metadata/end_related96_mask/direct target/variant from exactly15 source CONTROL_TRIPLET NPZ bindings discovered in all360 heldout metadata; no full_RGB/foreign_alpha/sourcebuffer/weights/cacheRGB/SAM bodies.',container_disclosure='Full cache metadata JSONL3144 and target-sidecar NPZ arrays3144 are deserialized to join fixed heldout indices. Only360heldout targets and metadata are interpreted/displayed; no TRAIN scores or pixels loaded. Each selected source packet is exactly3x8raw96, yielding only360image rows.',target_association='Saved constructive/weak finite-end mask and original F64 enclosure projection are diagnostic overlays. All315other-native targets remainUNKNOWN; no native physical recertification or new localization. Original clean input is never marked; no label admission.',requires_separate_root_actual_entry=True,automatic_retry=False,no_model_fit_forward_SAM_initializer_recrop_PnP_or_threshold_choice=True,no_source_holdout_training_promotion=True)
 write(PROTO,p);t=H/'ROOT_ACTUAL_ALL_SOURCE_HOLDOUT_VIEWER_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(t,dict(approved_all_source_holdout_viewer=False,source_only=True,no_model_or_new_sampling_or_training=True,all_holdout_rows_without_selection=True,code=p['code'],protocol=bind(PROTO),scope=p['scope'],source_contracts=p['source_contracts'],output=str(OUT)))
 r=dict(completed=True,PASS=True,FINAL_READY_FROZEN=True,prepared_only=True,code=p['code'],protocol=bind(PROTO),checks=p['checks'],false_entry=bind(t),actual_source_array_header_pixel_target_score_or_checkpoint_reads=0)
 write(H/'FINAL_READY.json',r);print(json.dumps(r))

def run(entry_path):
 e=read(entry_path);need(all(e.get(k)is True for k in('approved_all_source_holdout_viewer','source_only','no_model_or_new_sampling_or_training','all_holdout_rows_without_selection')),'Separate actual saved viewer entry')
 p=jd(e['protocol']);need(e['code']==bind(__file__)==p['code']and e['protocol']==bind(PROTO)and e['scope']==bind(SCOPE)==p['scope']and e['source_contracts']==p['source_contracts']and e['output']==p['output']==str(OUT),'Frozen source viewer rules')
 contracts=p['source_contracts'];fit=jd(contracts['source_fit_complete']);cache=jd(contracts['source_cache_complete']);jd(contracts['source_phase_freeze']);review=jd(contracts['source_fit_integrity_review'])
 need(fit['completed']and fit['PASS']and fit['source_only']and fit['all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs']and review['completed']and review['PASS']and review['source_fit_complete']==contracts['source_fit_complete'],'Frozen source model outputs and independent integrity')
 need(not OUT.exists(),'Exclusive viewer output/no automatic retry');OUT.mkdir()
 try:
  ib=p['input_bindings'];rows=lines(ib['cache_metadata']);plan=lines(ib['direct_split_rows']);allowed={ib['cache_targets']['path']:ib['cache_targets']}
  y=arrays(ib['cache_targets'],META+('direct_end_external_target','authority'),allowed);need(len(rows)==len(plan)==len(y['row_id'])==3144,'Original full source metadata/target container')
  selected,groups=select_all(rows,plan,y['direct_end_external_target']);need(Counter(r['direct_weak_target']for r in selected)==Counter({-1:315,0:30,1:15}),'All360/45known/315UNKNOWN')
  need(all(r['domain']=='SYNTH_FINITE_END'for r in selected),'Existing source heldout synthetic only; no real transfer claim')
  packets={r['source_packet']['path']:r['source_packet']for r in selected};need(len(packets)==15,'All15 source packets')
  for b in packets.values():allowed[b['path']]=b
  for a in ARMS:allowed[ib['heldout_outputs'][a]['outputs']['path']]=ib['heldout_outputs'][a]['outputs']
  write(OUT/'ALL360_AND_ALL15_BEFORE_PIXEL_SCORE_READS.json',dict(completed=True,selection='ALL_COMPONENT_HOLDOUT',rows360=360,known45=45,UNKNOWN315=315,cache_indices=[r['cache_index']for r in selected],source_packets=list(packets.values()),source_contracts=contracts,no_score_or_pixel_body_read_yet=True))
  predictions={};lineage=None
  for arm in ARMS:
   record=ib['heldout_outputs'][arm];z=arrays(record['outputs'],SCORE_KEYS,allowed);ll=lines(record['metadata']);need(len(ll)==len(z['row_id'])==360 and record['rows']==360,'Exact saved HOLDOUT output')
   need(lineage is None or ll==lineage,'Both heads same original heldout order');lineage=ll
   for j,r in enumerate(selected):
    i=r['cache_index'];need(ll[j]['cache_index']==i and ll[j]['role']=='HOLDOUT'and ll[j]['control_join_key']==r['control_join_key'],'Exact HOLDOUT cache index and role')
    for key in META:need(eq(z[key][j],y[key][i]),'Saved source output/target META6 bytes')
    q=qarray(r['query_xy']);need(np.array_equal(q,z['query_xy'][j],equal_nan=True)and np.array_equal(np.asarray(np.nan if r['L']is None else r['L']),z['L'][j],equal_nan=True),'Original q/L')
    need(str(y['authority'][i])==r['authority']and str(z['row_id'][j])==r['row_id'],'Source target authority and identity')
    raw=z['raw_direct_end_external_logit'][j];av=bool(z['original_query_available'][j]);dp=int(raw>=0)if av and np.isfinite(raw)else-1;need(dp==int(z['direct_end_prediction_D0_E1_Uminus1'][j]),'Frozen raw>=0 direct decoder/U')
    tag=arm[:3];r[tag+'_raw']=finite(raw);r[tag+'_D_E_U']={0:'D',1:'E',-1:'U'}[dp]
   predictions[arm]=record['outputs']
  proof_censuses={};mapping=[];image_records=[];triplets=[];contacts=[]
  for gi,(key,rr)in enumerate(sorted(groups.items())):
   rowby={(r['variant'],r['native_slot']):r for r in rr};selected_slot=next(r['native_slot']for r in rr if r['direct_weak_target']>=0);base=rr[0];b=base['source_packet'];a=arrays(b,SOURCE_KEYS,allowed)
   need(a['rgb96'].dtype==np.uint8 and a['rgb96'].shape==(3,8,96,96,3)and a['valid96'].dtype==bool and a['valid96'].shape==(3,8,96,96)and a['end_related96_mask'].dtype==bool and a['end_related96_mask'].shape==(96,96),'Saved exact raw96/valid/endmask packet')
   need(a['variant'].tolist()==list(VARS)and a['query_xy'].dtype==np.float64 and a['L'].dtype==np.float64 and a['native_slot'].tolist()==list(range(8)),'Original F64/source order')
   pb=base['source_proof_binding'];cb=pb['source_census']
   if cb['path']not in proof_censuses:proof_censuses[cb['path']]=lines(cb)
   source=proof_censuses[cb['path']][pb['source_census_row_index']];need(source['attempt_id']==base['attempt_id']and source['outputs']==b,'Exact saved proof row/source packet')
   geom=source.get('geometry_plan',{});endpoint=geom.get('endpoint_observation',{});bounds_raw=endpoint.get('candidate_enclosures_xyxy');q=qarray(rowby[('BEFORE',selected_slot)]['query_xy']);L=rowby[('BEFORE',selected_slot)]['L'];bounds=None
   if bounds_raw is not None and np.isfinite(q).all()and L is not None and math.isfinite(L)and L>0:bounds=[(96*(np.asarray(bb,np.float64).reshape(2,2)-q)/L+47.5).reshape(-1).tolist()for bb in bounds_raw]
   mask=a['end_related96_mask'];change=[int(np.any(a['rgb96'][vi,selected_slot]!=a['rgb96'][0,selected_slot],axis=2)[mask].sum())for vi in range(3)]
   proof=dict(attempt_id=base['attempt_id'],source_base_index=base.get('source_base_index'),selected_native_slot=selected_slot,role='HOLDOUT',authority=base['authority'],candidate_enclosures_fullRGB=bounds_raw,candidate_enclosures_original96_diagnostic=bounds,end_related_original96_samples=int(mask.sum()),end_mask_RGB_changed_samples_vs_BEFORE=dict(zip(VARS,change)),existing_geometry_status=geom.get('status','UNKNOWN'),existing_proof=source.get('closed_proof',{}),original_query_xy=q.tolist(),original_L=L,association_only_not_nativephysicalgold=True,all8_other_native_target_UNKNOWN=True,training_admission=False)
   triplets.append(proof);folder=OUT/('TRIPLET_%02d'%gi);folder.mkdir()
   selected_rows=[]
   for vi,var in enumerate(VARS):
    for k in range(8):
     r=rowby[(var,k)];i=r['cache_index'];rgb=a['rgb96'][vi,k];valid=a['valid96'][vi,k]
     need(ash(rgb)==r['raw_RGB96_sha256']and ash(valid)==r['raw_valid96_sha256'],'Saved raw/valid exact original cache lineage')
     need(np.array_equal(a['query_xy'][k],qarray(r['query_xy']),equal_nan=True)and a['L'][k]==r['L']and int(a['native_slot'][k])==k and int(a['original_global_index'][k])==r['original_global_index']and bool(a['original_query_available'][k])==r['original_query_available'],'Source packet original numeric/flags')
     need(int(a['direct_end_external_target'][vi,k])==r['direct_weak_target'],'Existing selected weak target, no promotion')
     rawfile=folder/(var+'_N%d_CLEAN96.png'%k);validfile=folder/(var+'_N%d_VALID96.png'%k);png(rawfile,Image.fromarray(rgb));png(validfile,Image.fromarray(valid.astype(np.uint8)*255))
     with Image.open(rawfile)as saved:need(np.array_equal(np.asarray(saved),rgb),'Generated clean PNG pixel exact')
     rec=dict(cache_index=i,row_id=r['row_id'],original_global_index=r['original_global_index'],attempt_id=r['attempt_id'],variant=var,native_slot=k,role='HOLDOUT',component_id=r['component_id'],original_query_xy=json.dumps(r['query_xy']),original_L=r['L'],original_query_available=r['original_query_available'],direct_weak_target=r['direct_weak_target'],target_authority=r['authority'],selected_weak_native=(k==selected_slot),target_association='SAVED_FINITE_END_MASK'if k==selected_slot else'UNKNOWN_NO_NATIVE_END_TARGET_ASSIGNED',end_related96_samples=int(mask.sum())if k==selected_slot elseNone,C56_raw=r['C56_raw'],C56_D_E_U=r['C56_D_E_U'],C57_raw=r['C57_raw'],C57_D_E_U=r['C57_D_E_U'],clean_path=rawfile.relative_to(OUT).as_posix(),valid_path=validfile.relative_to(OUT).as_posix(),training_admission=False)
     mapping.append(rec);image_records.append(dict(cache_index=i,clean=bind(rawfile),valid=bind(validfile),source_raw96_sha256=ash(rgb),source_valid96_sha256=ash(valid)))
    selected_rows.append(rowby[(var,selected_slot)])
   cp=folder/'CONTACT_SELECTED_WEAK_END.png';png(cp,contact(selected_rows,a['rgb96'][:,selected_slot],a['valid96'][:,selected_slot],mask,bounds));contacts.append(bind(cp));write(folder/'TARGET_ASSOCIATION.json',proof)
  need(len(mapping)==360 and sum(r['direct_weak_target']>=0 for r in mapping)==45 and len(contacts)==15,'All original counts after rendering')
  mapping.sort(key=lambda r:r['cache_index']);csvfile(OUT/'ALL360_HOLDOUT_TARGET_MAPPING.csv',mapping);write(OUT/'ALL360_HOLDOUT_TARGET_MAPPING.json',dict(completed=True,rows=mapping,source_contracts=contracts,no_new_label_or_training_admission=True));write(OUT/'ALL15_TARGET_ASSOCIATIONS.json',dict(completed=True,triplets=triplets,association_not_nativephysicalgold=True));write(OUT/'IMAGE_MANIFEST.json',dict(completed=True,images=image_records,contacts=contacts,actual_image_tool_view='PENDING_ROOT_OR_SEPARATE_QA'))
  intro='Source HOLDOUT360행 전체(45known/315UNKNOWN),15triplet×3variant×native8이다. 오답/점수/native로 선택·교체하지 않았다. 원q/L/clean96/valid를 유지한다. finite-end sampled mask와 bbox는 기존 구성적/weak 목표의 위치 설명이며 물리 native gold나 새 localization이 아니다. sourceHOLDOUT은 TRAIN으로 편입하지 않는다. source 설계가 이미 알려졌으며 자연 실사 일반화/모델 채택/해결 주장을 하지 않는다.'
  md=['# V196: 전체 source HOLDOUT 목표-입력 대응','',intro,'','[전체360 target/prediction CSV](ALL360_HOLDOUT_TARGET_MAPPING.csv) · [전체360 JSON](ALL360_HOLDOUT_TARGET_MAPPING.json) · [15개 목표 연관/기존proof](ALL15_TARGET_ASSOCIATIONS.json)','','원본clean96와valid는 각각 별도 파일이다. 아래contact 윗줄은clean96확대 표시, 아랫줄은마젠타 end-related sample/노란기존enclosure/cyan원q의 표시 사본이다. 색표시는 모델입력에들어가지 않는다.','']
  body='<h1>V196 all source HOLDOUT360</h1><p>'+html.escape(intro)+'</p><p><a href="ALL360_HOLDOUT_TARGET_MAPPING.csv">All360 CSV</a> | <a href="ALL15_TARGET_ASSOCIATIONS.json">All15 associations/proofs</a></p>'
  for gi,proof in enumerate(triplets):
   rel=Path(contacts[gi]['path']).relative_to(OUT).as_posix();md+=['## '+proof['attempt_id'],'',f'![all fixed source heldout triplet]({rel})','',f'[기존 목표 위치와 proof](TRIPLET_{gi:02d}/TARGET_ASSOCIATION.json)',''];body+='<h2>'+html.escape(proof['attempt_id'])+'</h2><img width="1400" src="'+rel+'">'
   rr=[r for r in mapping if r['attempt_id']==proof['attempt_id']];md+=['|variant|native|target|C56|C57|clean96|valid96|','|---|---|---|---|---|---|---|']
   for r in rr:
    md+=['|'+r['variant']+'|'+str(r['native_slot'])+'|'+str(r['direct_weak_target'])+'|'+r['C56_D_E_U']+'|'+r['C57_D_E_U']+'|[clean]('+r['clean_path']+')|[valid]('+r['valid_path']+')|'];body+='<div>'+html.escape(r['variant']+' native'+str(r['native_slot'])+' target'+str(r['direct_weak_target'])+' C56'+r['C56_D_E_U']+' C57'+r['C57_D_E_U'])+' <a href="'+r['clean_path']+'"><img width="96" src="'+r['clean_path']+'"></a> <a href="'+r['valid_path']+'">valid</a></div>'
   md+=['']
  text(OUT/'PORTABLE_GALLERY_KO.md','\n'.join(md));text(OUT/'index.html','<!doctype html><meta charset="utf-8"><style>body{font:18px sans-serif;margin:24px}img{max-width:100%;height:auto}</style>'+body)
  write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTO),entry=bind(entry_path),source_contracts=contracts,rows360=360,known45=45,UNKNOWN315=315,triplets15=15,native8_all_preserved=True,all_clean96_valid96_source_byteexact=True,all_HOLDOUT_roles_unchanged=True,source_packet_count15=15,all_target_and_prediction_META_exact=True,contacts=contacts,gallery=bind(OUT/'PORTABLE_GALLERY_KO.md'),HTML=bind(OUT/'index.html'),mapping=bind(OUT/'ALL360_HOLDOUT_TARGET_MAPPING.csv'),associations=bind(OUT/'ALL15_TARGET_ASSOCIATIONS.json'),actual_image_tool_view='PENDING',HTML_browser_rendered=False,new_labels_model_fit_or_selection=False,training_admission=False,selected_model=None,fully_solved=False,automatic_retry=False));print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'))))
 except BaseException:
  write(OUT/'FAILURE_NO_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare()if a.prepare else run(a.entry)

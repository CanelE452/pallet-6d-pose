"""Same V239 fixed12 SOURCE DEV cases, existing PNGs only, late Root entry."""
import argparse,csv,hashlib,html,io,json,math,os,stat,traceback
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation');H=Path(__file__).absolute().parent
BASE='BASELINE_V239_CANONICAL_CONTEXT';TREAT='TREAT_SCENE_CF_MATCHED_DATA'
PINS={
 'old_complete':{'path':str(R/'V239_SOURCE_GALLERY_A/gallery_v1/COMPLETE.json'),'sha256':'e5f1d0e211441a9e8860fbe0ab37ecee3d0fee0f7adbfeccb3c74bd1673b89e0'},
 'old_selection':{'path':str(R/'V239_SOURCE_GALLERY_A/gallery_v1/CASE_IDS_BEFORE_PIXELS.json'),'sha256':'709ccd36ce6076eb922fb2ea7e128ce33b83f5bc7618964126fa1635e91bc97c'},
 'old_case_data':{'path':str(R/'V239_SOURCE_GALLERY_A/gallery_v1/CASE_DATA.json'),'sha256':'d19b7f15b75696cebed7d62c9d522c59af1d48dd8b4ba151acd4dc7d3fe1070b'}}
EXPECTED=[0,382,148,7,7221,7266,7402,7220,149,1539,7262,7266]


def lexical(value):
 p=Path(value)
 if any(t in str(p).casefold() for t in ('v112','__pycache__','.pyc','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation')) or '..' in p.parts or not p.is_absolute():raise ValueError('Forbidden lexical path before FS')
 return p


def safe(value):
 p=lexical(value)
 for a in reversed((p,*p.parents)):
  try:mode=os.lstat(a).st_mode
  except FileNotFoundError:continue
  if stat.S_ISLNK(mode):raise ValueError('Symlink ancestor')
 return p


def ident(b):
 if not isinstance(b,dict) or not {'path','sha256'}<=set(b):raise ValueError('Exact binding required')
 return str(lexical(b['path'])),b['sha256']


def read(p):
 with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()


def binding(p):
 b=read(p);return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def checked(b,white,pin=None):
 if ident(b) not in {ident(x) for x in white} or (pin is not None and ident(b)!=ident(pin)):raise ValueError('Exact whitelist before FS')
 raw=read(b['path'])
 if hashlib.sha256(raw).hexdigest()!=b['sha256'] or ('size_bytes'in b and len(raw)!=b['size_bytes']):raise ValueError('Bound bytes changed')
 return raw


def exact(a,b):
 if ident(a)!=ident(b):raise ValueError('Exact metadata graph identity')


def require_PASS(v):
 if v.get('completed') is not True or v.get('PASS') is not True:raise ValueError('Completed/PASS required')


def write_json(path,obj):
 with safe(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 return binding(path)


def source_metadata(complete,selection,data):
 for v in (complete,selection,data):require_PASS(v)
 exact(complete['selection'],PINS['old_selection']);exact(data['selection'],PINS['old_selection'])
 if [c['sample_index'] for c in selection['cases']]!=EXPECTED or [c['sample_index'] for c in data['cases']]!=EXPECTED:raise ValueError('EXACT pretrial fixed12 IDs; no reselection')
 if selection['bucket_slots']!=12 or selection['unique_samples']!=11 or selection['missing_buckets']!=0 or len(data['cases'])!=12:raise ValueError('Historical fixed12 denominators')
 allowed={ident(b) for b in complete['outputs']};images={}
 for slot,case in zip(selection['cases'],data['cases']):
  if case['case_number']!=slot['case_number'] or case['missing']!=slot['missing'] or case['role']!='INTERNAL_SOURCE_DEV_REUSED_NOT_BLIND':raise ValueError('Historical exact cases/role/EMPTY state')
  if len(case['original_available8'])!=8 or len(case['q8_original'])!=8 or len(case['L8_original'])!=8 or not 0<=case['active_slot']<8:raise ValueError('Original geometry unchanged')
  proof=case['cache_pixel_proof']
  if len(proof['node_PNG_proofs'])!=8:raise ValueError('All8 original node crops')
  for node in proof['node_PNG_proofs']:
   b=node['PNG']
   if ident(b) not in allowed or node['exact'] is not True or node['cache_CHW_uint8_sha256']!=node['decoded_CHW_uint8_sha256']:raise ValueError('Prior exact RAW96 PNG closure')
   images[ident(b)]=b
  b=proof['context_display_PNG']
  if ident(b) not in allowed:raise ValueError('Prior context display PNG closure')
  images[ident(b)]=b
 if len(images)!=81:raise ValueError('Exact9groups x8RAWnodes+context display')
 return list(images.values())


def selected_CSV(raw,cases):
 wanted={c['sample_index']:c for c in cases};out={};count=0
 for i,row in enumerate(csv.DictReader(io.StringIO(raw.decode()))):
  count+=1
  if int(row['sample_index'])!=i:raise ValueError('Original all7617 order')
  if i not in wanted:continue
  c=wanted[i];f=int(row['family']);y=int(row['target']);z=[float(row['logit0']),float(row['logit1'])];selected=float(row['selected_family_logit'])
  if row['row_id']!=c['row_id'] or row['role']!='DEV' or row['available']!='True' or f!=c['family'] or y!=(0 if c['target_state']=='D' else 1):raise ValueError('Same original SOURCE DEV row/teacher/available')
  if not all(map(math.isfinite,z)) or selected!=z[f] or row['prediction']!=('E' if selected>=0 else 'D'):raise ValueError('Family-specific binarylogit0 boundary; no argmax')
  out[i]={'prediction':row['prediction'],'logit0':z[0],'logit1':z[1],'selected_family_logit':selected}
 if count!=7617 or set(out)!=set(wanted):raise ValueError('Alloriginal CSV denominator and exact11 unique fixed samples')
 return out


def decode_bound_PNG(b,white,shape):
 raw=checked(b,white)
 with Image.open(io.BytesIO(raw)) as im:
  if im.format!='PNG' or im.mode!='RGB' or im.size!=shape:raise ValueError('Existing RGB PNG exactsize')
  im.load();image=im.copy()
 return image,raw


def draw_cross(draw,xy,color,width=2,radius=9):
 x,y=xy;draw.line((x-radius,y,x+radius,y),fill=color,width=width);draw.line((x,y-radius,x,y+radius),fill=color,width=width)


def font(size):
 return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def render_card(case,old,new,nodes,context):
 board=Image.new('RGB',(2000,1220),'#f8fafc');d=ImageDraw.Draw(board);f=font(30);small=font(24);tiny=font(20)
 family='SYN intended-native witness0' if case['family']==0 else 'YCB associated-r AUX; native UNKNOWN'
 duplicate=' [DUPLICATE sample used in prior fixed slots]' if case['sample_index']==7266 else ''
 d.text((35,20),f"FIXED PRETRIAL CASE {case['case_number']:02d} | sample {case['sample_index']} | target {case['target_state']}"+duplicate,fill='#0f172a',font=f)
 d.text((35,66),family+' | SOURCE DEV exposed/reused; not blind',fill='#334155',font=small)
 d.text((35,105),f"row: {case['row_id']} | original q={case['original_q']} | L={case['original_L']:.4f} | active slot {case['active_slot']}",fill='#334155',font=tiny)
 truth=case['target_state'];oldok=old['prediction']==truth;newok=new['prediction']==truth
 change='RECOVERED' if not oldok and newok else 'DAMAGED' if oldok and not newok else 'BOTH CORRECT' if newok else 'BOTH WRONG'
 d.text((35,148),f"BASELINE V239 CONTEXT: {old['prediction']} z={old['selected_family_logit']:+.6f} | NEW TREAT: {new['prediction']} z={new['selected_family_logit']:+.6f} | {change}",fill='#166534' if newok else '#9f1239',font=f)
 # Display resizes/markers only. Original PNG copies remain encoded-byte exact.
 c=context.resize((560,560),Image.Resampling.BILINEAR);cd=ImageDraw.Draw(c);box=case['original_box'];addresses=[]
 valid_box=isinstance(box,list) and len(box)==4 and all(v is not None and math.isfinite(v) for v in box) and box[2]>box[0] and box[3]>box[1]
 for k,q in enumerate(case['q8_original']):
  valid=case['original_available8'][k]
  if not valid_box or not valid or q is None or len(q)!=2 or not all(v is not None and math.isfinite(v) for v in q):continue
  u=(q[0]-box[0])/(box[2]-box[0])*224-.5;v=(q[1]-box[1])/(box[3]-box[1])*224-.5
  # Edge clamp is display only, never a changed model query.
  px=(min(223,max(0,u))+.5)*2.5-.5;py=(min(223,max(0,v))+.5)*2.5-.5;color='#facc15' if k==case['active_slot'] else '#06d6ef'
  draw_cross(cd,(px,py),color,3,10);cd.text((min(525,px+7),min(530,max(0,py-26))),str(k),fill=color,font=tiny);addresses.append({'slot':k,'context_pixel_address':[u,v],'display_clamped':u<0 or u>223 or v<0 or v>223})
 board.paste(c,(35,245));d.text((35,208),'Original q8 context224 DISPLAY (F32 quantized)',fill='#334155',font=small)
 active=nodes[case['active_slot']].resize((560,560),Image.Resampling.NEAREST);ad=ImageDraw.Draw(active);draw_cross(ad,((47.5+.5)*560/96-.5,(47.5+.5)*560/96-.5),'#06d6ef',3,16);board.paste(active,(650,245));d.text((650,208),'Same original active RAW96 | cyan q=(47.5,47.5)',fill='#334155',font=small)
 lines=['New baseline is the OLD V239 TREAT,','86054 parameters; not old76098CONTROL.','No old CARD captions copied.','',f"Baseline logits: {old['logit0']:+.4f}, {old['logit1']:+.4f}",f"New TREAT logits: {new['logit0']:+.4f}, {new['logit1']:+.4f}",'Family-specific logit >=0 is E.','', 'Cyan: original q; yellow: activeq incontext.','Markers/resizes are display only.','No GT/reference/owner/pose overlay.','No new visibility labels or causal claim.','Context source F32 was already rounded','for this historical PNG display; input unchanged.']
 for j,line in enumerate(lines):d.text((1260,250+j*34),line,fill='#334155',font=small)
 d.text((35,842),'All8 same-scene original query crops (U flags preserved; no semantic state masking)',fill='#334155',font=small)
 for k,node in enumerate(nodes):
  xx=35+k*244;thumb=node.resize((180,180),Image.Resampling.NEAREST);td=ImageDraw.Draw(thumb)
  color='#06d6ef' if case['original_available8'][k] else '#94a3b8';draw_cross(td,((47.5+.5)*180/96-.5,(47.5+.5)*180/96-.5),color,2,7)
  board.paste(thumb,(xx,891));d.rectangle((xx-2,889,xx+181,1072),outline='#eab308' if k==case['active_slot'] else '#cbd5e1',width=3)
  d.text((xx,1080),f"slot{k} {'available' if case['original_available8'][k] else 'original U'}",fill='#334155',font=tiny)
  d.text((xx,1110),f"L={case['L8_original'][k]:.2f}" if case['L8_original'][k] is not None else 'L unavailable',fill='#334155',font=tiny)
 d.text((35,1165),'Fixed prior IDs only; 12 slots /11unique samples. These pictures are not representative performance evidence.',fill='#475569',font=small)
 return board,change,addresses


def run(entry_path):
 e=json.loads(read(entry_path))
 if e.get('approved_fixed12_SOURCE_savedPNG_gallery') is not True or e.get('source_only') is not True or e.get('automatic_retry') is not False:raise PermissionError('Separate Root saved-PNG gallery entry required')
 code,meta,payload=e['code_whitelist'],e['metadata_whitelist'],e['PNG_whitelist']
 checked(e['code'],code,binding(Path(__file__).absolute()));checked(e['implementation_protocol'],code)
 old={key:json.loads(checked(e[key],meta,pin)) for key,pin in PINS.items()};images=source_metadata(old['old_complete'],old['old_selection'],old['old_case_data'])
 if {ident(b) for b in payload}!={ident(b) for b in images}:raise ValueError('Only exact81 preexisting closed PNGs; no oldCARD/rawcache/sourceasset')
 fit=json.loads(checked(e['fit_complete'],meta));qa=json.loads(checked(e['fit_QA'],meta));freeze=json.loads(checked(e['freeze'],meta));report=json.loads(checked(e['source_report_complete'],meta))
 for value in (fit,qa,freeze,report):require_PASS(value)
 exact(qa['fit_complete'],e['fit_complete']);exact(report['fit_complete'],e['fit_complete']);exact(e['freeze'],fit['freeze'])
 if fit['new_optimizer_updates']!={BASE:0,TREAT:1000} or fit['head_parameters']!={BASE:86054,TREAT:86054} or fit['new_DEV_head_output_before_finalfreeze']!=0:raise ValueError('Frozen samearchitecture baseline0/newTREAT1000')
 for arm in (BASE,TREAT):
  exact(freeze['checkpoints'][arm],fit['checkpoints'][arm]);exact(freeze['logs'][arm],fit['logs'][arm]);exact(e['CSV'][arm],fit['outputs'][arm])
 if e['CSV'][BASE]['sha256']!=old['old_selection']['CSV'][1]['sha256']:raise ValueError('NewBASELINE must be exact oldV239TREAT CSV, not old76098CONTROL')
 out=safe(e['output_directory'])
 if out.parent!=H or out.name!='fixed12_gallery_v1' or out.exists():raise ValueError('Fresh owned output only')
 out.mkdir()
 try:
  cases=old['old_case_data']['cases'];selection={'completed':True,'PASS':True,'old_complete':e['old_complete'],'old_selection':e['old_selection'],'old_case_data':e['old_case_data'],'cases':[{'case_number':c['case_number'],'sample_index':c['sample_index'],'row_id':c['row_id'],'missing':c['missing']} for c in cases],'slots':12,'unique_samples':11,'unique_groups':9,'new_score_selection':False,'sealed_before_new_CSV_numeric_read_and_any_PNG_read':True}
  sel=write_json(out/'FIXED12_IDS_BEFORE_NEW_CSV_AND_PIXELS.json',selection)
  scores={arm:selected_CSV(checked(e['CSV'][arm],meta),cases) for arm in (BASE,TREAT)}
  decoded={};copied={};proofs=[]
  for c in cases:
   cp=c['cache_pixel_proof'];group=out/f"GROUP_{c['group_index']:04d}";group.mkdir(exist_ok=True)
   for k,npf in enumerate(cp['node_PNG_proofs']):
    b=npf['PNG'];key=ident(b)
    if key in decoded:continue
    im,raw=decode_bound_PNG(b,payload,(96,96));arr=np.asarray(im,dtype=np.uint8);sha=hashlib.sha256(np.ascontiguousarray(arr.transpose(2,0,1)).tobytes()).hexdigest()
    if sha!=npf['cache_CHW_uint8_sha256']:raise ValueError('Historical RAW96 CHW decoded pixels changed')
    dest=group/f'NODE_{k}_RAW96.png'
    with dest.open('xb') as f:f.write(raw)
    decoded[key]=im;copied[key]=dest.relative_to(out).as_posix();proofs.append({'source':b,'copy':binding(dest),'decoded_CHW_sha256':sha,'encoded_bytes_unchanged':True})
   b=cp['context_display_PNG'];key=ident(b)
   if key not in decoded:
    im,raw=decode_bound_PNG(b,payload,(224,224));sha=hashlib.sha256(np.asarray(im,dtype=np.uint8).tobytes()).hexdigest()
    if sha!=cp['display_HWC_uint8_sha256']:raise ValueError('Historical quantizedcontext display pixels changed')
    dest=group/'CONTEXT224_DISPLAY.png'
    with dest.open('xb') as f:f.write(raw)
    decoded[key]=im;copied[key]=dest.relative_to(out).as_posix();proofs.append({'source':b,'copy':binding(dest),'decoded_HWC_sha256':sha,'F32_context_quantization_display_only':True,'encoded_bytes_unchanged':True})
  case_data=[];table=[];thumbs=[]
  for c in cases:
   cp=c['cache_pixel_proof'];nodes=[decoded[ident(v['PNG'])] for v in cp['node_PNG_proofs']];context=decoded[ident(cp['context_display_PNG'])]
   card,change,addresses=render_card(c,scores[BASE][c['sample_index']],scores[TREAT][c['sample_index']],nodes,context)
   case_dir=out/f"CASE_{c['case_number']:02d}_SAMPLE_{c['sample_index']:04d}";case_dir.mkdir();path=case_dir/'CARD.png';card.save(path)
   item={key:c[key] for key in ('case_number','sample_index','row_id','family','target_state','target_referent','active_slot','group_index','original_q','original_L','q8_original','L8_original','original_available8','original_box','role')}
   item.update({'baseline_arm':BASE,'treatment_arm':TREAT,'baseline':scores[BASE][c['sample_index']],'treatment':scores[TREAT][c['sample_index']],'paired_change_this_trial':change,'historical_bucket_not_reapplied':True,'context_q_display_addresses':addresses,'card':binding(path),'raw_nodes':[copied[ident(v['PNG'])] for v in cp['node_PNG_proofs']],'context_display':copied[ident(cp['context_display_PNG'])]});case_data.append(item)
   table.append({'case_number':c['case_number'],'sample_index':c['sample_index'],'row_id':c['row_id'],'family':c['family'],'target':c['target_state'],'baseline_prediction':item['baseline']['prediction'],'baseline_selected_logit':item['baseline']['selected_family_logit'],'treatment_prediction':item['treatment']['prediction'],'treatment_selected_logit':item['treatment']['selected_family_logit'],'paired_change_this_trial':change,'duplicate_sample':c['sample_index']==7266,'card':path.relative_to(out).as_posix()});thumbs.append(card.resize((1000,610),Image.Resampling.LANCZOS))
  with (out/'CASES.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
  write_json(out/'CASE_DATA.json',{'completed':True,'PASS':True,'selection':sel,'cases':case_data,'slots':12,'unique_samples':11,'SOURCE_DEV_exposed_reused':True,'GT_reference_pose_owner_inputs':False,'new_model_forward':0})
  overview=Image.new('RGB',(2000,610*6),'white')
  for j,thumb in enumerate(thumbs):overview.paste(thumb,((j%2)*1000,(j//2)*610))
  overview.save(out/'FIXED12_OVERVIEW.png')
  body=['<!doctype html><meta charset="utf-8"><title>고정12 SOURCE 비교</title><style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:24px}img{max-width:100%}section{border-top:1px solid #ddd;padding:20px 0}code{overflow-wrap:anywhere}</style><h1>같은 사전 고정12 SOURCE DEV 사례</h1><p>이전 V239의12슬롯/11고유표본/9그룹을 그대로 사용했습니다. 새 점수로 사례를 선택하지 않았습니다. 새 baseline은 기존 V239 TREAT86054이며 옛76098 CONTROL과 다릅니다. SOURCE DEV는 이미 노출·재사용됐고 대표성·성능 향상·가림 원인·native gold의 증거가 아닙니다.</p><p>q/L/native/U는 원값입니다. 마커·표시 확대는 입력 변경이 아닙니다. context PNG는 기존 F32 context를 표시용으로 양자화한 이미지이며 원 F32 입력을 재생성하지 않았습니다. YCB native는 UNKNOWN이고 associated-r 교사와 literal q 상태를 구분합니다. 새 GT/reference/owner/pose overlay와 모델 forward는 없습니다.</p><p><a href="CASES.csv">사례 CSV</a> · <a href="CASE_DATA.json">사례 metadata</a> · <a href="FIXED12_OVERVIEW.png">overview</a></p>']
  for row in table:body.append(f'<section><h2>Case {row["case_number"]:02d} / sample {row["sample_index"]}</h2><p>교사 {row["target"]}; baseline {row["baseline_prediction"]} ({row["baseline_selected_logit"]:+.6f}), treatment {row["treatment_prediction"]} ({row["treatment_selected_logit"]:+.6f}); {html.escape(row["paired_change_this_trial"])}. '+('중복 sample입니다.' if row['duplicate_sample'] else '')+f'</p><img src="{html.escape(row["card"])}" alt="fixed SOURCE card"></section>')
  with (out/'PORTABLE_GALLERY_KO.html').open('x') as f:f.write('\n'.join(body))
  image_manifest=write_json(out/'IMAGE_MANIFEST.json',{'completed':True,'PASS':True,'source_PNG_proofs':proofs,'cards':[v['card'] for v in case_data],'rawcopies_encoded_and_decoded_exact':True})
  outputs=[binding(out/name) for name in ('FIXED12_IDS_BEFORE_NEW_CSV_AND_PIXELS.json','CASES.csv','CASE_DATA.json','FIXED12_OVERVIEW.png','PORTABLE_GALLERY_KO.html','IMAGE_MANIFEST.json')]+[p['copy'] for p in proofs]+[v['card'] for v in case_data]
  result={'completed':True,'PASS':True,'entry':binding(entry_path),'code':e['code'],'protocol':e['implementation_protocol'],'fit_complete':e['fit_complete'],'fit_QA':e['fit_QA'],'freeze':e['freeze'],'source_report_complete':e['source_report_complete'],'old_complete':e['old_complete'],'old_selection':e['old_selection'],'old_case_data':e['old_case_data'],'selection':sel,'CSV':e['CSV'],'slots':12,'unique_samples':11,'unique_groups':9,'closed_existing_PNGs':81,'outputs':outputs,'image_manifest':image_manifest,'same_prior_fixed_cases':True,'new_score_case_selection':False,'SOURCE_DEV_exposed_reused_not_blind':True,'rawcache_feature_checkpoint_originalRGB_GT_or_protected_reads':0,'new_model_forward_or_training':0,'new_visibility_label_or_cause_certification':False,'automatic_retry':False}
  complete=write_json(out/'COMPLETE.json',result);print(json.dumps({'completed':True,'PASS':True,'COMPLETE':complete}),flush=True)
 except BaseException:
  write_json(out/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()});raise


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)

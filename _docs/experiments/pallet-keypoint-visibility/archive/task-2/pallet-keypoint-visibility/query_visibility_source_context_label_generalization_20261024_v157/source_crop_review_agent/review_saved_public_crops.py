"""Source-only saved-crop review; metadata ID seal precedes any image bytes.
No torch/imports from training code; no models/features/weights/initializer bodies.
"""
from pathlib import Path
from collections import Counter
import json, hashlib, datetime, math, sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_source_context_label_generalization_20261024_v157'
O=V/'source_crop_review_agent'
MR=P/'query_visibility_mr6d_full_public_source_20261024_v153'
V154=P/'query_visibility_mr6d_source_supervision_bridge_20261024_v154'
V155=P/'query_visibility_independent_auxiliary_source_control_20261024_v155'
PRO=V155/'source_fit_preparation/PROTOCOL_BEFORE_SOURCE_FEATURE_OR_CHECKPOINT_BODIES.json'
PATHS={
'C35':V154/'runs/C35_C36_SEPARATE_QUERY_FIXED2000/C35_MATCHED_VISIBILITY_ONLY_CONTROL/FINAL_MR1840_OUTPUTS.npz',
'C36':V154/'runs/C35_C36_SEPARATE_QUERY_FIXED2000/C36_MATCHED_VISIBILITY_PLUS_ENTITY_AND_POINT_QUALITY/FINAL_MR1840_OUTPUTS.npz',
'C37':V155/'runs/C37_C38_INDEPENDENT_AUX_WARM_FRESH_FIXED8000/C37_WARM_VISIBILITY_INDEPENDENT_AUX_FIXED8000/FINAL_MR1840_OUTPUTS.npz',
'C38':V155/'runs/C37_C38_INDEPENDENT_AUX_WARM_FRESH_FIXED8000/C38_FRESH_VISIBILITY_INDEPENDENT_AUX_FIXED8000/FINAL_MR1840_OUTPUTS.npz'}
MAN=MR/'original_query_evidence_full230/ALL230_SAVED_CROP_EVIDENCE.json'
FACT=MR/'full230_native_DINO_v1/FEATURE_INPUT_FACTS1840_INPUT_ONLY.jsonl'
def allowed(p):
 p=Path(p).resolve(); s=str(p).lower()
 assert p.is_relative_to(P) and 'v112' not in s and 'current223' not in s and 'private' not in s,(p,'forbidden')
 return p
def read(p): return json.loads(allowed(p).read_text())
def bind(p):
 p=allowed(p); return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def check(b): assert bind(b['path'])==b; return allowed(b['path'])
def lines(p):return [json.loads(s) for s in allowed(p).read_text().splitlines() if s]
def put(p,x):allowed(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def loadmeta():
 scope=read(V/'ROOT_SOURCE_ONLY_SCOPE_BEFORE_BODIES.json'); pro=read(PRO)
 labels=lines(check(pro['source_labels']));facts=lines(check(pro['MR_inputfacts']))
 seal=read(MR/'original_query_evidence_full230/EVIDENCE_RENDER_COMPLETE.json'); recs=read(check(seal['evidence_manifest']))['records']
 packs={}
 for arm,p in PATHS.items():
  # Read only saved output metadata/logits/predictions; no features or checkpoint.
  with np.load(allowed(p),allow_pickle=False) as z:packs[arm]={k:z[k].copy() for k in ('global_index','frame_index','native_slot','query_id','frame_id','source_id','split','query_available','raw_three_logits','primary_visibility_prediction')}
  a=packs[arm]; assert np.array_equal(a['global_index'],np.arange(1840))
  assert np.array_equal(a['primary_visibility_prediction'],np.where(a['query_available'],(a['raw_three_logits'][:,0]>=0).astype(np.int8),-1))
  for g,r in enumerate(labels):
   assert g==r['global_index']==facts[g]['global_index'] and a['query_id'][g]==r['query_id']==facts[g]['query_id']
   assert a['frame_index'][g]==r['frame_index'] and a['split'][g]==r['split'] and bool(a['query_available'][g])==r['original_input_valid']
 return pro,labels,facts,seal,recs,packs

def prepare():
 pro,labels,facts,seal,recs,packs=loadmeta()
 held=[r['frame_index'] for r in recs if r['split']=='SOURCE_VALIDATION'];train=[r['frame_index'] for r in recs if r['split']=='SOURCE_DEVELOPMENT'][:12]
 assert len(held)==45 and len(train)==12 and held==sorted(held) and train==sorted(train)
 categories=[];empty=[]
 for split in ('SOURCE_DEVELOPMENT','SOURCE_VALIDATION'):
  for state in ('D','E','S','OOF','U'):
   for arm,a in packs.items():
    for answer in (0,1,-1):
     ix=[g for g,r in enumerate(labels) if r['split']==split and r['role_state']==state and int(a['primary_visibility_prediction'][g])==answer]
     cat=dict(split=split,weak_state=state,arm=arm,saved_primary={0:'D',1:'E',-1:'U'}[answer],error_category=('correct' if answer=={'D':0,'E':1}.get(state) else 'input_U' if answer==-1 else 'error') if state in ('D','E') else 'truth_not_DE_not_error_scored')
     if not ix: empty.append(cat);continue
     g=ix[0];r=labels[g];categories.append(dict(**cat,global_index=g,frame_index=r['frame_index'],native_slot=r['native_slot'],query_id=r['query_id'],source_id=r['source_id']))
 extra=sorted(set(x['frame_index'] for x in categories)-set(held+train));uniqueg=sorted(set(x['global_index'] for x in categories))
 selected=held+train+extra
 sources=[bind(V/'ROOT_SOURCE_ONLY_SCOPE_BEFORE_BODIES.json'),bind(PRO),pro['source_labels'],pro['MR_inputfacts'],bind(MR/'original_query_evidence_full230/EVIDENCE_RENDER_COMPLETE.json'),seal['evidence_manifest'],seal['code'],bind(MR/'initializer_feature_preparation/encode_full230.py')]+[bind(p) for p in PATHS.values()]
 notes=[]
 for f in selected:
  rr=labels[f*8:f*8+8]; record=recs[f];assert record['frame_index']==f
  counts={}
  for arm,a in packs.items():
   pred=a['primary_visibility_prediction'][f*8:f*8+8]; raw=a['raw_three_logits'][f*8:f*8+8];av=a['query_available'][f*8:f*8+8]
   available_ids=np.flatnonzero(av); consistent=bool(len(available_ids) and np.all(raw[available_ids,1]==raw[available_ids[0],1])); ent=(-1 if not len(available_ids) else int(raw[available_ids[0],1]>=0) if consistent else None)
   counts[arm]=dict(predicted_D=int(sum(pred==0)),predicted_E=int(sum(pred==1)),input_U=int(sum(pred==-1)),known_D_as_E=sum(int(r['role_state']=='D' and v==1) for r,v in zip(rr,pred)),known_E_as_D=sum(int(r['role_state']=='E' and v==0) for r,v in zip(rr,pred)),known_DE_input_U=sum(int(r['role_state'] in ('D','E') and v==-1) for r,v in zip(rr,pred)),entity_prediction_when_frame_shared=ent,entity_logit_frame_shared=consistent,entity_row_predictions=[int(x) for x in np.where(av,(raw[:,1]>=0).astype(np.int8),-1)],entity_truth=rr[0]['frame_entity_target'])
  notes.append(dict(frame_index=f,frame_id=record['frame_id'],source_id=record['source_id'],split=record['split'],selection=('ALL45_HELD' if f in held else 'FIRST12_TRAIN' if f in train else 'FIRST_CATEGORY_EXTRA'),query_ids=[r['query_id'] for r in rr],weak_states_native0_7=[r['role_state'] for r in rr],frame_entity_target=rr[0]['frame_entity_target'],frame_association_state=rr[0]['frame_association_state'],weak_state_counts=dict(Counter(r['role_state'] for r in rr)),available_count=sum(r['original_input_valid'] for r in rr),saved_counts=counts,source_board=record['PNG'],saved_crop_packet=record['saved_crop_packet']))
 idseal=dict(completed=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),images_read_before_this_seal=0,selection_rule='All45 held frames plus first12 TRAIN by original frame_index. Per arm/split/weakstate/saved primary D/E/U category: first global row. No score magnitudes or visual replacements. Unknown weak states never counted as D/E errors.',held45_frame_indices=held,first12TRAIN_frame_indices=train,category_extra_frame_indices=extra,selected_frame_ids=[dict(frame_index=x['frame_index'],frame_id=x['frame_id'],source_id=x['source_id']) for x in notes],category_first_rows=categories,empty_categories=empty,unique_category_global_indices=uniqueg,source_bindings=sources,weak_labels_preserved=True,no_GT_recenter_relabel_fit_or_new_forward=True)
 put(O/'ID_SEAL_BEFORE_RGB_READS.json',idseal);put(O/'SOURCE_BINDINGS.json',dict(completed=True,inputs=sources,saved_scores_original_bound_by_existing_source_fit_lineage=True,saved_npz_hashes_newly_recorded_here_not_falsely_claimed_prior_sealed=True,code=bind(__file__),provenance='Existing board local cells are exact saved rgb96 enlarged nearest x1.5. Encoder consumed local96 after preprocessing to full/central224; top scene is annotated original RGB, not target224 tensor. No exact target224 image was saved by producer; only its hash.'))
 put(O/'PERFRAME_SAVED_COUNTS_BEFORE_VISUAL_NOTES.json',notes)
 print(json.dumps(dict(held45=held,first12train=train,extra=extra,categories=len(categories),unique_category_rows=len(uniqueg),selected_frames=len(selected),idseal=bind(O/'ID_SEAL_BEFORE_RGB_READS.json'))))

def render():
 ids=read(O/'ID_SEAL_BEFORE_RGB_READS.json');notes=read(O/'PERFRAME_SAVED_COUNTS_BEFORE_VISUAL_NOTES.json');pro,labels,facts,seal,recs,packs=loadmeta()
 assert ids['images_read_before_this_seal']==0
 fnt='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(fnt,19);small=ImageFont.truetype(fnt,15);big=ImageFont.truetype(fnt,24)
 O.joinpath('contact_sheets').mkdir(exist_ok=True);outputs=[]
 groups=[('HELD45',ids['held45_frame_indices']),('FIRST12TRAIN',ids['first12TRAIN_frame_indices'])]
 for name,frames in groups:
  for start in range(0,len(frames),4):
   chunk=frames[start:start+4];canvas=Image.new('RGB',(1640,1940),'#f0f3f7');d=ImageDraw.Draw(canvas)
   d.text((20,12),f'PUBLIC SOURCE {name} | original IDs {start+1}..{start+len(chunk)} / {len(frames)}',font=big,fill='#16334c')
   d.text((20,45),'Original boards below: saved RGB96 inputs; scene overlay uses original q. Frozen zero predictions. Weak labels.',font=small,fill='#16334c')
   for j,f in enumerate(chunk):
    n=next(n for n in notes if n['frame_index']==f);r=recs[f];board=Image.open(check(r['PNG'])).convert('RGB');assert board.size==(800,820)
    x=20+(j%2)*810;y=85+(j//2)*925
    d.text((x,y),f'#{f:03d} {n["source_id"]} | labels '+','.join(n['weak_states_native0_7']),font=font,fill='#16334c')
    ct=[]
    for a,c in n['saved_counts'].items():ct.append(f'{a} E{c["predicted_E"]} D{c["predicted_D"]} U{c["input_U"]} | D>E{c["known_D_as_E"]} E>D{c["known_E_as_D"]}')
    for k,txt in enumerate(ct):d.text((x+(k%2)*400,y+26+(k//2)*20),txt,font=small,fill='#16334c')
    canvas.paste(board,(x,y+74))
   path=O/'contact_sheets'/f'{name}_{start//4+1:02d}.png';canvas.save(path);outputs.append(dict(kind=name,frame_indices=chunk,PNG=bind(path)))
 # Unique prescribed category rows; context clipped from existing board, saved query96 copied from bound packet.
 gs=ids['unique_category_global_indices']
 for start in range(0,len(gs),8):
  chunk=gs[start:start+8];canvas=Image.new('RGB',(1640,1530),'#f0f3f7');d=ImageDraw.Draw(canvas)
  d.text((20,10),f'FIRST GLOBAL ROW per split/state/arm/output category | rows {start+1}..{start+len(chunk)} / {len(gs)}',font=big,fill='#16334c')
  for j,g in enumerate(chunk):
   f,k=divmod(g,8);r=recs[f];row=labels[g];x=20+(j%2)*810;y=60+(j//2)*365
   d.text((x,y),f'g{g:04d} #{f:03d}/n{k} {row["split"].replace("SOURCE_","")} {row["source_id"]}',font=font,fill='#16334c')
   pred=' '.join(f'{a}:{ {0:"D",1:"E",-1:"U"}[int(v["primary_visibility_prediction"][g])]}' for a,v in packs.items())
   d.text((x,y+25),f'Weak={row["role_state"]} avail={row["original_input_valid"]} | {pred}',font=small,fill='#16334c')
   board=Image.open(check(r['PNG'])).convert('RGB');scene=board.crop((8,53,648,413)).resize((480,270),Image.Resampling.BILINEAR);canvas.paste(scene,(x,y+50))
   with np.load(check(r['saved_crop_packet']),allow_pickle=False) as z: rgb=z['rgb96'][k].copy();valid=z['valid96'][k].copy();idx=z['global_index'].copy()
   assert idx[k]==g and ah(rgb)==facts[g]['RGB96_HWC_sha256'] and ah(valid)==facts[g]['valid96_sha256']
   canvas.paste(Image.fromarray(rgb).resize((240,240),Image.Resampling.NEAREST),(x+510,y+50));d.text((x+510,y+295),f'exact RGB96 n{k}; x2.5 display',font=small,fill='#16334c')
   d.text((x+510,y+316),f'valid pixels {valid.mean():.1%}',font=small,fill='#16334c')
  path=O/'contact_sheets'/f'CATEGORY_FIRST_ROWS_{start//8+1:02d}.png';canvas.save(path);outputs.append(dict(kind='CATEGORY_FIRST_ROWS',global_indices=chunk,PNG=bind(path)))
 put(O/'CONTACT_SHEET_MANIFEST.json',dict(completed=True,PASS=True,idseal=bind(O/'ID_SEAL_BEFORE_RGB_READS.json'),sourcebindings=bind(O/'SOURCE_BINDINGS.json'),outputs=outputs,model_calls=0,coordinate_or_L_changes=0,held_relabels=0,exact_saved_rgb96_hashes_verified_for_category_rows=True))
 print(json.dumps(dict(completed=True,PNGs=len(outputs),categories=len(gs),manifest=bind(O/'CONTACT_SHEET_MANIFEST.json'))))
if __name__=='__main__': {'prepare':prepare,'render':render}[sys.argv[1]]()

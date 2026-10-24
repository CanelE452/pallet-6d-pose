"""Requirement amendment: explicit raw validity masks/count, preserving sealed base render."""
from pathlib import Path
import hashlib,json,traceback
import numpy as np
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent;P=O.parent.parent
PROV=O/'provenance/BASE_RENDER_BEFORE_EXPLICIT_VALIDITY'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def guard(p):
 p=Path(p);s=str(p).lower();assert p.is_absolute() and str(p).startswith(str(P)+'/')
 assert not any(x in s for x in ('v112','current223','/private','mr6d_native1840_features'))
 assert p.suffix.lower() not in ('.pt','.pth','.ckpt','.safetensors');return p
def sha(p):return hashlib.sha256(guard(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(b):p=guard(b['path']);assert sha(p)==b['sha256'];return p
def read(p):return json.loads(guard(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def move(p):
 p=Path(p);dest=PROV/p.relative_to(O);dest.parent.mkdir(parents=True,exist_ok=True)
 b=bind(p);p.rename(dest);return dict(original_path=str(p),preserved_path=str(dest),sha256=b['sha256'])
assert (O/'REVIEW_EVIDENCE_READY.json').exists()
assert not (O/'VALIDITY_AMENDMENT_STARTED.json').exists(),'Preserve prior attempt; no blind retry.'
rules=read(O/'EXPLICIT_CENTER_VALIDITY_RULES_BEFORE_AMENDMENT.json')
base_ready=read(O/'REVIEW_EVIDENCE_READY.json')
write(O/'VALIDITY_AMENDMENT_STARTED.json',dict(completed=False,code=bind(__file__),rules=bind(O/'EXPLICIT_CENTER_VALIDITY_RULES_BEFORE_AMENDMENT.json'),base_ready=bind(O/'REVIEW_EVIDENCE_READY.json'),automatic_retry=False))
try:
 base_inputs={g:read(O/f'REVIEW_INPUT_GROUP_{g}.json') for g in ('A','B')}
 source=read(verify(base_ready['source_bindings']));packets={}
 for f in source['selected_TRAIN_frames']:
  assert f['split']=='SOURCE_DEVELOPMENT'
  with np.load(verify(f['crop_packet']),allow_pickle=False) as a:valid=np.array(a['valid96'],copy=True)
  assert valid.shape==(8,96,96);packets[f['frame_index']]=valid
 moves=[]
 for name in ('REVIEW_EVIDENCE_READY.json','REVIEW_INPUT_GROUP_A.json','REVIEW_INPUT_GROUP_B.json','BLIND_CROSSREVIEW_FROM_A_BY_B.json','BLIND_CROSSREVIEW_FROM_B_BY_A.json'):
  moves.append(move(O/name))
 old_contacts=O/'contact_sheets';(PROV/'contact_sheets').parent.mkdir(parents=True,exist_ok=True);old_contacts.rename(PROV/'contact_sheets')
 records=[]
 for g in ('A','B'):
  for r in base_inputs[g]['cases']:
   case=r['case_id'];dest=O/'cases'/case;v=packets[r['frame_index']][r['native_slot']];c=v[47:49,47:49];count=int(c.sum())
   basepanel=Path(r['images']['panel']['path']);assert sha(basepanel)==r['images']['panel']['sha256']
   baseimage=Image.open(basepanel).convert('RGB');moves.append(move(basepanel));d=ImageDraw.Draw(baseimage)
   d.rectangle((1258,640,1595,822),fill='#f6f8fa')
   d.text((1270,648),f'CENTER4 VALID SAMPLES: {count}/4',font=ImageFont.truetype(FONT,18),fill='#142333')
   d.text((1270,678),'Raw 2x2 validity (white=valid)',font=ImageFont.truetype(FONT,17),fill='#142333')
   mask=Image.fromarray(c.astype(np.uint8)*255,'L').resize((80,80),Image.Resampling.NEAREST).convert('RGB')
   baseimage.paste(mask,(1310,708))
   d.text((1270,800),'Padding -> owner UNKNOWN' if count<4 else 'All 4 samples are valid',font=ImageFont.truetype(FONT,18),fill='#a00030' if count<4 else '#142333')
   baseimage.save(basepanel)
   vp=dest/'RAW_VALID96_MASK.png';cp=dest/'RAW_CENTER2X2_VALID_MASK.png'
   Image.fromarray(v.astype(np.uint8)*255,'L').save(vp);Image.fromarray(c.astype(np.uint8)*255,'L').save(cp)
   assert np.array_equal(np.asarray(Image.open(vp)),v.astype(np.uint8)*255)
   assert np.array_equal(np.asarray(Image.open(cp)),c.astype(np.uint8)*255)
   r['images']['panel']=bind(basepanel);r['images']['valid96_mask']=bind(vp);r['images']['center2x2_valid_mask']=bind(cp)
   r['center4_valid_count']=count;r['center4_valid_mask']=c.tolist();r['owner_must_be_UNKNOWN_due_to_padding']=count<4
   records.append(r)
 lookup={r['case_id']:r for r in records};contacts=[]
 for g in ('A','B'):
  group_records=base_inputs[g]['cases']
  for page,start in enumerate(range(0,195,4),1):
   chunk=group_records[start:start+4];sheet=Image.new('RGB',(3200,2000),'white')
   for j,r in enumerate(chunk):sheet.paste(Image.open(r['images']['panel']['path']).convert('RGB'),((j%2)*1600,(j//2)*1000))
   path=O/'contact_sheets'/f'GROUP_{g}_PAGE_{page:03d}.png';path.parent.mkdir(exist_ok=True);sheet.save(path)
   contacts.append(dict(review_group=g,page=page,case_ids=[r['case_id'] for r in chunk],panels=len(chunk),image=bind(path)))
  x=base_inputs[g];x['cases']=group_records;x['contact_sheets']=[r for r in contacts if r['review_group']==g]
  x['padding_rule']='If center4_valid_count<4,visible-owner observation must be UNKNOWN;blackpadding is not natural background. All390 remain.'
  write(O/f'REVIEW_INPUT_GROUP_{g}.json',x)
 sealed=read(O/'CASES_AND_PARTITIONS_BEFORE_TRAIN_RGB.json')
 for original,reviewer in [('A','B'),('B','A')]:
  ids=sealed[f'blind_crossreview_from_{original}_by_{reviewer}']
  write(O/f'BLIND_CROSSREVIEW_FROM_{original}_BY_{reviewer}.json',dict(schema='V158_BLIND_INPUT_ONLY_CROSSREVIEW',original_partition=original,
   crossreviewer=reviewer,count=20,cases=[lookup[i] for i in ids],first_reviewer_labels_exposed=False,
   selection_rule='Pre-RGB sealed first20 SHA256(query_id+V158_CROSS_REVIEW) peroriginalpartition',
   padding_rule='center4_valid_count<4 ->UNKNOWN;neverBACKGROUND from blackpadding'))
 write(O/'VALIDITY_AMENDMENT_COMPLETE.json',dict(completed=True,code=bind(__file__),rules=bind(O/'EXPLICIT_CENTER_VALIDITY_RULES_BEFORE_AMENDMENT.json'),
  cases=390,raw_masks_verified=390,center4_counts={str(k):sum(r['center4_valid_count']==k for r in records) for k in range(5)},
  invalid_center_cases_must_be_UNKNOWN=sum(r['center4_valid_count']<4 for r in records),all390_preserved=True,
  base_artifact_preservation=moves,base_contact_sheets_preserved=str(PROV/'contact_sheets'),new_held_or_private_or_model_reads=0))
 ready=base_ready;ready['review_group_A']=bind(O/'REVIEW_INPUT_GROUP_A.json');ready['review_group_B']=bind(O/'REVIEW_INPUT_GROUP_B.json')
 ready['blind_crossreview_from_A_by_B']=bind(O/'BLIND_CROSSREVIEW_FROM_A_BY_B.json');ready['blind_crossreview_from_B_by_A']=bind(O/'BLIND_CROSSREVIEW_FROM_B_BY_A.json')
 ready['explicit_center_validity_amendment']=bind(O/'VALIDITY_AMENDMENT_COMPLETE.json');ready['invalid_center_cases_must_be_UNKNOWN']=sum(r['center4_valid_count']<4 for r in records)
 ready['raw_valid96_mask_pngs']=390;ready['raw_center2x2_valid_mask_pngs']=390
 write(O/'REVIEW_EVIDENCE_READY.json',ready)
 print(json.dumps(dict(completed=True,ready=bind(O/'REVIEW_EVIDENCE_READY.json'),validity_amendment=bind(O/'VALIDITY_AMENDMENT_COMPLETE.json'))),flush=True)
except Exception as e:
 write(O/'VALIDITY_AMENDMENT_FAILED.json',dict(completed=False,error=str(e),traceback=traceback.format_exc(),automatic_retry=False));raise

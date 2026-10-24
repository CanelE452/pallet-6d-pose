"""Input-only TRAIN390 evidence rendering. No producer imports, model or feature loads."""
from pathlib import Path
import argparse, hashlib, json, traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_train_query_owner_witness_20261024_v158';O=Path(__file__).resolve().parent
S=P/'query_visibility_mr6d_full_public_source_20261024_v153'
FONT=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
READS=[]

def guard(path):
 p=Path(path);lo=str(p).lower()
 assert p.is_absolute() and str(p).startswith(str(P)+'/')
 assert not any(t in lo for t in ('v112','current223','/private','mr6d_native1840_features'))
 assert p.suffix.lower() not in ('.pt','.pth','.ckpt','.safetensors')
 return p
def sha(path): return hashlib.sha256(guard(path).read_bytes()).hexdigest()
def binding(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def verify(b):
 p=guard(b['path']);assert sha(p)==b['sha256'],str(p)
 READS.append(b);return p
def read(b):return json.loads(verify(b).read_text())
def direct(path):return json.loads(guard(path).read_text())
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def writejl(path,rows):
 with Path(path).open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')

def seal():
 assert not (O/'CASES_AND_PARTITIONS_BEFORE_TRAIN_RGB.json').exists()
 scope=direct(V/'ROOT_SCOPE_BEFORE_SOURCE_REVIEW_BODIES.json')
 protocol=read(scope['source_protocol'])
 # Selection examines only split/availability/known target fields. No held row review.
 source=verify(protocol['source_labels']);selected=[];hidden=[]
 for line in source.read_text().splitlines():
  r=json.loads(line)
  if r['split']!='SOURCE_DEVELOPMENT':continue
  if not r['original_input_valid'] or r['role_target'] not in (0,1):continue
  i=r['global_index'];assert r['role_state'] in ('D','E') and r['native_slot']==i%8
  q=r['query_xy'];assert len(q)==2 and np.isfinite(q).all() and np.isfinite(r['L']) and r['L']>0
  selected.append(dict(case_id=f'Q{i:04d}',global_index=i,frame_index=r['frame_index'],frame_id=r['id'],source_id=r['source_id'],
      query_id=r['query_id'],native_slot=r['native_slot'],query_xy=q,L=r['L'],split='SOURCE_DEVELOPMENT',
      partition_digest=hashlib.sha256((r['query_id']+'V158_OWNER_REVIEW').encode()).hexdigest()))
  hidden.append(dict(global_index=i,query_id=r['query_id'],original_role_state=r['role_state'],original_role_target=r['role_target'],
      original_entity_target=r['entity_target'],root_adjudicated_weak=r['root_adjudicated_weak'],human_gold=False,exact_real_point_gold=False))
 assert len(selected)==390 and len({r['query_id'] for r in selected})==390
 assert sum(r['original_role_state']=='D' for r in hidden)==341 and sum(r['original_role_state']=='E' for r in hidden)==49
 selected.sort(key=lambda r:r['global_index']);ordered=sorted(selected,key=lambda r:(r['partition_digest'],r['query_id']))
 group={r['query_id']:('A' if k<195 else 'B') for k,r in enumerate(ordered)}
 for r in selected:r['review_group']=group[r['query_id']]
 cross={}
 for g in ('A','B'):
  cross[g]=[r['case_id'] for r in sorted((r for r in selected if r['review_group']==g),
    key=lambda r:(hashlib.sha256((r['query_id']+'V158_CROSS_REVIEW').encode()).hexdigest(),r['query_id']))[:20]]
 rules=dict(schema='V158_TRAIN390_BLINDED_INPUT_RENDERING_RULES',case_count=390,selection='All original available known-role SOURCE_DEVELOPMENT rows; no case replacement',
    partition='SHA256(query_id+V158_OWNER_REVIEW), ascending digest/query_id; first195 A, remaining195 B',
    blinding='Review manifests/panels contain no original D/E,entity labels,GT reference,teacher reasons or model predictions/scores. Hidden ORIGINAL_ROLES_FOR_FINAL_JOIN.jsonl is root-only.',
    raw_views='Exact uint8 saved RGB96; central32 literal[32:64,32:64]; center4 literal[47:49,47:49]. Save unmodified PNG pixels, nearest display RGB96 3x,central32 6x,center2x2 40x.',
    whole_view='Original canonical RGB with selected q display overlay only; no boxes,other points or references.',
    display_overlay='Separate RGB96 3x display-copy with center cross/center4 outline; never mutate raw96/central32/center2x2.',
    contact_sheets='At most4 individual panels per sheet; groups remain separate; no image/score/label-driven ordering.',
    safety='No heldRGB/labelreview,forbidden/private access,fullDINOfeature deserialize,producer imports,model/initializer/encoder/fit/weight/PnP calls or q/L/order changes.',
    failure_policy='One render attempt; preserve STARTED/FAILED on error and stop. No blind retry.')
 write(O/'RULES_BEFORE_TRAIN_RGB_OR_CROP_NPZ_BODIES.json',rules)
 writejl(O/'ORIGINAL_ROLES_FOR_FINAL_JOIN.jsonl',hidden)
 write(O/'CASES_AND_PARTITIONS_BEFORE_TRAIN_RGB.json',dict(completed=True,TRAIN_RGB_or_crop_NPZ_hash_or_reads_so_far=0,
      code=binding(__file__),scope=binding(V/'ROOT_SCOPE_BEFORE_SOURCE_REVIEW_BODIES.json'),source_protocol=scope['source_protocol'],
      source_labels=protocol['source_labels'],rules=binding(O/'RULES_BEFORE_TRAIN_RGB_OR_CROP_NPZ_BODIES.json'),
      count=390,unique_frames=len({r['frame_index'] for r in selected}),group_counts={'A':195,'B':195},cases_original_order=selected,
      review_group_A=[r['case_id'] for r in ordered[:195]],review_group_B=[r['case_id'] for r in ordered[195:]],
      blind_crossreview_from_A_by_B=cross['A'],blind_crossreview_from_B_by_A=cross['B'],
      crossreview_selection_rule='Within each original partition,first20 ascending SHA256(query_id+V158_CROSS_REVIEW),query_id; no labels or images used'))
 print(json.dumps(dict(sealed=True,cases=390,groups={'A':195,'B':195},case_seal=binding(O/'CASES_AND_PARTITIONS_BEFORE_TRAIN_RGB.json'))),flush=True)

def text(draw,xy,value,size=22,fill='#182633'):
 draw.text(xy,str(value),font=ImageFont.truetype(str(FONT),size),fill=fill)
def put(image,xy,arr,scale=1):
 a=Image.fromarray(np.ascontiguousarray(arr),'RGB')
 if scale!=1:a=a.resize((a.width*scale,a.height*scale),Image.Resampling.NEAREST)
 image.paste(a,xy)

def panel(case,rgb,crop,valid,paths):
 im=Image.new('RGB',(1600,1000),'#f6f8fa');d=ImageDraw.Draw(im)
 text(d,(20,15),f"{case['case_id']} | {case['query_id']} | native {case['native_slot']} | group {case['review_group']}",27)
 q=case['query_xy'];text(d,(20,56),f"Original query xy=({q[0]:.6f}, {q[1]:.6f})   L={case['L']:.6f}   source={case['source_id']}",20)
 text(d,(20,93),'Whole source RGB: selected q display overlay only',21)
 whole=Image.fromarray(rgb,'RGB');wd=ImageDraw.Draw(whole);x,y=q
 wd.line((x-12,y,x+12,y),fill='#ff00ff',width=3);wd.line((x,y-12,x,y+12),fill='#ff00ff',width=3)
 whole.thumbnail((910,790),Image.Resampling.LANCZOS);im.paste(whole,(20,130))
 text(d,(960,105),'RAW saved RGB96 | nearest 3x',20);put(im,(960,140),crop,3)
 text(d,(1270,105),'RAW central32 [32:64]',19);put(im,(1270,140),crop[32:64,32:64],6)
 text(d,(1270,345),'Nearest 6x; literal pixels',17)
 text(d,(960,465),'DISPLAY COPY: center cross',20)
 over=Image.fromarray(crop,'RGB').resize((288,288),Image.Resampling.NEAREST);od=ImageDraw.Draw(over)
 od.line((120,143,167,143),fill='#00ffff',width=2);od.line((143,120,143,167),fill='#00ffff',width=2)
 od.rectangle((141,141,146,146),outline='#ff00ff',width=1);im.paste(over,(960,500))
 text(d,(1270,465),'RAW center2x2 [47:49]',19);put(im,(1310,515),crop[47:49,47:49],40)
 text(d,(1270,615),'Nearest 40x (80x80)',18)
 text(d,(1270,650),f'Center4 valid: {bool(valid[47:49,47:49].all())}',18)
 text(d,(960,830),'Raw pixel PNGs are separate and unchanged.',19)
 text(d,(960,866),'Judge visible owner only; ambiguity -> UNKNOWN.',19)
 text(d,(20,945),'Weak owner witness only. No native identity / exact keypoint / physical occlusion gold.',21)
 im.save(paths['panel'])
 Image.fromarray(crop,'RGB').save(paths['raw96']);Image.fromarray(crop[32:64,32:64],'RGB').save(paths['central32'])
 Image.fromarray(crop[47:49,47:49],'RGB').save(paths['center2x2']);over.save(paths['display96_overlay'])

def render():
 assert not (O/'STARTED.json').exists(),'Previous attempt exists; preserve it and do not blind retry.'
 case_seal=direct(O/'CASES_AND_PARTITIONS_BEFORE_TRAIN_RGB.json');assert case_seal['count']==390
 assert case_seal['code']==binding(__file__)
 write(O/'STARTED.json',dict(completed=False,case_seal=binding(O/'CASES_AND_PARTITIONS_BEFORE_TRAIN_RGB.json'),automatic_retry=False))
 try:
  scope=direct(V/'ROOT_SCOPE_BEFORE_SOURCE_REVIEW_BODIES.json');p=read(scope['source_protocol'])
  feature_complete=read(p['MR_feature_seal'])
  frames=read(feature_complete['inputs'])['frames'];cropmeta=read(feature_complete['crop_manifest'])
  assert len(frames)==230 and cropmeta['frames']==230
  crops={r['frame_index']:r['crop_packet'] for r in cropmeta['sheets']}
  facts={}
  for line in verify(p['MR_inputfacts']).read_text().splitlines():
   r=json.loads(line)
   if r['split']=='SOURCE_DEVELOPMENT':facts[r['global_index']]=r
  cases=case_seal['cases_original_order'];selected_frames=sorted({r['frame_index'] for r in cases})
  raw_cache={};crop_cache={};source_bindings=[]
  for fi in selected_frames:
   f=frames[fi];assert f['frame_index']==fi and f['split']=='SOURCE_DEVELOPMENT'
   rgb=np.load(verify(f['canonical_RGB_NPY']),allow_pickle=False)
   assert rgb.dtype==np.uint8 and list(rgb.shape)==f['raw_hw']+[3] and arrsha(rgb)==f['decoded_RGB_sha256']
   cp=crops[fi];assert Path(cp['path']).name==f'FRAME_{fi:04d}_FACTUAL96_INPUT_ONLY.npz'
   with np.load(verify(cp),allow_pickle=False) as a:
    assert set(a.files)=={'rgb96','valid96','global_index','crop_constructed'}
    packet={k:a[k] for k in a.files}
   assert packet['rgb96'].shape==(8,96,96,3) and packet['rgb96'].dtype==np.uint8 and packet['valid96'].shape==(8,96,96)
   assert np.array_equal(packet['global_index'],np.arange(fi*8,fi*8+8))
   raw_cache[fi]=rgb;crop_cache[fi]=packet
   source_bindings.append(dict(frame_index=fi,source_id=f['source_id'],split=f['split'],raw_RGB=f['canonical_RGB_NPY'],decoded_RGB_sha256=f['decoded_RGB_sha256'],raw_hw=f['raw_hw'],crop_packet=cp))
  records=[]
  for case in cases:
   i=case['global_index'];k=case['native_slot'];fi=case['frame_index'];f=facts[i];packet=crop_cache[fi]
   assert case['query_id']==f['query_id'] and case['query_xy']==f['query_xy'] and case['L']==f['L']
   assert f['native_continuous_valid'] and f['feature_constructed'] and packet['crop_constructed'][k]
   rgb96=packet['rgb96'][k];valid96=packet['valid96'][k]
   assert arrsha(rgb96)==f['RGB96_HWC_sha256'] and arrsha(rgb96.transpose(2,0,1))==f['RGB96_CHW_sha256']
   assert arrsha(valid96)==f['valid96_sha256']
   dest=O/'cases'/case['case_id'];dest.mkdir(parents=True)
   paths={key:dest/name for key,name in [('panel','REVIEW_PANEL.png'),('raw96','RAW_RGB96.png'),('central32','RAW_CENTRAL32.png'),('center2x2','RAW_CENTER2X2.png'),('display96_overlay','DISPLAY96_CENTER_OVERLAY.png')]}
   panel(case,raw_cache[fi],rgb96,valid96,paths)
   # Verify saved raw PNG bytes decode exactly to the bound source arrays.
   for key,a in [('raw96',rgb96),('central32',rgb96[32:64,32:64]),('center2x2',rgb96[47:49,47:49])]:assert np.array_equal(np.asarray(Image.open(paths[key]).convert('RGB')),a)
   rec={k:v for k,v in case.items() if k!='partition_digest'}
   rec['images']={key:binding(path) for key,path in paths.items()};rec['raw_RGB96_array_sha256']=arrsha(rgb96)
   rec['central32_array_sha256']=arrsha(rgb96[32:64,32:64]);rec['center2x2_array_sha256']=arrsha(rgb96[47:49,47:49]);rec['center4_valid']=bool(valid96[47:49,47:49].all())
   records.append(rec)
  lookup={r['case_id']:r for r in records};contacts=[]
  for group in ('A','B'):
   ids=case_seal['review_group_'+group];group_records=[lookup[i] for i in ids]
   for page,start in enumerate(range(0,195,4),1):
    chunk=group_records[start:start+4];sheet=Image.new('RGB',(3200,2000),'white')
    for j,r in enumerate(chunk):sheet.paste(Image.open(r['images']['panel']['path']).convert('RGB'),((j%2)*1600,(j//2)*1000))
    path=O/'contact_sheets'/f'GROUP_{group}_PAGE_{page:03d}.png';path.parent.mkdir(exist_ok=True);sheet.save(path)
    contacts.append(dict(review_group=group,page=page,case_ids=[r['case_id'] for r in chunk],panels=len(chunk),image=binding(path)))
   write(O/f'REVIEW_INPUT_GROUP_{group}.json',dict(schema='V158_BLINDED_OWNER_REVIEW_INPUT_ONLY',review_group=group,count=195,
     annotation_target='Coarse visible owner at literal RGB96 center2x2[47:49],with raw96,central32 and wholeframe context. No native/keypoint/occlusion gold.',
     owner_states=['TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED','UNKNOWN'],
     cases=group_records,contact_sheets=[r for r in contacts if r['review_group']==group]))
  for original,reviewer in [('A','B'),('B','A')]:
   ids=case_seal[f'blind_crossreview_from_{original}_by_{reviewer}']
   write(O/f'BLIND_CROSSREVIEW_FROM_{original}_BY_{reviewer}.json',dict(schema='V158_BLIND_INPUT_ONLY_CROSSREVIEW',original_partition=original,
     crossreviewer=reviewer,count=20,selection_rule='SHA256(query_id+V158_CROSS_REVIEW) ascending within original partition; first20',
     cases=[lookup[i] for i in ids],first_reviewer_labels_exposed=False))
  write(O/'SOURCE_BINDINGS_AND_ARRAY_PARITY.json',dict(completed=True,source_protocol=scope['source_protocol'],feature_complete=p['MR_feature_seal'],
    source_input_manifest=feature_complete['inputs'],crop_manifest=feature_complete['crop_manifest'],inputfacts=p['MR_inputfacts'],selected_TRAIN_frames=source_bindings,
    all390_raw_png_pixels_verified=True,selected_query_xy_L_unchanged=True,full_DINO_feature_files_hashed_or_deserialized=0,
    held_RGB_or_crop_NPZ_hash_or_reads=0,model_calls=0,read_bindings=READS))
  write(O/'REVIEW_EVIDENCE_READY.json',dict(completed=True,ready_for_fresh_blinded_owner_review=True,cases=390,groups={'A':195,'B':195},
    selected_TRAIN_frames=len(selected_frames),individual_panels=390,raw96_pngs=390,central32_pngs=390,center2x2_pngs=390,
    contact_sheets=len(contacts),max_panels_per_contact_sheet=4,
    case_seal=binding(O/'CASES_AND_PARTITIONS_BEFORE_TRAIN_RGB.json'),source_bindings=binding(O/'SOURCE_BINDINGS_AND_ARRAY_PARITY.json'),
    review_group_A=binding(O/'REVIEW_INPUT_GROUP_A.json'),review_group_B=binding(O/'REVIEW_INPUT_GROUP_B.json'),
    blind_crossreview_from_A_by_B=binding(O/'BLIND_CROSSREVIEW_FROM_A_BY_B.json'),blind_crossreview_from_B_by_A=binding(O/'BLIND_CROSSREVIEW_FROM_B_BY_A.json'),
    root_only_original_roles=binding(O/'ORIGINAL_ROLES_FOR_FINAL_JOIN.jsonl'),code=binding(__file__),
    original_roles_GT_reference_and_model_scores_exposed_in_panels_or_review_input=False,annotation_or_model_calls=0,
    forbidden_or_held_access=0,automatic_fit_admission=False))
  print(json.dumps(dict(completed=True,ready=binding(O/'REVIEW_EVIDENCE_READY.json'),groups={'A':195,'B':195},panels=390,contact_sheets=len(contacts))),flush=True)
 except Exception as e:
  write(O/'FAILED.json',dict(completed=False,error=str(e),traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=('seal','render'));args=a.parse_args()
 if args.stage=='seal':seal()
 else:render()

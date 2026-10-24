"""Freeze RGB-only catalog/96 assignments first, validate selected pixels second."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,re
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
Y=P/'query_visibility_public_real_20261022_v63/ycb'
ROOTDESIGN=V/'ROOT_DATA_DESIGN_BEFORE_CORPUS_RGB.json';ROOTSHA='4fbcff4606e0f2d437b8111d9081f50ef797e1503146928b273192a237579998'
INV=Y/'SELECTED_FILES_SHA256.jsonl';SOURCE=Y/'SOURCES_AND_LABELS.json'
TRAIN_AZ=[0,1,3,4,6,7,9,10];VAL_AZ=[2,5,8,11]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def jsonl(p,rs):
 with Path(p).open('x')as f:
  for r in rs:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
def freeze():
 assert sha(ROOTDESIGN)==ROOTSHA
 design=read(ROOTDESIGN);assert design['base_scenes']==96 and not design['current_allowed']and not design['fit_allowed_in_this_data_phase']
 provenance=read(SOURCE)
 protocol=dict(completed=True,metadata_only_before_selected_RGB=True,root_design=bind(ROOTDESIGN),code=bind(__file__),background_math=bind(H/'background_math.py'),inventory=bind(INV),source_provenance=bind(SOURCE),
  catalog_rule='Only relative_path matching data/train_real/SSSSSS/rgb/FFFFFF.png from cached SHA inventory; no instance/mask/depth/pose/score/label used. Original file bytes and hashes not recalculated until selected96 validation.',
  fold_rule='TRAIN scene0..7; VALIDATION scene8..9; same sourceframe/SHA prohibited across folds. No BOPtest. Catalog all cached frames, no quality filter.',
  ordering='base_index=(((azimuth_index*2+elevation_index)*2+aspect_index)*2+appearance_index), azimuthouter/appearanceinner; indexes0..95. TRAIN azindices0,1,3,4,6,7,9,10;VALIDATION2,5,8,11.',
  selection='For each base in canonicalorder, rank not-yet-used RGB frames in its fold by SHA256(20261023|V99_PUBLIC_YCB_BACKGROUND_V1|base_index|background_id|image_sha256), take lexicographicmin;64TRAIN+32VAL without replacement. Same chosen background for all3variants; no outcome-based reroll.',
  expected_raw_hw=[480,640],expected_canvas_hw=[640,640],resize='Literal FP64 gain=max(640/W,640/H); ceil resized W/H, at least640; Pillow12.2.0 RGB BILINEAR reducing_gap=None; integer center crop left/top=floor((resized-640)/2), no pad. Pure helper is authoritative.',
  exposure='Source annotation definitions were known from prior work; this metadata selection reads only RGB inventory filepath/SHA/size and source provenance. Not a blind-source claim.',
  raw_RGB_copied=False,source_masks_labels_selection=False,current_model_fit_PnP=0)
 write(H/'BACKGROUND_PROTOCOL_BEFORE_RGB.json',protocol)
 catalog=[]
 for row in rows(INV):
  m=re.fullmatch(r'data/train_real/(\d{6})/rgb/(\d{6})\.png',row['relative_path'])
  if m is None:continue
  scene,frame=map(int,m.groups());assert 0<=scene<=9
  path=Y/row['relative_path'];assert path.is_file()and path.stat().st_size==row['size_bytes']
  catalog.append(dict(background_id=f'YCB_BG_{scene:06d}_{frame:06d}',scene_id=scene,frame_id=frame,source_split='TRAIN'if scene<8 else'VALIDATION',image=dict(path=str(path),sha256=row['sha256']),source_archive_crc32=row['crc32_official_archive'],file_bytes=row['size_bytes'],image_format='PNG_RGB_UINT8',raw_hw_expected=[480,640],official_source='BOP train_real YCB-Video',source_repo=provenance['repo_id'],source_commit=provenance['resolved_commit']))
 catalog.sort(key=lambda r:(r['scene_id'],r['frame_id']))
 assert len(catalog)==1000 and Counter(r['source_split']for r in catalog)=={'TRAIN':800,'VALIDATION':200}
 assert len({r['background_id']for r in catalog})==1000
 train={r['image']['sha256']for r in catalog if r['source_split']=='TRAIN'};val={r['image']['sha256']for r in catalog if r['source_split']=='VALIDATION'};assert not train&val
 jsonl(H/'YCB_RGB_CATALOG_INPUT_ONLY.jsonl',catalog)
 used=set();assign=[]
 for az in range(12):
  for el in range(2):
   for aspect in range(2):
    for appearance in range(2):
     i=(((az*2+el)*2+aspect)*2+appearance);fold='TRAIN'if az in TRAIN_AZ else'VALIDATION'
     eligible=[r for r in catalog if r['source_split']==fold and r['background_id']not in used]
     def key(r):return hashlib.sha256('|'.join(map(str,[20261023,'V99_PUBLIC_YCB_BACKGROUND_V1',i,r['background_id'],r['image']['sha256']])).encode()).hexdigest()
     choice=min(eligible,key=key);used.add(choice['background_id'])
     assign.append(dict(base_index=i,base_id=f'V99_BASE_{i:03d}',azimuth_index=az,azimuth_offset_degrees=az*30,elevation_index=el,elevation_offset_degrees=[-10,10][el],aspect_index=aspect,appearance_index=appearance,source_split=fold,split=fold,fold='TRAIN'if fold=='TRAIN'else'VAL',background_id=choice['background_id'],image=choice['image'],image_format=choice['image_format'],raw_hw_expected=choice['raw_hw_expected'],canvas_hw=[640,640],ranking_sha256=key(choice),same_background_all_variants=['BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM'],background_catalog=bind(H/'YCB_RGB_CATALOG_INPUT_ONLY.jsonl')))
 assert len(assign)==len(used)==96 and Counter(r['source_split']for r in assign)=={'TRAIN':64,'VALIDATION':32}
 jsonl(H/'BACKGROUND_ASSIGNMENTS96_INPUT_ONLY.jsonl',assign)
 write(H/'CATALOG_SELECTION_SEAL_BEFORE_RGB.json',dict(completed=True,PASS=True,protocol=bind(H/'BACKGROUND_PROTOCOL_BEFORE_RGB.json'),catalog=bind(H/'YCB_RGB_CATALOG_INPUT_ONLY.jsonl'),assignments=bind(H/'BACKGROUND_ASSIGNMENTS96_INPUT_ONLY.jsonl'),counts=dict(Counter(r['source_split']for r in assign)),unique_backgrounds=96,RGB_decode_before_lock=0,no_source_frame_or_fileSHA_crossfold=True,raw_RGB_copied=False,current_models_fit_PnP=0))
 print(json.dumps(dict(metadata_lock=bind(H/'CATALOG_SELECTION_SEAL_BEFORE_RGB.json'))))
def validate():
 import numpy as np
 import PIL
 from PIL import Image
 from background_math import prepare_background
 lock=read(H/'CATALOG_SELECTION_SEAL_BEFORE_RGB.json');assert lock['PASS']and PIL.__version__=='12.2.0'
 for name in ['protocol','catalog','assignments']:assert sha(lock[name]['path'])==lock[name]['sha256']
 validation=[]
 for r in rows(lock['assignments']['path']):
  path=Path(r['image']['path']);assert sha(path)==r['image']['sha256']
  with Image.open(path)as image:
   assert image.format=='PNG'and image.mode=='RGB';hw=[image.height,image.width];assert hw==r['raw_hw_expected'];image.load();rgb=np.array(image,dtype=np.uint8,copy=True)
  canvas,facts=prepare_background(rgb,tuple(r['canvas_hw']))
  validation.append(dict(base_index=r['base_index'],background_id=r['background_id'],image=r['image'],source_split=r['source_split'],actual_format='PNG_RGB_UINT8',actual_raw_hw=hw,**facts))
  del canvas,rgb
 assert len(validation)==96
 jsonl(H/'SELECTED96_RGB_VALIDATION_AND_TRANSFORM_HASHES.jsonl',validation)
 write(H/'CATALOG_SEAL.json',dict(completed=True,PASS=True,metadata_selection_lock=bind(H/'CATALOG_SELECTION_SEAL_BEFORE_RGB.json'),background_math=bind(H/'background_math.py'),code=bind(__file__),catalog=bind(H/'YCB_RGB_CATALOG_INPUT_ONLY.jsonl'),assignments=bind(H/'BACKGROUND_ASSIGNMENTS96_INPUT_ONLY.jsonl'),selected96_validation=bind(H/'SELECTED96_RGB_VALIDATION_AND_TRANSFORM_HASHES.jsonl'),actual_selected_RGB_decode=96,source_selected_fileSHA_mismatches=0,all_raw_hw=[480,640],all_canvas_hw=[640,640],Pillow=PIL.__version__,numpy=np.__version__,copied_or_saved_RGB_arrays=0,source_masks_or_labels_used=0,source_backbone_head_fit_current_PnP=0))
 print(json.dumps(dict(completed=True,seal=bind(H/'CATALOG_SEAL.json'))))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['freeze','validate'],required=True);args=ap.parse_args();freeze()if args.stage=='freeze'else validate()

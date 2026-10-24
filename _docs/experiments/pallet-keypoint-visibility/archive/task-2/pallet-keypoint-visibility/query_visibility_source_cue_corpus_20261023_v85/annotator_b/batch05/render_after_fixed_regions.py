from pathlib import Path
from PIL import Image,ImageDraw
import json,hashlib
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');R=P/'query_visibility_source_cue_corpus_20261023_v85';D=R/'annotator_b/batch05'
def b(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
rows=json.loads((R/'ANNOTATOR_B_BATCH05_BEFORE_NEW_RGB.json').read_text())['rows'];idx=json.loads((D/'ALL5_BEFORE_SEALED_INDEX.json').read_text())['pairs'];out=[];assert (D/'AFTER_READ_ENTRY_AFTER_ALL5_BEFORE_SEALS.json').exists()
for row,a in zip(rows,idx):
 pd=Path(a['pair_dir']);assert row['pair_id']==a['pair_id'];assert b(pd/'BEFORE_SEAL.json')==a['before_seal'];before=json.loads((pd/'BEFORE_CUES.json').read_text());imgs={}
 for side in ['before','after']:
  z=row[side+'_RGB'];p=Path(z['path']);assert b(p)['sha256']==z['sha256'];imgs[side]=Image.open(p).convert('RGB');assert imgs[side].size==(640,480)
 contact=Image.new('RGB',(1200,560),(24,24,24));dr=ImageDraw.Draw(contact);crops=[]
 for k,e in enumerate(before['slots']):
  ox=(k%3)*400+4;oy=(k//3)*280+6;dr.text((ox,oy),e['part_slot'],fill=(255,255,255));q=e['reference_xy']
  if q is None:dr.multiline_text((ox,oy+35),'NULL before reference\nAfter cannot invent a reference',fill=(240,220,100),spacing=7);continue
  xy=[q[0]-32,q[1]-32,q[0]+32,q[1]+32];entry={'part_slot':e['part_slot'],'reference_xy_unchanged':q,'uncertainty_unchanged':e['uncertainty_bbox_xyxy_half_open'],'native_crop_xyxy':xy,'raw_crops':{}}
  for j,side in enumerate(['before','after']):
   raw=imgs[side].crop(xy)
   if side=='after':rp=pd/(e['part_slot']+'_AFTER_UNMARKED_NATIVE64.png');raw.save(rp)
   else:rp=pd/(e['part_slot']+'_BEFORE_UNMARKED_NATIVE64.png')
   z=raw.resize((192,192),Image.Resampling.NEAREST);dd=ImageDraw.Draw(z);box=e['uncertainty_bbox_xyxy_half_open'];bb=[(box[0]-xy[0])*3,(box[1]-xy[1])*3,(box[2]-xy[0])*3,(box[3]-xy[1])*3];dd.rectangle(bb,outline=(255,235,0),width=2);contact.paste(z,(ox+j*198,oy+40));dr.text((ox+j*198,oy+22),side.upper(),fill=(245,245,245));entry['raw_crops'][side]=b(rp)
  dr.text((ox,oy+237),f'r={q}; unchanged region',fill=(240,240,240));crops.append(entry)
 cp=pd/'AFTER_PAIRED_FIXED_REGIONS_CONTACT.png';contact.save(cp);r={'completed':True,'pair_id':row['pair_id'],'before_seal':b(pd/'BEFORE_SEAL.json'),'before_annotation':b(pd/'BEFORE_CUES.json'),'source_before_RGB':row['before_RGB'],'source_after_RGB':row['after_RGB'],'all_fixed_regions_without_registration':crops,'contact':b(cp),'labels_not_chosen_by_renderer':True,'raw_model_mask_sourceq_reads':0,'code':b(Path(__file__))};f=pd/'AFTER_PAIRED_RENDER_RECEIPT.json';assert not f.exists();f.write_text(json.dumps(r,indent=2)+'\n');out.append({'pair_id':row['pair_id'],'pair_dir':str(pd),'render_receipt':b(f),'contact':str(cp)})
f=D/'AFTER_PAIRED_RENDER_INDEX.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(b(f))

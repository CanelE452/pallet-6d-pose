from pathlib import Path
import json,hashlib,math
from PIL import Image,ImageDraw
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');R=P/'query_visibility_paired_cue_20261023_v84';D=R/'reviewer_b/after_review'
def b(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
policy=D/'AFTER_REVIEW_POLICY_BEFORE_RGB.json';assert policy.exists();union=R/'ROOT_BEFORE_REFERENCE_UNION_LOCK.json';assert b(union)['sha256']=='ae6e71a2a85733bcb7ed7081e3317a04d0a612d3e69fd27d51753d7d01e4dd4a';u=json.loads(union.read_text());m=json.loads((R/'FIXED_ONE_PAIR_METADATA_BEFORE_RGB.json').read_text());imgs={}
for side in ['before','after']:
 z=m['raw_files'][side]['image'];p=Path(z['path']);assert b(p)['sha256']==z['sha256'];imgs[side]=Image.open(p).convert('RGB');assert imgs[side].size==(640,480)
rows=[];panels=[]
for a in u['proposals']:
 name=a['union_id'];q=a['reference_xy_unchanged'];unc=a['uncertainty_bbox_xyxy_half_open_unchanged'];ox=int(math.floor(q[0]))-48;oy=int(math.floor(q[1]))-48;xy=[ox,oy,ox+96,oy+96];panel=Image.new('RGB',(800,430),(24,24,24));pd=ImageDraw.Draw(panel);pd.text((10,8),f'{name}  unchanged r={q}  unc={unc}',fill=(255,255,255));files={}
 for j,side in enumerate(['before','after']):
  raw=imgs[side].crop(xy);rp=D/(name+'_'+side+'_UNMARKED_NATIVE96.png');raw.save(rp);z=raw.resize((384,384),Image.Resampling.NEAREST);dz=ImageDraw.Draw(z);bb=[(unc[0]-ox)*4,(unc[1]-oy)*4,(unc[2]-ox)*4,(unc[3]-oy)*4];dz.rectangle(bb,outline=(255,235,0),width=2);px=(q[0]-ox)*4;py=(q[1]-oy)*4;dz.line([(px-16,py),(px-8,py)],fill=(255,70,70),width=2);dz.line([(px+8,py),(px+16,py)],fill=(255,70,70),width=2)
  zp=D/(name+'_'+side+'_FIXED_REGION_4X_NEAREST.png');z.save(zp);panel.paste(z,(8+j*400,36));pd.text((8+j*400,22),side.upper(),fill=(240,240,240));files[side]={'raw':b(rp),'marked':b(zp)}
 pp=D/(name+'_BEFORE_AFTER_FIXED_PANEL.png');panel.save(pp);panels.append(panel);rows.append({'union_id':name,'reference_xy_unchanged':q,'uncertainty_bbox_unchanged':unc,'native_crop_bounds_xyxy':xy,'padding':'PIL crop black only beyond raw image; unchanged coordinate display, not registration','files':files,'panel':b(pp)})
contact=Image.new('RGB',(1600,1720),(16,16,16))
for i,p in enumerate(panels):contact.paste(p,((i%2)*800,(i//2)*430))
cp=D/'ALL8_BEFORE_AFTER_FIXED_REGIONS_CONTACT.png';contact.save(cp)
r={'completed':True,'same_native_region_no_alignment':True,'policy':b(policy),'union':b(union),'raw_images':{s:m['raw_files'][s]['image'] for s in imgs},'raw_hw':[480,640],'regions':rows,'contact':b(cp),'mask_depth_model_reads':0,'other_reviewer_after_judgments_read':False,'code':b(Path(__file__))};p=D/'PAIRED_REGION_RENDER_MANIFEST.json';assert not p.exists();p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(b(p))

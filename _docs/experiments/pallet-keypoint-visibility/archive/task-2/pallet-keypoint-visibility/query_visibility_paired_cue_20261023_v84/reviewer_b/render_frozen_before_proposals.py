from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');B=P/'query_visibility_paired_cue_20261023_v84/reviewer_b'
def b(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
f=B/'BEFORE_ONLY_CUE_PROPOSALS.json';x=json.loads(f.read_text());src=Path(x['source_before_only']['path']);assert b(src)['sha256']==x['source_before_only']['sha256'];im=Image.open(src).convert('RGB');overview=im.copy();d=ImageDraw.Draw(overview);cols=[(255,90,70),(250,230,10),(0,230,230),(255,90,240)];contact=Image.new('RGB',(832,548),(24,24,24));dd=ImageDraw.Draw(contact);files=[]
for n,(r,col) in enumerate(zip(x['proposals'],cols)):
 q=r['reference_xy'];box=r['uncertainty_bbox_xyxy'];d.rectangle(box,outline=col,width=2);d.text((q[0]+8,q[1]-12),r['proposal_id'],fill=col)
 xy=[q[0]-48,q[1]-48,q[0]+48,q[1]+48];raw=im.crop(xy);rp=B/(r['proposal_id']+'_UNMARKED_NATIVE96.png');raw.save(rp);z=raw.resize((384,384),Image.Resampling.NEAREST);dz=ImageDraw.Draw(z);bb=[(box[0]-xy[0])*4,(box[1]-xy[1])*4,(box[2]-xy[0])*4,(box[3]-xy[1])*4];dz.rectangle(bb,outline=col,width=2);dz.line([(176,192),(184,192)],fill=col,width=2);dz.line([(200,192),(208,192)],fill=col,width=2)
 # Native96 is shown at2x in contact; full4x panel retained separately.
 zp=B/(r['proposal_id']+'_FROZEN_REFERENCE_4X_NEAREST.png');z.save(zp);ox=16+(n%2)*416;oy=16+(n//2)*266;contact.paste(z.resize((192,192),Image.Resampling.NEAREST),(ox,oy));label=f"{r['proposal_id']} r={tuple(q)}\nunc={tuple(box)}\nNATIVE8 UNKNOWN\nBEFORE ONLY / AI weak evidence";dd.multiline_text((ox+200,oy+10),label,fill=(245,245,245),spacing=8)
 files.append({'proposal_id':r['proposal_id'],'native_crop_xyxy':xy,'unmarked_native96':b(rp),'marked4x':b(zp)})
op=B/'BEFORE_FROZEN_PROPOSALS_OVERVIEW.png';overview.save(op);cp=B/'BEFORE_FROZEN_PROPOSALS_CONTACT.png';contact.save(cp)
r={'completed':True,'no_geometry_changes':True,'before_only':True,'proposal_binding':b(f),'source_before':b(src),'overview':b(op),'contact':b(cp),'panels':files,'code':b(Path(__file__))};p=B/'FROZEN_PROPOSAL_RENDER_RECEIPT.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print(b(p))

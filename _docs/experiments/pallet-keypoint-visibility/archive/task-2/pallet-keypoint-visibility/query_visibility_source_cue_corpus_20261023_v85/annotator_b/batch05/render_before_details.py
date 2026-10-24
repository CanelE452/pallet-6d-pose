from pathlib import Path
from PIL import Image,ImageDraw
import json,hashlib
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');D=P/'query_visibility_source_cue_corpus_20261023_v85/annotator_b/batch05';idx=json.loads((D/'BEFORE_INSPECTION_INDEX.json').read_text());windows=[[592,144,640,208],[216,208,440,352],[416,176,640,320],[184,200,464,376],[56,160,592,480]];out=[]
def b(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
for row,xy in zip(idx,windows):
 pd=Path(row['pair_dir']);im=Image.open(row['source_before']).convert('RGB');crop=im.crop(xy);s=3 if xy[2]-xy[0]<300 else 2;z=crop.resize((crop.width*s,crop.height*s),Image.Resampling.NEAREST);p=pd/'BEFORE_DETAIL_UNMARKED.png';z.save(p);dr=ImageDraw.Draw(z)
 for x in range(((xy[0]+7)//8)*8,xy[2],8):xx=(x-xy[0])*s;dr.line([(xx,0),(xx,z.height-1)],fill=(0,150,150),width=1);dr.text((xx+1,1),str(x),fill=(255,255,0))
 for y in range(((xy[1]+7)//8)*8,xy[3],8):yy=(y-xy[1])*s;dr.line([(0,yy),(z.width-1,yy)],fill=(0,150,150),width=1);dr.text((1,yy+9),str(y),fill=(255,255,0))
 gp=pd/'BEFORE_DETAIL_NATIVE8_GRID.png';z.save(gp);out.append({'pair_id':row['pair_id'],'native_inspection_xyxy':xy,'unmarked':b(p),'grid':b(gp),'coordinates_are_inspection_windows_only':True})
f=D/'BEFORE_DETAIL_RENDER_RECEIPT.json';assert not f.exists();f.write_text(json.dumps({'before_only':True,'windows':out,'code':b(Path(__file__))},indent=2)+'\n');print(b(f))

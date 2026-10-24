from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json,hashlib
R=Path(__file__).resolve().parent
P=Image.Resampling

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def font(s):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',s)
def fitted(im,wh):
 out=Image.new('RGB',wh,(20,26,36));a=im.copy();a.thumbnail(wh,P.BILINEAR);out.paste(a,((wh[0]-a.width)//2,(wh[1]-a.height)//2));return out

def main():
 ledger=R/'RGB_REFERENCE_PROPOSAL_LEDGER.json';assert sha(ledger)=='3ec1c83052dd57fd740855925fa9e492ff2629ce7df0925e6cd3fdefc4dd3664'
 data=json.loads(ledger.read_text());cards=R/'FRAME_CARDS';cards.mkdir(exist_ok=True)
 files=[]
 for f in data['frames']:
  n=f['extension_index'];im=Image.open(f['image']).convert('RGB');marked=im.copy();d=ImageDraw.Draw(marked)
  b=f['representative_annotation']['bbox'];d.rectangle([b[0],b[1],b[0]+b[2],b[1]+b[3]],outline='#42dce0',width=3)
  for j,r in enumerate(f['reference_proposals']):
   x,y=r['reference_xy'];u=r['uncertainty_bbox_half_open_xyxy'];d.rectangle(u,outline='#ffe04b',width=2);d.line([x-9,y,x+9,y],fill='#ff477d',width=2);d.line([x,y-9,x,y+9],fill='#ff477d',width=2);d.text((x+10,y-15),f'R{j}',font=font(17),fill='#ff477d',stroke_width=1,stroke_fill='black')
  bb=f['expanded_bbox_native_xyxy'];rawcrop=im.crop(bb);markedcrop=marked.crop(bb)
  card=Image.new('RGB',(1100,390),(12,18,28));cd=ImageDraw.Draw(card)
  state='CONDITIONAL VISIBLE 2D REFERENCE' if f['reference_proposals'] else 'U: no stable associated reference'
  cd.text((12,7),f"EXT {n:02d} | subset {f['source_subset']} | image {f['image_id']} | ann {f['representative_annotation']['id']} | {state}",font=font(17),fill='white')
  cd.text((12,32),'FULLFRAME locator (cyan official bbox)       RAW expanded native crop                          MARKED same crop (yellow range / pink cue)',font=font(14),fill='#c6d0dd')
  card.paste(fitted(marked,(360,285)),(10,57));card.paste(fitted(rawcrop,(350,285)),(375,57));card.paste(fitted(markedcrop,(350,285)),(740,57))
  tail='; '.join(f"R{j}={r['reference_xy']} range={r['uncertainty_bbox_half_open_xyxy']}" for j,r in enumerate(f['reference_proposals'])) if f['reference_proposals'] else f['U_reason']
  # At most two short display lines; full evidence is in immutable JSON.
  words=tail.split();lines=[];line=''
  for w in words:
   if len(line)+len(w)>132:lines.append(line);line=''
   line+=w+' '
  lines.append(line)
  for j,s in enumerate(lines[:2]):cd.text((12,348+j*18),s,font=font(13),fill='#d6dfeb')
  dest=cards/f'{n:02d}.png';assert not dest.exists();card.save(dest);files.append(dest)
 for p in range(5):
  page=Image.new('RGB',(1100,1600),(8,12,20));d=ImageDraw.Draw(page)
  d.text((12,8),f'LOCO extension B / fixed indices {20+4*p}–{23+4*p} / AI conditional proposals; native8 UNKNOWN',font=font(17),fill='white')
  for j in range(4):page.paste(Image.open(files[4*p+j]),(0,35+j*390))
  dst=R/f'FROZEN_REFERENCE_PAGE_{p+1}.png';assert not dst.exists();page.save(dst)
 # Each immutable reference receives equal native96 context, raw beside marked.
 refs=data['references'];sheet=Image.new('RGB',(1240,4*304+38),(10,15,24));d=ImageDraw.Draw(sheet)
 d.text((10,8),'Eight frozen RGB cues / native96 raw beside marked / AI proposals, not human or native8 gold',font=font(17),fill='white')
 for k,r in enumerate(refs):
  col=k%2;row=k//2;x0=col*620;y0=38+row*304;im=Image.open(r['image']).convert('RGB');x,y=r['reference_xy'];a,b=x-48,y-48;raw=im.crop((a,b,a+96,b+96));mark=raw.copy();md=ImageDraw.Draw(mark);u=r['uncertainty_bbox_half_open_xyxy'];md.rectangle([u[0]-a,u[1]-b,u[2]-a,u[3]-b],outline='#ffe04b',width=1);md.line([43,48,53,48],fill='#ff477d',width=1);md.line([48,43,48,53],fill='#ff477d',width=1)
  d.text((x0+6,y0+2),r['cue_refID']+f' xy={r["reference_xy"]}',font=font(14),fill='white')
  sheet.paste(raw.resize((240,240),P.NEAREST),(x0+8,y0+27));sheet.paste(mark.resize((240,240),P.NEAREST),(x0+264,y0+27))
  d.text((x0+8,y0+272),'RAW native96                  MARKED fixed uncertainty',font=font(13),fill='#c8d0dc')
 dst=R/'EIGHT_FROZEN_REFERENCE_NATIVE_RAW_MARKED.png';assert not dst.exists();sheet.save(dst)
 result={'schema':'V74_EXTENSION_B_RGB_REFERENCE_REVIEW_RESULT','geometry_unchanged':sha(ledger)=='3ec1c83052dd57fd740855925fa9e492ff2629ce7df0925e6cd3fdefc4dd3664',
         'counts':data['counts'],'ledger':bind(ledger),'geometry_lock':bind(R/'RGB_REFERENCE_GEOMETRY_LOCK.json'),
         'reference_contact_sheet':bind(dst),'frame_contact_pages':[bind(R/f'FROZEN_REFERENCE_PAGE_{i+1}.png') for i in range(5)],
         'frame_cards':[bind(p) for p in files], 'model_or_mask_reads':0,'current_data_reads':0,'forward_fit_PnP':0,
         'status':'READY_FOR_ROOT_VISUAL_REVIEW_CONDITIONAL_AI_PROPOSALS_ONLY','render_code':bind(Path(__file__))}
 out=R/'ACTUAL_REFERENCE_REVIEW_RESULT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'result':bind(out),'contact_sheet':bind(dst),'counts':data['counts']}))
if __name__=='__main__':main()

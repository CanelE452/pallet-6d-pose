"""Saved raw JPEG verification and display-only contact; no annotation/model code."""
import pathlib,json,hashlib,binascii,sys,datetime
import numpy as np
from PIL import Image,ImageDraw,ImageFont,__version__ as pillow_version
P=pathlib.Path(__file__).resolve().parents[1];R=P/'raw_pilot';O=P/'raw_visual_review';O.mkdir(exist_ok=True)
def H(b):return hashlib.sha256(b).hexdigest()
def bind(f):return {'path':str(f),'sha256':H(f.read_bytes()),'bytes':f.stat().st_size}
def write(f,x):assert not f.exists();f.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
x=json.loads((R/'EXTRACTION_COMPLETE_V2.json').read_text());assert x['completed'];rows=[];images=[]
for m in x['members']:
 if m['kind']!='RGB':continue
 f=pathlib.Path(m['extracted_file']['path']);raw=f.read_bytes();assert H(raw)==m['extracted_file']['sha256']
 assert f'{binascii.crc32(raw)&0xffffffff:08x}'==m['central_directory_record']['crc32']
 with Image.open(f) as im:im.verify()
 with Image.open(f) as im:
  assert im.mode=='RGB';a=np.array(im,copy=True)
 assert a.dtype==np.uint8 and a.shape==(1024,1296,3)
 images.append(Image.fromarray(a))
 rows.append({'selection_index':m['selection_index'],'original_member_name':m['original_member_name'],'raw_file':bind(f),'CRC32_verified':True,'native_HW':[1024,1296],'mode':'RGB','decoded_RGB_bytes_SHA256':H(a.tobytes()),'native_pixel_array_readonly_for_visual_review':True})
assert len(rows)==4
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
for name,W,Hh,iw,ih,toponly in [('FIXED4_RAW_CONTACT.png',1600,1408,800,632,False),('FIXED4_TOP_HALF_NATIVE_DETAIL.png',2592,1200,1296,512,True)]:
 f=O/name;assert not f.exists();canvas=Image.new('RGB',(W,Hh),'white');draw=ImageDraw.Draw(canvas)
 title='Fixed4 TOMIE unmarked raw frames | no model, q, cue labels or training admission'
 if toponly:title='Same fixed4 | image top-half at native 1:1 pixels | display-only, NOT query crops'
 draw.text((12,10),title,fill='#162230',font=font)
 for j,(im,r) in enumerate(zip(images,rows)):
  xx=(j%2)*iw;yy=52+(j//2)*(ih+36)
  short=r['original_member_name'].split('/')
  draw.text((xx+10,yy),f"[{j}] {short[1]}/{short[-1]} | raw1296x1024",fill='black',font=small)
  shown=im.crop((0,0,1296,512)) if toponly else im.resize((iw,ih),Image.Resampling.BILINEAR)
  canvas.paste(shown,(xx,yy+30))
 draw.text((12,Hh-24),'Source: Rutinowski et al. TOMIE, Zenodo7849183, CC BY4.0. Native originals unchanged.',fill='#52606d',font=small)
 canvas.save(f)
write(O/'RAW_RGB_DECODE_AND_CONTACT_PROOF.json',{'completed':True,'source_extraction':bind(R/'EXTRACTION_COMPLETE_V2.json'),'selection':bind(R/'SELECTED_MEMBERS_LOCK_BEFORE_RGB.json'),'code':bind(pathlib.Path(__file__).resolve()),'runtime':{'Python':sys.version,'NumPy':np.__version__,'Pillow':pillow_version},'records':rows,'contacts':[bind(O/n) for n in ['FIXED4_RAW_CONTACT.png','FIXED4_TOP_HALF_NATIVE_DETAIL.png']],'contact_transform':'Fullcontact bilinear display resize. Detailcontact fixed y0:512 native pixels for all4; no q/r/target/cue location is selected or produced. Text outside original image pixels.','training_RGB_candidate':'Original unmarked JPEG members only; contacts and text are display-only.','native_mapping':'UNKNOWN','labels_created':0,'model_features_fit_current_PnP':0,'decoder_parity':'Decoded SHA is exact for this recorded Pillow runtime, not a claim of equality to another JPEG decoder.'})
print(json.dumps({'proof':bind(O/'RAW_RGB_DECODE_AND_CONTACT_PROOF.json'),'contacts':[bind(O/n) for n in ['FIXED4_RAW_CONTACT.png','FIXED4_TOP_HALF_NATIVE_DETAIL.png']]},indent=2))

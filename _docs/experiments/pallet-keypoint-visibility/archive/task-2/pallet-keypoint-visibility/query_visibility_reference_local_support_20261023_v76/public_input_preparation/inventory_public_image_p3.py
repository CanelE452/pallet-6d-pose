"""Source image/P3 only inventory. No GT/queries/masks/model imports."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from PIL import Image
import PIL

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
V63=ROOT.parent/'query_visibility_public_real_20261022_v63'
CACHE=V63/'source_cache/YCB_P3_FRAMES';RGB=V63/'ycb/data/train_real'
WEIGHTS=Path('/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt')
WEIGHTSHA='970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(p.resolve()),sha256=sha(p))
start=time.monotonic();assert PIL.__version__=='12.0.0'and sha(WEIGHTS)==WEIGHTSHA
files=sorted(CACHE.glob('*.npz'));assert len(files)==1000
out=HERE/'IMAGES_AND_EXISTING_P3_INPUT_ONLY.jsonl';assert not out.exists()
rows=[];total=0
for i,p in enumerate(files):
 scene,frame=map(int,p.stem.split('_'));image=RGB/f'{scene:06d}/rgb/{frame:06d}.png'
 with np.load(p,allow_pickle=False)as z:
  assert set(z.files)=={'p3','gain','offset','added_border','raw_hw','input_shape','image_sha256','backbone_sha256'}
  a=z['p3'];assert a.shape==(1,64,80,80)and a.dtype==np.float16 and np.isfinite(a).all()
  assert int(z['added_border'])==100 and list(z['raw_hw'])==[480,640]and list(z['input_shape'])==[544,640]
  assert float(z['gain'])==640/840 and np.array_equal(z['offset'],[0.,13.])and str(z['backbone_sha256'])==WEIGHTSHA
  image_sha=str(z['image_sha256']);rowsha=hashlib.sha256(a.tobytes()).hexdigest()
 assert sha(image)==image_sha
 rgb=np.array(Image.open(image).convert('RGB'));assert rgb.shape==(480,640,3)and rgb.dtype==np.uint8
 if i==0:np.save(HERE/'FIRST_SOURCE_RGB_LOSSLESS.npy',rgb)
 rows.append(dict(frame_index=i,id=f'YCBFRAME:{scene:06d}:{frame:06d}',scene_id=scene,frame_id=frame,split='train'if scene<8 else'val',image=str(image),image_sha256=image_sha,
  raw_hw=[480,640],RGB_pixels_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),P3_cache=bind(p),P3_values_sha256=rowsha,P3_shape=[1,64,80,80],P3_dtype='float16',
  added_border=100,canvas_shape=[680,840],input_shape=[544,640],gain=640/840,offset=[0.,13.],backbone_sha256=WEIGHTSHA))
 total+=p.stat().st_size
 if (i+1)%200==0:print(json.dumps(dict(frames=i+1,total=1000,seconds=time.monotonic()-start)),flush=True)
out.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n'for r in rows))
receipt=HERE/'PUBLIC_IMAGE_AND_P3_INVENTORY.json';assert not receipt.exists()
receipt.write_text(json.dumps(dict(completed=True,PASS=True,code=bind(Path(__file__)),manifest=bind(out),first_lossless_RGB=bind(HERE/'FIRST_SOURCE_RGB_LOSSLESS.npy'),
 frames=1000,TRAIN_frames=800,VAL_frames=200,existing_P3_bytes=total,all_image_SHA_match=True,all_P3_finite=True,affine=dict(raw_hw=[480,640],added_border=100,canvas_shape=[680,840],input_shape=[544,640],gain=640/840,offset=[0.,13.]),
 weights=bind(WEIGHTS),prior_extraction_code=bind(V63/'extract_ycb_features.py'),prior_extraction_receipts=[bind(p)for p in sorted(CACHE.glob('EXTRACTION_*.json'))],
 decoder=dict(Pillow=PIL.__version__,numpy=np.__version__),GT_query_mask_reference_current_reads=0,model_forward_fit_PnP=0,seconds=time.monotonic()-start),indent=2)+'\n')
print(json.dumps(bind(receipt)),flush=True)

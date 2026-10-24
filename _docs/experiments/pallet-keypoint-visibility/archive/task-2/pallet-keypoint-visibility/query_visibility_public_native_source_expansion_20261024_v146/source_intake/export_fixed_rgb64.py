"""Canonical lossless source RGB transport; no learned or GT operators."""
from pathlib import Path
import json,hashlib,sys,time
import numpy as np
from PIL import Image
V=Path(__file__).resolve().parent.parent
SELECTION=V/'FIXED64_SELECTION_BEFORE_NEW_SOURCE_BODIES.json'
POLICY=V/'ROOT_PUBLIC64_NATIVE_EXPANSION_POLICY_BEFORE_NEW_BODIES.json'
OUT=V/'canonical_RGB_transport_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(p):return dict(path=str(p.resolve()),sha256=sha(p))
def w(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
assert sha(SELECTION)=='cf33c72e0d5f547d0ecfcf8849fbbfd30beda80724587bfef9972e9227aa26f2'
assert sha(POLICY)=='592ab0be3c8635cb75cbbd93c3457daf463892e7dfe34c5029c63213746d247e'
assert sys.version_info[:2]==(3,13)and not OUT.exists()
s=json.loads(SELECTION.read_text());assert len(s['rows'])==64
OUT.mkdir();rows=[];started=time.monotonic()
for i,r in enumerate(s['rows']):
 assert r['frame_index']==i and r['source']=='PALLOX'
 p=Path(r['image']);assert p.suffix.lower()=='.png' and '/public_pallet/data/'in str(p)
 rawsha=sha(p)
 with Image.open(p)as im:rgb=np.asarray(im.convert('RGB')).copy()
 assert rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3
 out=OUT/f'RGB_{i:03d}.npy';np.save(out,rgb,allow_pickle=False)
 back=np.load(out,allow_pickle=False);assert np.array_equal(back,rgb)and back.tobytes()==rgb.tobytes()and sha(p)==rawsha
 rows.append(dict(frame_index=i,id=r['id'],source=r['source'],source_id=r['source_id'],split=r['split'],image=str(p),image_sha256=rawsha,decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),raw_hw=list(rgb.shape[:2]),decoder='CANONICAL_PIL_RGB_TO_BGR',canonical_RGB_NPY=b(out)))
manifest=OUT/'CANONICAL64_RGB_MANIFEST_INPUT_ONLY.json';w(manifest,dict(completed=True,frames=64,rows=rows,selection=b(SELECTION),policy=b(POLICY),RGB_only=True,GT_pose_K_depth_mask_label_body_reads=0))
w(OUT/'TRANSPORT_COMPLETE.json',dict(completed=True,PASS=True,frames=64,lossless=True,all_raw_RGB_file_SHAs_unchanged=True,all_NPY_RGB_bytes_exact=True,manifest=b(manifest),selection=b(SELECTION),policy=b(POLICY),producer=b(__file__ and Path(__file__)),runtime=dict(executable=sys.executable,python=sys.version,np=np.__version__),GT_pose_K_depth_mask_label_body_reads=0,model_calls=0,seconds=time.monotonic()-started))
print(json.dumps(dict(completed=True,PASS=True,transport=b(OUT/'TRANSPORT_COMPLETE.json'),manifest=b(manifest))))

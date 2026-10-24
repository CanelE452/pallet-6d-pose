"""Frozen source-only YCB P3 extraction; masks/poses never enter the backbone."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
import numpy as np
import cv2
import torch

ROOT=Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'runtime/ultralytics')
SRC=Path('/tmp/pallet-bi-full-execution-20261019-v1')
sys.path.insert(0,str(SRC))
from scripts.research.pallet_line_pose_v1.features import FrozenYoloFeatures,canvas_affine,BASELINE_SHA


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);args=ap.parse_args()
manifest=Path(args.manifest)
raw=manifest.read_bytes()
rows=[json.loads(s)for s in raw.decode().splitlines()if s.strip()]
images={r['image']:r for r in rows}
out=ROOT/'source_cache/YCB_P3_FRAMES';out.mkdir(exist_ok=True,parents=True)
deny=set(json.load(open(ROOT/'contract/EVALUATION_SHA256_DENYLIST.json'))['image_sha256'])
weights=Path('/home/minjae/Documents/github/pallet-pose/challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt')
assert sha(weights)==BASELINE_SHA
torch.set_num_threads(1);cv2.setNumThreads(1)
torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
net=FrozenYoloFeatures(weights,device='cuda:0')
bn=[(m,m.running_mean.clone(),m.running_var.clone(),m.num_batches_tracked.clone())for m in net.yolo.model.modules()if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
calls=[0];hook=net.yolo.model.register_forward_pre_hook(lambda *_:calls.__setitem__(0,calls[0]+1))
start=time.monotonic();new=0;skipped=0
for i,(path,row)in enumerate(sorted(images.items())):
    p=out/f"{row['scene_id']:06d}_{row['frame_id']:06d}.npz"
    if p.exists():
        with np.load(p)as z:assert str(z['image_sha256'])==row['image_sha256'] and str(z['backbone_sha256'])==BASELINE_SHA
        skipped+=1;continue
    image_sha=sha(path);assert image_sha==row['image_sha256']and image_sha not in deny
    image=cv2.imread(path);assert image is not None and image.shape[:2]==(row['height'],row['width'])
    before=calls[0];captured=net.predict(image,cpu_features=False)
    gain,offset=canvas_affine(captured['canvas_shape'],captured['input_shape'])
    temporary=p.with_suffix('.partial.npz')
    np.savez_compressed(temporary,p3=captured['p3'].cpu().numpy(),gain=gain,offset=offset,added_border=captured['added_border'],raw_hw=image.shape[:2],input_shape=captured['input_shape'],image_sha256=image_sha,backbone_sha256=BASELINE_SHA)
    temporary.replace(p);new+=1
    if new%100==0:print('YCB_P3',new,'new',skipped,'reused','seconds',round(time.monotonic()-start,2),flush=True)
for m,a,b,c in bn:assert torch.equal(m.running_mean,a)and torch.equal(m.running_var,b)and torch.equal(m.num_batches_tracked,c)
assert all(not m.training for m in net.yolo.model.modules())and all(not p.requires_grad for p in net.yolo.model.parameters())
hook.remove();net.close()
record={'source_manifest':str(manifest),'source_manifest_sha256':hashlib.sha256(raw).hexdigest(),'input_images':len(images),'new_images':new,'reused_images':skipped,'forward_calls':calls[0],'backbone_sha256':BASELINE_SHA,'BN_unchanged':True,'optimizer_updates':0,'eval_image_overlap':0,'seconds':time.monotonic()-start}
stamp=time.time_ns();(out/f'EXTRACTION_{stamp}.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)

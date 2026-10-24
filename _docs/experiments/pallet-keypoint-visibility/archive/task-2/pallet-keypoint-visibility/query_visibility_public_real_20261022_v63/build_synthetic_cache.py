"""Build source-only training tensors from preserved synthetic inputs and M4."""
from pathlib import Path
import csv,gzip,json,hashlib,time
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
RUNS=ROOT.parent.parent
V60=RUNS/'external_ray_proxy_head_20261022_v60'
V61=RUNS/'target_presence_classifier_20261022_v61'
OUT=ROOT/'source_cache'/'v2'
OUT.mkdir(exist_ok=True,parents=True)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


inputs=json.load(gzip.open(V60/'PREPARED_INPUTS_SYNTH.json.gz','rt'))
metadata=json.load(open(V60/'SYNTH_FEATURE_METADATA.json'))
cache=np.load(V60/'SYNTH_ROI_CACHE.npz')
labels=np.load(V61/'PRESENCE_LABELS.npz')
x_cache=cache['x']; y_cache=labels['y']; valid_cache=labels['valid']
assert x_cache.shape==(1003,8,65,9,9)
v_channel=x_cache[:,:,64].astype(np.float32)
prototype=(x_cache[:,:,:64].astype(np.float32)*v_channel[:,:,None]).sum(axis=(1,3,4))/np.maximum(v_channel.sum(axis=(1,2,3))[:,None],1)
results={split:{k:[]for k in ['x','y','valid','dense_y','dense_valid','boundary','source','id','native_corner']}for split in ['train','validation']}
start=time.monotonic()
for i,u in enumerate(inputs):
    assert metadata[i]['id']==u['id']
    im=Path(u['image']);base=im.parent.parent;stem=im.name.replace('_rgb.png','')
    maskpath=base/'mask_visible'/(stem+'.png')
    mask=np.asarray(Image.open(maskpath))>127
    assert list(mask.shape)==u['raw_hw']
    height,width=mask.shape
    L=metadata[i]['side']
    offset=((np.arange(9)+.5)/9-.5)*L
    dx,dy=np.meshgrid(offset,offset)
    for k,q in enumerate(u['initial_points'][:8]):
        if not valid_cache[i,k]:continue
        xx=np.floor(q[0]+dx+.5).astype(int);yy=np.floor(q[1]+dy+.5).astype(int)
        continuous=(q[0]+dx>=0)&(q[0]+dx<width)&(q[1]+dy>=0)&(q[1]+dy<height)
        v=continuous&(xx>=0)&(xx<width)&(yy>=0)&(yy<height)
        yy_safe=np.clip(yy,0,height-1);xx_safe=np.clip(xx,0,width-1)
        dense=mask[yy_safe,xx_safe].astype(np.uint8)
        assert v[4,4] and dense[4,4]==y_cache[i,k]
        px,py=np.floor(np.asarray(q)+.5).astype(int)
        boundary=(px in (0,width-1)or py in (0,height-1)or mask[py-1:py+2,px-1:px+2].sum()not in (0,9))
        d=results[u['split']]
        for key,val in dict(x=x_cache[i,k],y=y_cache[i,k],valid=True,dense_y=dense,dense_valid=v,boundary=boundary,source='SYNTH',id=u['id'],sample_id=f"SYNTH:{u['id']}:{k}",native_corner=k,context=prototype[i],group_id='SYNTH:'+u['id'],image_sha256=u['image_sha256']).items():
            if key not in d:d[key]=[]
            d[key].append(val)
    if (i+1)%200==0:print('SYNTH_DENSE',i+1,'seconds',round(time.monotonic()-start,2),flush=True)
info={}
for split,d in results.items():
    arrays={k:np.array(v)for k,v in d.items()}
    path=OUT/f'SYNTH_{split.upper()}.npz'
    np.savez_compressed(path,**arrays)
    info[split]={'path':str(path),'queries':len(arrays['y']),'positive':int(arrays['y'].sum()),'frames':len(set(arrays['id'])),'sha256':sha(path)}
assert info['train']['queries']==5673 and info['validation']['queries']==1742
assert not(set(results['train']['id'])&set(results['validation']['id']))
report={'source':'preserved synthetic only; no current evaluation file read','new_backbone_forward':0,'coordinate_changes':0,'version':2,'dense_target':'M4 nearest pixel at each frozen ROI grid point; continuous and nearest-pixel invalid pixels both masked','target_context':'input-only valid-weighted mean of64feature channels over all8unchanged query ROI; no target mask or GT used','fit_updates':0,'source_bindings':{str(p):sha(p)for p in [V60/'SYNTH_ROI_CACHE.npz',V60/'PREPARED_INPUTS_SYNTH.json.gz',V61/'PRESENCE_LABELS.npz']},'outputs':info}
(OUT/'SYNTH_CACHE_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

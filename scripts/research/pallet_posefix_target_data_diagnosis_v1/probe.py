"""TRAIN-only frozen inference. Never loads evaluation reference coordinates."""
import copy
import gc
import time
from collections import defaultdict
import numpy as np
import torch
import cv2
from . import common as C
from scripts.research.pallet_posefix_crop_completion_v2.inference import load_model
from scripts.research.pallet_posefix_full_preserve_v1.run import load_fit as load_preserve
from scripts.research.pallet_posefix_crop_completion_v2.evaluate import peak_detail
from scripts.research.pallet_sensors_submission_v1.prior_model import expectation

DIRECTIONS={0:np.array([1.,0.]),90:np.array([0.,1.]),180:np.array([-1.,0.]),270:np.array([0.,-1.])}

def corrupt(item,j,radius,angle):
    points=item['original_points'].copy();points[j]=item['original_gt'][j]+radius*DIRECTIONS[angle]
    assert abs(np.linalg.norm(points[j]-item['original_gt'][j])-radius)<1e-9
    out=C.D.with_original_points(item,points,item['valid']);out['original_points']=points
    assert np.array_equal(points[8],item['original_points'][8])
    assert np.array_equal(points[np.arange(9)!=j],item['original_points'][np.arange(9)!=j])
    return out

def controlled_occ(entry,clean,exp):
    meta=entry['metadata'];image=cv2.imread(str(C.ROOT/meta['image']['path']))
    assert C.B.P.array_sha(image)==meta['clean_RGB_sha']
    tensor=torch.from_numpy(image[:,:,::-1].copy().transpose(2,0,1)).float()[None]/255
    C.D.apply_occlusion(tensor,copy.deepcopy(meta['plans']))
    occ=np.clip(np.rint(tensor[0].numpy().transpose(1,2,0)[:,:,::-1]*255),0,255).astype(np.uint8)
    assert C.B.P.array_sha(occ)==meta['occluded_RGB_sha']
    out=dict(clean);out['rgb']=C.D.crop(occ,clean['box'],clean['original_points'],clean['valid'],exp)['rgb']
    return out

def descriptors(protocol):
    selected=defaultdict(list)
    for r in protocol['selected']:selected[r['index']].append(r['corner_id'])
    for i in range(253):
        yield dict(index=i,mode='NATURAL_CLEAN',corner=None,radius=0,angle=0)
        yield dict(index=i,mode='NATURAL_OCC',corner=None,radius=0,angle=0)
        for j in selected[i]:
            for mode,radii in [('P1',protocol['P1_radii']),('P2',protocol['P2_radii'])]:
                for radius in radii:
                    for angle in protocol['angles']:yield dict(index=i,mode=mode,corner=j,radius=radius,angle=angle)

def make_inputs(protocol,exp):
    last=None
    for d in descriptors(protocol):
        i=d['index']
        if i!=last:
            entry=C.pair(i,exp);clean=entry['pair']['CLEAN'];occ=entry['pair']['OCC'];ctrl=None;last=i
        if d['mode']=='NATURAL_CLEAN':item=clean
        elif d['mode']=='NATURAL_OCC':item=occ
        else:
            if d['mode']=='P2' and ctrl is None:ctrl=controlled_occ(entry,clean,exp)
            item=corrupt(clean if d['mode']=='P1' else ctrl,d['corner'],d['radius'],d['angle'])
        yield d,item

def denied(*args,**kwargs):raise RuntimeError('Optimizer creation/step forbidden for diagnosis')

@torch.inference_mode()
def main():
    protocol=C.read(C.DOC/'CONTROLLED_PROBE_PROTOCOL.json');C.B.L.setup('cuda');cv2.setNumThreads(1)
    assert not (C.DOC/'PREDICTIONS_FROZEN.json').exists()
    torch.optim.Optimizer.__init__=denied
    selected=defaultdict(list)
    for r in protocol['selected']:selected[r['index']].append(r['corner_id'])
    bindings={};hashes={};start=time.monotonic()
    for arm in C.ARMS:
        dest=C.RAW/f'PROBE_{arm}.json';assert not dest.exists()
        m=load_preserve() if arm=='FULL_PRESERVE' else load_model({'PRIOR1':'PRIOR1','FULL125':'FULL','FULL150':'C'}[arm])
        m.eval().requires_grad_(False);assert not any(p.requires_grad for p in m.parameters());before=C.state_hash(m.state_dict());exp=1.5 if arm=='FULL150' else 1.25
        allrows=[];pending=[];done=0
        def flush():
            nonlocal done
            if not pending:return
            xs=[x for _,x in pending];args=[torch.as_tensor(np.stack([x[k] for x in xs]),device='cuda') for k in ('rgb','points','valid')]
            z=m(*args);q=expectation(z).cpu().numpy();logits=z.cpu().numpy()
            for k,(desc,x) in enumerate(pending):
                xy=C.D.transform_points(q[k],np.linalg.inv(x['matrix']));xy[~x['valid']]=x['original_points'][~x['valid']];xy[8]=x['original_points'][8]
                channels=np.flatnonzero(x['target_valid']) if desc['corner'] is None else [desc['corner']]
                cache=dict(logits=logits[k],matrix=x['matrix'],expectation=q[k],valid=x['valid'])
                for j in channels:
                    assert j<8
                    detail=peak_detail(cache,int(j),x['original_gt'][j])
                    original=x['original_points'][j]
                    allrows.append(dict(**desc,corner_id=int(j),output_xy=xy[j],input_xy=original,output_error=float(np.linalg.norm(xy[j]-x['original_gt'][j])),
                        input_error=float(np.linalg.norm(original-x['original_gt'][j])),input_valid=bool(x['valid'][j]),
                        input_out_of_image=bool(not (0<=original[0]<640 and 0<=original[1]<480)),
                        original_center_preserved=bool(np.array_equal(xy[8],x['original_points'][8])),
                        argmax_error=detail['argmax_error'],top5_error=detail['top5_nearest_error'],support_distance=detail['nearest_output_error'],
                        probability_mass=detail['GT_radius10_mass'],normalized_entropy=detail['normalized_entropy'],peak_ratio=detail['peak1_peak2_ratio'],
                        mass_over_uniform=detail['probability_mass_over_uniform']))
            done+=len(pending);pending.clear()
            if done%256==0:print('PROBE',arm,done,'elapsed_s',round(time.monotonic()-start,1),C.B.L.gpu_guard(),flush=True)
        for desc,item in make_inputs(protocol,exp):
            pending.append((desc,item))
            if len(pending)==8:flush()
        flush();after=C.state_hash(m.state_dict());assert before==after
        assert all(p.grad is None for p in m.parameters())
        C.save(dest,allrows);bindings[arm]=C.bind(dest);hashes[arm]=dict(before=before,after=after,examples=done,rows=len(allrows))
        del m;gc.collect();torch.cuda.empty_cache();print('ARM_FROZEN',arm,done,flush=True)
    C.save(C.DOC/'PREDICTIONS_FROZEN.json',dict(models=bindings,hashes=hashes,all_before_DEV_analysis=True,no_optimizer_created=True,optimizer_steps=0,checkpoint_updates=0,
        no_eval_GT_probe_generation=True,center8_preserved=True,elapsed_seconds=time.monotonic()-start,protocol=C.bind(C.DOC/'CONTROLLED_PROBE_PROTOCOL.json')))
    print('ALL_FROZEN',flush=True)

if __name__=='__main__':main()

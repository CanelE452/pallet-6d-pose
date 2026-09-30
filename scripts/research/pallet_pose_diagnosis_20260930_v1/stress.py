"""E5 fixed 4-corner 30px direction-correlation probe (116 forwards)."""
import copy
import time
import numpy as np
import cv2
import torch
from . import run as C
from .inference import load_refiner,count_save
from scripts.research.pallet_posefix_large_error_v1 import core as CORE

def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    start,cpu=time.monotonic(),time.process_time();count=0
    rows=[r for r in C.metadata() if r['severity']=='CLEAN'];base=C.read(C.RAW/'E3_CPU_QC.json');n=len(rows)
    rng=np.random.default_rng(42);angles=rng.uniform(0,2*np.pi,n)
    independent=np.column_stack([angles[rng.permutation(n)] for _ in range(4)])
    correlated=np.repeat(angles[:,None],4,axis=1)
    for j in range(4):np.testing.assert_array_equal(np.sort(independent[:,j]),np.sort(correlated[:,j]))
    inputs={k:{} for k in ('independent','correlated')};plan={}
    for ix,r in enumerate(rows):
        i=r['id'];c=CORE.selected(base[i]);q=np.asarray(c['keypoints_xy']);valid=np.isfinite(q).all(1)&~(q==-1).all(1);valid[8]=False
        selected=np.flatnonzero(valid)[:4];assert len(selected)==4
        plan[i]=dict(corners=selected.tolist(),target='clean R0 prediction; not teacher or physical GT',vectors={})
        for kind,arr in [('independent',independent),('correlated',correlated)]:
            delta=30*np.column_stack([np.cos(arr[ix]),np.sin(arr[ix])]);p=copy.deepcopy(base[i]);z=q.copy();z[selected]+=delta
            CORE.selected(p)['keypoints_xy']=z.tolist();inputs[kind][i]=p;plan[i]['vectors'][kind]=delta.tolist()
            np.testing.assert_allclose(np.linalg.norm(delta,axis=1),30,rtol=0,atol=1e-10)
    C.save(C.RAW/'E5_STRESS_PLAN.json',plan);C.save(C.RAW/'E5_STRESS_INPUTS.json',inputs)
    C.save(C.DOC/'E5_STRESS_PROTOCOL.json',dict(locked_before_forward=True,seed=42,frames=29,corner_count=4,magnitude_original_px=30,
        justification='Existing diagnosis: TRAIN hard mostly2 corners, DEV>=4 frequent; existing20-40px bin and r30 probe. E5B produced no >20px hard inputs, so mask transfer alone cannot isolate error correlation.',
        target='CPU clean R0 coordinates, fixed restoration reference only; not actual learning target or physical GT',
        controlled='Same clean RGB/bbox/confidence/validity/other points; first4 valid corners by index; center8 retained',
        marginal_direction_match='Each corner rank uses exactly same29 angles across arms via independent permutations; per-frame correlation differs. Finite shuffled design, not IID samples with replacement.',
        maximum_image_forwards=116,code=C.bind(__file__),plan=C.bind(C.RAW/'E5_STRESS_PLAN.json'),no_sweep=True))
    pred={'identity':inputs}
    for arm in ('PRIOR1','FULL125'):
        model=load_refiner(arm);before=C.L.state_hash(model.state_dict());pred[arm]={k:{} for k in inputs}
        for j,r in enumerate(rows):
            i=r['id'];im=cv2.imread(str(C.ROOT/r['image']['path']))
            for kind in inputs:
                pred[arm][kind][i]=CORE.predict(model,im,inputs[kind][i]);count+=1
            if (j+1)%7==0:print('STRESS',arm,j+1,count,flush=True)
        assert C.L.state_hash(model.state_dict())==before
        del model
    C.save(C.RAW/'E5_STRESS_PREDICTIONS.json',pred);assert count==116
    count_save('E5_stress_infer',start,cpu,count)

if __name__=='__main__':main()

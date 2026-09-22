"""Bounded paired placements: same size/fill, different location only."""
import cv2
import numpy as np
from scripts.paper.pose_metric_closure_v1.review_physical_axis import CUBOID_EDGES
from scripts.paper.pose_metric_closure_v1.run_pose_evaluation import cuboid

EDGES=[tuple(e) for e in CUBOID_EDGES]
x=cuboid(.8,.14,.59)
assert len(EDGES)==12 and all(np.count_nonzero(x[a]!=x[b])==1 for a,b in EDGES)


def cover(points,rect):
    l,t,w,h=rect
    return (points[:,0]>=l)&(points[:,0]<l+w)&(points[:,1]>=t)&(points[:,1]<t+h)


def overlap(rect,box):
    l,t,w,h=rect
    return max(0.,min(l+w,box[2])-max(l,box[0]))*max(0.,min(t+h,box[3])-max(t,box[1]))


def plan(points,mask,box,canvas,seed):
    rng=np.random.default_rng(seed)
    p=np.array(points);mask=np.array(mask,bool);mask[8]=False
    area=float(np.prod(np.maximum(0,np.array(box)[2:]-np.array(box)[:2])))
    ratio=float(rng.choice([.5,1.,2.]));fraction=float(rng.choice([.1,.2,.3]))
    w=max(1,round(np.sqrt(area*fraction*ratio)));h=max(1,round(np.sqrt(area*fraction/ratio)))
    fill_seed=int(rng.integers(0,2**31-1));scheduled=bool(rng.random()<.5)
    result=dict(seed=int(seed),scheduled=scheduled,area_fraction=fraction,aspect=ratio,size=[w,h],fill_seed=fill_seed,
        applied=False,reason='not_scheduled',S1=None,S2=None,candidates_checked=0)
    if not scheduled:return result
    if mask.sum()<4:result['reason']='fewer_than4_supervised';return result
    low=np.ceil(np.maximum(canvas[:2],0)).astype(int)
    high=np.floor(np.minimum(canvas[2:],640)-[w,h]).astype(int)
    if (high<low).any():result['reason']='shape_outside_native_canvas';return result
    def valid(rect):
        covered=cover(p,rect)&mask;n=int(covered.sum())
        return n>=1 and mask.sum()-n>=2 and overlap(rect,box)>0
    # First32 positions: no adjacency or face information used.
    randoms=[]
    for _ in range(32):
        pos=rng.integers(low,high+1);rect=[int(pos[0]),int(pos[1]),w,h];result['candidates_checked']+=1
        if valid(rect):randoms.append(rect)
    edges=[(a,b) for a,b in EDGES if mask[a] and mask[b]]
    rng.shuffle(edges)
    if not edges:result['reason']='no_supervised_edge';return result
    structures=[]
    for k in range(32):
        a,b=edges[k%len(edges)];pair=p[[a,b]]
        lo=np.maximum(low,np.ceil(pair.max(0)-[w,h]+2)).astype(int)
        hi=np.minimum(high,np.floor(pair.min(0)-2)).astype(int)
        if (hi<lo).any():continue
        pos=rng.integers(lo,hi+1);rect=[int(pos[0]),int(pos[1]),w,h];result['candidates_checked']+=1
        if valid(rect):structures.append((rect,[a,b]))
    tol=max(1.,area*.01)
    for r2,edge in structures:
        matches=[r1 for r1 in randoms if abs(overlap(r1,box)-overlap(r2,box))<=tol]
        if not matches:continue
        r1=matches[int(rng.integers(len(matches)))];result.update(applied=True,reason='paired',S1=r1,S2=r2,edge=edge,
            bbox_overlap_difference=abs(overlap(r1,box)-overlap(r2,box)),bbox_overlap_tolerance=tol)
        for arm,rect in [('S1',r1),('S2',r2)]:
            covered=cover(p,rect)&mask
            result[arm+'_masked']=np.flatnonzero(covered).tolist()
            result[arm+'_remaining']=int(mask.sum()-covered.sum())
            result[arm+'_edge_count']=sum(bool(covered[a] and covered[b]) for a,b in EDGES)
        assert result['S2_edge_count']>=1
        return result
    result['reason']='no_paired_position_within64';return result


def fill(plan):
    w,h=plan['size'];small=np.random.default_rng(plan['fill_seed']).integers(0,256,(8,8,3),dtype=np.uint8)
    return cv2.resize(small,(w,h),interpolation=cv2.INTER_LINEAR).transpose(2,0,1)


def apply(image,plan,arm):
    output=image.copy()
    if arm!='S0' and plan['applied']:
        l,t,w,h=plan[arm];output[:,t:t+h,l:l+w]=fill(plan)
    return output

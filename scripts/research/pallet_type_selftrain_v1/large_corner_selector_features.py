"""Pure GT-free features and actual candidate hypotheses; no filesystem access."""
import copy
import cv2
import numpy as np
from .large_corner_views import VARIANTS,pose_distance

C2=[list(range(9)),[5,4,7,6,1,0,3,2,8]]
REINDEX90=[4,0,3,7,5,1,2,6,8]


def hypotheses(views):
    out=[]
    for view in VARIANTS:
        if views.get(view) is None:continue
        for reindex in [False,True]:
            p=copy.deepcopy(views[view]);perm=REINDEX90 if reindex else list(range(9))
            p['keypoints_xy']=np.asarray(p['keypoints_xy'])[perm].tolist()
            p['keypoints_conf']=np.asarray(p['keypoints_conf'])[perm].tolist()
            p['keypoints_xy'][8]=copy.deepcopy(views['R0']['keypoints_xy'][8])
            out.append(dict(name=view+(':reindex90' if reindex else ':native'),view=view,reindex=reindex,candidate=p))
    assert out and out[0]['name']=='R0:native'
    return out


def image_evidence(image):
    grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    dx=cv2.Sobel(grey,cv2.CV_32F,1,0,ksize=3);dy=cv2.Sobel(grey,cv2.CV_32F,0,1,ksize=3)
    return grey,np.sqrt(dx*dx+dy*dy)


def patches(points,grey,edge):
    h,w=grey.shape;values=[]
    for x,y in np.asarray(points)[:8]:
        if not (0<=x<w and 0<=y<h):values.extend([0.,0.,0.,0.,0.]);continue
        x,y=int(round(x)),int(round(y));a=grey[max(0,y-5):min(h,y+6),max(0,x-5):min(w,x+6)]
        e=edge[max(0,y-5):min(h,y+6),max(0,x-5):min(w,x+6)]
        values.extend([1.,float(a.mean()),float(a.std()),float(e.mean()),float(e.max())])
    return values


def features(image,views,hypothesis,evidence=None):
    """Only RGB and model predictions enter this function, including symmetry metric."""
    grey,edge=image_evidence(image) if evidence is None else evidence
    h,w=image.shape[:2];base=views['R0'];p=hypothesis['candidate']
    box=np.asarray(base['box_xyxy'],float);center=(box[:2]+box[2:])/2;size=box[2:]-box[:2]
    diagonal=max(float(np.linalg.norm(size)),1.);q=np.asarray(p['keypoints_xy'],float);b=np.asarray(base['keypoints_xy'],float)
    conf=np.asarray(p['keypoints_conf'])[:8];bc=np.asarray(base['keypoints_conf'])[:8]
    f=[float(hypothesis['view']==v) for v in VARIANTS]+[float(hypothesis['reindex'])]
    f += [w/800,h/800,*list(center/[w,h]),*list(size/[w,h]),diagonal/np.hypot(h,w),float(size[0]/max(size[1],1))]
    f += [p['score'],base['score'],*conf.tolist(),*bc.tolist(),float(conf.mean()),float(conf.min()),float(bc.mean()),float(bc.min())]
    f += ((q[:8]-center)/diagonal).ravel().tolist()+((b[:8]-center)/diagonal).ravel().tolist()
    f += ((q[:8]-b[:8])/diagonal).ravel().tolist()
    for name in VARIANTS:
        other=views.get(name)
        f += [float(other is not None),pose_distance(q,other['keypoints_xy'],C2,diagonal) if other else 0.]
    shift=np.linalg.norm(q[:8]-b[:8],axis=1)/diagonal
    f += [float(shift.mean()),float(np.median(shift)),float(shift.max()),float(shift.std()),float(grey.mean()),float(grey.std())]
    f += patches(q,grey,edge)+patches(b,grey,edge)
    result=np.asarray(f,np.float32);assert result.shape==(183,) and np.isfinite(result).all(),result.shape
    return result


def stress_image(image,predicted_box,row):
    """Fixed degraded RGB; no coordinates or annotations are consulted."""
    rng=np.random.default_rng(np.random.SeedSequence([20260921,int(row)]))
    out=np.clip(image.astype(np.float32)*.65+rng.normal(0,12,image.shape),0,255).astype(np.uint8)
    out=cv2.GaussianBlur(out,(3,3),.8)
    box=np.asarray(predicted_box,float);size=np.maximum(box[2:]-box[:2],8)
    center=rng.uniform(box[:2],box[2:]);radius=.1*size
    lo=np.maximum(np.floor(center-radius),[0,0]).astype(int);hi=np.minimum(np.ceil(center+radius),[image.shape[1],image.shape[0]]).astype(int)
    if np.all(hi>lo):out[lo[1]:hi[1],lo[0]:hi[0]]=rng.integers(0,256,3,dtype=np.uint8)
    return out

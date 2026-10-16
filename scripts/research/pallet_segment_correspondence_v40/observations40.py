from context40 import *
from features25 import actual_image
import cv2

def canonical(segments):
    v=segments[...,1,:]-segments[...,0,:];length=np.linalg.norm(v,axis=-1);n=np.stack((-v[...,1],v[...,0]),-1)/np.maximum(length[...,None],1e-12);sign=np.where((n[...,1]<0)|((n[...,1]==0)&(n[...,0]<0)),-1.,1.);n*=sign[...,None];t=np.stack((-n[...,1],n[...,0]),-1);proj=(segments*t[...,None,:]).sum(-1);ordered=np.take_along_axis(segments,np.argsort(proj,axis=-1)[...,None],axis=-2);return ordered,n,length

def extract(p):
    cv2.setNumThreads(1);detector=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD);segments=np.zeros((p['N'],128,2,2));counts=np.zeros(p['N'],int)
    for i,row in enumerate(p['rows']):
        im=actual_image(row);lines=detector.detect(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))[0]
        if lines is None:continue
        q=lines.reshape(-1,2,2).astype(float);q,n,length=canonical(q);mid=q.mean(-2);bb=p['bb'][i];pad=.2*p['diag'][i];keep=(length>=8)&(mid[:,0]>=bb[0]-pad)&(mid[:,0]<=bb[2]+pad)&(mid[:,1]>=bb[1]-pad)&(mid[:,1]<=bb[3]+pad);q=q[keep];length=length[keep];q=q[np.argsort(-length,kind='stable')[:128]];counts[i]=len(q);segments[i,:len(q)]=q
    return dict(segments=segments,counts=counts)

def match(pool,n,rho,center,diag,physical=False,envelope=None):
    if len(pool)==0:return -1
    q,sn,length=canonical(pool);mid=q.mean(-2);d=np.abs(np.einsum('sed,d->se',q,n)-rho).mean(-1);sin=np.sqrt(np.maximum(0,1-(sn@n)**2));t=np.array([-n[1],n[0]]);along=np.abs((mid-center)@t)
    if physical:
        a=q@t;b=envelope@t;overlap=np.minimum(a.max(-1),b.max())-np.maximum(a.min(-1),b.min());ok=(d<=3)&(sin<=np.sin(np.deg2rad(10)))&(overlap>0)
    else:ok=(d<=max(3,.02*diag))&(sin<=np.sin(np.deg2rad(15)))&(along<=max(12,.25*diag))
    cost=d/diag+.05*sin+.1*np.linalg.norm(mid-center,axis=-1)/diag
    return int(np.where(ok,cost,np.inf).argmin()) if ok.any() else -1

def observed_roles(p,z,support,raw):
    # Function deliberately accepts no physical GT; only predictions, images and boxes.
    seg=np.zeros((p['N'],12,2,2));valid=np.zeros((p['N'],12),bool);indices=np.full((p['N'],12),-1,int)
    top=z['both']['score'].argmax(-1)
    for i in range(p['N']):
      for r in range(12):
        j=next(j for j in range(8) if r in INC[j]);slot=list(INC[j]).index(r)*8+top[i,r];center=support['support_center'][i,j,slot]*p['diag'][i]+p['center'][i];k=match(raw['segments'][i,:raw['counts'][i]],z['both']['normal'][i,r,top[i,r]],z['both']['rho'][i,r,top[i,r]],center,p['diag'][i])
        if k>=0:seg[i,r]=raw['segments'][i,k];valid[i,r]=True;indices[i,r]=k
    return seg,valid,indices

def privileged_roles(p,raw):
    g=p['g'];seg=np.zeros((p['N'],12,2,2));valid=np.zeros((p['N'],12),bool);indices=np.full((p['N'],12),-1,int)
    for i in range(p['N']):
      for r in range(12):
        if g['length'][i,r]<=0:continue
        envelope=g['support'][i,r];k=match(raw['segments'][i,:raw['counts'][i]],g['normal'][i,r],g['rho'][i,r],envelope.mean(0),p['diag'][i],True,envelope)
        if k>=0:seg[i,r]=raw['segments'][i,k];valid[i,r]=True;indices[i,r]=k
    return seg,valid,indices

def encode(segment,valid,anchor,normal,diag,line_only=False):
    inc=np.array(INC);q,n,length=canonical(segment);q=np.repeat(q[:,inc],8,axis=2);n=np.repeat(n[:,inc],8,axis=2);length=np.repeat(length[:,inc],8,axis=2);ok=np.repeat(valid[:,inc],8,axis=2);rel=(q-anchor[:,:,None,None])/diag[:,None,None,None,None];mid=rel.mean(-2);t=np.stack((-n[...,1],n[...,0]),-1);across=(mid*n).sum(-1);along=(mid*t).sum(-1);orient=(normal*n).sum(-1)
    line=np.concatenate((ok[...,None],n,across[...,None],orient[...,None]),-1);extent=np.concatenate((rel.reshape(*rel.shape[:3],4),(length/diag[:,None,None])[...,None],along[...,None],np.abs(along)[...,None],across[...,None],np.linalg.norm(rel,axis=-1).min(-1)[...,None]),-1)
    if line_only:extent=np.zeros_like(extent)
    result=np.concatenate((line,extent),-1);result=np.where(ok[...,None],result,0).astype(np.float32);assert result.shape[-1]==14 and np.isfinite(result).all();return result

def build(p,z,a,support,raw):
    predseg,predok,predidx=observed_roles(p,z,support,raw);gtseg,gtok,gtidx=privileged_roles(p,raw);g=p['g'];ideal=g['support'];valid=g['length']>0
    inputs=dict(pred_segment=(predseg,predok),GT_match_segment=(gtseg,gtok),GT_line=(ideal,valid),GT_segment=(ideal,valid),GT_shuffled_segment=(np.roll(ideal,4,axis=1),np.roll(valid,4,axis=1)))
    out={k:encode(q,ok,a['anchor'],a['candidate_normal'],p['diag'],k=='GT_line') for k,(q,ok) in inputs.items()}
    return out,dict(pred_segment=predseg,pred_valid=predok,pred_index=predidx,GT_match_segment=gtseg,GT_match_valid=gtok,GT_match_index=gtidx)

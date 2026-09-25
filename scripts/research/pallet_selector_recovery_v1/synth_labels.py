"""Connect exact synthetic labels only AFTER prediction freeze."""
import numpy as np
import cv2
from . import common as C
from . import features as F

def exact(rows):
    with np.load(C.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz') as z:t={k:z[k] for k in z.files}
    result={}
    for r in rows:
        i=r['table_index'];assert str(t['stems'][i])==r['id']
        X=t['Xcf'][i];dims=t['dims'][i];width=np.linalg.norm(X[1]-X[0]);depth=np.linalg.norm(X[4]-X[0]);height=np.linalg.norm(X[3]-X[0])
        assert np.allclose(sorted([width,depth]),sorted([dims[0],dims[2]]),atol=1e-6) and abs(height-dims[1])<1e-6
        assert abs(width-depth)>1e-6,'Square parity undefined'
        label=0 if width>depth else 1
        fx,fy,cx,cy=t['K'][i];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
        cam=X@t['R'][i].T+t['t'][i];q=(cam@K.T);q=q[:,:2]/q[:,2:]
        centroid=t['t'][i]@K.T;center=centroid[:2]/centroid[2];q=np.vstack([q,center]);up=(X[[0,1,4,5]].mean(0)-X[[2,3,6,7]].mean(0))@t['R'][i].T;ray=-t['t'][i]
        elevation=np.degrees(np.arcsin(np.clip(up@ray/(np.linalg.norm(up)*np.linalg.norm(ray)),-1,1)))
        result[r['id']]=dict(label=label,width=width,depth=depth,dims=dims,R=t['R'][i],t=t['t'][i],Xcf=X,K=K,keypoints=q,
            elevation=elevation,area=np.prod(np.maximum(0,q[:8].max(0)-q[:8].min(0)))/np.prod(r['hw']),exact_match_error=t['match_err'][i])
    return result

def addnorm(h,g):
    p=F.production_pose(h,g['dims'],source=True)
    if not p['available']:return float('inf')
    x=F.cuboid(*g['dims'])[:8];pr=x@np.array(p['R_physical']).T+np.array(p['centroid']);out=[]
    for sym in (np.eye(3),np.diag([-1.,1.,-1.])):
        target=x@(g['R']@sym).T+g['t'];out.append(np.linalg.norm(pr-target,axis=1).mean()/np.linalg.norm(g['dims']))
    return min(out)

def main():
    lock=C.read(C.sdoc(2)/'SYNTH_PREDICTION_LOCK.json');C.verify(lock['features']);C.verify(lock['predictions'])
    start=C.now();assert lock['created_at']<start
    rows=C.read(C.sraw(2)/'SYNTH_RECORDS.json');gt=exact(rows);source={r['id']:r for r in C.read(C.ROOT/'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json')['records']}
    maxerr=0.;perfect=[]
    for k,r in enumerate(rows):
        g=gt[r['id']];tar=np.array(source[r['id']]['targets'][0]['keypoints_normalized']);mask=tar[:,2]!=0;xy=tar[:,:2]*np.array(r['hw'][::-1]);error=np.linalg.norm(g['keypoints']-xy,axis=1)[mask]
        maxerr=max(maxerr,float(error.max()) if len(error) else 0)
        assert not len(error) or error.max()<.05
        if k<32:
            q=g['keypoints'];b=np.r_[q[:8].min(0),q[:8].max(0)];pred=dict(selected_index=0,candidates=[dict(keypoints_xy=q.tolist(),keypoints_conf=[1.]*9,score=1.,box_xyxy=b.tolist())]);out=F.extract(pred,g['K'],g['dims'],r['hw']);err=addnorm(out['hypotheses'][g['label']],g);assert err<1e-4,err;perfect.append(err)
    np.savez_compressed(C.sraw(2)/'SYNTH_LABELS.npz',ids=np.array([r['id'] for r in rows]),split=np.array([r['split'] for r in rows]),
        parity=np.array([gt[r['id']]['label'] for r in rows]),elevation=np.array([gt[r['id']]['elevation'] for r in rows]),size=np.array([gt[r['id']]['area'] for r in rows]))
    stats={}
    for part in ('TRAIN','VAL','TEST'):
        ii=[r['id'] for r in rows if r['split']==part];stats[part]=dict(n=len(ii),long=sum(gt[i]['label']==0 for i in ii),short=sum(gt[i]['label']==1 for i in ii),elevation_quantiles=np.quantile([gt[i]['elevation'] for i in ii],[0,.1,.5,.9,1]),area_quantiles=np.quantile([gt[i]['area'] for i in ii],[0,.1,.5,.9,1]))
    C.freeze(C.sdoc(2)/'EXACT_LABEL_AUDIT.json',dict(reference_read_time=start,prediction_lock_time=lock['created_at'],
        label='Renderer Xcf edge01 true metric width compared to physical width/depth. No image area heuristic; no prediction/free matching.',
        projection_max_px=maxerr,perfect_GT_candidate_ADD_max=max(perfect),splits=stats,labels=C.bind(C.sraw(2)/'SYNTH_LABELS.npz')))
    print('EXACT_LABELS',maxerr,stats,flush=True)

if __name__=='__main__':main()

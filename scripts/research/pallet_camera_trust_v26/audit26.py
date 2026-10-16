"""Independent projection, population, metric and lineage checks for V26."""
from common26 import *
import cv2,numpy as np,hashlib

def main():
    bindings={};n_bound=0
    for d,key in [(read(RUN/'PROTOCOL.json'),'source_sha256'),(read(RUN/'INFERENCE_COMPLETE.json'),'checkpoint_bindings'),(read(ROOT/'data/pallet/results/pallet_observation_v25/COMPLETION.json'),'files')]:
        for p,h in d[key].items():
            assert sha(ROOT/p)==h,p;bindings[p]=h;n_bound+=1
    assert sha(KIT/'infer26.py')==read(RUN/'INFERENCE_COMPLETE.json')['source_sha256']
    renders=read(RUN/'RENDER_COMPLETE.json')['rows'];manifest=read(RUN/'cache/manifest.json');rows=manifest['rows'];ts=read(RUN/'TARGETS.json')['rows'];det=np.array([r['detected'] for r in rows]);gt=np.array([t['points'] for t in ts]);maxproj=0.;maxmetric=0.
    assert len(renders)==180 and len(rows)==360 and det.sum()==172
    for r in renders:
        assert sha(ROOT/r['image'])==r['image_sha256'];T=np.asarray(r['camera_pose_opengl']);R=np.diag([1.,-1.,-1.])@T[:3,:3].T;t=-R@T[:3,3];rv,_=cv2.Rodrigues(R)
        q,_=cv2.projectPoints(np.asarray(r['points_world']),rv,t,np.asarray(r['K']),None);err=np.abs(q[:,0]-r['points']).max();maxproj=max(maxproj,float(err));assert err<1e-8
        w=np.asarray(r['points_world']);assert np.allclose(w[[0,1,4,5],2],w[:,2].max());assert np.allclose(w[[2,3,6,7],2],w[:,2].min());assert q[0,0,0]<=q[1,0,0] and q[4,0,0]<=q[5,0,0]
        z=np.load(ROOT/r['depth_path']);obj=z['object']>0;v=obj&(np.abs(z['scene']-z['object'])<.003);assert int(v.sum())==r['visible_pixels'] and int(obj.sum())==r['object_pixels']
        mask=cv2.imread(str(ROOT/r['image']).replace('.png','_visible.png'),0)>0;assert np.array_equal(v,mask)
        yy,xx=np.where(obj);uv=q[:,0];assert xx.min()>=uv[:,0].min()-2 and xx.max()<=uv[:,0].max()+2 and yy.min()>=uv[:,1].min()-2 and yy.max()<=uv[:,1].max()+2
    for i,(r,t) in enumerate(zip(rows,ts)):
        assert r['id']==t['id'];assert np.allclose(np.array(r['points'])-r['origin'],t['points'],atol=1e-9)
        im=cv2.imread(str(ROOT/r['image']));x,y=r['origin'];h,w=r['image_hw'];canvas=im[y:y+h,x:x+w];assert hashlib.sha256(canvas.tobytes()).hexdigest()==r['transformed_sha256']
        outside=(gt[i,:,0]<0)|(gt[i,:,0]>=w)|(gt[i,:,1]<0)|(gt[i,:,1]>=h);assert np.array_equal(outside,t['outside'])
    for p,h in manifest['chunk_sha256'].items():assert sha(RUN/'cache'/p)==h
    assert [r['observation_index'] for r in rows if r['detected']]==list(range(172));ids=[r['id'] for r in rows];assert len(set(ids))==360
    predictions={}
    for p in (RUN/'predictions').glob('*.npz'):
        d=np.load(p);assert d['ids'].tolist()==ids and np.array_equal(d['detected'],det);assert np.isfinite(d['points'][det]).all() and np.isnan(d['points'][~det]).all();predictions[p.stem]=d['points'].astype(float)
    per=read(RUN/'CAMERA_PER_IMAGE.json')['rows'];ix={r['id']:i for i,r in enumerate(rows)}
    for r in per:
        i=ix[r['id']];name='stock' if r['arm']=='stock' else f"{r['arm']}_seed{r['seed']}";mask=np.ones(8,bool) if r['subset']=='all' else np.arange(8)<4 if r['subset']=='front' else np.array(ts[i]['outside']);e=np.sqrt(((predictions[name][i]-gt[i])**2).sum(1))[mask]
        expected=float(e.mean()) if det[i] else None;norm=float(np.clip(e/800,0,1).mean()) if det[i] else 1
        if expected is not None:maxmetric=max(maxmetric,abs(expected-r['mean_px']))
        else:assert r['mean_px'] is None
        assert abs(norm-r['full_normalized'])<1e-10
    assert maxmetric<1e-10
    count=0;identity=0.
    with (RUN/'TRUST_POINTS.jsonl').open() as f:
        for line in f:
            r=json.loads(line);count+=1;assert r['oracle_predicted_px']<=r['base_px']+1e-9 and r['oracle_exact_incident_px']<=r['base_px']+1e-9
            e=abs(r['base_px']**2-r['normal_error_px']**2-r['tangent_error_px']**2);identity=max(identity,e);assert e<1e-7
            assert r['null_preferred']==(r['null_weight']>=r['association_score']);assert r['line_good']==(r['incident_line_rms_px']<=5)
    # An exact line can eliminate normal error while leaving all tangent error.
    p=np.array([12.,2.]);n=np.array([0.,1.]);q=p-(p@n)*n;assert np.array_equal(q,[12.,0.]) and np.linalg.norm(q)==12
    result=dict(status='PASS',rendered_images=180,observations=360,detected=172,prediction_files=len(predictions),independent_cv2_projection_max_px=maxproj,per_image_metric_replay_max_px=maxmetric,trust_rows=count,normal_tangent_squared_identity_max=identity,lineage_files_checked=n_bound,prior_V25_completion_unchanged=True,actual_masks_and_cuboid_enclosure_checked=True,new_training_updates=0,limits='Coordinate and cuboid-role checks do not certify manual real GT precision or physical mesh edge visibility. Forced top-line projection is a diagnostic, not the learned forward rule.',bindings=bindings)
    write(RUN/'AUDIT.json',result);record('독립 cv2 투영·180깊이 마스크·360GT/잘림·전체 예측 지표 재계산·접선/법선 항등식·V25 고정 파일 검사 PASS.');print('AUDIT26',maxproj,maxmetric,count,flush=True)

if __name__=='__main__':main()

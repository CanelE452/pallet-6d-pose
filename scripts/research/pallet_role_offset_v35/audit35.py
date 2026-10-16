from context35 import *
from analyze35 import INC,METHODS,fit_offsets
from followup35 import NEW
import cv2

def main():
    for marker in ('PRIOR_BINDINGS.json','EXTRACTION_COMPLETE.json','ANALYSIS_COMPLETE.json','FOLLOWUP_COMPLETE.json'):
        for p,h in read(RUN/marker)['files'].items():assert sha(ROOT/p)==h,p
    for marker in ('EXTRACTION_PROTOCOL.json','ANALYSIS_PROTOCOL.json'):
        for p,h in read(RUN/marker)['files'].items():assert sha(ROOT/p)==h,p
    f=read(RUN/'FOLLOWUP_PROTOCOL.json');assert f['source_sha256']==sha(KIT/'followup35.py');assert f['primary_summary_sha256']==sha(RUN/'SUMMARY.json')
    rows=read(RUN/'POPULATION.json')['rows'];g=np.load(RUN/'geometry.npz');gt=g['gt'];N=len(rows);bb=np.array([r['box'] for r in rows]);wh=np.maximum(bb[:,2:]-bb[:,:2],1);diag=np.linalg.norm(wh,axis=-1);aspect=np.log(wh[:,0]/wh[:,1]);train=np.array([r['split']=='train' for r in rows]);groups=np.array([r['source_group'] for r in rows]);assert not set(groups[train])&set(groups[~train]);assert len(set(groups[train]))==16 and len(set(groups[~train]))==8
    changed=gt.copy();changed[~train]+=1234567;refit=fit_offsets(g['normal'],g['rho'],changed,diag,aspect,g['length']>0,train);assert refit==read(RUN/'OFFSET_CALIBRATION.json')['models']
    maxproj=0.;maxline=0.;segments=0
    for i,r in enumerate(rows):
        T=np.array(r['camera_pose_opengl']);R=np.diag([1.,-1.,-1.])@T[:3,:3].T;tr=-R@T[:3,3];K=np.array(r['K']);rv=cv2.Rodrigues(R)[0];world=np.array(r['points_world']);uv=cv2.projectPoints(world,rv,tr,K,None)[0][:,0]-r['origin'];maxproj=max(maxproj,float(np.abs(uv-gt[i]).max()));assert sha(ROOT/r['label'])==r['label_sha256'];lab=read(ROOT/r['label']);ss={s['segment']:s for s in lab['segments']}
        for role in np.where(g['length'][i]>0)[0]:
            s=ss[int(g['segment'][i,role])];assert s['role']==role;end=cv2.projectPoints(np.array(s['world']),rv,tr,K,None)[0][:,0]-r['origin'];line=g['normal'][i,role];dist=np.abs(end@line-g['rho'][i,role]);maxline=max(maxline,float(dist.max()));assert np.isclose(np.linalg.norm(line),1);assert line[1]>=0;assert g['length'][i,role]>=8;segments+=1
            # Role proximity independently recomputed from original 3D endpoints.
            refs=world[np.array(EDGES)];vec=refs[:,1]-refs[:,0];mid=np.mean(s['world'],axis=0);t=np.clip(((mid-refs[:,0])*vec).sum(-1)/(vec*vec).sum(-1),0,1);distance=np.linalg.norm(mid-(refs[:,0]+t[:,None]*vec),axis=-1);axis=np.abs(np.diff(np.array(s['world']),axis=0)[0]).argmax();distance[np.abs(vec).argmax(-1)!=axis]=np.inf;assert distance.argmin()==role;assert abs(distance[role]-s['role_offset_m'])<1e-8
    assert maxproj<1e-6 and maxline<1e-6
    summary=read(RUN/'SUMMARY.json');supplement=read(RUN/'FOLLOWUP.json');checked=0;oracle=0.
    for seed in SEEDS:
        a=np.load(RUN/f'analysis_seed{seed}.npz');b=np.load(RUN/f'followup_seed{seed}.npz');assert a['base'].shape==(N,8,2);supported=a['supported'];targetnormal=((gt-a['base'])*a['gt_normal']).sum(-1);linehint=a['gt_rho']-(a['base']*a['gt_normal']).sum(-1);assert np.allclose((targetnormal-linehint)[supported],a['gt_bias_px'][supported],atol=1e-10)
        # Independent one-normal and full-rank least squares calculations.
        for i,j in zip(*np.where(supported)):
            n=a['gt_normal'][i,j];e=gt[i,j]-a['base'][i,j];out=a['base'][i,j]+n*np.dot(n,e);oracle=max(oracle,float(np.abs(out-a['normal_oracle'][i,j]).max()));rr=[r for r in INC[j] if g['length'][i,r]>0];A=g['normal'][i,rr]
            if a['rank'][i,j]==2:
                independently=a['base'][i,j]+np.linalg.solve(A.T@A,A.T@(A@e));assert np.allclose(independently,a['joint_oracle'][i,j],atol=1e-7)
        for name,methods,data,report in [('primary',METHODS,a,summary),('followup',NEW,b,supplement)]:
            for mi,k in enumerate(methods):
                error=np.linalg.norm(data[k]-gt,axis=-1);assert np.isfinite(error).all();assert np.allclose(error,data['errors'][mi],atol=1e-10)
                for split,take in [('train',train),('check',~train)]:
                    for typ,mask0 in [('supported',supported),('all_detected',np.ones_like(supported))]:
                        v=error[take[:,None]&mask0].mean();expected=report['results'][f'{split}/all/{typ}']['methods'][k]['per_seed_px'][seed-1];assert abs(v-expected)<1e-10
                checked+=1
    assert oracle<1e-8
    write(RUN/'AUDIT.json',dict(status='PASS',requested=600,detected=N,check_requested=200,check_detected=int((~train).sum()),groups_train=16,groups_check=8,source_groups_disjoint=True,shared_CAD_assets=2,physical_role_segments_checked=segments,independent_projection_max_px=maxproj,independent_line_equation_max_px=maxline,normal_identity_max_px=oracle,method_seed_metrics_checked=checked,changing_check_GT_does_not_change_fit=True,prior_files_unchanged=True,manual_GT_changed=False,new_real_evaluation=False,neural_updates=0,limits='Physical role proximity and projection consistency checked; not independently human-certified semantic correspondence. Visibility inherited from immutable V32 depth labels.'));print('AUDIT35 PASS',flush=True)
if __name__=='__main__':main()

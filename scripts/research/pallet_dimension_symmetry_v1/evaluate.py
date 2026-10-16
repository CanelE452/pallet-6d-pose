"""Identical symmetry-aware scoring, 2x2 contrasts, and GT-free dimension PnP."""
import cv2
from core import *

def iou(a,b):
    a=np.asarray(a);b=np.asarray(b)
    intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(intersection/max(1e-9,np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-intersection))

def pnp(xy,visibility,dims,K):
    """Two front-axis hypotheses using KNOWN fixed dimensions, never GT pose/order.

    Translation diagnostic only. W/D hypotheses differ by object-coordinate
    correspondences, not by a GT-directed dimension swap. Yaw ambiguity is not
    solved or scored as correct semantic rotation.
    """
    signs=np.array([[-1,1,1],[1,1,1],[1,-1,1],[-1,-1,1],[-1,1,-1],[1,1,-1],[1,-1,-1],[-1,-1,-1]])
    base=signs*np.asarray(dims)/2
    valid=(np.asarray(visibility[:8])>.5)&np.isfinite(xy[:8]).all(-1)
    if valid.sum()<4:return None
    best=None
    for perm in [np.arange(8),np.array([1,5,6,2,0,4,7,3])]:
        X=base[perm];Xp=np.ascontiguousarray(X[valid],dtype=np.float64)
        u=np.ascontiguousarray(xy[:8][valid],dtype=np.float64)
        try:
            ok,r,t=cv2.solvePnP(Xp,u,np.asarray(K,dtype=np.float64),None,flags=cv2.SOLVEPNP_SQPNP)
            if not ok:continue
            r,t=cv2.solvePnPRefineLM(Xp,u,np.asarray(K,dtype=np.float64),None,r,t)
            R=cv2.Rodrigues(r)[0]
            if (X@R.T+t.ravel())[:,2].min()<=0:continue
            proj=cv2.projectPoints(Xp,r,t,np.asarray(K,dtype=np.float64),None)[0].reshape(-1,2)
            error=float(np.linalg.norm(proj-u,axis=-1).mean())
            if best is None or error<best['reprojection_px']:
                best=dict(t=t.ravel().tolist(),reprojection_px=error)
        except cv2.error:continue
    return best

def score(xy,vis,records):
    rows=[]
    for pred,confidence,r in zip(xy,vis,records):
        gt=np.asarray(r['gt_raw'])[:8];mask=np.asarray(r['valid'])[:8]
        diag=float(np.linalg.norm(r['raw_hw']));match=r['confidence']>=.25 and iou(r['box_raw'],r['gt_box_raw'])>=.5
        cand=[]
        for branch,perm in enumerate([np.arange(8)]+([np.asarray(r['perm'])[:8]] if r['order']==2 else [])):
            err=np.linalg.norm(pred[:8]-gt[perm],axis=-1)
            mean=float(err[mask[perm]].mean()) if mask.any() else diag
            cand.append((mean,branch,err,mask[perm]))
        mean,branch,err,selected_mask=min(cand,key=lambda x:x[0])
        if not match or not np.isfinite(pred).all():mean=diag
        pose=pnp(pred,confidence,r['dims_xyz'],r['K']) if match else None
        t_error=float(np.linalg.norm(np.asarray(pose['t'])-r['t'])*100) if pose else None
        rows.append(dict(id=r['id'],group=r['group'],asset=r['asset'],order=r['order'],matched=match,
            normalized=min(mean/diag,1.),mean_px=mean,chosen_symmetry=branch,
            indexed_mean_px=float(np.linalg.norm(pred[:8]-gt,axis=-1)[mask].mean()) if mask.any() else diag,
            valid_corners=int(mask.sum()),translation_cm=t_error,
            point_error_px=err.tolist(),selected_mask=selected_mask.tolist()))
    return rows

def summary(rows):
    e=np.array([r['mean_px'] for r in rows]);n=np.array([r['normalized'] for r in rows])
    t=[r['translation_cm'] for r in rows if r['translation_cm'] is not None]
    return dict(frames=len(rows),matched=sum(r['matched'] for r in rows),mean_px=float(e.mean()),median_px=float(np.median(e)),
        p90_px=float(np.quantile(e,.9)),primary=float(n.mean()),bad_frames_gt20=int((e>20).sum()),
        alternate_symmetry_frames=sum(r['chosen_symmetry']!=0 for r in rows),
        pnp_coverage=len(t),translation_median_cm=float(np.median(t)) if t else None)

def ci(delta):
    rng=np.random.default_rng(4501)
    # One selected image per pair_group_id; sampling unit is that synthetic frame group.
    values=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(5000)])
    return dict(mean=float(delta.mean()),low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)),
                method='paired synthetic frame-group bootstrap, seed-average per group, 5000 replicates')

def main():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    assert read(OUT/'TRAINING_COMPLETE.json')['PASS']
    frozen=read(OUT/'TRAIN_SOURCE_LOCK.json')
    for name,digest in frozen['sources'].items():assert sha(KIT/name)==digest
    records=read(OUT/'val_records.json');assert len({r['group'] for r in records})==len(records)
    z=load_cache('val');m=base_model();h=m.model[-1];flow=h.flow_model
    stock=prepared_points(z['stock_grid'],z).numpy()
    stockvis=z['stock_grid'][:,:,2].sigmoid().numpy()
    rows0=score(stock,stockvis,records)
    allrows={'stock':rows0};predictions={'stock':stock};results={'stock':summary(rows0)}
    training=load_cache('train');train_identity={}
    with torch.no_grad():
        pred=OutputAdapter(h,1)(training,False)
        _,_,branch=objective(pred,training,flow,True)
        train_identity['initial_C2_objects']=int((training['order']==2).sum())
        train_identity['initial_alternate_targets']=int(branch.sum())
        v0=objective(pred,training,flow,False)[0];v1=objective(pred,training,flow,True)[0]
        train_identity['initial_mean_loss_reduction']=float((v0-v1).mean())
    for arm,(dims,sym) in ARMS.items():
        results[arm]=[]
        for seed in [1,2,3]:
            tag=f'{arm}_seed{seed}';model=OutputAdapter(h,seed)
            model.load_state_dict(torch.load(OUT/'models'/tag/'final.pt',weights_only=True)['state']);model.eval()
            with torch.no_grad():pred=model(z,dims)
            xy=prepared_points(pred,z).detach().numpy();vis=pred[:,:,2].sigmoid().detach().numpy()
            # Forward only consumes features, dims and decode coordinates; GT is read by score afterwards.
            rows=score(xy,vis,records);allrows[tag]=rows;predictions[tag]=xy
            result=summary(rows);result['seed']=seed;results[arm].append(result)
            if dims:
                zz=dict(z,dims_whd=z['dims_whd'].roll(1,0))
                with torch.no_grad():wrong=model(zz,True)
                # PnP/scoring retains the CORRECT dimensions. Only neural input is shuffled.
                wr=score(prepared_points(wrong,zz).detach().numpy(),wrong[:,:,2].sigmoid().numpy(),records)
                allrows[tag+'_shuffled']=wr
                result['shuffled_neural_dims_primary']=summary(wr)['primary']
    def arrays(arm,key):
        if arm=='stock':return np.array([[r[key] for r in rows0]]*3)
        return np.array([[r[key] for r in allrows[f'{arm}_seed{s}']] for s in [1,2,3]])
    contrasts={}
    for name,left,right in [('dims_without_sym','C_dimensions','A_indexed'),('dims_with_sym','D_both','B_symmetry'),
                            ('sym_without_dims','B_symmetry','A_indexed'),('sym_with_dims','D_both','C_dimensions')]:
        delta=arrays(left,'normalized')-arrays(right,'normalized')
        contrasts[name]=dict(bootstrap=ci(delta.mean(0)),per_seed_mean_delta=delta.mean(1).tolist())
    contrasts['interaction']=dict(bootstrap=ci((arrays('D_both','normalized')-arrays('C_dimensions','normalized')-
                                               arrays('B_symmetry','normalized')+arrays('A_indexed','normalized')).mean(0)))
    gates={}
    A=arrays('A_indexed','normalized');Ap=arrays('A_indexed','mean_px');S=arrays('stock','normalized')
    for arm in ARMS:
        delta=arrays(arm,'normalized')-A;p=arrays(arm,'mean_px')
        gain=1-arrays(arm,'normalized').mean(1)/A.mean(1)
        checks=dict(all_seed_gain_at_least_1pct=bool((gain>=.01).all()),
                    all_seed_mean_better_than_stock=bool((arrays(arm,'normalized').mean(1)<S.mean(1)).all()),
                    all_seed_p90_nonworse_than_A=bool((np.quantile(p,.9,axis=1)<=np.quantile(Ap,.9,axis=1)).all()),
                    paired_bootstrap_upper_below_zero=bool(ci(delta.mean(0))['high']<0),
                    no_new_bad_frames=bool(not ((Ap<=10)&(p>20)).any()))
        gates[arm]=dict(PASS=all(checks.values()),checks=checks,gain_over_A_per_seed=gain.tolist(),
                        new_bad_frames_per_seed=((Ap<=10)&(p>20)).sum(1).tolist())
    # Factor equality and activation are evidence, not a reason to change losses after scoring.
    exact={}
    for left,right in [('A_indexed','B_symmetry'),('C_dimensions','D_both')]:
        exact[f'{left}_vs_{right}']=[float(np.abs(predictions[f'{left}_seed{s}']-predictions[f'{right}_seed{s}']).max()) for s in [1,2,3]]
    np.savez_compressed(OUT/'predictions.npz',**predictions)
    write(OUT/'PER_FRAME.json',allrows)
    write(OUT/'SUMMARY.json',dict(complete=True,results=results,contrasts=contrasts,gates=gates,
        initial_symmetry_activation=train_identity,paired_prediction_max_abs_px=exact,
        decision='OUTPUT_LAYER_SCREEN_SIGNAL' if any(gates[a]['PASS'] for a in ['B_symmetry','C_dimensions','D_both']) else 'NO_STABLE_OUTPUT_LAYER_SCREEN_GAIN',
        full_network_training=False,real_evaluation=False,hough_training=False,
        pnp_note='GT-free two front-axis SQPnP hypotheses, known fixed dimensions for all methods, translation diagnostic only. Does not establish semantic rotation recovery.'))
    print(json.dumps(read(OUT/'SUMMARY.json'),ensure_ascii=False,indent=2))

if __name__=='__main__':main()

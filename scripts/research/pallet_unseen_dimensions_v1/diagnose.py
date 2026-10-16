"""Post-screen cause diagnostics. Oracle outputs are never promoted to model scores."""
from common import *
import sys,importlib.util,numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,str(ROOT))
from scripts.annotate.convert_to_camera_facing_v4 import compute_perm_v4

def main():
    rows=read(OUT/'test_records.json');frame=read(OUT/'PER_FRAME.json');pred=np.load(OUT/'predictions.npz')
    signs=np.array([[-1,1,1],[1,1,1],[1,-1,1],[-1,-1,1],[-1,1,-1],[1,1,-1],[1,-1,-1],[-1,-1,-1]])
    origin=signs[:,[0,2,1]] # world Z is height, no GT geometry required
    permutations=[]
    for a in [0,90,180,270]:
        c=np.cos(np.deg2rad(a));s=np.sin(np.deg2rad(a));rot=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        permutations.append(np.linalg.norm((signs@rot.T)[:,None]-signs[None],axis=-1).argmin(-1))
    target_identity=[];rng=np.random.default_rng(451);stability=[];out={};canonical={}
    for r in rows:
        gt=np.asarray(r['gt']);target_identity.append(compute_perm_v4(origin,gt)[:8]==list(range(8)))
        flips=sum(compute_perm_v4(origin,gt+rng.normal(0,1,gt.shape))[:8]!=list(range(8)) for _ in range(100))
        stability.append(flips/100)
    assert all(target_identity)
    for name in pred.files:
        xy=pred[name].copy();rec=[]
        for i,r in enumerate(rows):
            gt=np.array(r['gt']);p=xy[i,:8];d=gt-p
            cost=np.linalg.norm(p[:,None]-gt[None],axis=-1);a,b=linear_sum_assignment(cost)
            errors=[np.linalg.norm(p-gt[v],axis=-1).mean() for v in permutations]
            pp=compute_perm_v4(origin,p)
            if pp is not None:xy[i]=xy[i,pp]
            rec.append(dict(id=r['id'],seen=r['seen'],state=r['state'],matched=frame[name][i]['matched'],
                original_px=float(np.linalg.norm(d,axis=-1).mean()),
                capped_residual_GT_oracle_px=float(np.linalg.norm(d-d.clip(-64,64),axis=-1).mean()),
                unconstrained_assignment_GT_oracle_px=float(cost[a,b].mean()),
                four_role_GT_oracle_px=float(min(errors)),four_role_GT_oracle_branch=int(np.argmin(errors)),
                observation_only_canonical_px=float(np.linalg.norm(xy[i,:8]-gt,axis=-1).mean()),
                canonical_perm=pp,GT_1px_jitter_relabel_rate=stability[i]))
        canonical[name]=xy;out[name]=rec
    summary={}
    for name,rr in out.items():
        summary[name]={}
        for seen in [True,False]:
            selected=[r for r in rr if r['matched'] and r['seen']==seen]
            summary[name]['seen' if seen else 'unseen']=dict(n=len(selected),**{
                k:float(np.mean([r[k] for r in selected])) for k in ['original_px','capped_residual_GT_oracle_px',
                'unconstrained_assignment_GT_oracle_px','four_role_GT_oracle_px','observation_only_canonical_px','GT_1px_jitter_relabel_rate']},
                nonidentity_role_oracle_frames=sum(r['four_role_GT_oracle_branch']!=0 for r in selected),
                canonical_new_gt20_from_original_le10=sum(r['original_px']<=10 and r['observation_only_canonical_px']>20 for r in selected))
    write(OUT/'CAUSE_DIAGNOSTIC.json',dict(posthoc=True,changes_primary_decision=False,
        gt_canonical_identity_all_frames=True,summary=summary,
        distinction='Assignment/role/cap oracles read GT and are unattainable diagnostics. Observation-only canonicalization uses predicted points and a fixed cuboid topology, no GT, dims, or new training; this is postprocessing, not a new learned architecture.',
        symmetry_note='Quarter-turn relabeling is an ID-error oracle, NOT assumed physical C4 symmetry.',
        rows=out))
    np.savez_compressed(OUT/'canonical_diagnostic_predictions.npz',**canonical)
    print(json.dumps(summary['stock'],indent=2))
if __name__=='__main__':main()

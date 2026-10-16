from context39 import *

def main():
    write(RUN/'PHENOTYPE_PROTOCOL.json',dict(exploratory=True,primary_sha256=sha(RUN/'SUMMARY.json'),purpose='Describe hard cases without calling heuristic categories definitive causes. No fitting.',definitions='Near another GTcorner: hard>=10px, nearestGTindex differs from intendedj andnearest distance<=5px. This is alias-like proximity, not certified semantic swap. Rank from V35 visible incidentphysical normals SVD. Strong-parent error and GTbest-of-parent bound separate. All thresholds fixed for this post-primary descriptive probe, no policy changes.',source_sha256=sha(KIT/'phenotypes39.py')))
    p=population();part=partition(p);g=p['g'];result={}
    for seed in SEEDS:
        v=dict(np.load(RUN/'arrays'/f'both_seed{seed}.npz'));a=dict(np.load(V37/'arrays'/f'v34_both_seed{seed}.npz'));anchor=v['anchor'];gt=g['gt'];base=np.linalg.norm(anchor-gt,axis=-1);strong=np.linalg.norm(a['strong']-gt,axis=-1);pair=np.linalg.norm(anchor[:,:,None]-gt[:,None],axis=-1);nearest=pair.argmin(-1);alias=(nearest!=np.arange(8)[None])&(pair.min(-1)<=5)&v['hard'];rank=a['rank'];report={}
        for split in ('fit','calibration','check'):
            hard=part[split][:,None]&v['hard'];report[split]=dict(hard_points=int(hard.sum()),alias_like_points=int((hard&alias).sum()),alias_like_fraction=float(alias[hard].mean()),hidden_points=int((hard&g['hidden']).sum()),outside_points=int((hard&g['outside']).sum()),visible_physical_rank_counts={str(k):int((hard&(rank==k)).sum()) for k in (0,1,2)},anchor_hard_error_px=float(base[hard].mean()),strong_hard_error_px=float(strong[hard].mean()),strong_better_over1px=int(((base-strong>1)&hard).sum()),best_parent_GT_oracle_hard_px=float(np.minimum(base,strong)[hard].mean()),GT_physical_joint_normal_oracle_hard_px=float(np.linalg.norm(a['gt_joint_oracle']-gt,axis=-1)[hard].mean()))
        result[str(seed)]=report
    write(RUN/'PHENOTYPES.json',dict(complete=True,exploratory=True,results=result,limits='Description only; GTnearest relation does not prove point identity errors. GTnormals and GTparentselection are nondeployable diagnostic ceilings. Sourcegroups andpoints correlated.'));print('PHENOTYPES39', {k:v['check'] for k,v in result.items()},flush=True)
if __name__=='__main__':main()

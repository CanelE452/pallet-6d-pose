"""Posthoc illustrations of both improvement and damage, chosen deterministically."""
import cv2
import numpy as np
from . import common as C

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.research.pallet_selector_recovery_v1.features import cuboid
    from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
    metrics=C.read(C.RAW/'REAL_SELECTED_METRICS_PRIVATE.json');dec=C.read(C.RAW/'REAL_SELECTOR_DECISIONS_PRIVATE.json')
    raw=C.read(C.HARDRAW/'RAW_PREDICTIONS.json')['real'];poses=C.read(C.HARDRAW/'POSE_DECISIONS.json')['real']
    meta={r['id']:r for r in C.read(C.HARDRAW/'INFERENCE_INPUTS.json')['real']};truth=C.read(V.E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json')
    old=metrics['H_MANUAL_OLD_GEO'];new=metrics['H_MANUAL_HMANSPEC_GEO'];delta={i:new[i]['ADDsym_normalized']-v['ADDsym_normalized'] for i,v in old.items() if v['available'] and new[i]['available']}
    selected=[('improved',i) for i in sorted([i for i,v in delta.items() if v<0],key=lambda i:(delta[i],i))[:3]]
    selected += [('worsened',i) for i in sorted([i for i,v in delta.items() if v>0],key=lambda i:(-delta[i],i))[:3]]
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)];manifest=[]
    for j,(tag,fid) in enumerate(selected):
        r=meta[fid];im=cv2.cvtColor(cv2.imread(str(C.ROOT/r['image']['path'])),cv2.COLOR_BGR2RGB);fig,axs=plt.subplots(1,3,figsize=(15,5))
        for ax,combo in zip(axs,('S1_OLD_GEO','H_MANUAL_OLD_GEO','H_MANUAL_HMANSPEC_GEO')):
            model=dec[combo][fid]['model'];arm=C.ARM[model];name=dec[combo][fid]['selected'];p=next((h['pose'] for h in poses[arm][fid]['hypotheses'] if h['name']==name),{})
            ax.imshow(im);q=C.U.selected(raw[arm][fid]);t=truth[fid];gt=np.array(t['gt']);valid=np.array(t['valid'],bool);valid[8:]=False
            ax.scatter(gt[valid,0],gt[valid,1],s=24,c='lime',marker='x',label='annotation reference')
            if q:
                xy=np.array(q['keypoints_xy']);ax.scatter(xy[:8,0],xy[:8,1],s=12,c='cyan',label='raw keypoints')
            if p.get('available'):
                X=cuboid(*p['cf_extents'])@np.array(p['R_cf']).T+p['centroid'];proj=X@np.array(r['K']).T;proj=proj[:,:2]/proj[:,2:]
                for a,b in edges:ax.plot(proj[[a,b],0],proj[[a,b],1],color='yellow',lw=1)
            v=metrics[combo][fid];ax.set_title(f'{combo}\nADDnorm {v["ADDsym_normalized"]:.4f} | {name}' if v['available'] else combo+' unavailable',fontsize=9)
            ax.set_xlim(0,im.shape[1]);ax.set_ylim(im.shape[0],0);ax.axis('off')
        axs[0].legend(fontsize=7);fig.suptitle(f'{tag}: {fid}\nPOSTHOC largest actual ADD change; H_MANUAL raw keypoints unchanged between middle and right',fontsize=10);fig.tight_layout()
        path=C.stage(3)/'examples'/f'{j+1:02d}_{tag}.jpg';path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=130);plt.close(fig)
        manifest.append(dict(id=fid,category=tag,ADDnorm_delta=delta[fid],image=C.bind(path),old_selection=dec['H_MANUAL_OLD_GEO'][fid]['selected'],new_selection=dec['H_MANUAL_HMANSPEC_GEO'][fid]['selected']))
    C.save(C.stage(3)/'EXAMPLE_MANIFEST.json',dict(rows=manifest,rule='Top three negative and positive actual per-frame ADDnorm changes, ties frame ID',posthoc_explanatory_only=True,not_model_selection=True))
    print('FINAL_EXAMPLES',len(manifest),flush=True)

if __name__=='__main__':main()

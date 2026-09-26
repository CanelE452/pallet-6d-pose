"""Deterministic posthoc explanatory examples, never used for fitting/selection."""
import cv2
import numpy as np
from . import common as C

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.research.pallet_selector_recovery_v1.features import cuboid
    from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
    dec=C.read(C.RAW/'stage1/FRAME_DECOMPOSITION_PRIVATE.json');pred=C.read(C.HARDRAW/'RAW_PREDICTIONS.json')['real']
    pose=C.read(C.HARDRAW/'POSE_DECISIONS.json')['real'];meta={r['id']:r for r in C.read(C.HARDRAW/'INFERENCE_INPUTS.json')['real']}
    truth=C.read(V.E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json');picks=[]
    for tag,fn in [('selector_recoverable',lambda i:dec['models']['H_MANUAL'][i]['category']=='SELECTOR_RECOVERABLE'),('unrealized_gain',lambda i:dec['paired'][i]['UNREALIZED_MANUAL_GAIN']),('tail_worsened',lambda i:dec['paired'][i]['tail']=='TAIL_WORSENED')]:
        ids=sorted([i for i in dec['paired'] if fn(i)],key=lambda i:(-(dec['models']['H_MANUAL'][i]['recoverable_gap'] or 0),i))[:2]
        picks += [(tag,i) for i in ids]
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    manifest=[]
    for k,(tag,fid) in enumerate(picks):
        r=meta[fid];im=cv2.cvtColor(cv2.imread(str(C.ROOT/r['image']['path'])),cv2.COLOR_BGR2RGB);fig,axs=plt.subplots(1,3,figsize=(15,5))
        for ax,(model,mode) in zip(axs,[('S1','current'),('H_MANUAL','current'),('H_MANUAL','oracle')]):
            a=C.ARM[model];f=dec['models'][model][fid];name=f['selected'] if mode=='current' else f['oracle'];p=next((h['pose'] for h in pose[a][fid]['hypotheses'] if h['name']==name),{})
            ax.imshow(im);q=C.U.selected(pred[a][fid]);t=truth[fid];gt=np.array(t['gt']);valid=np.array(t['valid'],bool);valid[8:]=False
            ax.scatter(gt[valid,0],gt[valid,1],s=24,c='lime',marker='x',label='annotation reference')
            if q:
                xy=np.array(q['keypoints_xy']);ax.scatter(xy[:8,0],xy[:8,1],s=10,c='cyan',label='raw keypoints')
            if p.get('available'):
                X=cuboid(*p['cf_extents'])@np.array(p['R_cf']).T+p['centroid'];pr=X@np.array(r['K']).T;pr=pr[:,:2]/pr[:,2:]
                for b,e in edges:ax.plot(pr[[b,e],0],pr[[b,e],1],color='yellow',lw=1)
            add=f['current_ADDnorm'] if mode=='current' else f['best_candidate_ADDnorm']
            ax.set_title(f'{model} {mode}\nADDnorm {add:.4f}' if add is not None else f'{model} unavailable');ax.set_xlim(0,im.shape[1]);ax.set_ylim(im.shape[0],0);ax.axis('off')
        axs[0].legend(fontsize=7);fig.suptitle(f'{tag}: {fid}\nPOSTHOC explanatory choice; oracle is GT-dependent and NONDEPLOYABLE',fontsize=10);fig.tight_layout()
        path=C.stage(1)/'examples'/f'{k+1:02d}_{tag}.jpg';path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=130);plt.close(fig)
        manifest.append(dict(category=tag,id=fid,image=C.bind(path)))
    C.save(C.stage(1)/'EXAMPLE_MANIFEST.json',dict(rule='top two actual H_MANUAL old-GEO recoverable ADD gaps within each fixed category, ties frame ID',posthoc_only=True,rows=manifest))
    report=C.stage(1)/'STAGE1_REPORT_KO.md';text=report.read_text()
    if '## Explanatory cases' not in text:
        text+='\n## Explanatory cases\n\n초록 X=기존 annotation reference, 청록 점=raw keypoints, 노랑 선=선택된 PnP pose 재투영. 각 고정 범주에서 실제 ADD gap 큰 순으로 선택한 사후 설명 사례이며, 학습·모델 선택 근거가 아니다.\n'
        for r in manifest:text+=f'\n### {r["category"]}: {r["id"]}\n\n![{r["category"]}](examples/{r["image"]["path"].split("/")[-1]})\n'
        C.save(report,text,immutable=False)
    print('EXAMPLES',len(manifest),flush=True)

if __name__=='__main__':main()

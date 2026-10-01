"""Input-chosen synthetic RGB and frozen R0 overlays; no quality selection."""
from . import common as C
C.source_guard()
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=C.protocol();smoke=C.read(C.DOC/'R0_SMOKE.json');assert smoke['complete']
    rows=C.read(C.RAW/'SOURCE_INPUTS.json');lookup={r['id']:(j,r) for j,r in enumerate(rows)}
    receipt=C.read(C.DOC/'R0_SOURCE_COMPLETE.json');assert receipt['complete']
    contract=C.read(C.DOC/'SOURCE_CONTRACT.json');eligible=set(sum(contract['fit_eligibility']['eligible_ids'].values(),[]))
    fig,axes=plt.subplots(4,2,figsize=(14,17),constrained_layout=True)
    images=[]
    for ax,fid in zip(axes.flat,p['smoke_ids']):
        j,row=lookup[fid];C.verify(row['image']);C.verify(receipt['files'][j])
        pred=C.read(C.ROOT/receipt['files'][j]['path'])['prediction']
        im=cv2.imread(str(C.ROOT/row['image']['path']));assert im is not None
        ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        if pred['selected_index'] is not None:
            selected=pred['candidates'][pred['selected_index']]
            x1,y1,x2,y2=selected['box_xyxy'];points=np.asarray(selected['keypoints_xy'])
            ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color='#ffbb33',lw=1.5))
            ax.scatter(points[:8,0],points[:8,1],c=np.arange(8),cmap='tab10',vmin=0,vmax=9,s=16,edgecolors='white',linewidths=.5)
            ax.scatter(points[8,0],points[8,1],marker='+',color='red',s=45)
        ax.set_title(f"{row['split']} | {fid}\n{row['hw'][1]} x {row['hw'][0]} | C2 proper-fit eligible: {fid in eligible}",fontsize=10)
        ax.set_xlim(0,im.shape[1]);ax.set_ylim(im.shape[0],0);ax.axis('off')
        images.append(dict(id=fid,image=row['image'],prediction=receipt['files'][j]))
    fig.suptitle('Fixed SHA-selected runtime smoke: original synthetic RGB + R0 predictions\nAll 8 exactly reproduce cached box / keypoints / confidence / candidate selection. No GT overlays or performance selection.',fontsize=13)
    path=C.DOC/'figures/01_fixed_source_smoke.png';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
    fig.savefig(path,dpi=120);plt.close(fig)
    C.save(C.DOC/'SOURCE_GALLERY_BINDINGS.json',dict(figure=C.bind(path),inputs=images,smoke=C.bind(C.DOC/'R0_SMOKE.json'),selection='Fixed before outcomes: four TRAIN and four VAL cached examples by RGB SHA; unchanged after failed smoke.',code=C.bind(__file__),quality_claim=False))
    print('SOURCE_GALLERY_COMPLETE',C.bind(path),flush=True)


if __name__=='__main__':main()

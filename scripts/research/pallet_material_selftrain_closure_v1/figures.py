"""Measured material deltas and native student overlays, never AI-generated RGB."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-material-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from . import common as C
from scripts.research.pallet_selftraining_paper_closure_v1.report import overlay

def save(fig,name):
    dst=C.DOC/'figures';dst.mkdir(exist_ok=True)
    fig.savefig(dst/(name+'.png'),dpi=140,bbox_inches='tight',facecolor='white')
    fig.savefig(C.PAPER/'figures'/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    wr=C.read(C.DOC/'WOOD_RESULTS.json');pr=C.read(C.P.DOC/'CORE_RESULTS.json')
    C.verify(wr['prediction_lock'])
    pred=C.read(C.RAW/'PREDICTIONS.json');fm=C.read(C.RAW/'FRAME_METRICS.json');truth=C.read(C.P.TRUTH)
    rows={r['id']:r for r in C.read(C.RAW/'EVAL_METADATA.json')}
    arms=('R0','WOOD_RAW_LR5','WOOD_REF_LR5');oldarms=('R0','RAW_LR5','REF_LR5')
    names=('R0','Raw student','Corrected student');colors=('#00a8db','#dbaa00','#c232be')
    fig,axs=plt.subplots(1,2,figsize=(9,3.4))
    for ax,metric,label in zip(axs,('PCK10','AUC'),('Full-denominator PCK10 (%)','Common-D9 ADDsym AUC')):
        for i in range(3):
            vals=[]
            for res,aa in ((pr,oldarms),(wr,arms)):
                a=res['groups']['ALL'][aa[i]]
                vals.append(100*a['twoD']['PCK']['10'] if metric=='PCK10' else a['sixD']['ADDsym_AUC'])
            ax.bar(np.arange(2)+(i-1)*.24,vals,.24,color=colors[i],label=names[i])
        ax.set_xticks([0,1],['Plastic\n128 images / 7 recordings','Wood\n45 images / 2 recordings'])
        ax.set_title(label);ax.spines[['top','right']].set_visible(False)
    axs[0].legend(fontsize=8);fig.suptitle('Within-material matched contrasts; both panels are reused DEV',fontsize=11)
    fig.tight_layout();save(fig,'material_main')
    delta=lambda fid:float(np.mean(fm[arms[2]][fid]['errors'])-np.mean(fm[arms[1]][fid]['errors']))
    ranked=sorted(rows,key=lambda f:(delta(f),f));selected=[]
    for tag,ii in [('improved',[f for f in ranked if delta(f)<-1e-8][:2]),
                   ('worsened',[f for f in reversed(ranked) if delta(f)>1e-8][:2]),
                   ('least_changed',sorted(rows,key=lambda f:(abs(delta(f)),f))[:2])]:
        for fid in ii:
            if fid not in [f for _,f in selected]:selected.append((tag,fid))
    examples=[]
    for n,(tag,fid) in enumerate(selected,1):
        im=Image.open(C.ROOT/rows[fid]['image']['path']);gt=truth[fid]
        targets={j:q for j,q in enumerate(gt['gt'][:8]) if gt['valid'][j]}
        fig,axs=plt.subplots(1,3,figsize=(13.2,4))
        for ax,arm,name,color in zip(axs,arms,names,colors):
            overlay(ax,im,pred[arm][fid],targets,f'{name} | scored mean {np.mean(fm[arm][fid]["errors"]):.2f}px',color)
        fig.suptitle(f'{tag}: {fid} | {rows[fid]["severity"]} | corrected - raw {delta(fid):+.2f}px\nGreen crosses: legacy reference, NOT verified visible truth. Lines: native 2D, NOT PnP.',fontsize=9)
        name=f'wood_{n:02d}_{tag}';save(fig,name)
        examples.append(dict(name=name,id=fid,category=tag,delta_mean_px=delta(fid),image=rows[fid]['image'],
            matched=fm[arms[1]][fid]['matched'],reference='legacy unknown provenance'))
    C.save(C.DOC/'FIGURE_MANIFEST.json',dict(examples=examples,
        selection='Two largest mean-scored-error improvements, two largest deteriorations, two least changes; unique IDs. No selection feeds inference or training.',
        native_predictions=True,GT_used_only_after_prediction_lock=True,
        source_artifacts=[C.bind(C.RAW/n) for n in ('PREDICTIONS.json','FRAME_METRICS.json')]+[C.bind(C.DOC/'WOOD_RESULTS.json'),C.bind(C.P.DOC/'CORE_RESULTS.json')],
        files=[C.bind(p) for p in sorted((C.DOC/'figures').glob('*.png'))]),True)
    appendix='\n## 실제 예측 이미지: 개선·악화·변화 적음\n\n![material 주 비교](figures/material_main.png)\n\n같은 사후 순위 규칙으로 고른 사례이며 학습이나 평가 집합 선택에는 사용하지 않았다. 모든 선은 native2D 예측이고 PnP 투영이 아니다. 초록X는 출처가 혼합된 기존 reference이며 검수 가시점으로 해석하지 않는다.\n\n'
    appendix+=''.join(f'### {e["category"]}: {e["id"]}\n\n![{e["name"]}](figures/{e["name"]}.png)\n\n' for e in examples)
    report=C.DOC/'REPORT_KO.md';text=report.read_text()
    if '## 실제 예측 이미지: 개선·악화·변화 적음' not in text:C.save(report,text+appendix)
    print('MATERIAL_FIGURES',len(examples),flush=True)

if __name__=='__main__':main()

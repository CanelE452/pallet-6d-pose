"""Evidence figures from real cached images/coordinates; never generated imagery."""
import json
import copy
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from . import run as C
from . import scoring as S
from .inference import masked
from .review_analysis import write
from scripts.research.pallet_clean19_structured_easyhard_v1.augmentation import EDGES

FIG=C.DOC/'figures'
plt.rcParams.update({'font.family':[FontProperties(fname='/usr/share/fonts/truetype/nanum/NanumGothic.ttf').get_name(),'DejaVu Sans'],
    'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def savefig(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/name,dpi=130,bbox_inches='tight',pil_kwargs={'quality':88} if name.endswith('.jpg') else {})
    plt.close(fig)

def points(ax,q,color,alpha=1,lw=1.2,labels=False):
    q=np.asarray(q,float);valid=np.isfinite(q).all(1)&~(q==-1).all(1)
    for a,b in EDGES:
        if valid[a] and valid[b]:ax.plot(q[[a,b],0],q[[a,b],1],color=color,lw=lw,alpha=alpha)
    ax.scatter(q[:8,0][valid[:8]],q[:8,1][valid[:8]],s=14,c=color,edgecolors='black',linewidths=.3,alpha=alpha)
    if labels:
        for j in range(8):
            if valid[j]:ax.text(q[j,0]+3,q[j,1]-3,str(j),color=color,fontsize=7,clip_on=True,bbox=dict(facecolor='black',alpha=.5,pad=.4,edgecolor='none'))

def photo(ax,image,pred=None,ref=None,title='',box=True):
    ax.imshow(image[:,:,::-1]);h,w=image.shape[:2]
    if ref is not None:points(ax,ref,'#34e5e0',alpha=.8,lw=1)
    if pred is not None:
        selected=S.CORE.selected(pred)
        if selected:
            points(ax,selected['keypoints_xy'],'#ffb22c',labels=True)
            if box:
                x1,y1,x2,y2=selected['box_xyxy'];ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color='white',ls='--',lw=.8))
    ax.set(xlim=(0,w),ylim=(h,0));ax.set_title(title,fontsize=9);ax.axis('off')

def main():
    start,cpu=time.monotonic(),time.process_time()
    rows=C.metadata();meta={r['id']:r for r in rows};natural=[r for r in rows if r['severity']!='CLEAN'];clean=[r for r in rows if r['severity']=='CLEAN']
    e1=C.read(C.RAW/'E1_POSE_METRICS.json');pred=C.read(C.RAW/'INPUT_PREDICTIONS.json');refs=S.references()
    summaries=C.read(C.DOC/'E1_SUMMARY.json');ft=C.read(C.DOC/'E6_SUMMARY.json');ftpred=C.read(C.RAW/'E6_PREDICTIONS.json');ftm=C.read(C.RAW/'E6_POSE_METRICS.json')
    colors=['#64748b','#4292c6','#d97706','#9370ad','#169873'];arms=['identity','PRIOR1','FULL125','OLD_REF217','REALFT_A']
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for row,pop in enumerate(('NATURAL99','CLEAN29')):
        for col,key in enumerate(('translation_cm','rotation_deg')):
            vals=[(ft[pop]['pose'] if a=='REALFT_A' else summaries[pop]['models'][a])['conditional'][key]['median'] for a in arms]
            ax=axes[row,col];bars=ax.bar(arms,vals,color=colors);ax.bar_label(bars,fmt='%.2f',padding=3);ax.set_ylim(0,max(vals)*1.2)
            ax.set_title(pop+' / '+('T 중앙값 (cm)' if col==0 else 'R 중앙값 (°)'));ax.tick_params(axis='x',rotation=15);ax.grid(axis='y',alpha=.2)
    fig.suptitle('동일 GEO 평가 — 낮을수록 좋음 | REALFT_A는 다른 학습 구성의 참고 모델',fontsize=14);fig.tight_layout()
    savefig(fig,'01_pose_overview.png')
    dt=np.array([e1['FULL125'][r['id']]['translation_cm']-e1['identity'][r['id']]['translation_cm'] for r in natural]);dr=np.array([e1['FULL125'][r['id']]['rotation_deg']-e1['identity'][r['id']]['rotation_deg'] for r in natural])
    fig,ax=plt.subplots(figsize=(9,6));recnames=sorted({r['recording'] for r in natural})
    for rec in recnames:
        ix=[j for j,r in enumerate(natural) if r['recording']==rec];ax.scatter(dt[ix],dr[ix],label=f'{rec} (N={len(ix)})',alpha=.8,s=35)
    ax.axhline(0,c='black',lw=.7);ax.axvline(0,c='black',lw=.7);ax.set_xscale('symlog',linthresh=1)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}'))
    ax.set(xlabel='FULL125 - identity: 프레임별 ΔT (cm), symlog 축',ylabel='프레임별 ΔR (°)',title='natural99: 동시 개선 34 / 동시 악화 27 / T만 개선 23 / R만 개선 15')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();savefig(fig,'02_paired_changes.png')
    byrec=C.read(C.DOC/'BY_RECORDING.json');fig,axes=plt.subplots(1,2,figsize=(12,5))
    for col,key in enumerate(('translation_cm','rotation_deg')):
        labels=[];values=[];cs=[]
        for pop,recs in byrec.items():
            for rec,v in recs.items():
                labels.append(f'{pop} / {rec} / N={v["N"]}');values.append(v['models']['FULL125']['conditional'][key]['median']-v['models']['identity']['conditional'][key]['median']);cs.append('#d97706' if pop=='NATURAL99' else '#4292c6')
        ax=axes[col];ax.barh(labels,values,color=cs);ax.axvline(0,c='black',lw=.7);ax.set_title('집계 중앙값 차이 '+('ΔT (cm)' if col==0 else 'ΔR (°)'));ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('recording마다 결과 방향이 다름 — 프레임별 차이의 중앙값과 구별');fig.tight_layout();savefig(fig,'03_recording_changes.png')
    e3=C.read(C.DOC/'E3_SUMMARY.json');fig,axes=plt.subplots(2,2,figsize=(12,7));conds=['CC','CO','OC','OO','nativeOO']
    for row,kind in enumerate(('cover','avoid')):
        for col,key in enumerate(('translation_cm','rotation_deg')):
            ax=axes[row,col]
            for a,color in zip(arms[:3],colors[:3]):
                vals=[e3[kind]['CLEAN29']['models'][a][c]['pose']['conditional'][key]['median'] for c in conds];ax.plot(conds,vals,'o-',color=color,label=a)
            ax.set_title(kind+' / '+('T (cm)' if col==0 else 'R (°)'));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('clean29: 첫 글자 RGB, 둘째 글자 좌표 | C=clean, O=가림 | 3 recording');fig.tight_layout();savefig(fig,'04_rgb_coordinate_controls.png')
    plans=C.read(C.RAW/'E3_MASK_PLANS.json');qc=C.read(C.RAW/'E3_CPU_QC.json');qo=C.read(C.RAW/'E3_R0_PREDICTIONS.json')
    casebook=[]
    for rec in sorted({r['recording'] for r in clean}):
        r=sorted([r for r in clean if r['recording']==rec],key=lambda x:x['id'])[0];i=r['id'];im=C.cv2.imread(str(C.ROOT/r['image']['path']))
        fig,axs=plt.subplots(2,3,figsize=(13,7))
        for j,kind in enumerate(('cover','avoid')):
            occ=masked(im,plans[i],kind)
            photo(axs[j,0],im,qc[i],refs[i]['gt'],f'clean qC | {kind}')
            photo(axs[j,1],occ,qo[i][kind],refs[i]['gt'],f'실제 {kind} 가림 입력 + qO')
            # Same qO laid on clean RGB demonstrates CO without making a new prediction.
            co=copy.deepcopy(qc[i]);S.CORE.selected(co)['keypoints_xy']=copy.deepcopy(S.CORE.selected(qo[i][kind])['keypoints_xy'])
            photo(axs[j,2],im,co,refs[i]['gt'],f'CO 입력: clean RGB + 동일 qO / clean 박스')
        fig.suptitle(rec+' | '+i+'\n주황=입력 예측, 청록=저장 참조 / 흰 점선=검출 박스',fontsize=11)
        fig.tight_layout();name=f'05_e3_inputs_{rec}.jpg';savefig(fig,name);casebook.append(dict(type='E3_INPUT',id=i,rule='lexicographically first clean frame per recording',figure=name,image=r['image']))
    # Representative examples of all four outcome directions, with no best-case ranking.
    selected=[]
    for category,mask in [('both_improve',(dt<0)&(dr<0)),('both_worsen',(dt>0)&(dr>0)),('T_only',(dt<0)&(dr>0)),('R_only',(dt>0)&(dr<0))]:
        eligible=[r for j,r in enumerate(natural) if mask[j]];r=sorted(eligible,key=lambda x:x['id'])[0];selected.append((category,r))
    fig,axs=plt.subplots(4,3,figsize=(13,12))
    for row,(category,r) in enumerate(selected):
        i=r['id'];im=C.cv2.imread(str(C.ROOT/r['image']['path']))
        for col,a in enumerate(('identity','FULL125','REALFT_A')):
            p=ftpred[i] if a=='REALFT_A' else pred[a][i];m=ftm[i] if a=='REALFT_A' else e1[a][i]
            photo(axs[row,col],im,p,refs[i]['gt'],f'{category} / {a}\nT={m["translation_cm"]:.2f} cm, R={m["rotation_deg"]:.2f}°')
        casebook.append(dict(type='NATURAL_DIRECTION',id=i,category=category,rule='lexicographically first ID within each sign quadrant',image=r['image']))
    fig.suptitle('자연 가림의 개선·악화·tradeoff 사례 | 주황=모델 출력, 청록=저장 참조',fontsize=13);fig.tight_layout();savefig(fig,'06_natural_examples.jpg')
    box=C.read(C.RAW/'E2_BOX_DIAGNOSTIC.json')['identity'];tail=sorted(natural,key=lambda r:(-e1['identity'][r['id']]['translation_cm'],r['id']))[:10]
    fig,axs=plt.subplots(5,2,figsize=(12,15))
    for ax,r in zip(axs.flat,tail):
        i=r['id'];im=C.cv2.imread(str(C.ROOT/r['image']['path']));m=e1['identity'][i]
        photo(ax,im,pred['identity'][i],refs[i]['gt'],f'{r["recording"]} / T={m["translation_cm"]:.1f} cm\n{box[i]["category"]}')
        x1,y1,x2,y2=refs[i]['box'];ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,color='#34e5e0',lw=1.5))
        casebook.append(dict(type='TOP10_T',id=i,rule='top ceil(10% of natural99) identity T, tie ID',category=box[i]['category'],image=r['image']))
    fig.suptitle('큰 T 오류 상위 10장 전부 — 선택 박스: 흰 점선 / 참조 박스: 청록',fontsize=13);fig.tight_layout();savefig(fig,'07_detection_tail.jpg')
    stress=C.read(C.DOC/'E5_STRESS_SUMMARY.json')['CLEAN29']['models'];fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,kind in zip(axes,('independent','correlated')):
        values=[stress[a][kind]['clean_R0_restoration']['recovered30to10'] for a in arms[:3]]
        bars=ax.bar(arms[:3],values,color=colors[:3]);ax.bar_label(bars,labels=[f'{v}/116' for v in values],padding=3);ax.set(ylim=(0,125),title=kind+' / 30 px → 10 px 이하 복구 코너',ylabel='코너 수');ax.grid(axis='y',alpha=.2)
    fig.suptitle('같은 clean29·4코너·오차 30 px — 방향 상관성만 변경');fig.tight_layout();savefig(fig,'08_correlated_stress.png')
    # Complete natural99 visual index, ordered by recording then ID.
    gallery=['# 자연 가림 99장 전체 비교','', '모든 프레임을 recording·ID 순으로 표시한다. 왼쪽 identity, 오른쪽 FULL125. 주황은 모델 코너, 청록은 저장 평가 참조이며 독립 실측 pose를 뜻하지 않는다. 흰 점선은 선택 검출 박스다. T/R은 전체 pose 지표로 IoU gate를 적용하지 않는다. 이미지 패널 밖으로 벗어난 예측점은 그림에서만 잘리며 평가에서는 제거하지 않았다.','']
    for rec in recnames:
        rr=sorted([r for r in natural if r['recording']==rec],key=lambda r:r['id'])
        for page in range(0,len(rr),6):
            batch=rr[page:page+6];fig,axs=plt.subplots(len(batch),2,figsize=(11,2.9*len(batch)),squeeze=False)
            for row,r in enumerate(batch):
                i=r['id'];im=C.cv2.imread(str(C.ROOT/r['image']['path']))
                for col,a in enumerate(('identity','FULL125')):
                    m=e1[a][i];photo(axs[row,col],im,pred[a][i],refs[i]['gt'],f'{i}\n{a}: T {m["translation_cm"]:.2f} cm / R {m["rotation_deg"]:.2f}°')
            fig.tight_layout();name=f'gallery_{rec}_{page//6+1:02d}.jpg';savefig(fig,name)
            gallery.extend([f'## {rec} — {page+1}–{page+len(batch)} / {len(rr)}','',f'![{rec} 프레임 비교](figures/{name})',''])
    write(C.DOC/'GALLERY_NATURAL99.md','\n'.join(gallery))
    write(C.DOC/'FIGURE_CASE_SELECTION.json',dict(gallery_rule='all natural99, recording then ID; six frames per page',gallery_frames=99,casebook=casebook,reference='Stored geometry-linked evaluation annotations, not independently measured physical pose',no_new_predictions=True))
    write(C.DOC/'REVIEW_COST_figures.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,image_forwards=0,new_fits=0,GPU_seconds=0))
    print('FIGURES_DONE',len(list(FIG.iterdir())),flush=True)

if __name__=='__main__':main()

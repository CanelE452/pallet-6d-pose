"""Generate tables and honest illustrative cases from frozen measured results."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-paper-closure-mpl')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from . import common as C

MAIN=('R0','RAW_LR5','REF_LR5')
DISPLAY={'R0':'Synthetic-only R0','RAW_LR5':'Raw-pseudo student','REF_LR5':'Corrected-pseudo student','TEACHER':'Frozen Replay teacher'}
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
FIG=C.DOC/'figures'

def fmt(x,precision=3):return f'{x:.{precision}f}'
def texescape(x):return str(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def tab(name,headers,rows,caption,label,wide=False):
    C.save(C.DOC/(name+'.md'),'# '+caption+'\n\n'+C.table(headers,rows))
    env='table*' if wide else 'table'
    lines=[rf'\begin{{{env}}}[t]',r'\centering\small',r'\caption{'+caption+'}',r'\label{tab:'+label+'}',r'\resizebox{'+(r'\textwidth' if wide else r'\columnwidth')+r'}{!}{%',r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}',r'\toprule',
        ' & '.join(texescape(v) for v in headers)+r' \\',r'\midrule']
    lines+=[' & '.join(texescape(v) for v in row)+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}}',rf'\end{{{env}}}']
    C.save(C.PAPER/'generated_tables'/(name+'.tex'),'\n'.join(lines)+'\n')

def savefig(fig,name):
    FIG.mkdir(parents=True,exist_ok=True);(C.PAPER/'figures').mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/(name+'.png'),dpi=145,bbox_inches='tight',facecolor='white')
    fig.savefig(C.PAPER/'figures'/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)

def overlay(ax,im,pred,gt,title,color):
    ax.imshow(im);c=C.selected(pred)
    if c is not None:
        q=np.array(c['keypoints_xy'])[:8]
        for a,b in EDGES:
            if np.isfinite(q[[a,b]]).all():ax.plot(q[[a,b],0],q[[a,b],1],color=color,lw=1.1)
        ax.scatter(q[:,0],q[:,1],c=color,s=12)
    for j,xy in gt.items():
        ax.scatter(*xy,c='#34ef58',marker='x',s=26,linewidths=1.5)
    ax.set_xlim(0,im.width);ax.set_ylim(im.height,0);ax.axis('off');ax.set_title(title,fontsize=9)

def main():
    if (C.DOC/'TABLE_FIGURE_FREEZE.json').exists():
        for b in C.read(C.DOC/'TABLE_FIGURE_FREEZE.json')['tables']:C.verify(b)
        C.verify(C.read(C.DOC/'TABLE_FIGURE_FREEZE.json')['figures'])
        print('TABLES_FIGURES_ALREADY_FROZEN');return
    res=C.read(C.DOC/'CORE_RESULTS.json');q=C.read(C.DOC/'PSEUDO_LABEL_QUALITY.json');pa=C.read(C.DOC/'PAIRED_ANALYSIS.json')
    g=res['groups'];allg=g['ALL'];points=C.read(C.RAW/'ANCHOR_POINTS.json');fm=C.read(C.RAW/'FRAME_METRICS.json');pred=C.read(C.RAW/'PREDICTIONS.json');truth=C.read(C.TRUTH)
    rr={r['id']:r for r in C.records()};anchor={}
    for p in points:anchor.setdefault(p['frame_id'],{})[p['corner_id']]=p['verified_xy']
    q1=q['groups']['ALL'];delta=pa['REF_LR5-minus-RAW_LR5'];rdelta=pa['REF_LR5-minus-R0']
    def qualityrow(a):
        m=q1[a];return [DISPLAY[a],m['n'],*[f"{m['PCK'][str(k)]['correct']}/66 ({100*m['PCK'][str(k)]['fraction']:.2f})" for k in (5,10,20)],fmt(m['median_px']),fmt(m['p90_px']),m['gt20']]
    tab('TABLE1_pseudo_quality',['Output','Points','PCK5 %','PCK10 %','PCK20 %','Med px','P90 px','Above20'],[qualityrow(a) for a in ('R0','TEACHER')],
        'Pseudo-label quality on 66 verified visible points in 16 reused DEV images; fixed native identity. A labeled proxy, not accuracy measured on unlabeled adaptation images.','quality',True)
    def twodrow(a):
        m=allg[a]['twoD'];return [a,*[fmt(100*m['PCK'][str(k)],2) for k in (5,10,20)],fmt(m['matched_pooled_corner8_median_px']),fmt(m['matched_pooled_corner8_P90_px'])]
    tab('TABLE2_main_2d',['Arm','PCK5 %','PCK10 %','PCK20 %','Med px','P90 px'],[twodrow(a) for a in MAIN],
        'Main 2D comparison on 128 images and 985 supported corners. PCK includes detection failures; medians and P90 are conditional on matched detection. LR5 was historically DEV-selected.','main2d')
    def poserow(a):
        m=allg[a]['sixD'];return [a,f"{m['available']}/128",f"{m['axis_correct_count']}/128",fmt(m['rotation_deg']['median']),fmt(m['yaw_deg']['median']),fmt(m['translation_cm']['median']),fmt(m['IoU3D']['median'],4),fmt(m['ADDsym_AUC'],5)]
    tab('TABLE2_main_6d',['Arm','Pose','Axis','R deg','Yaw deg','t cm','IoU3D','ADDsym AUC'],[poserow(a) for a in MAIN],
        'Final 6D under the SAME frozen D9 selector and corner-only solver. Reference is geometry-derived, not independently measured physical pose.','main6d',True)
    tab('TABLE3_severity',['Group','N','Arm','PCK10 %','P90 px','ADDsym AUC','t cm'],[[name,g[name][a]['sixD']['frames'],a,fmt(100*g[name][a]['twoD']['PCK']['10'],2),fmt(g[name][a]['twoD']['matched_pooled_corner8_P90_px']),fmt(g[name][a]['sixD']['ADDsym_AUC'],5),fmt(g[name][a]['sixD']['translation_cm']['median'])] for name in ('CLEAN','MODERATE','SEVERE') for a in MAIN],
        'All severity strata retained. Improved pooled accuracy does not imply improved tails or all pose statistics.','severity',True)
    tab('TABLE4_all_historical_controls',['Arm','PCK10 %','P90 px','ADDsym AUC'],[[a,fmt(100*allg[a]['twoD']['PCK']['10'],2),fmt(allg[a]['twoD']['matched_pooled_corner8_P90_px']),fmt(allg[a]['sixD']['ADDsym_AUC'],5)] for a in C.ARMS],
        'Complete historical grid and order sensitivity. SYN is additional source-only training. ORDER43/44 change data order, not initialization seed. No best-repeat selection.','controls')
    tab('TABLE5_fairness',['Contract','RAW and corrected'],[['Initialization','Same R0 checkpoint'],['Trainable state','Pose branches and flow only'],['Real RGB','217 identical unique images'],['Real/synthetic exposure','2560 / 2560 each'],['Synthetic pool','512 identical images; no negatives'],['Supervised support','Identical raw/ref confidence intersection'],['Optimizer / updates','AdamW / 320'],['Learning rate','1e-5 main; 1e-4 sensitivity'],['Augmentation / seed','Same settings / 42'],['Checkpoint choice','Fixed final last.pt'],['Teacher supervision','9 images / 38 manual corners'],['Only changed target','Supervised pseudo coordinates']],
        'Matched coordinate intervention. Shared image membership is conditioned on the same raw-plus-refined filter acceptance, so this is not a comparison of different selection policies.','fairness')
    tab('TABLE6_recordings',['Recording','N','Raw PCK10','Corr PCK10','Raw AUC','Corr AUC'],[[n,g[n]['R0']['sixD']['frames'],fmt(100*g[n]['RAW_LR5']['twoD']['PCK']['10'],2),fmt(100*g[n]['REF_LR5']['twoD']['PCK']['10'],2),fmt(g[n]['RAW_LR5']['sixD']['ADDsym_AUC'],5),fmt(g[n]['REF_LR5']['sixD']['ADDsym_AUC'],5)] for n in g if n.startswith('REC_')],
        'Every reused recording group, including small or adverse groups. Percentages are descriptive, not independent frame trials.','recordings')
    fig,ax=plt.subplots(figsize=(10.4,3.5));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off')
    boxes=[(.1,1.8,'Real RGB + R0\nraw pseudo'),(3.55,1.8,'Frozen Replay teacher\ncorrected pseudo'),(7.0,1.8,'Matched student training\n+ synthetic replay'),(7.0,.1,'Deployment: RGB student\n+ fixed D9 PnP')]
    for x,y,t in boxes:
        ax.add_patch(FancyBboxPatch((x,y),2.8,.9,boxstyle='round,pad=.06',fc='#e8f2f8',ec='#315b76'));ax.text(x+1.4,y+.45,t,ha='center',va='center',fontsize=10)
    for x1,y1,x2,y2 in ((2.95,2.25,3.45,2.25),(6.4,2.25,6.9,2.25),(8.4,1.7,8.4,1.1)):
        ax.annotate('',(x2,y2),(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.8))
    ax.text(.1,.9,'Raw control: same accepted RGB, masks, updates, replay;\nonly the target coordinates differ.',fontsize=10)
    ax.text(.1,.18,'Teacher adaptation: 9 real images / 38 clicked corners.\nTeacher is absent from student inference.',fontsize=10)
    savefig(fig,'pipeline')
    fig,axs=plt.subplots(1,2,figsize=(9,3.4))
    for ax,key,title in zip(axs,('PCK10','AUC'),('PCK10 (%)','Final ADDsym AUC')):
        names=('CLEAN','MODERATE','SEVERE','ALL');xx=np.arange(4)
        for k,a in enumerate(MAIN):
            vals=[100*g[n][a]['twoD']['PCK']['10'] if key=='PCK10' else g[n][a]['sixD']['ADDsym_AUC'] for n in names]
            ax.bar(xx+(k-1)*.25,vals,.25,label=DISPLAY[a])
        ax.set_xticks(xx,names,fontsize=8);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
    axs[0].legend(fontsize=7);fig.suptitle('Fixed128 reused DEV — no population or selector changes');fig.tight_layout();savefig(fig,'severity')
    manifest=[]
    # Anchor examples: strongest improvement and strongest worsening, both retained.
    af=list(anchor)
    def ad(fid):
        rs=[p for p in points if p['frame_id']==fid]
        return np.mean([p['errors']['TEACHER']-p['errors']['R0'] for p in rs])
    ranked=sorted(af,key=lambda i:(ad(i),i));chosen=ranked[:2]+ranked[-2:]
    for n,fid in enumerate(chosen):
        im=Image.open(C.ROOT/rr[fid]['image']['path']);fig,ax=plt.subplots(1,2,figsize=(10,4))
        for aa,arm,col in zip(ax,('R0','TEACHER'),('#00b7ff','#ec42c8')):
            er=np.mean([p['errors'][arm] for p in points if p['frame_id']==fid])
            overlay(aa,im,pred[arm][fid],anchor[fid],f'{DISPLAY[arm]} | visible mean {er:.2f}px',col)
        fig.suptitle(f'{fid} | verified green crosses | corrected - raw {ad(fid):+.2f}px',fontsize=10)
        name=f'pseudo_quality_{n+1:02d}';savefig(fig,name);manifest.append(dict(name=name,frame_id=fid,type='pseudo',selection='2 strongest visible improvements + 2 strongest deteriorations',delta_px=float(ad(fid))))
    def sd(fid):return float(np.mean(fm['REF_LR5'][fid]['errors'])-np.mean(fm['RAW_LR5'][fid]['errors']))
    ranked=sorted(rr,key=lambda i:(sd(i),i));selected=[]
    for tag,pool in [('improved',ranked[:3]),('worsened',ranked[-3:]),('least_changed',sorted(rr,key=lambda i:(abs(sd(i)),i))[:3])]:
        for fid in pool:selected.append((tag,fid))
    for n,(tag,fid) in enumerate(selected):
        im=Image.open(C.ROOT/rr[fid]['image']['path']);t=truth[fid]
        target={j:xy for j,xy in enumerate(t['gt'][:8]) if t['valid'][j]}
        fig,ax=plt.subplots(1,3,figsize=(13.6,4.2))
        for aa,arm,col in zip(ax,MAIN,('#00b7ff','#f2c430','#ec42c8')):
            er=np.mean(fm[arm][fid]['errors'])
            overlay(aa,im,pred[arm][fid],target,f'{DISPLAY[arm]}\nmean scored error {er:.2f}px',col)
        fig.suptitle(f'{tag}: {fid} | {rr[fid]["severity"]} | corrected - raw {sd(fid):+.2f}px\nGreen: legacy reference (mixed provenance); lines: native 2D predictions, NOT PnP projection',fontsize=9)
        name=f'student_{n+1:02d}_{tag}';savefig(fig,name);manifest.append(dict(name=name,frame_id=fid,type='student',selection=tag,delta_px=sd(fid)))
    C.save(C.DOC/'FIGURE_MANIFEST.json',dict(examples=manifest,source_artifacts=[C.bind(C.RAW/n) for n in ('PREDICTIONS.json','ANCHOR_POINTS.json','FRAME_METRICS.json')],
        native_predictions=True,GT_used_for_posthoc_example_selection_only=True,inference_GT_used=False,
        scalar_errors='Symmetry-aware scoring for full128; native fixed identity for verified66. Lines are native, not remapped by evaluation oracle.',files=[C.bind(p) for p in sorted(FIG.glob('*.png'))]))
    claims={
        'C1':dict(status='SUPPORTED',scope='66 verified visible DEV points only',raw_correct10=44,corrected_correct10=50,evidence='PSEUDO_LABEL_QUALITY.json'),
        'C2':dict(status='SUPPORTED',scope='matched frozen-backbone pose-only historical fits; reused plastic DEV',PCK10_delta_pp=delta['PCK10_delta_pp'],evidence='CORE_RESULTS.json + PAIRED_ANALYSIS.json'),
        'C3':dict(status='PARTIAL',scope='PCK10 and ADDsym AUC improve; P90, rotation/axis and some conditions do not',PCK10_delta_pp=rdelta['PCK10_delta_pp'],evidence='CORE_RESULTS.json'),
        'C4':dict(status='PARTIAL',scope='geometry-reference final ADDsym AUC improves; no all-metric or physical6D generalization',AUC_delta=delta['ADDsym_AUC_delta'],evidence='CORE_RESULTS.json'),
        'C5':dict(status='SUPPORTED',scope='9 teacher-adaptation images / 38 manual corners; evaluation labels extra, not zero-label',evidence='CORE_COMPARABILITY_AUDIT.json'),
        'C6':dict(status='SUPPORTED',scope='Repeated DEV clearly separated from independent confirmation; independence not claimed',evidence='PAPER_CLOSURE_PROTOCOL.json')}
    C.save(C.DOC/'CLAIM_MATRIX.json',claims)
    C.save(C.DOC/'FINAL_DECISION.json',dict(status='PAPER_CORE_SUPPORTED',claim_scope='Descriptive reused ordinary-plastic recording-disjoint DEV evidence only; not independent confirmation',
        method_development='STOP',new_fits=0,extra_seed_runs=0,models_promoted=False,claims=claims,
        remaining='Writing, audit, reproduction checks, reserve provenance audit; no new methods'),True)
    C.save(C.DOC/'PSEUDO_LABEL_QUALITY_REPORT_KO.md','# Q1: 같은 점에서 raw와 보정 비교\n\n'+(C.DOC/'TABLE1_pseudo_quality.md').read_text()+'\n기존 학생을 만든 Replay9/38 보정기만 사용했다. Clean19 보정기 수치와 혼합하지 않았다. 66점은 16장 가시점이며 숨은 점·unlabeled217장의 정확도를 직접 검증한 수치가 아니다.\n\n'+''.join(f'![{m["frame_id"]}](figures/{m["name"]}.png)\n\n' for m in manifest if m['type']=='pseudo'))
    report='# Corrected pseudo-label self-training — 논문 핵심 증거 마감\n\n'
    report+='## 결론\n\n**PAPER_CORE_SUPPORTED — 단, 반복 사용한 일반 플라스틱 DEV 범위.** 새 학습 0회; 기존 공정 비교 12개 학생 결과를 동일128장에서 재평가했다. 모든 지표 우월성이나 새 세션 독립 일반화는 주장하지 않는다. 방법 개발은 STOP한다.\n\n'
    report+=f'Q1: 같은 66점에서 PCK10 **44/66 → 50/66**, median **{q1["R0"]["median_px"]:.3f} → {q1["TEACHER"]["median_px"]:.3f}px**. Q2: 학생 PCK10 **47.51% → 51.47%**, ADDsym AUC **0.33472 → 0.35902**. Q3: R0 **49.14% / 0.33796**보다 두 주 지표는 개선. Q4: 최종6D도 측정했지만 P90·회전·축 선택·일부 조건 악화는 남는다.\n\n'
    report+='![실제 사용한 학습·추론 흐름](figures/pipeline.png)\n\n## 과정과 감독 예산\n\n1. 기존 synthetic-pallet R0를 고정한다. R0에는 upstream COCO-pose pretraining이 있다.\n2. 평가 밖 일반 플라스틱 후보1000장에 R0 confidence/flip·LOO를 적용: 259장.\n3. 고정 Replay9/38 보정 후 LOO로 249장. RAW/REF 모두 동일 승인집합을 사용한다. 필터 통과=정답이라는 의미가 아니다.\n4. 동일 RNG로 512 real 슬롯을 추출한 결과 unique217장. 512 synthetic 슬롯과 5epoch, 320update. raw와 보정의 공통 mask에 좌표만 다르게 준다.\n5. 같은 R0에서 pose head/flow만 학습한 기존 학생을 재사용한다. Backbone/검출/buffer는 고정.\n6. teacher 없이 학생 단독 RGB 추론을 비교한다. 모든6D는 같은 D9와 corner0..7 solver다.\n7. teacher는 실사9장38 manual corner와 합성 replay로 학습된 고정 PoseFix-derived RGB ResNet152이다. 새 refiner, cap, PnP hidden-label 추가, router 또는 loss는 없다.\n\n'
    for n in ('TABLE1_pseudo_quality','TABLE2_main_2d','TABLE2_main_6d','TABLE3_severity','TABLE5_fairness','TABLE4_all_historical_controls','TABLE6_recordings'):
        report+='## '+n+'\n\n'+(C.DOC/(n+'.md')).read_text().split('\n\n',1)[1]+'\n'
    report+='## 짝지은 변화와 한계\n\n'
    report+=f'Corrected vs raw: frame 평균오차 개선92 / 악화28 / 동일8. 10px 진입51점 / 이탈12점(분모985). PCK10 차이 {delta["PCK10_delta_pp"]:.2f}pp, 7 recording cluster bootstrap 95% interval [{delta["recording_bootstrap_CI95_pp"][0]:.2f}, {delta["recording_bootstrap_CI95_pp"][1]:.2f}]pp. 반복 DEV·historical search 미보정 기술통계이며 독립 검증 p-value가 아니다.\n\n'
    report+='전체 corner P90은 raw41.958 → corrected42.085px, R0는40.902px다. Severe translation median은 R0보다 악화되고 Moderate PCK10도 R0보다 낮다. 큰 오류 복구는 해결됐다고 하지 않는다. Teacher 수동9/38 및 shared teacher-based selection 효과를 제거한 순수 zero-real-label 비교가 아니다. 좌표 보정 intervention만 격리한 조건부 비교다.\n\n'
    report+='![난도별 모든 결과](figures/severity.png)\n\n## 실제 이미지: 개선·악화·변화 적음\n\n정량 결과에서 정해진 순위로 고른 설명용 사례다. 원본 native 2D 예측선이며 PnP 투영이 아니다. 초록 X는 reference, R0 파랑 / raw 학생 노랑 / 보정 학생 자홍.\n\n'
    report+=''.join(f'### {m["selection"]}: {m["frame_id"]}\n\n![{m["name"]}](figures/{m["name"]}.png)\n\n' for m in manifest)
    report+='## 보조 실험과 제외 범위\n\nClean19 S0/S1/S2는 occlusion 확장, hard8/H_MANUAL은 추가 manual 확장, GEO_LINEAR는 selector 확장으로만 분류한다. 다른 teacher·감독 budget·학습 조건을 섞어 main self-training 이득이라고 하지 않는다. 초록·목재 전체 및 독립 미사용 촬영 일반화는 이번 main으로 검증하지 않았다.\n\n원고: [`selftraining_submission_v1`](../../paper/selftraining_submission_v1/manuscript.tex). 전체 수치·체크포인트·분모·해시는 JSON과 재현 문서를 참조한다.\n'
    C.save(C.DOC/'REPORT_KO.md',report)
    sources=[C.bind(C.DOC/n) for n in ('CORE_RESULTS.json','PSEUDO_LABEL_QUALITY.json','PAIRED_ANALYSIS.json','CORE_COMPARABILITY_AUDIT.json')]
    C.save(C.PAPER/'generated_tables/NUMBER_PROVENANCE.json',dict(sources=sources,table_mapping={'TABLE1':'PSEUDO_LABEL_QUALITY.groups.ALL','TABLE2':'CORE_RESULTS.groups.ALL','TABLE3':'CORE_RESULTS.groups.CLEAN/MODERATE/SEVERE','TABLE4':'CORE_RESULTS.groups.ALL all13arms','TABLE5':'CORE_COMPARABILITY_AUDIT','TABLE6':'CORE_RESULTS.groups.REC_*'},tables=[C.bind(p) for p in sorted((C.PAPER/'generated_tables').glob('*.tex'))]))
    C.save(C.DOC/'TABLE_FIGURE_FREEZE.json',dict(created_at=C.now(),decision=C.bind(C.DOC/'FINAL_DECISION.json'),tables=[C.bind(p) for p in sorted(C.DOC.glob('TABLE*.md'))],figures=C.bind(C.DOC/'FIGURE_MANIFEST.json')),True)
    print('TABLES_FIGURES_CLAIM_FROZEN',len(manifest),flush=True)

if __name__=='__main__':main()

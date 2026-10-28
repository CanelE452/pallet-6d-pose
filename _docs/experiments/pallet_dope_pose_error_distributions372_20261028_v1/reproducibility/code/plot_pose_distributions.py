"""Descriptive scatter plots from already-published per-frame pose scalars only."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import datetime as dt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/minjae/Documents/pallet-research/dope-wood640-eval80-20261027-v1/combined372/evaluation/final_v1/supplemental-pose')
KINDS=['PLASTIC','WOOD','GREEN']
NAMES={'PLASTIC':'Rectangular plastic','WOOD':'Small wooden','GREEN':'Square plastic'}
TOTAL={'PLASTIC':128,'WOOD':125,'GREEN':119}
STAGES=['R0','R1','R2']
LABELS={'R0':'Initial (R0)','R1':'Self-training 1 (R1)','R2':'Self-training 2 (R2)'}
COLORS={'R0':'#6B7280','R1':'#E28B16','R2':'#007FB5'}
MARKERS={'R0':'o','R1':'^','R2':'s'}
METRICS=['translation_cm','rotation_symmetry_deg']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvread(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def savejson(p,obj):
    with p.open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def num(s):
    if s=='':return None
    v=float(s);assert math.isfinite(v) and v>=0;return v
def truth(s):
    assert s in ['True','False'];return s=='True'
def stats(values):
    a=np.asarray(values,dtype=float);assert len(a)>0
    return dict(n=len(a),mean=float(a.mean()),population_SD=float(a.std(ddof=0)),
        median=float(np.median(a)),q1=float(np.quantile(a,.25)),q3=float(np.quantile(a,.75)),
        p90=float(np.quantile(a,.90)),p95=float(np.quantile(a,.95)),maximum=float(a.max()))
def jitter(kind,frameid):
    return (int(hashlib.sha256((kind+'|'+frameid).encode()).hexdigest()[:12],16)/(16**12-1)-.5)*.36

def main():
    assert not (OUT/'COMPLETE.json').exists(),'Preserve completed output'
    bindings={str(SOURCE/n):sha(SOURCE/n) for n in ['FRAME_ERRORS.csv','MODEL_SUMMARY.csv','RESULTS.md']}
    raw=csvread(SOURCE/'FRAME_ERRORS.csv');summaries=csvread(SOURCE/'MODEL_SUMMARY.csv')
    selected=[];index={};source_summary={}
    for x in summaries:
        stage='R0' if x['condition']=='R0' and x['round']=='0' else 'R'+x['round'] if x['condition']=='full' and x['round'] in ['1','2'] else None
        if stage:
            key=(x['kind'],stage);assert key not in source_summary;source_summary[key]=x
    for line,x in enumerate(raw,2):
        stage='R0' if x['condition']=='R0' and x['round']=='0' else 'R'+x['round'] if x['condition']=='full' and x['round'] in ['1','2'] else None
        if stage is None:continue
        kind=x['kind'];assert kind in KINDS
        t=num(x['translation_error_m']);r=num(x['rotation_symmetry_deg'])
        ref=truth(x['reference_PnP_success']);pred=truth(x['model_PnP_success']);cross=truth(x['cross_hypothesis_rotation_NA'])
        assert (t is not None)==(ref and pred)
        assert (r is not None)==(ref and pred and not cross)
        v=dict(kind=kind,stage=stage,id=x['id'],session=x['session'],image_sha256=x['image_sha256'],
            translation_cm=None if t is None else 100*t,rotation_symmetry_deg=r,
            reference_PnP_success=ref,model_PnP_success=pred,
            cross_hypothesis_rotation_NA=cross,source_line=line,
            prediction_hypothesis=x['prediction_hypothesis'],reference_hypothesis=x['reference_hypothesis'])
        key=(kind,stage,x['id']);assert key not in index;index[key]=v;selected.append(v)
    assert len(selected)==1116 and len(source_summary)==9
    common={};common_rows=[];coverage=[];summary=[];checks=0
    for kind in KINDS:
        idsets=[{x['id'] for x in selected if x['kind']==kind and x['stage']==stage} for stage in STAGES]
        assert all(s==idsets[0] and len(s)==TOTAL[kind] for s in idsets)
        for metric in METRICS:
            common[kind,metric]={id for id in idsets[0] if all(index[kind,s,id][metric] is not None for s in STAGES)}
        for id in sorted(idsets[0]):
            hashes={index[kind,s,id]['image_sha256'] for s in STAGES};assert len(hashes)==1
            common_rows.append(dict(kind=kind,id=id,image_sha256=hashes.pop(),
                translation_all3=id in common[kind,'translation_cm'],
                rotation_all3=id in common[kind,'rotation_symmetry_deg'],
                both_errors_all3=id in common[kind,'translation_cm'] & common[kind,'rotation_symmetry_deg']))
        for stage in STAGES:
            rr=[x for x in selected if x['kind']==kind and x['stage']==stage]
            old=source_summary[kind,stage]
            coverage.append(dict(kind=kind,stage=stage,n_evaluation=TOTAL[kind],
                n_reference_PnP=sum(x['reference_PnP_success'] for x in rr),
                n_predicted_PnP=sum(x['model_PnP_success'] for x in rr),
                n_translation=sum(x['translation_cm'] is not None for x in rr),
                n_rotation=sum(x['rotation_symmetry_deg'] is not None for x in rr),
                n_joint_scatter=sum(all(x[m] is not None for m in METRICS) for x in rr),
                n_cross_hypothesis_rotation_NA=sum(x['cross_hypothesis_rotation_NA'] for x in rr),
                n_common_translation=len(common[kind,'translation_cm']),
                n_common_rotation=len(common[kind,'rotation_symmetry_deg'])))
            assert int(old['n_DEV'])==TOTAL[kind]
            for metric in METRICS:
                a=stats([x[metric] for x in rr if x[metric] is not None])
                prefix='translation_error_cm' if metric=='translation_cm' else metric
                for key,oldkey in [('n','n'),('mean','mean'),('population_SD','std'),('median','median'),('q1','q1'),('q3','q3')]:
                    assert math.isclose(a[key],float(old[prefix+'_'+oldkey]),rel_tol=2e-12,abs_tol=2e-12),(kind,stage,metric,key)
                    checks+=1
                for population in ['all_observed','common_three_stages']:
                    values=[x[metric] for x in rr if x[metric] is not None and (population=='all_observed' or x['id'] in common[kind,metric])]
                    summary.append(dict(kind=kind,stage=stage,metric=metric,population=population,n_evaluation=TOTAL[kind],**stats(values)))
    csvwrite(OUT/'data/CASEWISE_ERRORS.csv',selected)
    csvwrite(OUT/'data/COMMON_SUBSET_IDS.csv',common_rows)
    csvwrite(OUT/'data/COVERAGE.csv',coverage)
    csvwrite(OUT/'data/DISTRIBUTION_SUMMARY.csv',summary)
    savejson(OUT/'INPUTS.json',dict(allowed_files_sha256=bindings,raw_images_or_annotations_opened=0,
        training_or_inference=False,selected_conditions=['R0/0','full/1','full/2'],
        translation='100*stored translation_error_m; camera-coordinate center Euclidean error, cm',
        rotation='stored rotation_symmetry_deg; same geometry hypothesis only; symmetry rule unchanged',
        scope='SUPPLEMENTAL_2D_RECONSTRUCTED_REFERENCE',
        population_SD_ddof=0,source_summary_scalar_checks=checks))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titlesize':13,'axes.labelsize':11,'savefig.facecolor':'white'})
    max_t=max(x['translation_cm'] for x in selected if x['translation_cm'] is not None)
    max_r=max(x['rotation_symmetry_deg'] for x in selected if x['rotation_symmetry_deg'] is not None)
    lim_t=max_t*1.10;lim_r=max_r*1.08
    tick_t=[0,1,3,10,30,100,300];tick_r=[0,1,3,10,30,90,180]
    legend=[Line2D([],[],marker=MARKERS[s],color=COLORS[s],ls='',markersize=7,label=LABELS[s]) for s in STAGES]
    files=[]
    def savepng(fig,name):
        p=OUT/'figures'/name;assert not p.exists();fig.savefig(p,dpi=250);files.append(str(p))
    def scale(ax,which,limit):
        getattr(ax,'set_'+which+'scale')('symlog',linthresh=1,linscale=1)
        getattr(ax,'set_'+which+'lim')(0,limit)
        ticks=tick_t if which=='x' else tick_r
        getattr(ax,'set_'+which+'ticks')([t for t in ticks if t<=limit])
        getattr(ax,'set_'+which+'ticklabels')([str(t) for t in ticks if t<=limit])
    def scatter(kinds,log=True):
        fig,axs=plt.subplots(1,len(kinds),figsize=(14.2,6.0) if len(kinds)==3 else (6.4,6.0),squeeze=False)
        fig.subplots_adjust(left=.065 if len(kinds)==3 else .13,right=.985,top=.72,bottom=.24,wspace=.25)
        fig.text(.065,.955,'Pose-error distributions: initial model and self-training',fontsize=17,weight='bold') if len(kinds)==3 else fig.text(.13,.955,'Pose-error distributions',fontsize=17,weight='bold')
        fig.text(.065 if len(kinds)==3 else .13,.905,'One point = one frame/model with both errors available; smaller errors are at the lower left.',fontsize=10 if len(kinds)==3 else 8)
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.51,.875),ncol=3,frameon=False,fontsize=10 if len(kinds)==3 else 8)
        for ax,kind in zip(axs[0],kinds):
            counts=[]
            for stage in STAGES:
                rr=sorted([x for x in selected if x['kind']==kind and x['stage']==stage and all(x[m] is not None for m in METRICS)],key=lambda x:x['id'])
                counts.append(len(rr))
                ax.scatter([x['translation_cm'] for x in rr],[x['rotation_symmetry_deg'] for x in rr],
                    color=COLORS[stage],marker=MARKERS[stage],s=28,alpha=.58,linewidths=.3,edgecolors='white',label=stage)
            ax.set_title(f"{NAMES[kind]}\nEvaluation frames: {TOTAL[kind]}",pad=12)
            if log:
                scale(ax,'x',lim_t);scale(ax,'y',lim_r)
            else:ax.set_xlim(0,lim_t);ax.set_ylim(0,lim_r)
            ax.set_xlabel('Translation error (cm)');ax.set_ylabel('Rotation error (deg)')
            ax.grid(alpha=.22,linewidth=.6);ax.set_axisbelow(True)
            ax.text(.03,.97,'Available paired errors\n'+', '.join(f'{s}: {n}' for s,n in zip(STAGES,counts)),
                transform=ax.transAxes,va='top',fontsize=8.5,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=3))
        x=.065 if len(kinds)==3 else .13
        notes=[('Both axes use a symmetric-log scale: 0-1 is linear; larger errors are logarithmic.' if log else 'Both axes are linear; the full error range is retained.'),
            'All available paired errors are shown. Missing/failed poses and undefined rotations are not zero.',
            'Errors use the existing 2D-reconstructed reference; this is a supplemental pose assessment.',
            'Per-model frame sets differ. See the all-observed and common-frame distribution panels.']
        for i,line in enumerate(notes):fig.text(x,.15-i*.029,line,fontsize=8.6 if len(kinds)==3 else 7.0,color='#4B5563')
        return fig
    def strip(common_only):
        fig,axs=plt.subplots(2,3,figsize=(14.2,9.6))
        fig.subplots_adjust(left=.065,right=.985,top=.80,bottom=.19,wspace=.25,hspace=.40)
        title='Same-frame comparison across R0, R1 and R2' if common_only else 'All available errors: individual-frame distributions'
        fig.text(.065,.955,title,fontsize=18,weight='bold')
        subtitle='Each metric uses frames with a valid error in all three models.' if common_only else 'Translation and rotation retain their own available frame sets, matching Table 4-1.'
        fig.text(.065,.913,subtitle,fontsize=11)
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.51,.89),ncol=3,frameon=False)
        for j,kind in enumerate(KINDS):
            for i,metric in enumerate(METRICS):
                ax=axs[i,j];counts=[]
                for z,stage in enumerate(STAGES):
                    rr=sorted([x for x in selected if x['kind']==kind and x['stage']==stage and x[metric] is not None and (not common_only or x['id'] in common[kind,metric])],key=lambda x:x['id'])
                    values=np.asarray([x[metric] for x in rr]);counts.append(len(rr))
                    ax.scatter([z+jitter(kind,x['id']) for x in rr],values,color=COLORS[stage],marker=MARKERS[stage],s=20,alpha=.63,edgecolors='white',linewidths=.2)
                    q1,med,q3=np.quantile(values,[.25,.5,.75])
                    ax.plot([z,z],[q1,q3],color='#111827',lw=2.0,zorder=5)
                    ax.plot([z-.23,z+.23],[med,med],color='#111827',lw=2.0,zorder=5)
                ax.set_yscale('symlog',linthresh=1,linscale=1)
                ticks=tick_t if metric=='translation_cm' else tick_r
                limit=lim_t if metric=='translation_cm' else lim_r
                ax.set_ylim(0,limit);ax.set_yticks([t for t in ticks if t<=limit]);ax.set_yticklabels([str(t) for t in ticks if t<=limit])
                ax.set_xlim(-.5,2.5);ax.set_xticks([0,1,2]);ax.set_xticklabels([f'{s}\nn={n}' for s,n in zip(STAGES,counts)])
                ax.set_ylabel('Translation error (cm)' if i==0 else 'Rotation error (deg)')
                ax.set_title(NAMES[kind] if i==0 else '')
                ax.grid(axis='y',alpha=.22,linewidth=.6);ax.set_axisbelow(True)
        notes=['One point = one available frame error. Black bars: median and interquartile range; all tails retained.',
            'Vertical axes: symmetric-log (linear 0-1, logarithmic above 1). Horizontal jitter is for visibility only.',
            'Errors vs. existing 2D-reconstructed reference; rotation uses the original symmetry and geometry masks.',
            ('Common-frame sets differ between translation and rotation. Pose failures remain outside these distributions.' if common_only else 'Missing errors are excluded, not zero. Model-specific n changes as more poses become available.'),
            'WOOD125 includes the approved 80 additional frames; mixed resolutions and shared adaptation recordings.']
        for i,line in enumerate(notes):fig.text(.065,.137-i*.025,line,fontsize=9,color='#4B5563')
        return fig
    pdf=OUT/'figures/POSE_ERROR_DISTRIBUTIONS.pdf';assert not pdf.exists()
    with PdfPages(pdf,metadata={'Title':'Pallet pose-error distributions, R0/R1/R2','Author':'Pallet pose research','Subject':'Existing scalar errors only; no retraining or inference'}) as pp:
        f=scatter(KINDS,True);savepng(f,'POSE_ERROR_SCATTER_symlog.png');pp.savefig(f);plt.close(f)
        f=scatter(KINDS,False);savepng(f,'POSE_ERROR_SCATTER_linear.png');pp.savefig(f);plt.close(f)
        f=strip(False);savepng(f,'ALL_OBSERVED_ERROR_POINTS.png');pp.savefig(f);plt.close(f)
        f=strip(True);savepng(f,'COMMON_FRAME_ERROR_POINTS.png');pp.savefig(f);plt.close(f)
    files.append(str(pdf))
    for kind in KINDS:
        f=scatter([kind],True);savepng(f,f'{kind}_POSE_ERROR_SCATTER.png');plt.close(f)
    note=('프레임별 자세 오차 분포: 초기 R0, 기존 본 실험 LOO+flip self-training R1/R2.\n'
        '최근 개수 비례 비율 실험의 R2 또는 합성 대조군은 이 그림에 포함하지 않았다.\n'
        '점 한 개는 한 영상의 한 모델 오차다. 회색 R0, 주황 R1, 파랑 R2.\n'
        '2차원 산점도는 이동/회전 오차가 모두 있는 표본만 표시한다. 이동 전체 표본 수와 다를 수 있다.\n'
        'ALL_OBSERVED는 각 오차의 전체 관측값, COMMON_FRAME은 지표별 세 모델 공통 프레임이다.\n'
        '실패/NA는 오차0으로 표시하지 않았다. 누락과 가설 불일치는 data/COVERAGE.csv에 있다.\n'
        '군집 수를 가정하거나 유리한 프레임을 선택하지 않았다. 전체 꼬리와 선형 축 버전도 제공한다.\n'
        '오차 축의 symlog는 0~1 선형, 그 이상 로그다. 오차 수치 자체에는 jitter를 적용하지 않았다.\n'
        '점 분포도의 범주 축 jitter는 종류+frame id SHA로 고정되며 세 모델에서 같은 frame offset이다.\n'
        '원본 참조는 독립6D 실측이 아니라 2D 기준점·K·등록 치수에서 재구성한 보조 자세다.\n'
        '추가 목재80장 K 품질 UNKNOWN/자세 부호 UNCONFIRMED 등 기존 한계는 변하지 않는다.\n'
        '입력 파일은 허용된 기존 공개 파생 FRAME_ERRORS/MODEL_SUMMARY/RESULTS 세 개뿐이다.\n'
        '원본 영상·원시 정답·다른 보호 FINAL·가중치 열람 없음. 새 학습·추론·기존 결과 수정 없음.\n'
        '입력 SHA를 실행 전후 비교했다. population SD(ddof=0)는 원 표와 동일하다.\n')
    with (OUT/'README.txt').open('x') as f:f.write(note)
    assert all(sha(p)==h for p,h in bindings.items())
    outputs={str(p.relative_to(OUT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in OUT.rglob('*') if p.is_file()}
    result=dict(complete=True,utc=dt.datetime.now(dt.timezone.utc).isoformat(),input_sha256=bindings,
        input_hashes_unchanged=True,selected_frame_model_rows=1116,evaluation_frames=372,model_groups=9,
        source_summary_scalar_checks=checks,population_rows=len(summary),
        common_frames={k:{m:len(common[k,m]) for m in METRICS} for k in KINDS},
        all_errors_retained_in_marginal_panels=True,outliers_removed=0,metric_jitter_applied=False,
        raw_images_or_annotations_opened=0,training_or_inference=False,
        output_sha256=outputs,plot_review_pending=True)
    savejson(OUT/'COMPLETE.json',result)
    print(json.dumps({k:result[k] for k in ['complete','selected_frame_model_rows','source_summary_scalar_checks','common_frames']}),flush=True)

if __name__=='__main__':main()

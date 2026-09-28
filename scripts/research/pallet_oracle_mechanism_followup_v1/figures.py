"""Measured plots only; never change historical paper/figures or invent imagery."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-oracle-followup-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import common as C

def c2_figures():
    from PIL import Image
    from scripts.research.pallet_selftraining_paper_closure_v1.report import overlay
    path=C.DOC/'cycles/C2_REAL_AFFINE_OFF/RESULTS.json'
    if not path.exists():return []
    result=C.read(path);raw=C.RAW/'cycles/C2_REAL_AFFINE_OFF'
    fig,axs=plt.subplots(1,2,figsize=(10,3.8));colors=['#aaa','#d8af55','#8565aa','#c98944','#498f83']
    for ax,metric in zip(axs,('PCK10','AUC')):
        for j,arm in enumerate(('R0','OLD_RAW','OLD_REF','NEW_RAW','NEW_REF')):
            values=[]
            for mat in ('PLASTIC','WOOD'):
                row=result['materials'][mat]['groups']['ALL'][arm]
                values.append(100*row['twoD']['PCK']['10'] if metric=='PCK10' else row['sixD']['ADDsym_AUC'])
            ax.bar(np.arange(2)+(j-2)*.15,values,.15,color=colors[j],label=arm)
        ax.set_xticks([0,1],['Plastic128','Wood45']);ax.set_title(metric);ax.spines[['top','right']].set_visible(False)
    axs[0].legend(fontsize=7,ncol=2);fig.suptitle('C2: real affine-off did not improve the previous corrected student',fontsize=11)
    fig.tight_layout();save(fig,'c2_matched_comparison')
    examples=[];truth=C.read(C.P.TRUTH)
    for mat in ('PLASTIC','WOOD'):
        previous=C.P if mat=='PLASTIC' else C.M
        allowed_manifest=C.read(previous.DOC/'FIGURE_MANIFEST.json')
        key='frame_id' if mat=='PLASTIC' else 'id'
        allowed={r[key] for r in allowed_manifest['examples']}
        metadata={r['id']:r for r in C.read(raw/f'{mat}_METADATA.json')}
        pred=C.read(raw/f'{mat}_PREDICTIONS.json');fm=C.read(raw/f'{mat}_FRAME_METRICS.json')
        oldpred=C.read(previous.RAW/'PREDICTIONS.json')
        oldarm='REF_LR5' if mat=='PLASTIC' else 'WOOD_REF_LR5'
        eligible=sorted(allowed&set(metadata))
        delta=lambda fid:float(np.mean(fm['NEW_REF'][fid]['errors'])-np.mean(fm['OLD_REF'][fid]['errors']))
        ranked=sorted(eligible,key=lambda fid:(delta(fid),fid))
        picks=[('improved',ranked[0]),('worsened',ranked[-1])]
        for tag,fid in picks:
            value=delta(fid)
            # Honest fallback labels if all approved examples move in one direction.
            if tag=='improved' and value>=0:tag='least_worsened'
            if tag=='worsened' and value<=0:tag='least_improved'
            im=Image.open(C.ROOT/metadata[fid]['image']['path']);C.verify(metadata[fid]['image'])
            targets={ci:q for ci,q in enumerate(truth[fid]['gt'][:8]) if truth[fid]['valid'][ci]}
            fig,axs=plt.subplots(1,3,figsize=(13.2,4))
            pp=[oldpred['R0'][fid],oldpred[oldarm][fid],pred['NEW_REF'][fid]]
            mm=['R0','OLD_REF','NEW_REF']
            for ax,p,a,color in zip(axs,pp,mm,('#00a8db','#c232be','#e69f00')):
                overlay(ax,im,p,targets,f'{a} | scored mean {np.mean(fm[a][fid]["errors"]):.2f}px',color)
            fig.suptitle(f'{mat} {tag}: {fid} | NEW_REF - OLD_REF {value:+.3f}px\nGreen crosses = legacy reference; colored lines = native 2D, NOT PnP. Previously published RGB only.',fontsize=9)
            name=f'c2_{mat.lower()}_{tag}';save(fig,name)
            examples.append(dict(name=name,id=fid,material=mat,category=tag,delta_mean_px=value,
                previous_publication=C.bind(previous.DOC/'FIGURE_MANIFEST.json'),image=metadata[fid]['image'],
                selection='min/max scored mean error difference within already published IDs only; not used in training or metric selection'))
    return examples

def save(fig,name):
    folder=C.DOC/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=150,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def c3_figures():
    path=C.DOC/'cycles/C3_MANUAL38_CAPABILITY/RESULTS.json'
    if not path.exists():return []
    result=C.read(path);fig,axs=plt.subplots(1,3,figsize=(12,3.6))
    for ax,group in zip(axs,('ALL','PLASTIC','WOOD')):
        rows=result['TRAIN_capability']['groups'][group]
        colors=['#888','#beaa62','#4b9283']
        values=[rows[a]['MANUAL9']['mean_px'] for a in ('R0','RAW9','MANUAL9')]
        ax.bar(range(3),values,color=colors)
        for i,v in enumerate(values):ax.text(i,v+.3,f'{v:.2f}px',ha='center',fontsize=8)
        ax.set_ylim(0,32);ax.set_xticks(range(3),['R0','RAW9','MANUAL9']);ax.set_title(group+' | TRAIN manual support')
        ax.set_ylabel('Mean residual to stored manual coordinates');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('C3 finite TRAIN fitting: some gain, three Wood gross errors remain; NOT a generalization bound',fontsize=10)
    fig.tight_layout();save(fig,'c3_train_capability')
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for ax,metric in zip(axs,('PCK10','AUC')):
        for j,arm in enumerate(('R0','OLD_REF','RAW9','MANUAL9')):
            values=[]
            for mat in ('PLASTIC','WOOD'):
                row=result['materials'][mat]['groups']['ALL'][arm]
                values.append(100*row['twoD']['PCK']['10'] if metric=='PCK10' else row['sixD']['ADDsym_AUC'])
            ax.bar(np.arange(2)+(j-1.5)*.18,values,.18,color=['#aaa','#9670ba','#d0a044','#498f83'][j],label=arm)
        ax.set_xticks([0,1],['Plastic128','Wood45']);ax.set_title(metric);ax.spines[['top','right']].set_visible(False)
    axs[0].legend(fontsize=8,ncol=2);fig.suptitle('C3: RAW9/MANUAL9 matched; R0 and old REF are descriptive across different TRAIN sets',fontsize=10)
    fig.tight_layout();save(fig,'c3_development_tradeoff')
    return [path]

def main():
    poses=C.read(C.DOC/'ORACLE_POSE_RESULTS.json')['materials']
    coords=C.read(C.DOC/'ORACLE_COORDINATE_RESULTS.json')['materials']
    train=C.read(C.DOC/'TRAIN_TARGET_TRANSFER.json')
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for ax,mat in zip(axs,('PLASTIC','WOOD')):
        a='REF_LR5' if mat=='PLASTIC' else 'WOOD_REF_LR5';v=poses[mat]['groups']['ALL'][a]
        # Whole-output oracle has a distinct candidate set. Never add the gaps.
        w=coords[mat]['whole_production_pose_expert_oracle']
        whole=w['whole_output_oracle_AUC']
        values=[v['current']['ADDsym_AUC'],v['oracle']['ADDsym_AUC'],whole]
        ax.bar(range(3),values,color=['#3267a8','#de9553','#5a9c78'])
        for i,n in enumerate(values):ax.text(i,n+.005,f'{n:.4f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(values)+.08);ax.set_xticks(range(3),['REF\nproduction','Same W/D\nGT-choice','4 whole outputs\nGT-choice'])
        ax.set_title(mat+' | reused DEV');ax.set_ylabel('Normalized ADDsym AUC');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Different oracle sets: diagnostic headroom, NOT deployable accuracy',fontsize=11);fig.tight_layout();save(fig,'oracle_headroom')
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for ax,mat in zip(axs,('PLASTIC','WOOD')):
        v=train[mat]['all']['corners'];values=[v['residuals'][a]['ref']['mean_px'] for a in C.ARMS]
        ax.bar(range(3),values,color=['#8b8f94','#d0a044','#9670ba'])
        for i,n in enumerate(values):ax.text(i,n+.04,f'{n:.3f}px',ha='center')
        ax.set_xticks(range(3),['R0','RAW student','REF student']);ax.set_ylim(0,4.8);ax.set_ylabel('Native mean residual to corrected TRAIN target')
        ax.set_title(f"{mat}: {train[mat]['unique_images']} images / {v['points']} corners")
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Partial target transfer is not physical correctness',fontsize=11);fig.tight_layout();save(fig,'train_target_transfer')
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for ax,mat in zip(axs,('PLASTIC','WOOD')):
        v=C.read(C.DOC/f'SIGNAL_DIAGNOSTIC_{mat}.json')['results']['REF_LR5']
        vals=[]
        for mode in ('EXISTING_AFFINE','NO_RANDOM_AFFINE'):
            e=[z for r in v[mode]['target_following'] for z in r['normalized_errors']]
            vals.append([np.mean(e),np.median(e),np.quantile(e,.9)])
        for j,values in enumerate(vals):ax.bar(np.arange(3)+(j-.5)*.3,values,.3,label=['Original affine','Affine off'][j])
        ax.set_xticks(range(3),['Mean','Median','P90']);ax.set_title(mat+' | 32 TRAIN real exposures');ax.set_ylabel('Residual / target box diagonal');ax.spines[['top','right']].set_visible(False)
    axs[0].legend(fontsize=8);fig.suptitle('Frozen probe: Plastic mean dominated by one outlier; not a causal conclusion',fontsize=10)
    fig.tight_layout();save(fig,'augmentation_probe')
    examples=c2_figures();extra_sources=c3_figures()
    from pathlib import Path
    sources=[C.DOC/n for n in ('ORACLE_POSE_RESULTS.json','ORACLE_COORDINATE_RESULTS.json','TRAIN_TARGET_TRANSFER.json',
                              'SIGNAL_DIAGNOSTIC_PLASTIC.json','SIGNAL_DIAGNOSTIC_WOOD.json')]
    sources.extend([Path(__file__),C.P.TRUTH,C.DOC/'cycles/C2_REAL_AFFINE_OFF/RESULTS.json',*extra_sources])
    for mat,previous in [('PLASTIC',C.P),('WOOD',C.M)]:
        sources.extend([previous.DOC/'FIGURE_MANIFEST.json',previous.RAW/'PREDICTIONS.json'])
        sources.extend(C.RAW/'cycles/C2_REAL_AFFINE_OFF'/f'{mat}_{name}.json' for name in ('METADATA','PREDICTIONS','FRAME_METRICS'))
    C.save(C.DOC/'FIGURE_MANIFEST.json',dict(kind='measured plots and previously published RGB overlays only',examples=examples,
        files=[C.bind(p) for p in sorted((C.DOC/'figures').glob('*.png'))],sources=[C.bind(p) for p in sources]))
    print('DIAGNOSTIC_FIGURES_READY',flush=True)

if __name__=='__main__':main()

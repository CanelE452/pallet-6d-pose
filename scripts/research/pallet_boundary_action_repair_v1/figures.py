"""Scientific plots from sealed evaluated rows; no fitting or pose changes."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import resource
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import read, rows, sha, write
from .registry import B_ARMS, L_GROUPS, SEEDS, ALL_ARMS, COMPARATORS

LABELS={'N0':'N3 + SubPix / N0','E0':'Historical statewise / E0',
 'E0_REGENERATED':'Regenerated common E0','R0':'Raw YOLO + F',
 'POSEFIX_RAW':'PoseFix raw','POSEFIX_SUBPIX_PF_ANCHOR':'PoseFix + SubPix (PF)',
 'POSEFIX_SUBPIX_YOLO_ANCHOR':'PoseFix + SubPix (YOLO)',
 'LEGACY_L_HEAD_s20261018':'Historical L, seed18','LEGACY_B_FIXED_BANK':'Historical B fixed bank',
 'LEGACY_B_FULL':'Historical B full','LEGACY_EDGE':'Historical edge full','LEGACY_AB_FULL':'Historical AB full',
 'B_FINITE_BANK':'B finite / fixed bank','B_FINITE_FULL':'B finite / full solve',
 'B_FINITE_LOCAL':'B finite / local','EDGE_FINITE_FULL':'Edge finite / full solve'}
COLORS={'N0':'#238b45','E0':'#2376b7','E0_REGENERATED':'#31a6be',
 'POSEFIX_SUBPIX_PF_ANCHOR':'#c78b00','POSEFIX_SUBPIX_YOLO_ANCHOR':'#877500',
 'LEGACY_L_HEAD_s20261018':'#7c58a7','B_FINITE_FULL':'#da6a18','B_FINITE_LOCAL':'#b54b15',
 'D1_H1_MLP_s20261018':'#c33552','D1_H1_PAIR_MLP_s20261018':'#a43ca8'}


def label(arm):
    if arm in LABELS:return LABELS[arm]
    return arm.replace('_s20261018',' / seed18').replace('_s20261019',' / seed19').replace('_s20261020',' / seed20').replace('_',' ')


def csvfile(path,records):
    if not records:return
    with Path(path).open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def save(fig,path):
    fig.savefig(path,dpi=150,bbox_inches='tight',facecolor='white');plt.close(fig)


def representatives():return [g+'_s20261018'for g in L_GROUPS]


def success_plot(result,arms,path,title):
    fig,axes=plt.subplots(1,3,figsize=(15,max(4.6,.34*len(arms))),sharey=True)
    data=[];y=np.arange(len(arms))
    for ax,pop in zip(axes,('PRIMARY192','TEST49','REMAINDER143')):
        methods=result['populations'][pop];n=next(iter(methods.values()))['denominator']
        counts=[methods[a]['successes']for a in arms]
        ax.barh(y,counts,color=[COLORS.get(a,'#647e94')for a in arms]);ax.invert_yaxis()
        ax.set_xlim(0,n*1.16);ax.set_xlabel('Joint success count / full denominator')
        ax.set_title(pop+' (n='+str(n)+')');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        for j,(a,k)in enumerate(zip(arms,counts)):
            ax.text(k+n*.015,j,f'{k}/{n} ({100*k/n:.1f}%)',va='center',fontsize=8)
            data.append(dict(population=pop,arm=a,successes=k,denominator=n,percent=100*k/n))
    axes[0].set_yticks(y,[label(a)for a in arms],fontsize=8)
    fig.suptitle(title+'\nSame-pose center <=5 cm AND C2 rotation <=5 deg; all failures retained',fontsize=12)
    fig.tight_layout();save(fig,path);csvfile(path.with_suffix('.csv'),data)


def cdf_plot(scored,panels,key,path):
    primary=[r for r in scored if r['partition']!='CAL'];fig,axes=plt.subplots(1,len(panels),figsize=(7.8*len(panels),5.5),squeeze=False)
    data=[];unit='cm'if key=='translation_cm'else'deg'
    for ax,(name,arms)in zip(axes[0],panels.items()):
        xmax=0.
        for arm in arms:
            valid=sorted(r['metrics'][arm][key]for r in primary if r['metrics'][arm][key]is not None)
            n=len(valid);xmax=max(xmax,max(valid,default=0.))
            ax.step([0.]+valid,[0.]+[(i+1)/len(primary)for i in range(n)],where='post',label=label(arm)+f' (pose {n}/192)',color=COLORS.get(arm))
            data.extend(dict(id=r['id'],partition=r['partition'],arm=arm,error=r['metrics'][arm][key],unit=unit,
                full_denominator=192,pose_available=r['metrics'][arm]['pose_available'])for r in primary)
        ax.set_xscale('symlog',linthresh=1. if unit=='cm'else .1);ax.set_xlim(0,max(xmax,1.)*1.08);ax.set_ylim(0,1.02)
        ax.set_xlabel(f'Full-range {key.replace("_"," ")} ({unit}; symlog, zero retained)')
        ax.set_ylabel('Cumulative fraction of all 192 images');ax.grid(alpha=.2);ax.set_title(name)
        ax.legend(fontsize=7,loc='lower right')
    fig.suptitle('No tail trimming; missing poses never enter the CDF and remain failures',fontsize=11)
    fig.tight_layout();save(fig,path);csvfile(path.with_suffix('.csv'),data)


def paired_plots(result,path):
    arms=list(B_ARMS)+representatives();fig,axes=plt.subplots(1,len(COMPARATORS),figsize=(20,6.6),sharey=True)
    data=[];y=np.arange(len(arms));population='PRIMARY192'
    for ax,before in zip(axes,COMPARATORS):
        rec=[];damage=[]
        for after in arms:
            p=result['paired'][population][before+'__TO__'+after];r=p['counts']['RECOVERY'];d=p['counts']['DAMAGE']
            rec.append(r);damage.append(-d);data.append(dict(population=population,before=before,after=after,recovery=r,damage=d,denominator=192))
        ax.barh(y,rec,color='#1b9e77',label='Failure -> success');ax.barh(y,damage,color='#d95f4b',label='Success -> failure')
        ax.axvline(0,color='#555',lw=.7);ax.set_title(label(before),fontsize=9);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Damage < 0 | recovery > 0')
        for j,(r,d)in enumerate(zip(rec,damage)):
            ax.text(r+.3,j,str(r),va='center',fontsize=7);ax.text(d-.3,j,str(-d),va='center',ha='right',fontsize=7)
    axes[0].set_yticks(y,[label(a)for a in arms],fontsize=7);axes[0].invert_yaxis();axes[-1].legend(fontsize=7,loc='lower right')
    fig.suptitle('Paired binary recovery and damage / all 192 images');fig.tight_layout();save(fig,path);csvfile(path.with_suffix('.csv'),data)
    fig,axes=plt.subplots(1,2,figsize=(14,6.5),sharey=True);data=[]
    for ax,key,unit in zip(axes,('translation_cm','rotation_deg'),('cm','deg')):
        for j,arm in enumerate(arms):
            p=result['paired']['PRIMARY192']['E0__TO__'+arm]['both_failure_continuous'][key]
            ax.barh(j,p['improvement_sum'],color='#1b9e77');ax.barh(j,p['worsening_sum'],color='#d95f4b')
            data.append(dict(before='E0',after=arm,metric=key,unit=unit,**{k:v for k,v in p.items()if not isinstance(v,(list,dict))}))
        ax.set_xscale('symlog',linthresh=1);ax.axvline(0,color='#555',lw=.6);ax.set_xlabel(f'Sum of paired changes ({unit}; symlog)');ax.set_title('Both methods fail: '+key.replace('_',' '));ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(y,[label(a)for a in arms],fontsize=7);axes[0].invert_yaxis()
    fig.suptitle('Continuous changes among joint failures vs historical E0; improvement <0 / worsening >0')
    fig.tight_layout();target=path.with_name('both_failure_continuous.png');save(fig,target);csvfile(target.with_suffix('.csv'),data)


def action_plot(scored,path):
    data=[];arms=list(B_ARMS)+representatives();primary=[r for r in scored if r['partition']!='CAL']
    roles=['N0','E0_REGENERATED','HISTORICAL_E0','OTHER','NO_POSE'];colors=['#238b45','#31a6be','#2376b7','#c33552','#777777']
    fig,ax=plt.subplots(figsize=(13,5.8));left=np.zeros(len(arms));y=np.arange(len(arms))
    for role,color in zip(roles,colors):
        counts=[sum(r['action_sources'][a]['role']==role for r in primary)for a in arms]
        ax.barh(y,np.array(counts)/192*100,left=left,color=color,label=role)
        for a,n in zip(arms,counts):data.append(dict(arm=a,source=role,n=n,denominator=192))
        left+=np.array(counts)/192*100
    ax.set_yticks(y,[label(a)for a in arms],fontsize=8);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Percent of all 192 outputs');ax.legend(ncol=3,fontsize=8,loc='lower left',bbox_to_anchor=(0,1.02))
    ax.set_title('Actual numerical output source / N0 takes display precedence for overlapping aliases',pad=52)
    fig.tight_layout();save(fig,path);csvfile(path.with_suffix('.csv'),data)


def factor_plot(result,path):
    fig,axes=plt.subplots(1,2,figsize=(14,5.5));data=[];groups=list(L_GROUPS)
    for ax,pop in zip(axes,('PRIMARY192','REMAINDER143')):
        for i,g in enumerate(groups):
            for j,s in enumerate(SEEDS):
                arm=g+'_s'+str(s);m=result['populations'][pop][arm];x=m['successes'];y=i+(j-1)*.14
                ax.scatter(x,y,s=62 if j==0 else 27,marker='D'if j==0 else'o',color=['#c33552','#4c78a8','#59a14f'][j])
                ax.text(x+.35,y,str(x),fontsize=7,va='center');data.append(dict(population=pop,group=g,seed=s,successes=x,denominator=m['denominator']))
        ax.set_yticks(range(len(groups)),[g.replace('_',' ')for g in groups],fontsize=8);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_xlabel('Joint success / full denominator');ax.set_title(pop)
    fig.suptitle('Same new common real bank: 2 x 2, role affine, independently trained pair\nDiamond = predeclared seed20261018; circles = fixed seeds19/20, no best-seed selection',fontsize=11)
    fig.tight_layout();save(fig,path);csvfile(path.with_suffix('.csv'),data)


def training_plot(run_dir,path):
    manifest=read(run_dir/'action_training_v1/TRAINING_MANIFEST.json');runs=manifest['runs'];groups=list(L_GROUPS)
    fig,axes=plt.subplots(len(groups),2,figsize=(13,3.1*len(groups)));data=[]
    for i,group in enumerate(groups):
        for record in runs:
            if record['arm']!=group:continue
            checkpoint=Path(record['checkpoint']);history=checkpoint.parent/'HISTORY.json';value=read(history)
            history_rows=value.get('epochs',value.get('history',[]))if isinstance(value,dict)else value
            for r in history_rows:
                epoch=r.get('epoch');tr=r.get('train_loss');vl=r.get('validation_loss')
                # Both original and new histories are read by actual schema.
                if tr is None and isinstance(r.get('train'),dict):tr=r['train'].get('loss')
                if vl is None and isinstance(r.get('validation'),dict):vl=r['validation'].get('loss')
                val=r.get('validation',{});train=r.get('train',{})
                success=val.get('success_5cm5deg') if isinstance(val,dict) else None
                train_success=train.get('success_5cm5deg') if isinstance(train,dict) else None
                denom=val.get('frames',235) if isinstance(val,dict) else 235
                entry=dict(group=group,seed=record['seed'],epoch=epoch,train_loss=tr,validation_loss=vl,
                    train_joint_success=train_success,train_denominator=768,validation_joint_success=success,validation_denominator=denom)
                for prefix,metrics in [('train',train),('validation',val)]:
                    for metric in ('translation_cm','rotation_deg'):
                        for statistic in ('mean','sample_SD','P90','P95','max'):
                            entry[prefix+'/'+metric+'/'+statistic]=metrics.get(metric,{}).get(statistic)
                    for key in ('frame_balanced_action_accuracy','frame_balanced_Brier','good_action_absent','zero_action_frames'):
                        entry[prefix+'/'+key]=metrics.get(key)
                data.append(entry)
            selected=[r for r in data if r['group']==group and r['seed']==record['seed']]
            if not selected:continue
            x=[r['epoch']for r in selected];axes[i,0].plot(x,[r['validation_loss']for r in selected],label=str(record['seed'])+' val')
            axes[i,0].plot(x,[r['train_loss']for r in selected],ls=':',alpha=.65,label=str(record['seed'])+' train')
            valid=[r for r in selected if r['validation_joint_success']is not None]
            if valid:
                axes[i,1].plot([r['epoch']for r in valid],[100*r['validation_joint_success']/r['validation_denominator']for r in valid],label=str(record['seed'])+' val')
                axes[i,1].plot([r['epoch']for r in valid],[100*r['train_joint_success']/768 for r in valid],ls=':',alpha=.65,label=str(record['seed'])+' train')
        axes[i,0].set_title(group+' / own objective loss',fontsize=9);axes[i,1].set_title(group+' / joint success % (train768 / val235)',fontsize=9)
        if group=='D0_H0_MLP':axes[i,1].text(.5,.5,'Historical epoch pose metrics not archived\nN/A; no invented curve',ha='center',va='center',transform=axes[i,1].transAxes)
        for ax in axes[i]:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.suptitle('All seeds. Checkpoint chosen by own validation loss; CE and BCE magnitudes are not compared.',fontsize=11,y=1.002)
    fig.tight_layout();save(fig,path);csvfile(path.with_suffix('.csv'),data)


def run(run_dir,evaluation,output):
    started,cpu=time.monotonic(),time.process_time()
    run_dir=Path(run_dir);evaluation=Path(evaluation);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result=read(evaluation/'RESULTS.json');scored=list(rows(evaluation/'SCORED_ROWS.jsonl.gz'))
    main=['N0','E0','E0_REGENERATED','POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR','LEGACY_L_HEAD_s20261018']+list(B_ARMS)+representatives()
    success_plot(result,main,output/'joint_success.png','Predeclared representatives / same original denominators')
    success_plot(result,list(ALL_ARMS),output/'joint_success_all_arms.png','All registered outputs, including all three seeds')
    panels={'Baselines and learned actions':['N0','E0','E0_REGENERATED','POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR','LEGACY_L_HEAD_s20261018','D1_H1_MLP_s20261018','D1_H1_PAIR_MLP_s20261018'],
            'B geometry repair':['N0','E0','LEGACY_B_FULL','B_FINITE_BANK','B_FINITE_FULL','B_FINITE_LOCAL','EDGE_FINITE_FULL']}
    cdf_plot(scored,panels,'translation_cm',output/'translation_cdf.png')
    cdf_plot(scored,panels,'rotation_deg',output/'rotation_cdf.png')
    paired_plots(result,output/'paired_recovery_damage.png');action_plot(scored,output/'actual_action_sources.png')
    factor_plot(result,output/'same_bank_2x2_pair.png');training_plot(run_dir,output/'training_curves_all_seeds.png')
    write(output/'FIGURE_MANIFEST.json',dict(schema='SCIENTIFIC_FIGURES_V1',evaluation_seal_sha256=sha(evaluation/'EVALUATION_SEAL.json'),
        source_sha256=sha(__file__),no_new_fit_or_pose_change=True,artifacts={p.name:sha(p)for p in sorted(output.iterdir())if p.is_file()},
        visual_review='not implied by generation; separate actual pixel review required',
        cost=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.run_dir,a.evaluation,a.output)


if __name__=='__main__':main()

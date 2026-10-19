"""Plots from actual six fit histories and postseal full-denominator metrics."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.research.pallet_risk_information_v1.common import read,rows,write,sha
from .registry import SEEDS,ARMS


def run(run_dir,output):
    run_dir,output=Path(run_dir),Path(output);output.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
    for i,arm in enumerate(('COMMON','BI')):
        for j,seed in enumerate(SEEDS):
            folder=run_dir/'training'/f'OBS_{arm}_s{seed}';h=read(folder/'HISTORY.json');r=read(folder/'RUN_RECEIPT.json');ax=axes[i,j]
            epochs=[v['epoch_index'] for v in h]
            ax.plot(epochs,[v['train']['loss'] for v in h],label='train')
            ax.plot(epochs,[v['validation']['loss'] for v in h],label='validation')
            ax.axvline(r['best_epoch'],ls='--',c='#555555',label='selected checkpoint')
            ax.set_title(f'{arm} seed{seed}: stop{r["epochs"]}, best{r["best_epoch"]}');ax.set_xlabel('actual epoch');ax.set_ylabel('hierarchical SmoothL1');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'six_training_curves.png',dpi=120);plt.close(fig)
    summary=read(run_dir/'evaluation/SUMMARY.json')['PRIMARY192']
    fig,ax=plt.subplots(figsize=(15,6));ax.bar(range(len(ARMS)),[summary[a]['successes'] for a in ARMS],color=['#b7c6d0']*3+['#e28a00','#087cca']*3+['#aaaacc']*4)
    ax.set_xticks(range(len(ARMS)),ARMS,rotation=60,ha='right',fontsize=8);ax.set_ylabel('joint success / 192');ax.set_ylim(0,192)
    for j,a in enumerate(ARMS):ax.text(j,summary[a]['successes']+2,str(summary[a]['successes']),ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(output/'whole192_all_systems.png',dpi=120);plt.close(fig)
    scored=[r for r in rows(run_dir/'evaluation/SCORED_ROWS.jsonl.gz') if r['partition']!='CAL']
    C='OBS_COMMON_s20261018';B='OBS_BI_s20261018'
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,arm,color in zip(axes,(C,B),('#e28a00','#087cca')):
        available=[r for r in scored if r['metrics'][arm]['pose_available']]
        ax.scatter([r['metrics'][arm]['translation_cm'] for r in available],[r['metrics'][arm]['rotation_deg'] for r in available],s=17,alpha=.7,c=color)
        ax.axvline(5,c='gray',ls='--');ax.axhline(5,c='gray',ls='--');ax.set_title(arm+f' / all192, available{len(available)}')
        ax.set_xlabel('whole-pose position error (cm)');ax.set_ylabel('whole-pose C2 rotation error (deg)')
    xmax=max(r['metrics'][a]['translation_cm'] or 0 for r in scored for a in (C,B));ymax=max(r['metrics'][a]['rotation_deg'] or 0 for r in scored for a in (C,B))
    for ax in axes:ax.set_xlim(-1,xmax*1.03);ax.set_ylim(-1,ymax*1.03)
    fig.tight_layout();fig.savefig(output/'pose_position_rotation_scatter.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(14,7),sharex=True)
    for ax,axis,label in zip(axes,('translation_cm','rotation_deg'),('position delta cm','rotation delta deg')):
        d=[None if r['metrics'][B][axis] is None or r['metrics'][C][axis] is None else r['metrics'][B][axis]-r['metrics'][C][axis] for r in scored]
        ax.scatter([j+1 for j,v in enumerate(d) if v is not None],[v for v in d if v is not None],s=16,c='#087cca');ax.axhline(0,c='gray');ax.set_ylabel(label+' (BI - COMMON)')
    axes[1].set_xlabel('image index in unchanged PRIMARY192 order');fig.tight_layout();fig.savefig(output/'paired_frame_deltas.png',dpi=130);plt.close(fig)
    p=read(run_dir/'evaluation/PAIRED.json')['PRIMARY192']
    fig,ax=plt.subplots(figsize=(10,5));positions=np.arange(3);width=.18
    for j,(key,color) in enumerate((('BOTH_SUCCESS','#2a9d8f'),('RECOVERY','#087cca'),('DAMAGE','#e28a00'),('BOTH_FAILURE','#9aa3a8'))):
        values=[p[f'OBS_BI_s{s}_minus_OBS_COMMON_s{s}']['counts'][key] for s in SEEDS]
        ax.bar(positions+(j-1.5)*width,values,width,label=key,color=color)
    ax.set_xticks(positions,[str(s) for s in SEEDS]);ax.set_ylabel('original images / 192 per seed');ax.legend();fig.tight_layout();fig.savefig(output/'paired_success_transitions.png',dpi=130);plt.close(fig)
    write(output/'FIGURES.json',dict(source_sha256=sha(__file__),files={p.name:sha(p) for p in output.glob('*.png')},all_denominators_preserved=True,model_oracle_results_mixed=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.run_dir,a.output)


if __name__=='__main__':main()

"""Non-selected numerical figures generated from measured JSON only."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from env import *
def run():
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    data=read(DOC/'P_VS_R0_PAIRED.json');rows=[r for r in data['per_session_and_LOSO'] if r['mode']=='only_session']
    fig,ax=plt.subplots(figsize=(8,4));ax.barh([r['session'] for r in rows],[r['delta_px'] for r in rows]);ax.axvline(0,color='black',lw=.7);ax.set_xlabel('P − R0 pooled median error (px), seed mean');ax.set_title('All observed development sessions; negative favors P');fig.tight_layout()
    fig.savefig(PAPER/'FIG_SESSION_EFFECTS.pdf');fig.savefig(PAPER/'FIG_SESSION_EFFECTS.png',dpi=160);plt.close(fig)
    train=read(DOC/'D_TRAINING_AUDIT.json');fig,axes=plt.subplots(1,2,figsize=(8,3))
    for r in train['runs']:
        for ax,partition in zip(axes,['train','cal']):ax.plot([p['step'] for p in r['probes']],[p[partition] for p in r['probes']],label=f"D seed{r['seed']}");ax.set_title(f'Fixed {partition} probe');ax.set_xlabel('Optimizer updates');ax.set_ylabel('Normalized residual L1')
    axes[0].legend();fig.tight_layout();fig.savefig(PAPER/'FIG_D_PROBES.pdf');fig.savefig(PAPER/'FIG_D_PROBES.png',dpi=160);plt.close(fig)
    rt=read(DOC/'RUNTIME_PANEL.json');names=list(rt['summary']);fig,ax=plt.subplots(figsize=(8,3))
    ax.boxplot([[r['image_to_2d_ms'] for r in rt['records'] if r['model']==name] for name in names],labels=names,showfliers=True);ax.set_ylabel('BGR → original 2D (ms)');ax.set_title('RTX3080; all130 measurements per model');fig.tight_layout();fig.savefig(PAPER/'FIG_RUNTIME.pdf');fig.savefig(PAPER/'FIG_RUNTIME.png',dpi=160);plt.close(fig)
    write(PAPER/'FIGURE_SOURCE_BINDING.json',{p.name:bound(p) for p in [DOC/'P_VS_R0_PAIRED.json',DOC/'D_TRAINING_AUDIT.json',DOC/'RUNTIME_PANEL.json']})
    print('FIGURES_READY',flush=True)
if __name__=='__main__':run()

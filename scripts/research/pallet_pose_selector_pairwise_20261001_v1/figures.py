"""Measured experiment figures and the fixed supervision graph."""
from . import common as C
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def supervision_graph():
    folder = C.DOC / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.4), layout='constrained')
    positions = {0:(.2,.76), 1:(.2,.26), 2:(.8,.76), 3:(.8,.26)}
    labels = ['R0 / long hypothesis', 'R0 / short hypothesis',
              'DIVERSE / long hypothesis', 'DIVERSE / short hypothesis']
    for a, b in [(0,1),(2,3)]:
        x1,y1=positions[a];x2,y2=positions[b]
        ax.plot([x1,x2],[y1,y2],color='#3579a2',lw=3,zorder=1)
    for a,b in [(0,2),(1,3)]:
        x1,y1=positions[a];x2,y2=positions[b]
        ax.plot([x1,x2],[y1,y2],color='#a06b32',lw=3,zorder=1)
    for i,(x,y) in positions.items():
        ax.add_patch(FancyBboxPatch((x-.17,y-.06),.34,.12,boxstyle='round,pad=0.015',
                     facecolor='#f4f6f8',edgecolor='#576873',zorder=2))
        ax.text(x,y,labels[i],ha='center',va='center',fontsize=10,zorder=3)
    ax.text(.20,.51,'WD comparison\nweight 1/4',ha='center',va='center',fontsize=10,
            color='#286785',bbox=dict(facecolor='white',edgecolor='none'))
    ax.text(.80,.51,'WD comparison\nweight 1/4',ha='center',va='center',fontsize=10,
            color='#286785',bbox=dict(facecolor='white',edgecolor='none'))
    ax.text(.5,.81,'Expert comparison: weight 1/4',ha='center',fontsize=10,color='#8a5627')
    ax.text(.5,.16,'Expert comparison: weight 1/4',ha='center',fontsize=10,color='#8a5627')
    ax.text(.5,.03,'Training: one coherent cost / Pareto / tie order supplies every edge label.\n'
            'Inference: all four candidates remain; shared Linear94 argmin selects one complete pose.',
            ha='center',fontsize=10)
    ax.set(xlim=(0,1),ylim=(-.04,1))
    ax.axis('off')
    ax.set_title('Supervision changes; candidate pool and inference rule stay fixed',fontsize=13)
    fig.savefig(folder/'supervision_graph.png',dpi=160)
    plt.close(fig)


def measured(gate, logs):
    folder=C.DOC/'figures'
    folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,4.3),layout='constrained')
    for model,color in zip(C.NEW_MODELS,['#26847a','#3c75b0','#b24e45']):
        rows=[r for r in logs if r['model']==model]
        ax.plot([r['call'] for r in rows],[r['objective'] for r in rows],label=model,color=color)
    ax.set(xlabel='Objective / gradient evaluations',ylabel='Pairwise logistic loss + explicit ridge',
           title='Three new fits; certified R0_ONLY baseline reused without optimization')
    ax.legend();ax.grid(alpha=.18)
    fig.savefig(folder/'training_objective.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    models=['R0_GEO',*C.MODEL_NAMES]
    labels=['R0 GEO','R0 ONLY','UNION 1','UNION 2','UNION 3']
    for ax,(axis,q,unit) in zip(axs.flat,[('translation_cm','median','Translation median (cm)'),
            ('rotation_deg','median','Rotation median (degrees)'),
            ('translation_cm','P90','Translation P90 (cm)'),('rotation_deg','P90','Rotation P90 (degrees)')]):
        values=[gate['summaries'][m]['full_population'][axis][q] for m in models]
        bars=ax.bar(labels,values,color=['#526979','#95a7b3','#26847a','#3c75b0','#b24e45'])
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
        ax.set_ylim(0,max(values)*1.16);ax.set_ylabel(unit)
        ax.tick_params(axis='x',rotation=15);ax.grid(axis='y',alpha=.18)
        if q=='P90':
            ax.axhline(values[1]*1.05,ls='--',color='#666',lw=1,label='1.05 x R0 ONLY')
            ax.legend(fontsize=8)
    verdict='PASS' if gate['PASS'] else 'FAIL'
    fig.suptitle(f"Source VAL: 1,024 frames retained / {verdict} ({gate['checks_passed']} / 45 checks passed)")
    fig.savefig(folder/'source_val_results.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    supervision_graph()

"""Directive figures from frozen aggregates; no model inference or new scoring."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from . import common as C

ARMS=('BASE','H_PSEUDO','H_MANUAL')
COLORS=('#67788a','#d99832','#228f8b')
FIGURES=('01_tagging_queue_recordings.png','02_human_difficulty_distribution.png',
         '03_selected_hard_frames_contact.jpg','04_manual_corner_coverage.png','05_pair_integrity.png',
         '06_clean_pck_auc.png','07_moderate_pck_auc.png','08_severe_pck_auc.png',
         '09_oracle_vs_current.png','10_verified_visible.png','11_recording_breakdown.png',
         '12_pseudo_vs_manual.png','13_source_preservation.png','14_decision.png')


def render():
    root=C.DOC/'figures';r=C.read(C.DOC/'RESULTS.json')['groups']
    anchor=C.read(C.DOC/'VERIFIED_VISIBLE.json')['groups'];source=C.read(C.DOC/'SOURCE_PRESERVATION.json')['groups']
    def save(fig,name):
        fig.tight_layout();fig.savefig(root/name,dpi=150);plt.close(fig)
    def bars(ax,vals,title,percent=False):
        ax.bar(ARMS,vals,color=COLORS);ax.set_title(title);ax.tick_params(axis='x',labelsize=8)
        for i,y in enumerate(vals):ax.text(i,y,f'{y:.2f}' if percent else f'{y:.4f}',ha='center',va='bottom',fontsize=9)
        ax.set_ylim(0,max(vals)*1.2 if max(vals)>0 else 1)
    fig,axes=plt.subplots(1,2,figsize=(11,3.5));x=np.arange(3)
    bottom=np.zeros(3)
    for label,values,color in [('Synthetic',[512,512,512],'#67788a'),('Clean',[512,448,448],'#9dbdcc'),('Hard',[0,64,64],'#228f8b')]:
        axes[0].bar(x,values,bottom=bottom,label=label,color=color);bottom+=values
    axes[0].set_xticks(x,ARMS);axes[0].set_title('Same 1,024 occurrences per epoch');axes[0].legend(fontsize=8)
    axes[1].axis('off');axes[1].text(.02,.92,'H_PSEUDO  =  H_MANUAL',fontsize=15,va='top')
    axes[1].text(.02,.75,'SAME: original R0 init / seed42 / 320 updates\nRGB / PnP box / affine / support / order\n\nDIFFERENT: 320 hard-slot coordinate targets only\n\n4,800 other slots: exact original S1 tensors\nHard: xy supervision only, no automatic corners',va='top',fontsize=11)
    save(fig,'05_pair_integrity.png')
    for i,g in enumerate(('CLEAN','MODERATE','SEVERE'),6):
        fig,axes=plt.subplots(1,2,figsize=(10,3.7))
        bars(axes[0],[100*r[g][a]['twoD']['PCK']['10'] for a in ARMS],g+' PCK10 (%)',True)
        bars(axes[1],[r[g][a]['current']['ADDsym_AUC'] for a in ARMS],g+' CURRENT ADDsym AUC')
        save(fig,f'{i:02d}_{g.lower()}_pck_auc.png')
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,g in zip(axes,('CLEAN','MODERATE','SEVERE')):
        for i,a in enumerate(ARMS):
            cur=r[g][a]['current']['ADDsym_AUC'];ora=r[g][a]['oracle']['ADDsym_AUC']
            ax.bar(i,cur,color=COLORS[i]);ax.bar(i,ora-cur,bottom=cur,color=COLORS[i],alpha=.25,hatch='//')
        ax.set_xticks(x,ARMS,fontsize=7);ax.set_title(g);ax.set_ylim(0,.85)
    fig.suptitle('Solid = frozen selector; hatched = posthoc oracle headroom (not deployable)',fontsize=11)
    save(fig,'09_oracle_vs_current.png')
    fig,ax=plt.subplots(figsize=(9,3.8));gg=('ALL','CLEAN','MODERATE','SEVERE','HARD');x=np.arange(len(gg))
    for j,a in enumerate(ARMS):
        vals=[100*anchor[g][a]['PCK']['10']['fraction'] for g in gg]
        ax.bar(x+(j-1)*.25,vals,width=.25,color=COLORS[j],label=a)
    ax.set_xticks(x,[f'{g}\nn={anchor[g]["BASE"]["n"]}' for g in gg]);ax.set_ylim(0,100);ax.set_ylabel('Fixed-ID PCK10 (%)');ax.legend();ax.set_title('Verified visible reference, no symmetry minimization')
    save(fig,'10_verified_visible.png')
    recs=sorted(set(r)-{'ALL','CLEAN','MODERATE','SEVERE'});fig,axes=plt.subplots(1,3,figsize=(13,4.6))
    for ax,key in zip(axes,('PCK10','CURRENT AUC','ORACLE AUC')):
        def value(g,a):return 100*r[g][a]['twoD']['PCK']['10'] if key=='PCK10' else r[g][a]['current' if key=='CURRENT AUC' else 'oracle']['ADDsym_AUC']
        data=np.array([[value(g,a)-value(g,'BASE') for a in ARMS[1:]] for g in recs]);scale=max(abs(data).max(),.001)
        im=ax.imshow(data,cmap='RdYlGn',vmin=-scale,vmax=scale,aspect='auto');ax.set_xticks([0,1],ARMS[1:],fontsize=8);ax.set_yticks(range(len(recs)),recs);ax.set_title('Delta '+key+(' (pp)' if key=='PCK10' else ''))
        for (i,j),v in np.ndenumerate(data):ax.text(j,i,f'{v:+.2f}' if key=='PCK10' else f'{v:+.3f}',ha='center',va='center',fontsize=8)
        fig.colorbar(im,ax=ax,fraction=.05)
    save(fig,'11_recording_breakdown.png')
    fig,axes=plt.subplots(1,3,figsize=(12,3.6));gg=('CLEAN','MODERATE','SEVERE');x=np.arange(3)
    for ax,key in zip(axes,('PCK10','CURRENT AUC','ORACLE AUC')):
        for j,a in enumerate(ARMS[1:]):
            vals=[100*(r[g][a]['twoD']['PCK']['10']-r[g]['BASE']['twoD']['PCK']['10']) if key=='PCK10' else r[g][a]['current' if key=='CURRENT AUC' else 'oracle']['ADDsym_AUC']-r[g]['BASE']['current' if key=='CURRENT AUC' else 'oracle']['ADDsym_AUC'] for g in gg]
            ax.bar(x+(j-.5)*.35,vals,width=.35,label=a,color=COLORS[j+1])
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,gg,fontsize=8);ax.set_title('vs BASE: '+key);ax.legend(fontsize=7)
    save(fig,'12_pseudo_vs_manual.png')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    bars(axes[0],[100*source[a]['twoD']['PCK']['10']['fraction'] for a in ARMS],'Fixed synthetic256 PCK10 (%)',True)
    bars(axes[1],[source[a]['pose']['ADDsym_AUC'] for a in ARMS],'Exact synthetic ADDsym AUC')
    save(fig,'13_source_preservation.png')
    fig,ax=plt.subplots(figsize=(11,3.5));ax.axis('off')
    checks=[('Manual hard supervision','Train clicks within10px: 24/36 -> 36/36'),('Held-out localization signal','Verified HARD: 19/36 -> 24/36'),('Final selected 6D','ALL AUC: 0.3650 -> 0.3586; no deployment gain')]
    for i,(title,body) in enumerate(checks):
        ax.text(.01+i*.34,.78,title,fontsize=12,weight='bold',bbox=dict(facecolor='#d8eeee',edgecolor='none',pad=8))
        ax.text(.01+i*.34,.53,body.replace('; ','\n'),fontsize=9,va='top')
    ax.text(.5,.12,'KEEP S1  |  STOP MORE LABELING  |  NEXT: candidate-to-selector failure analysis',ha='center',fontsize=12)
    save(fig,'14_decision.png')
    for name in FIGURES:assert (root/name).is_file(),name
    C.save(C.DOC/'FIGURE_MANIFEST.json',dict(source='frozen aggregate results; no new training or scoring',
           figures=[C.bind(root/name) for name in FIGURES],example_images=12))


if __name__=='__main__':render()

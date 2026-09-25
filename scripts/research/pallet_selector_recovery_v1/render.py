"""Posthoc charts. This module is never imported by synthetic learners."""
import argparse
import numpy as np
import cv2
from . import common as C
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def figpath(stage):
    p=C.DOC/'figures'/f'stage{stage}';p.mkdir(parents=True,exist_ok=True);return p
def finish(stage,name):
    plt.tight_layout();plt.savefig(figpath(stage)/name,dpi=150);plt.close()
def stage1():
    data=C.read(C.sdoc(1)/'MODERATE_SELECTOR_DIAGNOSTIC.json');rr=data['moderate'];groups=data['groups']
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(3);gn=['CLEAN','MODERATE','SEVERE']
    for k,(field,label) in enumerate([('CURRENT','CURRENT D9'),('ORACLE','GT ORACLE (nondeployable)')]):ax.bar(x+(k-.5)*.35,[groups[g][field]['ADDsym_AUC'] for g in gn],.35,label=label)
    ax.set(xticks=x,xticklabels=gn,ylabel='ADDsym AUC',title='Frozen S1 / HELDOUT128');ax.legend();finish(1,'01_current_vs_oracle.png')
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(np.arange(21),[r['score_margin'] for r in rr],color=['#da713b' if r['alternate_ADD_better'] else '#407ca3' for r in rr]);ax.set(title='Orange: alternative has lower ADD; no margin threshold added',ylabel='Absolute production score margin',xlabel='Fixed Moderate21 frame order');finish(1,'02_selector_margin.png')
    dd=C.read(C.sdoc(1)/'FEATURE_DISTRIBUTIONS.json');fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,k in zip(axs,('confidence','upright','spread')):
        ax.boxplot([dd[c][k]['values'] for c in ('CURRENT_BEST','ALTERNATE_BETTER')],tick_labels=['Current best','Alt better']);ax.set(title=k)
    finish(1,'03_component_distributions.png')
    from scripts.research.pallet_existing_data_transfer_v1.report import crop,tile
    from scripts.research.pallet_clean19_pose_mismatch_v1.render import draw_edges,projection
    from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
    records={r['id']:r for r in V.records()};truth=C.read(V.E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json');meta={r['id']:r for r in C.read(V.E.V.RAW/'INFERENCE_METADATA.json')};pred=C.read(C.PREV_RAW/'PREDICTIONS.json')['S1'];panels=[]
    for r in rr:
        if not (r['axis_wrong'] and r['alternate_ADD_better']):continue
        fid=r['id'];base,off,s=crop(records[fid],truth[fid]['box']);pair=[]
        for label,name in [('CURRENT',r['current']),('POSTHOC BEST',r['best'])]:
            im=base.copy();h=next(h for h in r['hypotheses'] if h['name']==name);q=np.array(C.selected(pred[fid])['keypoints_xy'])
            draw_edges(im,(q-off)*s,(0,220,255));draw_edges(im,(projection(h['pose'],np.array(meta[fid]['K']))-off)*s,(20,30,255))
            pair.append(tile(im,[label+' | '+fid,name,f"ADDnorm {h['metric']['ADDsym_normalized']:.4f} | axis {h['metric']['axis_correct']}",'Yellow raw2D / red candidate; oracle not deployed'],height=340))
        panels.append(np.hstack(pair))
    cv2.imwrite(str(figpath(1)/'04_wrong_candidate_cases.jpg'),np.vstack(panels))
    recs=sorted({r['recording'] for r in rr});fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(recs))
    ax.bar(x-.18,[sum(r['recording']==rec and r['axis_wrong'] for r in rr) for rec in recs],.36,label='Current axis wrong');ax.bar(x+.18,[sum(r['recording']==rec and r['alternate_ADD_better'] for r in rr) for rec in recs],.36,label='Alternate ADD better');ax.set(xticks=x,xticklabels=recs,ylabel='Frames');ax.legend();finish(1,'05_recording_breakdown.png')

def stage2():
    val=C.read(C.sdoc(2)/'SCORER_VAL_RESULTS.json')['variants'];test=C.read(C.sdoc(2)/'SCORER_SYNTH_TEST.json')
    fig,ax=plt.subplots(figsize=(8,4))
    for name,v in val.items():ax.plot([r['epoch'] for r in v['curve']],[r['val_accuracy'] for r in v['curve']],label=name)
    ax.set(xlabel='Epoch',ylabel='Synthetic VAL parity accuracy');ax.legend();finish(2,'01_validation.png')
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(2)
    for k,key in enumerate(('current_accuracy','learned_accuracy')):ax.bar(x+(k-.5)*.35,[test['by_expert'][a]['TEST'][key] for a in C.ARMS],.35,label=key)
    ax.set(xticks=x,xticklabels=C.ARMS,ylim=(0,1),ylabel='Synthetic TEST parity accuracy');ax.legend();finish(2,'02_test.png')

def stage3():
    j=C.read(C.sdoc(3)/'REAL_SCORER_RESULTS.json')['groups'];t=C.read(C.sdoc(3)/'REAL_SCORER_TRANSITIONS.json')
    m=j['MODERATE']['S1'];fig,ax=plt.subplots(figsize=(8,4));ax.bar(['CURRENT','SCORER','ORACLE'],[m[k]['ADDsym_AUC'] for k in ('current','scorer','oracle')]);ax.set(ylabel='ADDsym AUC',title='S1 Moderate21 / frozen scorer');finish(3,'01_moderate_auc_current_scorer_oracle.png')
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(['Current correct','Scorer correct','Recovered','Regressed'],[m['current']['axis_correct_count'],m['scorer']['axis_correct_count'],t['MODERATE']['S1']['recoveries'],t['MODERATE']['S1']['regressions']]);ax.set(ylabel='Frames / 21');finish(3,'02_axis_recovery.png')
    for filename,gg in [('03_clean_severe_safeguards.png',['CLEAN','SEVERE']),('04_recording_breakdown.png',[g for g in j if g.startswith('REC')])]:
        fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(gg))
        for k,key in enumerate(('current','scorer')):ax.bar(x+(k-.5)*.35,[j[g]['S1'][key]['ADDsym_AUC'] for g in gg],.35,label=key)
        ax.set(xticks=x,xticklabels=gg,ylabel='S1 ADDsym AUC');ax.legend();finish(3,filename)
    fig,ax=plt.subplots(figsize=(8,4));gg=['CLEAN','MODERATE','SEVERE','ALL'];ax.bar(gg,[j[g]['S1']['gap_recovery'] or 0 for g in gg]);ax.axhline(0,color='black');ax.set(ylabel='Unclamped oracle gap recovery',title='Zero-height Clean has no positive oracle gap (N/A)');finish(3,'05_gap_recovery.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=int);a=p.parse_args();globals()[f'stage{a.stage}']()

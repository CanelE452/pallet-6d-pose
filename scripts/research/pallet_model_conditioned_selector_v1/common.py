import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from scripts.research.pallet_selector_recovery_v1 import common as U

ROOT=U.ROOT
NAME='pallet_model_conditioned_selector_v1'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
OUT=ROOT/'outputs'/NAME
OLD=U.DOC
OLDRAW=U.RAW
HARD=ROOT/'_docs/experiments/pallet_min_hard_ab_v1'
HARDRAW=ROOT/'data/pallet/results/pallet_min_hard_ab_v1'
MODELS=('S1','H_MANUAL')
ARM={'S1':'BASE','H_MANUAL':'H_MANUAL'}
SELECTORS=('D9','OLD_GEO','S1SPEC_GEO','HMANSPEC_GEO')
NEW={'S1':'S1_SPECIFIC_GEO_LINEAR','H_MANUAL':'HMAN_SPECIFIC_GEO_LINEAR'}
HYP=U.HYP
PRIMARY=('CLEAN','MODERATE','SEVERE','ALL')
read=U.read;sha=U.sha;bind=U.bind;verify=U.verify;now=U.now;clean=U.clean
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-model-selector-mpl')

def save(p,x,immutable=True):
    p=Path(p);assert any(p.resolve().is_relative_to(r) for r in (DOC,RAW,OUT))
    data=x if isinstance(x,str) else json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if immutable and p.exists():
        assert p.read_text()==data,f'Immutable conflict: {p}'
        return
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name('.'+p.name+'.tmp');tmp.write_text(data);os.replace(tmp,p)

def stage(n):return DOC/f'stage{n}'
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def immutable():
    b=read(DOC/'INPUT_BINDINGS.json')
    for f in b['files']:verify(f)
    return len(b['files'])

def guard(mode):
    """Deny real reference opens in synth/decision processes; record permitted repo reads."""
    paths=[]
    tokens=['TRUTH_FOR_DISPLAY','GEOMETRY_RESOLVED','VERIFIED_LABELS','FRAME_METRICS','POSE_METRICS','ANCHOR_METRICS',
            'REAL_SELECTOR_RESULTS','REAL_SCORER_RESULTS','RESULTS.json','FINAL_V2','/data/evaluation/',
            'FRAME_DECOMPOSITION','SELECTOR_CATEGORY_COUNTS','MANUAL_GAIN_DECOMPOSITION','LOCALIZATION_TAIL_DECOMPOSITION']
    if mode=='fit':tokens += ['/stage1/','/stage3/','RAW_PREDICTIONS.json','POSE_DECISIONS.json','INFERENCE_INPUTS.json','SCORER_SYNTH_TEST.json','SYNTH_COMPATIBILITY_MATRIX.json']
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=os.path.abspath(os.fsdecode(args[0]));writing=isinstance(args[1],str) and any(c in args[1] for c in 'wax+')
        if not writing:
            assert not any(t in p for t in tokens),f'{mode.upper()}_REFERENCE_READ_DENIED: {p}'
            if p.startswith(str(ROOT)):paths.append(p)
    sys.addaudithook(hook)
    return paths

def table(headers,rows):
    def fmt(v):return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in rows])+'\n'

def figure(n,name,title,labels,series,ylabel='Count'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(max(8,len(labels)*1.05),4.8));x=np.arange(len(labels));width=.8/max(1,len(series))
    for j,(label,values) in enumerate(series.items()):ax.bar(x+(j-(len(series)-1)/2)*width,values,width,label=label)
    ax.set_xticks(x,labels,rotation=20,ha='right');ax.set_title(title);ax.set_ylabel(ylabel);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();p=stage(n)/'figures'/name;p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=150);plt.close(fig)
    return p

def metric_for(row,name):
    return next((h['metric'] for h in row['hypotheses'] if h['name']==name),dict(available=False))

def category(row,selected):
    hh=[h for h in row['hypotheses'] if h['metric']['available']]
    if len(hh)!=2 or selected not in [h['name'] for h in hh]:return 'POSE_UNAVAILABLE'
    if abs(hh[0]['metric']['ADDsym_normalized']-hh[1]['metric']['ADDsym_normalized'])<=1e-12:return 'ORACLE_TIE'
    return 'SELECTOR_CORRECT' if selected==row['oracle_name'] else 'SELECTOR_RECOVERABLE'

def oracle_delta(s,h):
    if s is None or h is None:return 'POSE_UNAVAILABLE'
    if abs(h-s)<=1e-12:return 'ORACLE_TIE'
    return 'ORACLE_GAIN' if h<s else 'ORACLE_LOSS'

def decision(groups,synth):
    hard=('MODERATE','SEVERE')
    compatibility={g:groups[g]['H_MANUAL_HMANSPEC_GEO']['ADDsym_AUC']-groups[g]['H_MANUAL_OLD_GEO']['ADDsym_AUC'] for g in hard}
    base={g:groups[g]['H_MANUAL_HMANSPEC_GEO']['ADDsym_AUC']-groups[g]['S1_OLD_GEO']['ADDsym_AUC'] for g in ('CLEAN',*hard)}
    if any(v>0 for v in compatibility.values()) and all(v>=0 for v in compatibility.values()):q='MODEL_CONDITIONED_SELECTOR_RECOVERY'
    elif any(v>0 for v in compatibility.values()) and any(v<0 for v in compatibility.values()):q='PARTIAL_SELECTOR_COMPATIBILITY_RECOVERY'
    else:q='NO_SELECTOR_COMPATIBILITY_RECOVERY'
    beaten=all(v>=0 for v in base.values()) and any(base[g]>0 for g in hard)
    p='HMAN_PIPELINE_RECOVERS_AND_BEATS_BASE' if beaten else 'HMAN_SELECTOR_GAIN_BUT_BASE_NOT_BEATEN' if q=='MODEL_CONDITIONED_SELECTOR_RECOVERY' else 'NO_PIPELINE_RECOVERY'
    gap=synth['H_MANUAL_HMANSPEC_GEO']['accuracy']>synth['H_MANUAL_OLD_GEO']['accuracy'] and q!='MODEL_CONDITIONED_SELECTOR_RECOVERY'
    return dict(Q_SELECTOR_COMPATIBILITY=q,Q_PIPELINE=p,SYNTH_REAL_COMPATIBILITY_GAP=gap,compatibility_deltas=compatibility,base_deltas=base,
                current_deployment_candidate='HMAN_HMANSPEC_GEO' if beaten else 'S1_OLD_GEO',additional_labeling=0,keypoint_model_training=0,new_selector_trainings=2,
                next_one_step='untouched/reserved recording inventory for independent confirmation' if beaten else 'descriptive synthetic-to-real feature-family shift audit' if gap else 'candidate/localization structural diagnosis of remaining tails',
                independent_final_test=False,already_viewed_DEV=True,oracle='POSTHOC / GT-dependent / NONDEPLOYABLE',auto_extra_experiment=False)

import argparse,time
import numpy as np
from . import common as C
from . import pose_oracle as O

ID='C1_HUBER_D9'
DELTA=12.

def robust_score(cue,delta=DELTA):
    r=np.asarray(cue['residual9_px'],float)
    rmse=float(np.sqrt(np.mean(r*r)))
    huber=float(np.sqrt(np.mean(np.where(r<=delta,r*r,2*delta*r-delta*delta))))
    return float(cue['selector_score'])-rmse+huber

def choice(cues):
    return min(cues,key=lambda name:(robust_score(cues[name]),name)) if cues else None

def main():
    d=C.DOC/'cycles'/ID;r=C.RAW/'cycles'/ID;start=time.monotonic()
    if (d/'RESULTS.json').exists():print('C1_ALREADY_COMPLETE');return
    source=O.RAW/'D9_RESIDUAL9_CUES.json';cues=C.read(source)
    spec=d/'SPEC.md';assert spec.exists()
    dst=r/'SELECTIONS.json';lock=r/'PREDICTIONS_LOCK.json'
    if not dst.exists():
        selections={mat:{a:{fid:choice(h) for fid,h in frames.items()} for a,frames in arms.items()} for mat,arms in cues['materials'].items()}
        for mat,arms in cues['materials'].items():
            for a,frames in arms.items():
                for fid,h in frames.items():assert selections[mat][a][fid]==choice(dict(reversed(list(h.items()))))
        C.save(dst,selections,True)
        C.save(lock,dict(utc=C.now(),source=C.bind(source),spec=C.bind(spec),selection=C.bind(dst),
                        GT_reference_read=False,delta_px=DELTA,same_candidate_set=True,no_fit=True),True)
    C.verify(C.read(lock)['selection']);selections=C.read(dst)
    # Scoring-only join below. GT-derived metric caches do not enter choice().
    oracle=C.read(O.RAW/'POSE_ORACLE_RESULTS.json');out={}
    for mat in O.ARMS:
        cand=C.read(O.RAW/f'{mat}_CANDIDATES.json');metrics=C.read(O.RAW/f'{mat}_METRICS.json')['arms'];out[mat]={}
        groups={'ALL':cand['ids']}
        for k in ('recording','severity'):
            for g in sorted({v[k] for v in cand['groups'].values()}):groups[g]=[i for i in cand['ids'] if cand['groups'][i][k]==g]
        for group,ids in groups.items():
            out[mat][group]={}
            for a in O.ARMS[mat]:
                old=[];new=[];best=[];swaps=0
                for fid in ids:
                    m=metrics[a][fid];old.append(m['current']);best.append(m['oracle'])
                    selected=selections[mat][a][fid]
                    matched=next((h['metric'] for h in m['hypotheses'] if h['name']==selected),None)
                    new.append(matched or dict(id=fid,available=False));swaps+=selected!=cand['arms'][a][fid]['selected_name']
                summaries=[O.D.aggregate(x) for x in (old,new,best)];base,method,ceiling=summaries
                gap=ceiling['ADDsym_AUC']-base['ADDsym_AUC'];delta=method['ADDsym_AUC']-base['ADDsym_AUC']
                out[mat][group][a]=dict(old=base,new=method,fixed_set_oracle=ceiling,delta_AUC=delta,oracle_gap=gap,
                     recovery=delta/gap if abs(gap)>1e-12 else None,selected_changes=swaps,native2D_unchanged=True)
    C.save(d/'RESULTS.json',dict(cycle=ID,groups=out,predictions_lock=C.bind(lock),new_fits=0,updates=0,
         GPU_seconds=0,wall_seconds=time.monotonic()-start,reference='Legacy geometry-derived, reusedDEV',
         no_posthoc_scale_selection=True,delta_px=DELTA),True)
    lines=['# C1 강건 D9 선택 결과','',C.table(['재료','모델','기존AUC','HuberAUC','차이','oracle gap 회수율','선택변경'],[
        [mat,a,f"{v['old']['ADDsym_AUC']:.6f}",f"{v['new']['ADDsym_AUC']:.6f}",f"{v['delta_AUC']:+.6f}",
         'NA' if v['recovery'] is None else f"{100*v['recovery']:.2f}%",v['selected_changes']]
        for mat,gg in out.items() for a,v in gg['ALL'].items()]),
        'native2D·검출·후보생성은 동일하다. 음의 회수율은 그대로 보고하며0으로clamp하지 않는다. GT-free선택을 먼저 고정했다. 이득이 있어도 독립일반화나 정확한원인확정이 아니다.']
    C.save(d/'REPORT_KO.md','\n'.join(lines),True);C.resource(ID,time.monotonic()-start,details='CPU score-only; no fit/GPU')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()

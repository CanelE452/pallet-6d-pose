"""Post-lock reference scoring, with raw 2D recalculated for the chosen expert."""
import numpy as np
from . import common as C

def main():
    lock=C.read(C.sdoc(4)/'REAL_ROUTER_DECISION_LOCK.json');C.verify(lock['decisions']);C.verify(lock['router']);start=C.now();assert start>lock['created_at']
    C.freeze(C.sdoc(4)/'ROUTER_REFERENCE_READ_LOCK.json',dict(created_at=start,decision_lock=C.bind(C.sdoc(4)/'REAL_ROUTER_DECISION_LOCK.json')))
    from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
    from scripts.research.pallet_recording_disjoint_transfer_v1.evaluate import summary
    rr=V.records();groups=V.groups(rr);dec=C.read(C.sraw(4)/'REAL_ROUTER_DECISIONS.json');pm=C.read(C.PREV_RAW/'POSE_METRICS.json');pred=C.read(C.PREV_RAW/'PREDICTIONS.json');truth=C.read(V.E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json')
    arms=['S0','S1','ROUTED','POSTHOC_BEST_EXPERT'];fm={a:{} for a in arms};metrics={a:{} for a in arms};selection={};old2d=C.read(C.PREV_RAW/'FRAME_METRICS.json')
    for r in rr:
        fid=r['id'];selection[fid]={}
        for a in C.ARMS:
            h=next((h for h in pm[a][fid]['hypotheses'] if h['name']==dec[fid]['selected_hypotheses'][a]),None)
            metrics[a][fid]=h['metric'] if h else dict(id=fid,available=False)
        oracle=min(C.ARMS,key=lambda a:(metrics[a][fid].get('ADDsym_normalized') if metrics[a][fid].get('available') else float('inf'),a))
        for a,expert in [('S0','S0'),('S1','S1'),('ROUTED',dec[fid]['chosen']),('POSTHOC_BEST_EXPERT',oracle)]:
            selection[fid][a]=expert;metrics[a][fid]=metrics[expert][fid];t=truth[fid];c=C.selected(pred[expert][fid]);matched=c is not None and V.E.C.H.E.O.iou(c['box_xyxy'],t['box'])>=.5
            q=np.full((9,2),np.nan) if c is None else c['keypoints_xy'];fm[a][fid]=dict(id=fid,**V.E.P.M.measure(q,t['gt'],t['valid'],t['permutations'],t['hw'],matched,c is not None))
            V.E.D.close(fm[a][fid],old2d[expert][fid])
    out={};route={}
    for g,ids in groups.items():
        out[g]={a:dict(twoD=summary([fm[a][i] for i in ids]),sixD=V.E.D.aggregate([metrics[a][i] for i in ids])) for a in arms}
        counts={a:sum(dec[i]['chosen']==a for i in ids) for a in C.ARMS};route[g]=dict(n=len(ids),counts=counts,fraction={a:counts[a]/len(ids) for a in C.ARMS})
    delta={g:out[g]['ROUTED']['sixD']['ADDsym_AUC']-out[g]['S1']['sixD']['ADDsym_AUC'] for g in ('CLEAN','MODERATE','SEVERE','ALL')}
    clean=delta['CLEAN']>0;hardloss=delta['SEVERE']<0 or delta['MODERATE']<0
    primary='CLEAN_HARD_TRADEOFF_SEPARABLE_SIGNAL' if clean and not hardloss else 'CLEAN_RECOVERY_BUT_HARD_LOSS' if clean and hardloss else 'ROUTER_NO_BENEFIT'
    syn=C.read(C.sdoc(4)/'ROUTER_SYNTH_RESULTS.json')['groups']['ALL'];sgain=syn['ROUTED']['ADDsym_AUC']>max(syn['S0']['ADDsym_AUC'],syn['S1']['ADDsym_AUC'])
    C.freeze(C.sraw(4)/'REAL_ROUTER_FRAME_RESULTS.json',dict(metrics=metrics,twoD=fm,expert=selection))
    C.freeze(C.sdoc(4)/'REAL_ROUTER_RESULTS.json',dict(groups=out,routing=route,base_selector=lock['base_selector'],PCK_recomputed=True,raw2D_parity_with_chosen_original=True,GT_oracle='POSTHOC best expert by lower ADD; diagnostic only, not PCK-optimal',references_after_decision_lock=True))
    C.freeze(C.sdoc(4)/'STAGE4_DECISION.json',dict(primary=primary,secondary='SYNTH_ROUTER_REAL_GAP' if sgain and primary!='CLEAN_HARD_TRADEOFF_SEPARABLE_SIGNAL' else None,
        actual_delta_vs_S1=delta,synthetic_better_than_both=sgain,real_used_to_fit_or_select=False,already_viewed_DEV=True,deployable_final=False))
    print('ROUTER_REAL',primary,delta,flush=True)

if __name__=='__main__':main()

"""References enter only after all real scorer decisions are frozen."""
from . import common as C

def main():
    lock=C.read(C.sdoc(3)/'REAL_SCORER_DECISION_LOCK.json');C.verify(lock['decisions']);C.verify(lock['scorer']);start=C.now();assert start>lock['created_at']
    C.freeze(C.sdoc(3)/'REFERENCE_READ_LOCK.json',dict(created_at=start,decisions=C.bind(C.sdoc(3)/'REAL_SCORER_DECISION_LOCK.json')))
    from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
    dec=C.read(C.sraw(3)/'REAL_SCORER_DECISIONS.json');pm=C.read(C.PREV_RAW/'POSE_METRICS.json');records=V.records();groups=V.groups(records)
    out={};trans={};chosen={}
    for arm in C.ARMS:
        chosen[arm]={}
        for fid,r in pm[arm].items():
            selected=next((h for h in r['hypotheses'] if h['name']==dec[arm][fid]['selected']),None)
            chosen[arm][fid]=selected['metric'] if selected else dict(id=fid,available=False)
    for g,ids in groups.items():
        out[g]={};trans[g]={}
        for arm in C.ARMS:
            baseline=V.E.D.aggregate([pm[arm][i]['current'] for i in ids]);scorer=V.E.D.aggregate([chosen[arm][i] for i in ids]);oracle=V.E.D.aggregate([pm[arm][i]['oracle'] for i in ids]);gap=oracle['ADDsym_AUC']-baseline['ADDsym_AUC']
            out[g][arm]=dict(current=baseline,scorer=scorer,oracle=oracle,oracle_current_gap=gap,
                gap_recovery=(scorer['ADDsym_AUC']-baseline['ADDsym_AUC'])/gap if gap>0 else None)
            trans[g][arm]=dict(selected_changed=sum(pm[arm][i]['current'].get('id') is not None and next(h['name'] for h in pm[arm][i]['hypotheses'] if h['metric']==pm[arm][i]['current'])!=dec[arm][i]['selected'] for i in ids),
                recoveries=sum(pm[arm][i]['current'].get('axis_correct') is False and chosen[arm][i].get('axis_correct') is True for i in ids),
                regressions=sum(pm[arm][i]['current'].get('axis_correct') is True and chosen[arm][i].get('axis_correct') is False for i in ids))
    syn=C.read(C.sdoc(2)/'SCORER_SYNTH_TEST.json')['aggregate'];m=out['MODERATE']['S1'];gain=m['scorer']['ADDsym_AUC']>m['current']['ADDsym_AUC']
    collateral=any(out[g]['S1']['scorer']['ADDsym_AUC']<out[g]['S1']['current']['ADDsym_AUC'] for g in ('CLEAN','SEVERE'))
    axis_ok=m['scorer']['axis_correct_count']>=m['current']['axis_correct_count']
    primary='SELECTOR_RECOVERY_WITHOUT_COLLATERAL' if gain and not collateral and axis_ok else 'SELECTOR_RECOVERY_WITH_COLLATERAL' if gain else 'SELECTOR_NO_REAL_RECOVERY'
    secondary='SELECTOR_SYNTH_REAL_GAP' if not gain and syn['learned_accuracy']>syn['current_accuracy'] else None
    s0safe=all(out[g]['S0']['scorer']['ADDsym_AUC']>=out[g]['S0']['current']['ADDsym_AUC'] for g in ('CLEAN','MODERATE','SEVERE'))
    C.freeze(C.sraw(3)/'CHOSEN_METRICS.json',chosen)
    C.freeze(C.sdoc(3)/'REAL_SCORER_RESULTS.json',dict(groups=out,scorer_frozen=True,real_reference_after_lock=True,oracle='POSTHOC NONDEPLOYABLE'))
    C.freeze(C.sdoc(3)/'REAL_SCORER_TRANSITIONS.json',trans)
    C.freeze(C.sdoc(3)/'STAGE3_DECISION.json',dict(primary=primary,secondary=secondary,moderate_gain=gain,axis_count_not_reduced=axis_ok,
        clean_severe_collateral=collateral,S0_valid_for_router=s0safe,
        S0_safeguard_definition='same scorer valid for S0 if CURRENT AUC not reduced on Clean/Moderate/Severe; sign-only conservative common-selector gate',
        already_viewed_DEV=True,no_independent_validation=True))
    # A selector cannot change the frozen 2D coordinates or their final visible-anchor scores.
    C.freeze(C.sdoc(3)/'VERIFIED_ANCHOR_SANITY.json',dict(raw_predictions_unchanged=True,binding=C.bind(C.PREV_RAW/'PREDICTIONS.json'),
        final_anchor=C.bind(C.PREV_DOC/'VERIFIED_ANCHOR_TRANSFER.json'),PCK_identical=True,reason='Only a W/D candidate name is selected; no keypoint output is overwritten.'))
    print('STAGE3',primary,secondary,m,flush=True)

if __name__=='__main__':main()

"""Join frozen decisions to immutable freshly reproduced candidate reference metrics."""
from collections import Counter
import numpy as np
from . import common as C

def main():
    C.immutable();lock=C.read(C.stage(3)/'REAL_SELECTOR_DECISION_LOCK.json');C.verify(lock['decisions']);C.verify(lock['scorer_lock'])
    assert not (C.stage(3)/'REAL_SELECTOR_RESULTS.json').exists(),'Results already frozen'
    start=C.now();assert start>lock['created_at']
    C.save(C.stage(3)/'REFERENCE_READ_START.json',dict(created_at=start,decision_lock=C.bind(C.stage(3)/'REAL_SELECTOR_DECISION_LOCK.json'),
        note='Fresh process starts GT-dependent candidate-metric reads only after all eight decisions frozen. Stage1 historical GT exposure disclosed.'))
    from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
    from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
    from scripts.research.pallet_recording_disjoint_transfer_v1.evaluate import transitions
    decisions=C.read(C.ROOT/lock['decisions']['path']);pm=C.read(C.RAW/'stage1/POSE_METRICS.json');fm=C.read(C.HARDRAW/'FRAME_METRICS.json')
    first=C.read(C.RAW/'stage1/FRAME_DECOMPOSITION_PRIVATE.json');groups=V.groups(V.records());base=C.read(C.stage(1)/'BASELINE_PARITY.json')['groups'];results={};oracles={};tt={};cross={};metrics={}
    for combo,rows in decisions.items():
        model=next(iter(rows.values()))['model'];metrics[combo]={i:C.metric_for(pm[model][i],r['selected']) for i,r in rows.items()}
    for group,ids in groups.items():
        results[group]={};oracles[group]={};tt[group]={};cross[group]={}
        for model in C.MODELS:
            oracles[group][model]=D.aggregate([pm[model][i]['oracle'] for i in ids])
            own='S1SPEC_GEO' if model=='S1' else 'HMANSPEC_GEO';old=decisions[model+'_OLD_GEO'];new=decisions[model+'_'+own]
            oc={i:C.category(pm[model][i],old[i]['selected']) for i in ids};nc={i:C.category(pm[model][i],new[i]['selected']) for i in ids}
            tt[group][model]=dict(wrong_to_correct=[i for i in ids if oc[i]=='SELECTOR_RECOVERABLE' and nc[i]=='SELECTOR_CORRECT'],
                correct_to_wrong=[i for i in ids if oc[i]=='SELECTOR_CORRECT' and nc[i]=='SELECTOR_RECOVERABLE'],
                selection_changed=[i for i in ids if old[i]['selected']!=new[i]['selected']],old_counts=dict(Counter(oc.values())),new_counts=dict(Counter(nc.values())))
            cross[group][model]=dict(Counter(first['paired'][i]['tail']+'__'+nc[i] for i in ids))
        for combo,rows in decisions.items():
            model=next(iter(rows.values()))['model'];agg=D.aggregate([metrics[combo][i] for i in ids]);categories=Counter(C.category(pm[model][i],rows[i]['selected']) for i in ids)
            margins=[rows[i]['score_margin'] for i in ids if rows[i]['score_margin'] is not None]
            agg.update(pose_coverage=agg['available']/len(ids),selection_loss=oracles[group][model]['ADDsym_AUC']-agg['ADDsym_AUC'],
                selector_category_counts=dict(categories),selector_correct_count=categories['SELECTOR_CORRECT'],
                selector_correct_fraction=categories['SELECTOR_CORRECT']/len(ids),selector_correct_denominator='all frames; ties/unavailable reported separately',
                selection_change_vs_old=sum(rows[i]['selected']!=decisions[model+'_OLD_GEO'][i]['selected'] for i in ids),
                score_margin_quantiles=np.quantile(margins,[.1,.5,.9]).tolist() if margins else [],fallback_count=sum(rows[i]['fallback_reason'] is not None for i in ids))
            results[group][combo]=agg
            if combo.endswith('_OLD_GEO') or combo.endswith('_D9'):
                reference=base[group][model]['current' if combo.endswith('_OLD_GEO') else 'D9']
                D.close({k:agg[k] for k in reference},reference)
    twod={g:{m:base[g][m]['twoD'] for m in C.MODELS} for g in groups}
    tr={g:transitions([fm['BASE'][i] for i in ids],[fm['H_MANUAL'][i] for i in ids]) for g,ids in groups.items()}
    anchors=C.read(C.HARD/'VERIFIED_VISIBLE.json');synth=C.read(C.stage(2)/'SYNTH_COMPATIBILITY_MATRIX.json')['combinations'];d=C.decision(results,synth)
    C.save(C.stage(3)/'REAL_SELECTOR_RESULTS.json',dict(groups=results,model_oracles=oracles,twoD=twod,twoD_transitions=tr,verified_visible=anchors,
        twoD_selector_independent=True,oracle_label='POSTHOC / GT-dependent / NONDEPLOYABLE',already_viewed_DEV=True,independent_final_test=False))
    C.save(C.stage(3)/'SELECTOR_TRANSITIONS.json',tt);C.save(C.stage(3)/'TAIL_AND_SELECTOR_CROSSTAB.json',cross)
    C.save(C.stage(3)/'RECORDING_BREAKDOWN.json',{g:dict(combinations=results[g],model_oracles=oracles[g],model_twoD=twod[g]) for g in groups if g.startswith('REC_')})
    C.save(C.RAW/'REAL_SELECTED_METRICS_PRIVATE.json',metrics);C.save(C.stage(3)/'DECISION.json',d)
    render(results,oracles,twod,tt,d)
    print('FINAL_DECISION',d,flush=True)

def render(results,oracles,twod,tt,d):
    labels=['CLEAN','MODERATE','SEVERE'];combos=list(results['ALL'])
    C.figure(3,'01_current_auc_matrix.png','Real DEV: frozen eight-combination CURRENT AUC',labels,{s:[results[g][s]['ADDsym_AUC'] for g in labels] for s in combos},'ADDsym AUC')
    for n,g in [(2,'MODERATE'),(3,'SEVERE'),(4,'CLEAN')]:
        fn={2:'02_moderate_selector_recovery.png',3:'03_severe_selector_recovery.png',4:'04_clean_safeguard.png'}[n]
        C.figure(3,fn,g+' CURRENT vs model oracle',combos,{'CURRENT':[results[g][s]['ADDsym_AUC'] for s in combos],
            'POSTHOC oracle':[oracles[g]['H_MANUAL' if s.startswith('H_MANUAL') else 'S1']['ADDsym_AUC'] for s in combos]},'ADDsym AUC')
    C.figure(3,'05_selection_loss.png','Oracle minus CURRENT (nondeployable headroom)',labels,{s:[results[g][s]['selection_loss'] for g in labels] for s in ('S1_OLD_GEO','S1_S1SPEC_GEO','H_MANUAL_OLD_GEO','H_MANUAL_HMANSPEC_GEO')},'AUC gap')
    C.figure(3,'06_wrong_to_correct.png','H_MANUAL old -> model-specific selector transitions',labels,{k:[len(tt[g]['H_MANUAL'][k]) for g in labels] for k in ('wrong_to_correct','correct_to_wrong')})
    rec=[g for g in results if g.startswith('REC_')]
    C.figure(3,'07_recording_breakdown.png','Per-recording CURRENT AUC (no recording excluded)',rec,{s:[results[g][s]['ADDsym_AUC'] for g in rec] for s in ('S1_OLD_GEO','H_MANUAL_OLD_GEO','H_MANUAL_HMANSPEC_GEO')},'ADDsym AUC')
    C.figure(3,'08_localization_tail_warning.png','Unchanged raw localization tail: selectors cannot fix this',labels,{m:[twod[g][m]['matched_pooled_corner8_P90_px'] for g in labels] for m in C.MODELS},'Matched corner P90 (px)')
    C.figure(3,'09_final_decision.png',d['Q_PIPELINE'],labels,{'HMAN-specific minus S1 old':[d['base_deltas'][g] for g in labels]},'CURRENT AUC delta')
    text='# Stage 3 — frozen real selector compatibility\n\n'+d['Q_SELECTOR_COMPATIBILITY']+'\n\n'+d['Q_PIPELINE']+'\n\n모든 8개 조합 × 128장 = 1,024개 decision을 reference 읽기 전 고정했다. Stage1에서 이미 평가 reference를 열람했다는 사실은 숨기지 않으며, Stage3의 새 decision process는 GT 파일 접근을 차단했다.\n\n'
    for g in C.PRIMARY:
        text+=f'## {g}\n\n'+C.table(['Combination','CURRENT AUC','Axis correct','Coverage','Selection loss','Selector correct'],[[s,v['ADDsym_AUC'],v['axis_correct_count'],v['pose_coverage'],v['selection_loss'],v['selector_correct_count']] for s,v in results[g].items()])+'\n'
    text+='## Localization tail (model-level, not changed by selector)\n\n'+C.table(['Group','Model','PCK10','PCK20','Median px','P90 px','Gross20','Missing'],[[g,m,twod[g][m]['PCK']['10'],twod[g][m]['PCK']['20'],twod[g][m]['matched_pooled_corner8_median_px'],twod[g][m]['matched_pooled_corner8_P90_px'],twod[g][m]['gross20'],twod[g][m]['missing']] for g in C.PRIMARY for m in C.MODELS])
    text+='\n아래 도표는 이미 열람한 recording-disjoint DEV이다. 6D reference는 독립 측정 GT가 아니라 기존 annotation/기하 기반이다. 후보 oracle은 GT-dependent/nondeployable. 난도·recording별 selector routing이나 TEST 후 재학습은 하지 않았다.\n'
    for p in sorted((C.stage(3)/'figures').glob('*.png')):text+=f'\n![{p.stem}](figures/{p.name})\n'
    C.save(C.stage(3)/'STAGE3_REPORT_KO.md',text)

if __name__=='__main__':main()

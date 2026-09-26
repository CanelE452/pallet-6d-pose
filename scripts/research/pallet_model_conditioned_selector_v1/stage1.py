"""Frozen real posthoc decomposition. Does not train or infer any keypoint model."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from . import common as C

def main():
    C.immutable()
    if (C.stage(1)/'BASELINE_PARITY.json').exists():
        assert C.read(C.stage(1)/'BASELINE_PARITY.json')['passed'];print('STAGE1_ALREADY_FROZEN');return
    from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
    from scripts.research.pallet_recording_disjoint_transfer_v1.evaluate import summary
    from scripts.research.pallet_clean19_pose_sensitive_diag_v1.evaluate import pose_row
    from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
    rows=V.records();groups=V.groups(rows);pose=C.read(C.HARDRAW/'POSE_DECISIONS.json')['real'];fm=C.read(C.HARDRAW/'FRAME_METRICS.json')
    oldpm=C.read(C.HARDRAW/'POSE_METRICS.json');published=C.read(C.HARD/'RESULTS.json')['groups'];_,gt=D.Pose.metadata('REAL_DEV')
    pm={};base={};frames={};paritychecks=[]
    for model in C.MODELS:
        arm=C.ARM[model]
        with ProcessPoolExecutor(max_workers=4) as pool:
            pm[model]=dict(pool.map(pose_row,[(r['id'],pose[arm][r['id']],gt[r['id']]) for r in rows],chunksize=8))
        D.close(pm[model],oldpm[arm]);paritychecks.append(model+'_fresh_candidate_metrics')
        frames[model]={}
        for r in rows:
            fid=r['id'];m=pm[model][fid];p=pose[arm][fid];f=fm[arm][fid];errors=np.array(f['errors'],float)
            oldname=p['current'].get('selected_hypothesis');d9name=p['D9'].get('selected_hypothesis')
            cat=C.category(m,oldname);dcat=C.category(m,d9name)
            if cat in ('POSE_UNAVAILABLE','ORACLE_TIE') or dcat in ('POSE_UNAVAILABLE','ORACLE_TIE'):
                compat='ONE_OR_BOTH_UNAVAILABLE' if cat=='POSE_UNAVAILABLE' or dcat=='POSE_UNAVAILABLE' else 'ORACLE_TIE'
            else:
                right=cat=='SELECTOR_CORRECT';dr=dcat=='SELECTOR_CORRECT'
                compat='BOTH_RIGHT' if right and dr else 'BOTH_WRONG' if not right and not dr else 'D9_RIGHT_OLD_GEO_WRONG' if dr else 'OLD_GEO_RIGHT_D9_WRONG'
            cur=m['current'].get('ADDsym_normalized');best=m['oracle'].get('ADDsym_normalized')
            frames[model][fid]=dict(id=fid,severity=r['severity'],recording=r['recording_group'],category=cat,D9_category=dcat,compatibility=compat,
                selected=oldname,D9_selected=d9name,oracle=m['oracle_name'],candidate_ADDnorm={h['name']:h['metric'].get('ADDsym_normalized') for h in m['hypotheses']},
                current_ADDnorm=cur,best_candidate_ADDnorm=best,recoverable_gap=cur-best if cur is not None and best is not None else None,
                corner_errors=errors.tolist(),canonical_errors=f['canonical_errors'],PCK={str(t):float(np.mean(errors<=t)) if len(errors) else None for t in (5,10,20)},
                correct={str(t):int((errors<=t).sum()) for t in (5,10,20)},gross20=int((errors>20).sum()),frame_median_px=float(np.median(errors)) if len(errors) else None,
                frame_mean_px=float(np.mean(errors)) if len(errors) else None,frame_max_px=float(np.max(errors)) if len(errors) else None,matched=f['matched'],detected=f['detected'])
        print('FRESH_BASELINE_SCORED',model,flush=True)
    categories={};gain={};tail={};paired={}
    for fid in frames['S1']:
        s,h=frames['S1'][fid],frames['H_MANUAL'][fid]
        od=C.oracle_delta(s['best_candidate_ADDnorm'],h['best_candidate_ADDnorm'])
        unrealized=od=='ORACLE_GAIN' and (h['current_ADDnorm'] is None or (s['current_ADDnorm'] is not None and h['current_ADDnorm']>=s['current_ADDnorm']))
        td=h['gross20']-s['gross20'];tc='TAIL_WORSENED' if td>0 else 'TAIL_IMPROVED' if td<0 else 'TAIL_SAME'
        paired[fid]=dict(id=fid,oracle_change=od,UNREALIZED_MANUAL_GAIN=unrealized,tail=tc,
            NO_SELECTOR_HEADROOM=h['category'] in ('SELECTOR_CORRECT','ORACLE_TIE'),
            correct_delta={str(t):h['correct'][str(t)]-s['correct'][str(t)] for t in (5,10,20)},gross20_delta=td)
    for g,ids in groups.items():
        base[g]={};categories[g]={}
        for model in C.MODELS:
            arm=C.ARM[model];current=D.aggregate([pm[model][i]['current'] for i in ids]);oracle=D.aggregate([pm[model][i]['oracle'] for i in ids])
            d9=D.aggregate([D.metric(i,pose[arm][i]['D9'],gt[i]) for i in ids]);two=summary([fm[arm][i] for i in ids])
            for name,v in (('current',current),('oracle',oracle),('D9',d9),('twoD',two)):D.close(v,published[g][arm][name]);paritychecks.append(f'{g}/{model}/{name}')
            base[g][model]=dict(current=current,oracle=oracle,D9=d9,twoD=two)
            categories[g][model]=dict(selector=dict(Counter(frames[model][i]['category'] for i in ids)),compatibility=dict(Counter(frames[model][i]['compatibility'] for i in ids)))
        selected=[i for i in ids if paired[i]['UNREALIZED_MANUAL_GAIN']]
        def contribution(model,which):
            # Existing trapezoidal grid AUC is linear across frames, including unavailable failures.
            return D.aggregate([pm[model][i][which] for i in selected])['ADDsym_AUC']*len(selected)/len(ids) if selected else 0.
        gain[g]=dict(frames=len(ids),categories=dict(Counter(paired[i]['oracle_change'] for i in ids)),unrealized_count=len(selected),unrealized_ids=selected,
            unrealized_oracle_AUC_contribution=contribution('H_MANUAL','oracle')-contribution('S1','oracle'),
            unrealized_current_AUC_contribution=contribution('H_MANUAL','current')-contribution('S1','current'))
        tail[g]=dict(categories=dict(Counter(paired[i]['tail'] for i in ids)),cross={
            'UNREALIZED_MANUAL_GAIN_AND_TAIL_WORSENED':sum(paired[i]['UNREALIZED_MANUAL_GAIN'] and paired[i]['tail']=='TAIL_WORSENED' for i in ids),
            'UNREALIZED_MANUAL_GAIN_NOT_TAIL_WORSENED':sum(paired[i]['UNREALIZED_MANUAL_GAIN'] and paired[i]['tail']!='TAIL_WORSENED' for i in ids),
            'NO_SELECTOR_HEADROOM_AND_TAIL_WORSENED':sum(paired[i]['NO_SELECTOR_HEADROOM'] and paired[i]['tail']=='TAIL_WORSENED' for i in ids)},
            model={m:base[g][m]['twoD'] for m in C.MODELS})
    C.save(C.RAW/'stage1/FRAME_DECOMPOSITION_PRIVATE.json',dict(models=frames,paired=paired,groups=groups))
    C.save(C.RAW/'stage1/POSE_METRICS.json',pm)
    C.save(C.stage(1)/'BASELINE_PARITY.json',dict(passed=True,method='fresh GT-based candidate/selected/D9 pose metrics; frozen raw 2D reaggregation',tolerance=dict(atol=1e-7,rtol=1e-7),checks=paritychecks,groups=base))
    for name,val in [('SELECTOR_CATEGORY_COUNTS',categories),('MANUAL_GAIN_DECOMPOSITION',gain),('LOCALIZATION_TAIL_DECOMPOSITION',tail)]:C.save(C.stage(1)/(name+'.json'),val)
    C.save(C.stage(1)/'FRAME_DECOMPOSITION_PRIVATE.json',dict(private_artifact=C.bind(C.RAW/'stage1/FRAME_DECOMPOSITION_PRIVATE.json'),reason='Large per-frame records remain local; summaries and selected explanatory images published.'))
    render(base,categories,gain,tail,groups)
    print('STAGE1_BASELINE_REPRODUCED',len(paritychecks),flush=True)

def render(base,categories,gain,tail,groups):
    labels=['CLEAN','MODERATE','SEVERE'];rec=[g for g in groups if g.startswith('REC_')]
    C.figure(1,'01_selector_categories.png','Old GEO selector-recoverable frames',labels,{m:[categories[g][m]['selector'].get('SELECTOR_RECOVERABLE',0) for g in labels] for m in C.MODELS})
    C.figure(1,'02_d9_vs_oldgeo.png','H_MANUAL D9 vs old GEO',labels,{s:[base[g]['H_MANUAL'][s]['ADDsym_AUC'] for g in labels] for s in ('D9','current','oracle')},'ADDsym AUC')
    C.figure(1,'03_manual_oracle_gain.png','H_MANUAL vs S1 oracle quality',labels,{s:[gain[g]['categories'].get(s,0) for g in labels] for s in ('ORACLE_GAIN','ORACLE_LOSS','ORACLE_TIE')})
    C.figure(1,'04_unrealized_manual_gain.png','Oracle gains not realized by old GEO',labels,{'Frames':[gain[g]['unrealized_count'] for g in labels]})
    C.figure(1,'05_localization_tail.png','Raw corner tail (>20 px), independent of selector',labels,{s:[tail[g]['categories'].get(s,0) for g in labels] for s in ('TAIL_WORSENED','TAIL_IMPROVED','TAIL_SAME')})
    C.figure(1,'06_recording_breakdown.png','Per-recording H_MANUAL candidate vs current quality',rec,{s:[base[g]['H_MANUAL'][s]['ADDsym_AUC'] for g in rec] for s in ('current','oracle')},'ADDsym AUC')
    text='# Stage 1 — frozen failure decomposition\n\n기존 수치 재현 통과. 후보와 선택 결과의 6D 지표는 frozen pose와 기존 reference로 재계산했고, 2D는 frozen per-corner error를 재집계했다. 새 모델 추론·학습 없음.\n\n'
    text+=C.table(['Group','Model','D9 AUC','Old GEO AUC','Oracle AUC','Recoverable'],[[g,m,base[g][m]['D9']['ADDsym_AUC'],base[g][m]['current']['ADDsym_AUC'],base[g][m]['oracle']['ADDsym_AUC'],categories[g][m]['selector'].get('SELECTOR_RECOVERABLE',0)] for g in C.PRIMARY for m in C.MODELS])
    text+='\n'+C.table(['Group','D9 right / GEO wrong','GEO right / D9 wrong','Unrealized manual gain','Tail worsened'],[[g,categories[g]['H_MANUAL']['compatibility'].get('D9_RIGHT_OLD_GEO_WRONG',0),categories[g]['H_MANUAL']['compatibility'].get('OLD_GEO_RIGHT_D9_WRONG',0),gain[g]['unrealized_count'],tail[g]['categories'].get('TAIL_WORSENED',0)] for g in C.PRIMARY])
    text+='\nSELECTOR_RECOVERABLE은 두 유효 후보 중 낮은 ADDnorm을 선택하지 않은 경우이다. 두 후보가 모두 나쁘다는 임의 threshold는 만들지 않았으며 best_candidate_ADDnorm 연속값을 저장했다. NO_SELECTOR_HEADROOM은 현재 이미 oracle 후보를 고르거나 동률인 경우이며, candidate quality가 좋다는 뜻은 아니다. 동률은 별도 ORACLE_TIE로 분리했다.\n\nORACLE는 사후 GT 의존·비배포 가능 분석이다. HELDOUT128은 이미 열람한 recording-disjoint DEV이다. selector는 raw keypoint tail을 바꾸지 못한다.\n'
    for p in sorted((C.stage(1)/'figures').glob('*.png')):text+=f'\n![{p.stem}](figures/{p.name})\n'
    C.save(C.stage(1)/'STAGE1_REPORT_KO.md',text)

if __name__=='__main__':main()

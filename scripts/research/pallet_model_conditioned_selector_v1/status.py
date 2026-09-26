"""Read-only full requested CLI status, or save an explicitly timestamped snapshot."""
import argparse
import json
from . import common as C

def main(save=False):
    b=C.read(C.DOC/'INPUT_BINDINGS.json');base=C.read(C.stage(1)/'BASELINE_PARITY.json')['groups'];cat=C.read(C.stage(1)/'SELECTOR_CATEGORY_COUNTS.json');gain=C.read(C.stage(1)/'MANUAL_GAIN_DECOMPOSITION.json');tail=C.read(C.stage(1)/'LOCALIZATION_TAIL_DECOMPOSITION.json')
    r=C.read(C.stage(3)/'REAL_SELECTOR_RESULTS.json');d=C.read(C.stage(3)/'DECISION.json');syn=C.read(C.stage(2)/'SYNTH_COMPATIBILITY_MATRIX.json')['combinations'];shift=C.read(C.stage(2)/'FEATURE_SHIFT.json');tr=C.read(C.stage(3)/'SELECTOR_TRANSITIONS.json');sl=C.read(C.stage(2)/'SCORER_LOCK.json')
    receipts={n:C.read(C.OUT/f'STAGE{n}_GIT.json') if (C.OUT/f'STAGE{n}_GIT.json').exists() else None for n in (1,2,3)}
    v=dict(STATUS='COMPLETE_EXPERIMENT_PUBLICATION_'+('VERIFIED' if receipts[3] else 'PENDING'),HEAD_START=b['head_start'],HEAD_END=C.git('rev-parse','HEAD'),BRANCH=C.git('branch','--show-current'),snapshot_at=C.now(),
        STAGE1=dict(BASELINE_REPRODUCED=True,MODERATE_FRAMES=21,SEVERE_FRAMES=78))
    for group,short in [('MODERATE','MOD'),('SEVERE','SEV')]:
        for model,name in [('S1','S1'),('H_MANUAL','HMAN')]:
            a=base[group][model];c=cat[group][model]
            v['STAGE1'][name+'_'+short]=dict(OLD_GEO_CURRENT_AUC=a['current']['ADDsym_AUC'],D9_CURRENT_AUC=a['D9']['ADDsym_AUC'],ORACLE_AUC=a['oracle']['ADDsym_AUC'],
                SELECTOR_RECOVERABLE=c['selector'].get('SELECTOR_RECOVERABLE',0),D9_RIGHT_OLD_GEO_WRONG=c['compatibility'].get('D9_RIGHT_OLD_GEO_WRONG',0),OLD_GEO_RIGHT_D9_WRONG=c['compatibility'].get('OLD_GEO_RIGHT_D9_WRONG',0))
            if model=='H_MANUAL':v['STAGE1'][name+'_'+short].update(UNREALIZED_MANUAL_GAIN=gain[group]['unrealized_count'],TAIL_WORSENED_FRAMES=tail[group]['categories'].get('TAIL_WORSENED',0))
    v['STAGE2']=dict(SYNTH_SPLIT='exact existing renderer groups, seed20260925',TRAIN=4096,VAL=1024,TEST=1024,SYNTH_MATRIX={k:s['accuracy'] for k,s in syn.items()},FEATURE_SHIFT=dict(MEDIAN_L2=shift['median_L2'],P90_L2=shift['P90_L2']))
    for model,name in [('S1','S1_SPECIFIC'),('H_MANUAL','HMAN_SPECIFIC')]:
        val=C.read(C.stage(2)/('S1_SCORER_VAL.json' if model=='S1' else 'HMAN_SCORER_VAL.json'))
        v['STAGE2'][name]=dict(VAL_ACC=val['best_val_accuracy'],TEST_ACC=syn[model+('_S1SPEC_GEO' if model=='S1' else '_HMANSPEC_GEO')]['accuracy'],CHECKPOINT=sl['checkpoints'][model])
    for g in ('CLEAN','MODERATE','SEVERE'):
        a=r['groups'][g];v['STAGE3_'+g]={name+'_AUC':a[key]['ADDsym_AUC'] for name,key in [('S1_OLD_GEO','S1_OLD_GEO'),('S1_S1SPEC','S1_S1SPEC_GEO'),('HMAN_OLD_GEO','H_MANUAL_OLD_GEO'),('HMAN_D9','H_MANUAL_D9'),('HMAN_S1SPEC','H_MANUAL_S1SPEC_GEO'),('HMAN_HMANSPEC','H_MANUAL_HMANSPEC_GEO')]}
        v['STAGE3_'+g].update(HMAN_ORACLE_AUC=r['model_oracles'][g]['H_MANUAL']['ADDsym_AUC'],HMAN_HMANSPEC_SELECTION_LOSS=a['H_MANUAL_HMANSPEC_GEO']['selection_loss'],WRONG_TO_CORRECT_VS_OLD=len(tr[g]['H_MANUAL']['wrong_to_correct']),CORRECT_TO_WRONG_VS_OLD=len(tr[g]['H_MANUAL']['correct_to_wrong']))
    v['LOCALIZATION_TAIL']={}
    for g,short in [('MODERATE','MOD'),('SEVERE','SEV')]:
        for m,n in [('S1','S1'),('H_MANUAL','HMAN')]:
            a=r['twoD'][g][m];v['LOCALIZATION_TAIL'].update({n+'_'+short+'_PCK20':a['PCK']['20'],n+'_'+short+'_P90':a['matched_pooled_corner8_P90_px']})
    for m,n in [('S1','S1'),('H_MANUAL','HMAN')]:v['LOCALIZATION_TAIL'][n+'_DETECTIONS']=r['twoD']['ALL'][m]['detected']
    for k in ('Q_SELECTOR_COMPATIBILITY','Q_PIPELINE','SYNTH_REAL_COMPATIBILITY_GAP'):v[k]=d[k]
    v.update(CURRENT_DEPLOYMENT_CANDIDATE=d['current_deployment_candidate'],DEPLOYMENT_CONFIGURATION_CHANGED=False,ADDITIONAL_LABELING=0,KEYPOINT_MODEL_TRAINING=0,NEW_SELECTOR_TRAININGS=2,NEXT_ONE_STEP=d['next_one_step'],REPORT=str((C.DOC/'REPORT_KO.md').relative_to(C.ROOT)),
        COMMITS={f'STAGE{n}':receipts[n]['commit'] if receipts[n] else None for n in receipts},
        LATEST_REMOTE_HEAD=next((receipts[n]['remote_HEAD'] for n in (3,2,1) if receipts[n]),None),REMOTE_HEAD_NOTE='latest verified receipt, not a live network query',
        GIT_STATUS=C.git('status','--short','--branch'),TASK_GIT_STATUS=C.git('status','--short','--',str(C.DOC.relative_to(C.ROOT)),str((C.ROOT/'scripts/research'/C.NAME).relative_to(C.ROOT))))
    for n in receipts:v['STAGE'+str(n)+'_COMMIT']=receipts[n]['commit'] if receipts[n] else None;v['STAGE'+str(n)+'_PUSH']=bool(receipts[n] and receipts[n]['push_verified'])
    text=json.dumps(v,ensure_ascii=False,indent=2)+'\n'
    if save:C.save(C.DOC/'FINAL_CLI_OUTPUT.json',v,immutable=False);C.save(C.DOC/'FINAL_CLI_OUTPUT.txt',text,immutable=False)
    print(text)
    return v

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');main(p.parse_args().save)

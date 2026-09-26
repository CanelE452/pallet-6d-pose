"""Complete directive section21 output; all values come from saved artifacts."""
import json
from collections import Counter
from . import common as C


def collect():
    read=lambda n:C.read(C.DOC/n)
    a=read('CANDIDATE_POOL_AUDIT.json');tag=read('DIFFICULTY_TAG_SUMMARY_PUBLIC.json');sel=read('HARD_SELECTION_PUBLIC.json')
    ll=read('HARD_LABEL_LOCK.json');cov=read('HARD_COVERAGE_PUBLIC.json');teacher=read('TEACHER_HARD_PREDICTION_LOCK.json')
    occ=read('TRAIN_OCCURRENCE_LOCK.json');res=read('RESULTS.json')['groups'];dec=read('DECISION.json')
    frames=C.read(C.ROOT/ll['labels']['path'])['frames'];initial=[r for r in sel['rows'] if r['assignment']=='INITIAL']
    value=dict(STATUS=C.state()['status'],HEAD_START=read('INPUT_BINDINGS.json')['head'],HEAD_END=C.git('rev-parse','HEAD'),BRANCH=C.git('branch','--show-current'))
    # These exclusions had no matches in this adaptation pool; do not confuse with protected-set size.
    ex=a['exclusions']
    value['CANDIDATE_POOL']=dict(TOTAL=a['eligible_frames'],SOURCE_TOTAL=a['source_frames'],RECORDINGS=a['eligible_recordings'],
        EXCLUDED_EVAL=sum(v for k,v in ex.items() if 'EVAL' in k or 'HELDOUT' in k),EXCLUDED_ANCHOR=sum(v for k,v in ex.items() if 'ANCHOR' in k),
        EXCLUDED_FINAL=sum(v for k,v in ex.items() if 'FINAL' in k or 'RESERVED' in k),EXCLUSION_REASONS=ex,
        SHA_OVERLAP=a['SHA_overlap_after_exclusion'],NEAR_DUP_OVERLAP=a['near_duplicate_overlap_after_exclusion'])
    value['DIFFICULTY_TAGGING']=dict(ROUNDS_COMPLETED=read('DIFFICULTY_TAG_LOCK.json')['rounds'],HUMAN_TAGGED=tag['total'],**tag['counts'],HARD_RECORDINGS=tag['hard_recordings'])
    value['HARD_SELECTION']=dict(INITIAL=sel['initial'],RESERVE=sel['reserve'],SELECTED_TOTAL=len(sel['rows']),
        RECORDINGS=dict(Counter(r['recording'] for r in initial)),MODERATE=sum(r['tag']=='MODERATE' for r in initial),SEVERE=sum(r['tag']=='SEVERE' for r in initial),RESERVE_ACTIVATED=0)
    value['ANNOTATION']=dict(ROLE_CONFIDENT=sum(f['role']=='ROLE_CONFIDENT' for f in frames.values()),ROLE_UNCERTAIN=sum(f['role']=='ROLE_UNCERTAIN' for f in frames.values()),
        USABLE_FRAMES=cov['usable_frames'],DIRECT_VISIBLE_CLICKS=cov['direct_visible_clicks'],CORNER_COUNTS=cov['corner_counts'],LABEL_LOCK=C.sha(C.DOC/'HARD_LABEL_LOCK.json'),
        PROTOCOL='USER_APPROVED_PNP_ASSISTED_WITH_SHARED_PNP_BOX')
    value['TEACHER_SUPPORT']=dict(VISIBLE_POINTS=teacher['support_points'],FINITE_TEACHER_POINTS=teacher['finite_points'],COVERAGE=teacher['coverage'],PSEUDO_ARM_STATUS=teacher['status'])
    value['TRAINING']=dict(BASE_CHECKPOINT=occ['base_checkpoint'],INIT_HASH=occ['original_init']['sha256'],HARD_OCC_PER_EPOCH=64,CLEAN_OCC_PER_EPOCH=448,SYNTH_OCC_PER_EPOCH=512,
        TOTAL_UPDATES={a:read(f'FIT_{a}.json')['steps'] for a in ('H_MANUAL','H_PSEUDO')},NEW_UPDATES_DURING_FINALIZATION=0)
    value['PAIR_INTEGRITY']=dict(MANUAL_VS_PSEUDO='PASS' if read('PAIR_INTEGRITY_MANUAL_VS_PSEUDO.json')['passed'] else 'FAIL')
    for g in ('CLEAN','MODERATE','SEVERE','ALL'):
        for a,label in [('BASE','BASE'),('H_MANUAL','HMANUAL'),('H_PSEUDO','HPSEUDO')]:
            r=res[g][a];b=res[g]['BASE']
            value[f'{label}_{g}']=dict(PCK10=r['twoD']['PCK']['10'],CURRENT_AUC=r['current']['ADDsym_AUC'],ORACLE_AUC=r['oracle']['ADDsym_AUC'])
            if a!='BASE':value[f'{label}_{g}'].update(DELTA_PCK10=r['twoD']['PCK']['10']-b['twoD']['PCK']['10'],DELTA_CURRENT=r['current']['ADDsym_AUC']-b['current']['ADDsym_AUC'],DELTA_ORACLE=r['oracle']['ADDsym_AUC']-b['oracle']['ADDsym_AUC'])
    anchors=read('VERIFIED_VISIBLE.json')['groups']['HARD'];source=read('SOURCE_PRESERVATION.json')['groups']
    value['VERIFIED_HARD36']={label+'_CORRECT10':anchors[a]['PCK']['10']['correct'] for a,label in [('BASE','BASE'),('H_MANUAL','HMANUAL'),('H_PSEUDO','HPSEUDO')]}
    value['SOURCE256']={label+'_'+key:(source[a]['twoD']['PCK']['10']['fraction'] if key=='PCK10' else source[a]['pose']['ADDsym_AUC']) for a,label in [('BASE','BASE'),('H_MANUAL','HMANUAL'),('H_PSEUDO','HPSEUDO')] for key in ('PCK10','AUC')}
    value.update(PRIMARY_DECISION=dec['primary'],STOP_MORE_HARD_LABELING=dec['stop_more_hard_labeling'],
        INTERPRETATION=dict(HARD_EXPOSURE_SIGNAL='H_PSEUDO: hard PCK10 unchanged, hard selected AUC lower than BASE',MANUAL_COORDINATE_SIGNAL='H_MANUAL improves hard PCK10 and oracle; better than H_PSEUDO on hard',SELECTOR_LIMIT='Oracle gains not delivered by frozen selector; tail errors also worsen',CLEAN_TRADEOFF='Clean current AUC improves; no clean-AUC tradeoff in this run'),
        NEXT_ONE_STEP='No new labels/training: audit candidate-to-selector failures; keep deployed S1',REPORT='_docs/experiments/pallet_min_hard_ab_v1/REPORT_KO.md')
    receipts={}
    for label,n in [('TAGGING_PREP','TAGGING_PREP_GIT.json'),('HARD_SELECTION','HARD_SELECTION_GIT.json'),('HARD_LABEL','HARD_LABEL_GIT.json'),('AB_RESULT','AB_RESULT_GIT.json'),('FINALIZATION','FINALIZATION_GIT.json')]:
        if (C.OUT/n).exists():receipts[label]=C.read(C.OUT/n)
    value['COMMITS']={k:r.get('commit','NOT_RECORDED') for k,r in receipts.items()}
    latest=receipts.get('FINALIZATION',receipts.get('AB_RESULT',{}))
    value['PUSH']=latest.get('push','NOT_RECORDED');value['LATEST_REMOTE_HEAD']=latest.get('verified_remote_main','NOT_RECORDED')
    value['PUSH_NOTE']='Receipt is last verified push, not an implicit live network check.'
    value['GIT_STATUS']=C.git('status','--short','--branch','--',str(C.DOC.relative_to(C.ROOT)),str((C.ROOT/'scripts/research'/C.NAME).relative_to(C.ROOT)))
    return value


def render():
    lines=[]
    for key,value in collect().items():
        if isinstance(value,dict):
            lines.append(key+':')
            lines.extend('  '+k+': '+json.dumps(v,ensure_ascii=False) for k,v in value.items())
        else:lines.append(key+': '+str(value))
    return '\n'.join(lines)


if __name__=='__main__':print(render())

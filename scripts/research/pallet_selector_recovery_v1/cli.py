"""Print the requested final handoff, using live local HEAD (no mutation)."""
import subprocess
from . import common as C

def main():
    one=C.read(C.sdoc(1)/'MODERATE_SELECTOR_DIAGNOSTIC.json')['groups']['MODERATE'];two=C.read(C.sdoc(2)/'SCORER_SYNTH_TEST.json');three=C.read(C.sdoc(3)/'REAL_SCORER_RESULTS.json')['groups'];trans=C.read(C.sdoc(3)/'REAL_SCORER_TRANSITIONS.json')['MODERATE']['S1'];d3=C.read(C.sdoc(3)/'STAGE3_DECISION.json');four=C.read(C.sdoc(4)/'REAL_ROUTER_RESULTS.json');syn=C.read(C.sdoc(4)/'ROUTER_SYNTH_RESULTS.json')['groups']['ALL'];d4=C.read(C.sdoc(4)/'STAGE4_DECISION.json');final=C.read(C.DOC/'FINAL_DECISION.json');head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    def emit(k,v=''):print(f'{k}: {v}')
    def git(st):
        r=C.read(C.DOC/f'STAGE{st}_GIT.json');emit(f'STAGE{st}_COMMIT',r['commit']);emit(f'STAGE{st}_PUSH',r['push'])
    emit('STATUS',final['status']);emit('HEAD_START',C.read(C.DOC/'INPUT_BINDINGS.json')['head_start']);emit('HEAD_END',head);emit('BRANCH',subprocess.check_output(['git','branch','--show-current'],text=True).strip())
    emit('STAGE1');emit('MODERATE_FRAMES',21);emit('CURRENT_AUC',one['CURRENT']['ADDsym_AUC']);emit('ORACLE_AUC',one['ORACLE']['ADDsym_AUC']);emit('SELECTION_LOSS',one['selection_loss']);emit('AXIS_CURRENT_CORRECT',one['CURRENT']['axis_correct_count']);emit('ALT_BETTER_COUNT',one['alternate_ADD_better']);emit('CURRENT_WRONG_ALT_BETTER',one['current_wrong_alternate_better']);git(1)
    emit('STAGE2');emit('SYNTH_TRAIN',4096);emit('SYNTH_VAL',1024);emit('SYNTH_TEST',1024);emit('FEATURE_MODE',C.read(C.sdoc(2)/'SYNTH_PREDICTION_LOCK.json')['feature_mode']);emit('SELECTED_SCORER',two['winner']);emit('CURRENT_SELECTOR_SYNTH_TEST_ACC',two['aggregate']['current_accuracy']);emit('LEARNED_SCORER_SYNTH_TEST_ACC',two['aggregate']['learned_accuracy'])
    for a in C.ARMS:emit(a+'_SYNTH_TEST_ACC',two['by_expert'][a]['TEST']['learned_accuracy'])
    git(2);m=three['MODERATE']['S1'];emit('STAGE3_MODERATE')
    for k in ('current','scorer','oracle'):emit(k.upper()+'_AUC',m[k]['ADDsym_AUC'])
    emit('GAP_RECOVERY',m['gap_recovery']);emit('CURRENT_AXIS_CORRECT',m['current']['axis_correct_count']);emit('SCORER_AXIS_CORRECT',m['scorer']['axis_correct_count']);emit('RECOVERIES',trans['recoveries']);emit('REGRESSIONS',trans['regressions'])
    for g in ('CLEAN','SEVERE'):
        emit('STAGE3_'+g);emit('CURRENT_AUC',three[g]['S1']['current']['ADDsym_AUC']);emit('SCORER_AUC',three[g]['S1']['scorer']['ADDsym_AUC'])
    emit('STAGE3_DECISION',d3['primary']);git(3);emit('STAGE4');emit('BASE_SELECTOR_FOR_ROUTER',four['base_selector']);emit('STAGE4_SYNTH');emit('ROUTER_TEST_ACC',syn['accuracy'])
    for a,k in [('S0','S0_ADD'),('S1','S1_ADD'),('ROUTED','ROUTED_ADD'),('ORACLE','ORACLE_EXPERT_ADD')]:emit(k,syn[a]['mean_ADDnorm'])
    emit('ADD_UNITS','mean exact synthetic C2 ADD / object diameter')
    for g in ('CLEAN','MODERATE','SEVERE'):
        emit('STAGE4_REAL_'+g)
        for a in ('S0','S1','ROUTED'):
            if g=='CLEAN':emit(a+'_PCK10',four['groups'][g][a]['twoD']['PCK']['10'])
            emit(a+'_AUC',four['groups'][g][a]['sixD']['ADDsym_AUC'])
    emit('ROUTE_FRACTION')
    for g in ('CLEAN','MODERATE','SEVERE'):emit(g+'_S0_S1',four['routing'][g])
    emit('STAGE4_DECISION',d4['primary']);emit('DUAL_INFERENCE_LATENCY_MS',C.read(C.sdoc(4)/'COMPUTE_COST.json')['timings_ms']['dual_total_ms']);git(4)
    emit('FINAL')
    for k,v in final['questions'].items():emit(k,v)
    emit('PRIMARY_BOTTLENECK_NOW',final['primary_bottleneck']);emit('SECONDARY_BOTTLENECK_NOW',final['secondary_bottleneck']);emit('NEXT_ONE_STEP',final['next_one_step']);emit('REPORT',str((C.DOC/'REPORT_KO.md').relative_to(C.ROOT)))
    emit('GIT_STATUS_THIS_TASK',subprocess.check_output(['git','status','--short','--',str(C.DOC.relative_to(C.ROOT)),str((C.ROOT/'scripts/research'/C.NAME).relative_to(C.ROOT))],text=True).strip() or 'CLEAN')
    emit('OTHER_WORKTREE','Pre-existing unrelated untracked work preserved; not included in this push.')

if __name__=='__main__':main()

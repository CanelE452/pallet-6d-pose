import subprocess
from . import common as C
def main():
    b=C.read(C.DOC/'BASELINE_LOCK.json');f=C.read(C.DOC/'FIT.json');r=C.read(C.DOC/'RESULTS.json')['groups'];a=C.read(C.DOC/'VERIFIED_VISIBLE.json')['groups'];d=C.read(C.DOC/'DECISION.json');gap=C.read(C.DOC/'SUPERVISION_GAP_DIAGNOSTIC.json') if (C.DOC/'SUPERVISION_GAP_DIAGNOSTIC.json').exists() else None
    labelpath=C.ROOT/'_docs/experiments/pallet_min_hard_labels_v1/STATUS.json';label=C.read(labelpath) if labelpath.exists() else None
    def emit(k,v=''):print(f'{k}: {v}')
    emit('STATUS','PRESERVATION_COMPLETE / '+(label['status'] if label else d['hard_labeling']));emit('HEAD_START',C.read(C.DOC/'INPUT_BINDINGS.json')['head']);emit('HEAD_END',subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());emit('BRANCH',subprocess.check_output(['git','branch','--show-current'],text=True).strip());emit('BASELINE');emit('MODEL','S1');emit('SELECTOR','frozen GEO_LINEAR')
    for g in ('CLEAN','MODERATE','SEVERE'):emit(g+'_AUC',b['groups'][g]['current']['ADDsym_AUC'])
    emit('VERIFIED_HARD_PCK10',b['anchors']['HARD']['PCK']['10']['fraction']);emit('PRESERVATION');emit('ZERO_INIT_PARITY',C.read(C.DOC/'ZERO_INIT_PARITY.json')['passed']);emit('ADAPTER_PARAMS',f['adapter_params']);emit('TRAIN_STEPS',f['steps']);emit('CLEAN_LOSS_FINAL',f['epochs'][-1]['L_clean']);emit('OCC_PRESERVE_LOSS_FINAL',f['epochs'][-1]['L_occ_preserve']);emit('LOSS_AGGREGATION','final epoch mean')
    for g in ('CLEAN','MODERATE','SEVERE'):
        emit('PRES1_'+g);emit('PCK10',r[g]['PRES1']['twoD']['PCK']['10']);emit('AUC',r[g]['PRES1']['current']['ADDsym_AUC']);emit('DELTA_AUC_VS_S1',d['delta_AUC'][g]);emit('ORACLE_AUC',r[g]['PRES1']['oracle']['ADDsym_AUC'])
    emit('PRES1_VERIFIED_HARD');emit('POINTS',36);emit('PCK10',a['HARD']['PRES1']['PCK']['10']['fraction']);emit('DELTA_CORRECT_COUNT',d['hard_correct_delta']);emit('PRESERVATION_DECISION',d['primary']);emit('SUPERVISION_GAP');emit('RUN',gap is not None)
    if gap:
        for k,field in [('HARD_VISIBLE_POINTS','hard_visible_points'),('STUDENT_WRONG10','student_wrong10'),('TEACHER_CAN_TEACH','teacher_can_teach'),('BOTH_FAIL_VISIBLE','both_fail_visible'),('GROSS_BOTH_FAIL','gross_both_fail'),('MISSING_TRUSTED_SUPPORT','missing_trusted_support'),('DISTINCT_FRAMES','distinct_frames'),('DISTINCT_CORNERS','distinct_corners'),('GAP_DECISION','decision')]:emit(k,gap[field])
    emit('MIN_HARD_LABELING');emit('JUSTIFIED',gap is not None and gap['decision']=='MIN_HARD_LABELING_JUSTIFIED')
    if label:
        for k in ('status','initial_images','reserve_images','candidate_recordings','candidate_moderate','candidate_severe','usable_frames','direct_visible_clicks','role_uncertain_frames','label_lock'):emit(k.upper(),label[k])
    emit('USER_ACTION_REQUIRED','YES' if label and label['user_action_required'] else 'NO');emit('IF_YES_COMMAND','N/A — no valid annotation queue yet' if label else 'N/A');emit('PRESERVATION_REPORT',str((C.DOC/'REPORT_KO.md').relative_to(C.ROOT)));emit('HARD_LABEL_REPORT','_docs/experiments/pallet_min_hard_labels_v1/REPORT_KO.md' if label else 'N/A')
    receipt=C.OUT/'GIT_RECEIPTS.json'
    if receipt.exists():
        for k,v in C.read(receipt).items():emit(k,v)
    emit('LABEL_FINAL_COMMIT','N/A — no human labels');emit('LABEL_FINAL_PUSH','N/A');emit('NEXT_AFTER_THIS','Need independently tagged hard candidates from >=3 recordings; do not infer labeling sufficiency or relax the gate automatically.' if label else 'Freeze candidate and plan independent recording.')
    emit('GIT_STATUS',subprocess.check_output(['git','status','--short','--branch','--',str(C.DOC.relative_to(C.ROOT)),str((C.ROOT/'scripts/research'/C.NAME).relative_to(C.ROOT)), '_docs/experiments/pallet_min_hard_labels_v1','scripts/research/pallet_min_hard_labels_v1'],text=True).strip())
if __name__=='__main__':main()

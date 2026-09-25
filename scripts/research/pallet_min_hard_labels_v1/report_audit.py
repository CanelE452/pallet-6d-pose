"""Publish honest blocked-annotation status, without a pretend annotation queue."""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .inventory import P,DOC,RAW,save
def main():
    a=P.read(DOC/'INVENTORY_AUDIT.json');e=P.read(DOC/'EXPANDED_POOL_AUDIT.json');p=P.read(RAW/'CANDIDATES_PRIVATE.json');assert a['status']==e['status']=='HARD_CANDIDATE_METADATA_INSUFFICIENT';assert e['additional_hard_tagged_candidates']==0
    figures=DOC/'figures';figures.mkdir(exist_ok=True)
    def finish(name):plt.tight_layout();plt.savefig(figures/name,dpi=150);plt.close()
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(list(a['recordings']),list(a['recordings'].values()));ax.set(ylabel='Eligible hard-tagged frames before selection',title='Only2 recordings; required >=3 / max3 selected per recording');finish('01_candidate_recordings.png')
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(['MODERATE','SEVERE'],[a['severity'].get(s,0) for s in ('MODERATE_OCCLUSION','SEVERE_OCCLUSION')]);ax.set(ylabel='Eligible candidates (not selected)',title='Existing human/capture metadata only');finish('02_severity_distribution.png')
    for name,text in [('03_corner_coverage.png','NEW TRUSTED CORNER COVERAGE: NOT COLLECTED\nNo candidate queue; no invented coordinates'),('04_annotation_status.png','ANNOTATION NOT STARTED\nHARD_CANDIDATE_METADATA_INSUFFICIENT\nInitial0 / reserve0 / human clicks0'),('05_qa_summary.png','LABEL QA: NOT APPLICABLE YET\nNo human labels and no final label lock')]:
        fig,ax=plt.subplots(figsize=(9,3));ax.axis('off');ax.text(.5,.5,text,ha='center',va='center',fontsize=13);finish(name)
    for b in a['bindings']+e['bindings']:P.verify(b)
    assert not (RAW/'LABELS_PRIVATE.json').exists() and not (DOC/'MIN_HARD_LABELS_LOCK.json').exists()
    excluded=set(p['excluded_sha']);assert all(r['image']['sha256'] not in excluded and r['min_MAD_to_exclusions']>2 for r in p['records'])
    assert all(r['recording'] not in set(p['reserved_recordings'])|set(p['heldout_recordings']) for r in p['records'])
    save(DOC/'PREPARATION_AUDIT.json',dict(status='PASS_FOR_METADATA_STOP',source_hashes_checked=len(a['bindings'])+len(e['bindings']),
        tests=dict(selection_no_model_predictions=a['no_model_prediction_reads'],eval_anchor_clean10_h10_excluded=True,sha_overlap_zero=True,near_duplicate_zero=True,reserved_recordings_excluded=True,
            no_candidate_selection_when_recording_quota_insufficient=len(a['recordings'])<3,no_fabricated_labels=True,no_public_exact_coordinates=True),
        conditional_NOT_RUN=['GUI_no_prediction_overlay','GUI_no_PnP_label','visible_click_required','nonvisible_xy_none','role_uncertain_excluded','label_lock_hash'],reason='Metadata eligibility stop; GUI/labels not created. Not a completed annotation phase.'))
    save(DOC/'STATUS.json',dict(status=a['status'],justified=True,initial_images=0,reserve_images=0,usable_frames=0,direct_visible_clicks=0,role_uncertain_frames=0,label_lock=None,
        candidate_recordings=len(a['recordings']),candidate_moderate=a['severity'].get('MODERATE_OCCLUSION',0),candidate_severe=a['severity'].get('SEVERE_OCCLUSION',0),user_action_required=False,
        next='Need model-independent hard metadata from at least one additional non-evaluation/non-reserved recording before fixed8-frame selection. No relabel/retraining or invented GUI task.'))
    path=DOC/'REPORT_KO.md';path.write_text(path.read_text()+'\n## 상태 그림\n\n아래 16장은 선택된 annotation queue가 아니라 적격성 검사 후보 수다. 3개 recording 조건이 충족되지 않아 실제 선택0장/새 클릭0개다.\n\n'+'\n\n'.join(f'![{f.stem}](figures/{f.name})' for f in sorted(figures.glob('*.png')))+'\n')
    print('LABEL_PREPARATION_METADATA_STOP_AUDITED',flush=True)
if __name__=='__main__':main()

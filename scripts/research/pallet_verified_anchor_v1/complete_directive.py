"""Audit remaining directive conditions without changing prior labels/results.

Legacy visibility is a QA trigger, not authority to overwrite a human label.
No models, optimizer or inference are created. Teacher scoring waits for QA.
"""
import copy
from collections import Counter
from pathlib import Path
import hashlib
from . import common as C
from .review_saved_keypoints import REVIEW

QA=C.RAW/'metadata_conflict_qa'

def put(path,obj):
    if path.exists():assert C.read(path)==obj, f'Existing audit differs: {path}'
    else:C.save_new(path,obj)

def main():
    labels=C.read(REVIEW/'FIRST_PASS_SNAPSHOT.json');selection=C.read(REVIEW/'ANCHOR_SELECTION.json')
    first=C.read(REVIEW/'FIRST_PASS_LOCK.json')
    assert C.sha(REVIEW/'LABELS.json')==first['labels_sha256']
    assert labels==C.read(REVIEW/'LABELS.json')
    records={r['frame_id']:r for r in C.read(C.OUT/'keypoints_first/WORKSPACE.json')['frames']}
    conflicts=[];categories=Counter();bindings=[]
    for fi,ci in labels['review_queue']:
        frame=labels['frames'][fi];r=records[frame['frame_id']];b=r['source_annotation']
        assert C.sha(C.ROOT/b['path'])==b['sha256']
        old=C.read(C.ROOT/b['path'])['objects'][0]['keypoint_annotations'][ci]
        new=frame['corners'][ci];reason=str(old.get('reason','')).lower()
        categories[(new['status'],str(old.get('source')),int(old.get('visibility',-1)),reason)]+=1
        conflict=(new['status']=='DIRECT_VISIBLE' and (old.get('visibility') in (0,1) or old.get('in_frame') is False)) or (
            new['status'] in ('SELF_OCCLUDED','EXTERNAL_OCCLUDED','OUT_OF_FRAME') and old.get('visibility')==2 and reason=='visible')
        if conflict:
            conflicts.append(dict(frame_index=fi,corner_id=ci,frame_id=frame['frame_id'],
                new_status=new['status'],first_pass_click=new.get('input_xy'),
                legacy_visibility=old.get('visibility'),legacy_reason=reason,legacy_source=old.get('source'),
                cause='HUMAN_STATUS_VS_EXISTING_VISIBILITY_METADATA',
                old_reference_may_be_wrong=True,model_errors_used=False))
        bindings.append(b)
    assert len(conflicts)==2
    teacher_lock=C.ROOT/'_docs/experiments/pallet_visible_refine_hidden_pnp_v1/PREDICTION_LOCK.json'
    teacher=C.read(teacher_lock)['predictions'];assert C.sha(C.ROOT/teacher['path'])==teacher['sha256']
    sources=dict(first_pass=first,source_annotations=list({b['path']:b for b in bindings}.values()),
        teacher=teacher,teacher_arm='TYPE_REPLAY_PIPELINE',teacher_scoring='WAIT_FOR_METADATA_QA',
        original_results_sha256=C.sha(C.DOC/'VERIFIED_RESULTS.json'),selection_sha256=C.sha(REVIEW/'ANCHOR_SELECTION.json'))
    put(QA/'INPUT_LOCK.json',sources)
    put(QA/'QA_QUEUE_PRIVATE.json',dict(points=conflicts,model_outputs_used_for_queue=False,
        previously_omitted_criterion='existing corner status/identity conflict',legacy_coordinates_shown=False))
    put(QA/'ANCHOR_SELECTION.json',selection)
    template=copy.deepcopy(labels);template['review_queue']=[[r['frame_index'],r['corner_id']] for r in conflicts]
    template['protocol']='METADATA_CONFLICT_SECOND_PASS_QA_PNP_ASSISTED_FIRST_PASS'
    for fi,ci in template['review_queue']:template['frames'][fi]['corners'][ci]['qa_decision']=None
    put(QA/'LABEL_TEMPLATE.json',template)
    summary=dict(status='WAITING_FOR_HUMAN_QA',qa_points=2,qa_images=2,new_images=0,
        new_keypoints_required=0,current66_results='PROVISIONAL_PENDING_METADATA_QA',
        previous_zero_QA_claim='CORRECTED: coordinate distance gate was complete; visibility conflict gate omitted',
        source_unknown_not_GT_authority=True,teacher_cache_available=True,teacher_supplement_pending=True,
        task_training=0,new_inference=0,original_labels_or_results_overwritten=False,
        protocol='Keep user-requested existing annotation.py+PnP then status workflow; cannot retrospectively call it blind',
        full144_statuses=False,unreviewed_points_excluded=True,
        source_status_combinations=[dict(new_status=k[0],old_source=k[1],old_visibility=k[2],old_reason=k[3],count=v) for k,v in categories.items()])
    put(C.DOC/'DIRECTIVE_COMPLETION_AUDIT.json',summary)
    print(summary,flush=True)

if __name__=='__main__':main()

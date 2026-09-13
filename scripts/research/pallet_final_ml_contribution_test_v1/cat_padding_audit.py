"""Read-only selected-instance/coordinate diagnosis, never a new selector."""
import cv2
import numpy as np
from common import *
from task_risk_audit import CAT

def run():
    record=next(r for r in read(TASK_DOC/'SPLIT_BINDING.json')['evaluation'] if r['frame_id']==CAT)
    image=ROOT/record['image_path'];h,w=cv2.imread(str(image)).shape[:2]
    decomposition=read(A/'CAT_FRAME_DECOMPOSITION.json');gt=np.array(decomposition['reference_keypoints'])
    output={};perm=[2,3,0,1,6,7,4,5,8]
    for seed in (1,2,3):
        name=f'proposed_seed{seed}';path=TASK/'evaluation'/name/'PREDICTIONS.json';rows=read(path)['frames'][record['image_path']]
        top=max(range(len(rows)),key=lambda j:rows[j]['score']);selected=rows[top];points=np.array(selected['keypoints_xy'])
        box=np.array(selected['box_xyxy']);entirely_right=bool(box[0]>=w)
        output[name]=dict(selected_index=top,candidate_count=len(rows),selected_box=box.tolist(),selected_score=selected['score'],
            selected_box_entirely_right_of_original=entirely_right,all9_points_right_of_original=bool((points[:,0]>w).all()),
            points_at_original_width_plus100=int((points[:,0]==w+100).sum()),
            C2_reindex_diagnostic_mean9_px=float(np.linalg.norm(points[perm]-gt,axis=-1).mean()),
            identity_mean9_px=float(np.linalg.norm(points-gt,axis=-1).mean()),
            candidates=[dict(index=i,score=c['score'],box_xyxy=c['box_xyxy']) for i,c in enumerate(rows)],
            source_sha256=sha(path))
        assert entirely_right
    write(A/'CAT_PADDING_INSTANCE_AUDIT.json',dict(frame_id=CAT,original_image_shape=[h,w],image_sha256=sha(image),
        source_decomposition_sha256=sha(A/'CAT_FRAME_DECOMPOSITION.json'),runs=output,
        measured='All proposed highest-score boxes and all9 predicted points lie to the right of the original640px image; multiple points equal740px, the original width plus the existing100px reflected border.',
        interpretation='Selected-instance/padding-region detection failure precedes PnP. Canonical C2 reindexing and width/depth hypothesis changes do not rescue it.',
        inference_note='Calling this a reflected-padding detection is an inference from exact coordinate range and the existing reflected-border recipe, not a new forward pass.',
        no_selector_changed=True,no_candidate_substitution=True,no_new_training=True,official_verdict_unchanged=True))
    print('CAT_PADDING_DIAGNOSIS_CONFIRMED',flush=True)

if __name__=='__main__':run()

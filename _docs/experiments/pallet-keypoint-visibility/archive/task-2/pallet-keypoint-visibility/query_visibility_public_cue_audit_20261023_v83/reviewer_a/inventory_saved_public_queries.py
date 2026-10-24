"""Inventory selected saved source proposals only; no RGB/model/GT matching."""
from pathlib import Path
import hashlib,json
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent.parent
S=P/'query_visibility_cue_occlusion_20261023_v72/public_pallet_initializer'
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
policy=json.loads((R/'REVIEW_POLICY_BEFORE_RGB.json').read_text())
result=json.loads((S/'INFERENCE_RESULT.json').read_text());assert result['completed'] and result['source_only'] and result['current223_access']==0
entry=json.loads((S/'RUN_ENTRYLOCK.json').read_text())
for b in entry['bindings']:
    if Path(b['path']).name in ('INPUTS_RGB_ONLY.jsonl','infer_frozen_initializer.py','PROTOCOL.json'):
        assert bind(Path(b['path']))==b
proposal_binding=next(b for b in result['outputs'] if Path(b['path']).name=='FROZEN_PROPOSALS.jsonl')
assert bind(Path(proposal_binding['path']))==proposal_binding
props=[json.loads(x) for x in Path(proposal_binding['path']).read_text().splitlines()]
inputs=[json.loads(x) for x in (S/'INPUTS_RGB_ONLY.jsonl').read_text().splitlines()]
assert len(inputs)==len(props)==48
by_sha={r['image_sha256']:(r,i) for r,i in zip(props,inputs)}
rows=[]
for cue in policy['fixed_records']:
    row,inp=by_sha[cue['image_sha256']]
    assert row['input_id']==inp['id'] and inp['image']==cue['image']
    cs=row['candidates'];j=row['selected_index'];selected=None
    if cs:
        assert j==int(np.argmax([c['score'] for c in cs]))
        selected=cs[j];assert selected['candidate_index']==j
        q=np.asarray(selected['keypoints_xy'],np.float64);assert q.shape==(9,2)
        h,w=cue['raw_hw'];finite=np.isfinite(q).all(1)
        valid=finite&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
        box=np.asarray(selected['box_xyxy']);r=np.asarray(cue['original_xy'])
        in_box=bool(box[0]<=r[0]<box[2] and box[1]<=r[1]<box[3])
    else:
        assert j is None and row['no_detection'];q=None;finite=valid=None;in_box=None
    rows.append(dict(cue_id=cue['id'],image_sha256=cue['image_sha256'],input_id=row['input_id'],candidate_count=len(cs),selected_index=j,no_detection=row['no_detection'],saved_selection_rule='first numpy.argmax candidate score; unchanged from V72, no reference-based replacement',selected_candidate=selected,selected_raw9_xy=None if q is None else q.tolist(),selected_logical_first8_xy=None if q is None else q[:8].tolist(),logical_first8_finite_N=0 if q is None else int(finite[:8].sum()),logical_first8_continuous_inframe_N=0 if q is None else int(valid[:8].sum()),reference_inside_selected_detector_box_GT_only=in_box,native8_semantic_association=[None]*8,reference_to_selected_slot_association='UNKNOWN',stored_roi_side_in_V72_proposal=False,crop_availability_for_reference='NOT_EVALUATED_no_storedL_here',raw_coordinate_convention='V72 frozen wrapper original native coordinates after exactly100px source border removed; no coordinates modified'))
out=dict(completed=True,scope='source saved q availability only; no model/features/current/GT-based selection',records=rows,all48_run_summary=dict(frames=48,no_detection_frames=result['no_detection_frames'],candidate_count=result['candidate_count'],source_family=result['source_family'],split=result['split']),fixed3_summary=dict(selected_raw9_available_frames=sum(not r['no_detection'] for r in rows),no_detection_frames=sum(r['no_detection'] for r in rows),first8_finite_N=sum(r['logical_first8_finite_N'] for r in rows),first8_continuous_inframe_N=sum(r['logical_first8_continuous_inframe_N'] for r in rows),certified_native8_associations=0),bindings=dict(policy=bind(R/'REVIEW_POLICY_BEFORE_RGB.json'),result=bind(S/'INFERENCE_RESULT.json'),entry=bind(S/'RUN_ENTRYLOCK.json'),proposal=proposal_binding,input_manifest=bind(S/'INPUTS_RGB_ONLY.jsonl'),producer_protocol=bind(S/'PROTOCOL.json'),code=bind(Path(__file__))),new_RGB_model_feature_current_forward_fit_PnP=0)
with (R/'SAVED_INITIALIZER_QUERY_INVENTORY.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(bind(R/'SAVED_INITIALIZER_QUERY_INVENTORY.json')))

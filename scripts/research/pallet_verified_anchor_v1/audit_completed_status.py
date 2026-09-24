"""Freeze human status review, then build legacy-only blind QA queue.

Never reads model predictions. The revised protocol is PnP-assisted, visible-only;
unreviewed/nonmanual points remain excluded, not fabricated as human statuses.
"""
from collections import Counter
import numpy as np
from . import common as C
from .review_saved_keypoints import REVIEW, prepare

def same_or_new(path, obj):
    if path.exists():assert C.read(path)==obj, f'Frozen artifact changed: {path}'
    else:C.save_new(path,obj)

def main():
    prepare()  # Recheck exact saved-keypoint bindings and original GT preservation.
    source=REVIEW/'LABELS.json';labels=C.read(source);selection=C.read(REVIEW/'ANCHOR_SELECTION.json')
    template=C.read(REVIEW/'LABEL_TEMPLATE.json');queue=template['review_queue']
    assert labels['review_queue']==queue and labels['selection_sha256']==C.sha(REVIEW/'ANCHOR_SELECTION.json')
    assert len(labels['frames'])==len(selection['frames'])==18
    points=[];statuses=Counter();severity=Counter();ids=Counter();recordings=set()
    for fi,ci in queue:
        f=labels['frames'][fi];s=selection['frames'][fi];c=f['corners'][ci];original=template['frames'][fi]['corners'][ci]
        assert f['frame_id']==s['frame_id'] and f['image_sha256']==s['image']['sha256']
        assert c['id']==ci and c['coordinate_source']=='manual_click'
        assert c['input_xy']==original['input_xy']
        assert C.valid_corner(c,s['hw']), f'Unfinished status {fi}:{ci}'
        if c['status'] in ('DIRECT_VISIBLE','VIRTUAL_INFERABLE'):assert c['xy']==original['xy']
        statuses[c['status']]+=1
        if c['status']=='DIRECT_VISIBLE':
            points.append(dict(frame_id=f['frame_id'],frame_index=fi,corner_id=ci,xy=c['xy'],
                severity=s['severity'],recording=s['recording']))
            severity[s['severity']]+=1;ids[ci]+=1;recordings.add(s['recording'])
    coverage=dict(direct_visible=len(points),reviewed_statuses=len(queue),statuses=dict(statuses),
        severity={s:severity[s] for s in C.SEVERITIES},corner_counts=[ids[i] for i in range(8)],
        recordings=len(recordings),images_with_direct_visible=len({p['frame_id'] for p in points}),
        original_selected_images=18,extra_images=0,full_144_status_annotation=False,
        nonmanual_unreviewed_excluded=True,protocol='PNP_ASSISTED_KEYPOINTS_FIRST_THEN_STATUS',
        minimum_visible_coverage_pass=len(points)>=60 and min(severity[s] for s in C.SEVERITIES)>=12
            and sum(ids[i]>=3 for i in range(8))>=5 and len(recordings)>=3)
    lock=dict(labels_sha256=C.sha(source),history_sha256=C.sha(REVIEW/'HUMAN_HISTORY.jsonl'),
              selection_sha256=C.sha(REVIEW/'ANCHOR_SELECTION.json'),input_lock_sha256=C.sha(REVIEW/'INPUT_LOCK.json'),
              models_read=False,legacy_coordinates_read_before_lock=False,protocol=coverage['protocol'])
    # Exact snapshot before accessing legacy reference; don't replace a locked first pass.
    same_or_new(REVIEW/'FIRST_PASS_SNAPSHOT.json',labels)
    same_or_new(REVIEW/'FIRST_PASS_LOCK.json',lock)
    same_or_new(C.DOC/'STATUS_COVERAGE.json',coverage)
    if not coverage['minimum_visible_coverage_pass']:
        print('COVERAGE_ROUTING_REQUIRED',coverage);return
    legacy_path=C.ROOT/'data/pallet/results/pallet_replay_clean19_v1/TRUTH_FOR_DISPLAY_ONLY.json'
    legacy=C.read(legacy_path);qa=[];differences=[]
    for p in points:
        row=legacy[p['frame_id']];ci=p['corner_id'];q=np.asarray(row['gt'][ci],dtype=float)
        valid=bool(row['valid'][ci]) and q.shape==(2,) and np.isfinite(q).all()
        distance=float(np.linalg.norm(q-np.array(p['xy']))) if valid else None
        reasons=[]
        if distance is not None and distance>20:reasons.append('FIRST_PASS_VS_LEGACY_GT20')
        note=labels['frames'][p['frame_index']]['corners'][ci].get('note')
        if note and any(v in note.lower() for v in ('uncertain','애매','불확실')):reasons.append('HUMAN_NOTE_UNCERTAIN')
        differences.append(dict(**p,legacy_xy=q.tolist() if valid else None,distance_px=distance))
        if reasons:qa.append(dict(**p,reasons=reasons))
    # QA UI gets no legacy coordinate or model output, only the user's first click.
    same_or_new(REVIEW/'QA_QUEUE.json',dict(points=qa,first_pass=lock,legacy_sha256=C.sha(legacy_path),
        queue_rule='new DIRECT_VISIBLE vs legacy >20px OR human uncertainty note; no model outputs',
        legacy_status_conflicts='Not inferred from old unknown-provenance visibility',models_read=False))
    same_or_new(REVIEW/'LEGACY_DISAGREEMENT_PRIVATE.json',differences)
    summary=dict(qa_points=len(qa),qa_images=len({r['frame_id'] for r in qa}),
        primary_evaluation='PENDING_BLIND_SECOND_PASS_QA',model_predictions_read=False,
        extra_labeling_needed_for_minimum_coverage=False,legacy_not_automatically_declared_wrong=True)
    same_or_new(C.DOC/'STATUS_QA_SUMMARY.json',summary)
    print(coverage,flush=True);print(summary,flush=True)

if __name__=='__main__':main()

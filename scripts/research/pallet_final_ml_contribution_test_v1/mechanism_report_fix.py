"""Reporting-only completion: missing domain and empty matched subgroups."""
import csv
import numpy as np
from common import *
from statistics_and_mechanism import load,distribution

def run():
    stores,poses,preds,targets,metadata,summaries=load()
    source=old('paper_evaluation').OLDCSV
    with source.open() as f:domain={r['frame_id']:r['domain'] for r in csv.DictReader(f) if r['kind']=='POSITIVE'}
    records=[]
    for frame,target in targets.items():
        gt=target.keypoints_xy;valid=target.keypoint_supervision_mask;item=metadata[frame]
        errors=stores['R0'][frame]['errors'];average=float(errors.mean()) if len(errors) else None
        difficulty='UNAVAILABLE' if average is None else 'easy' if average<=5 else 'mid' if average<=10 else 'hard'
        for role,(a,b) in enumerate(old('model').SIDE_EDGES):
            if not valid[a] or not valid[b]:continue
            vector=gt[b]-gt[a];length=np.linalg.norm(vector)
            if length<2:continue
            tangent=vector/length;normal=np.array([-tangent[1],tangent[0]])
            values={}
            for name,prediction in preds.items():
                if prediction[frame] is None:continue
                error=prediction[frame][[a,b]]-gt[[a,b]]
                values[name]=dict(normal=float(np.abs(error@normal).mean()),tangent=float(np.abs(error@tangent).mean()))
            records.append(dict(frame=frame,role=str(role),session=item['session_id'],material=item['object_type'],
                day_night=item.get('domain',domain[frame] or 'UNTAGGED'),difficulty=difficulty,
                visibility='ANNOTATION_2' if target.visibility[a]==target.visibility[b]==2 else 'ANNOTATION_1_OR_MIXED',
                values=values,endpoint_indices=[a,b]))
    groups={'overall':{'all':records}}
    for field in ('role','session','material','day_night','difficulty','visibility'):
        groups[field]={v:[r for r in records if r[field]==v] for v in sorted({r[field] for r in records})}
    report={};empty=[]
    for field,grouping in groups.items():
        report[field]={}
        for group,rows in grouping.items():
            methods={m:{a:distribution([r['values'][m][a] for r in rows if m in r['values']]) for a in ('normal','tangent')} for m in preds}
            estimable=all(methods[m]['normal']['n']>0 for m in preds)
            delta={a:{s:float(np.mean([methods[f'L{i}'][a][s]-methods[f'P{i}'][a][s] for i in (1,2,3)])) if estimable else None
                for s in ('median','mean','p90')} for a in ('normal','tangent')}
            report[field][group]=dict(edge_frames=len(rows),methods=methods,L_minus_P=delta,
                status='ESTIMATED' if estimable else 'NOT_ESTIMABLE_NO_MATCHED_PREDICTION')
            if not estimable:empty.append(dict(field=field,group=group,edge_frames=len(rows)))
    write(BRAW/'MECHANISM_PER_EDGE.json',dict(records=records,GT_assisted_diagnostic_only=True))
    write(B/'MECHANISM_NORMAL_TANGENT.json',dict(complete=True,groups=report,GT_assisted_diagnostic_only=True,
        endpoints_averaged_within_each_edge=True,shared_endpoint_repeated_across_roles=True,
        visibility_scope='Annotation visibility codes only; physical observed/occluded provenance not independently validated, so no physical occlusion mechanism claim.',
        subgroup_selection=False,primary_gate_not_overridden=True,rows_sha256=sha(BRAW/'MECHANISM_PER_EDGE.json')))
    write(B/'MECHANISM_COMPLETION_CORRECTION.json',dict(complete=True,code_sha256=sha(Path(__file__)),
        canonical_domain_source_sha256=sha(source),original_locked_source_unchanged=True,
        corrections=['Missing optional manifest domain uses existing canonical CSV value','Empty matched-prediction subgroups retain n0, null statistics and NOT_ESTIMABLE; never substitute0 error'],
        empty_groups=empty,predictions_training_selection_primary_statistics_unchanged=True))
    print('MECHANISM_COMPLETED',report['overall']['all']['L_minus_P'],'EMPTY',empty,flush=True)

if __name__=='__main__':run()

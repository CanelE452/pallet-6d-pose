"""Fixed24 structural review artifacts, explicitly automatic and uncertified.

Selection uses recording cluster and sorted identity only, before S3/S4 scoring.
The gallery presents original pixels, model predictions and actual measurements
separately. It does not create independent physical-owner or human labels.
"""
from __future__ import annotations
import argparse,csv,html,json
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import numpy as np
from . import image_batch as I,material_assets as A


def select(inputs,output):
    frames=list(I.rows(inputs));groups=defaultdict(list)
    for frame in frames:
        if frame['partition']!='CAL':groups[frame['cluster_id']].append(frame)
    if len(groups)!=6:raise ValueError('Original PRIMARY six clusters expected')
    selected=[];used=set()
    for cluster,records in sorted(groups.items()):
        for frame in sorted(records,key=lambda r:r['id'])[:4]:
            selected.append(frame);used.add(frame['id'])
    while len(selected)<24:
        for cluster,records in sorted(groups.items()):
            remaining=[r for r in sorted(records,key=lambda r:r['id'])if r['id']not in used]
            if remaining and len(selected)<24:
                selected.append(remaining[0]);used.add(remaining[0]['id'])
    result=dict(schema='S3_FIXED24_AUTOMATIC_STRUCTURE_REVIEW_SELECTION_V1',
        selection='up to4 lexicographically sorted PRIMARY IDs/cluster, then one extra/cluster in sorted cluster order until24',
        reason_not_exact6x4='REC_007 contains only1 original PRIMARY frame; retain it and fill deterministic extras',
        input_sha256=A.sha(inputs),ordered_ids=[r['id']for r in selected],
        cluster_counts=dict(Counter(r['cluster_id']for r in selected)),
        used_accuracy_or_GT=False,independent_physical_labels_available=False,human_confirmation_count=0)
    I.write(output,result);return result


def _factor_overlay(image,packet,archive):
    result=image.copy()
    # Predicted proxy visibility is grey, never a reference projection.
    for record in archive:
        q=np.asarray(record['predicted_projection_xy'],float)
        visible=np.asarray(record['predicted_visibility'],bool)
        for a,b,valid_a,valid_b in zip(q[:-1],q[1:],visible[:-1],visible[1:]):
            if valid_a and valid_b and np.isfinite([a,b]).all():
                cv2.line(result,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),(135,135,135),1,cv2.LINE_AA)
    for factor in packet['factors']:
        xy=np.asarray(factor['xy'],float)
        if not np.isfinite(xy).all():continue
        point=tuple(np.rint(xy).astype(int));color=(220,190,0)if factor['role']=='B'else(0,150,255)
        if factor['observation_dimension']==2:
            covariance=np.asarray(factor['effective_covariance_px2'])
            values,vectors=np.linalg.eigh(covariance);order=np.argsort(values)[::-1]
            values,vectors=values[order],vectors[:,order]
            angle=float(np.rad2deg(np.arctan2(vectors[1,0],vectors[0,0])))
            axes=tuple(np.clip(np.rint(2*np.sqrt(np.maximum(values,0))),1,30).astype(int))
            cv2.ellipse(result,point,axes,angle,0,360,color,1,cv2.LINE_AA)
            cv2.circle(result,point,3,color,-1,cv2.LINE_AA)
        else:
            normal=np.asarray(factor['normal_unit_xy']);sigma=np.sqrt(factor['effective_variance_px2'])
            end=tuple(np.rint(xy+normal*min(2*sigma,25)).astype(int))
            cv2.arrowedLine(result,point,end,color,1,cv2.LINE_AA,tipLength=.25)
            cv2.circle(result,point,2,color,-1,cv2.LINE_AA)
        for raw in factor.get('raw_image_support_xy',[]):
            cv2.circle(result,tuple(np.rint(raw).astype(int)),1,color,-1)
    return result


def render(inputs,production,selection,output):
    frames=list(I.rows(inputs));index={r['id']:(i,r)for i,r in enumerate(frames)}
    selected=I.read(selection)
    if selected['input_sha256']!=A.sha(inputs):raise ValueError('Review selection input differs')
    out=Path(output);out.mkdir(parents=True,exist_ok=True);cards=[];records=[];bindings=[]
    for order,ident in enumerate(selected['ordered_ids']):
        i,frame=index[ident];path=Path(production)/'frames/real'/f'{i:04d}'/'FRAME.jsonl.gz'
        result=next(I.rows(path));image=cv2.imread(frame['image'])
        if A.sha(frame['image'])!=frame['image_sha256']:raise ValueError('Original RGB differs')
        panels=[]
        original=image.copy()
        for k,point in enumerate(np.asarray(frame['initial_points'])[:8]):
            if np.isfinite(point).all()and min(point)>=0:
                xy=tuple(np.rint(point).astype(int));cv2.circle(original,xy,3,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(original,str(k),(xy[0]+4,xy[1]),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1,cv2.LINE_AA)
        panels.append((original,'Original RGB + original N3/SubPix queries'))
        observations=result['material_observations']
        if observations is None:raise ValueError('Selected structure frame has no valid initialization')
        for arm in ('COMMON','BI'):
            packet=observations['policy_packets'][arm]
            overlay=_factor_overlay(image,packet,observations['shared_mesh_support_archive'])
            panels.append((overlay,arm+' | grey=proxy prediction, cyan=B, orange=I'))
            for factor in packet['factors']:
                records.append(dict(id=ident,cluster_id=frame['cluster_id'],arm=arm,
                    support_id=factor['support_id'],predicted_role=factor['role'],
                    evidence_dimension=factor['observation_dimension'],raw_xy=factor['xy'],
                    object_xyz_m=factor['object_xyz'],physical_source_ids=factor['physical_source_ids'],
                    object_other_occluder_background='UNKNOWN_WITHOUT_INDEPENDENT_PHYSICAL_LABEL',
                    actual_material_id='UNVERIFIED_PROXY_SUPPORT_ID_ONLY',
                    actual_point_or_virtual='AUTOMATIC_REGISTERED_PROXY_SUPPORT; ORIGINAL_CUBOID_QUERY_SHOWN_SEPARATELY',
                    evidence_source='AUTOMATIC_RENDER_AND_RGB_MEASUREMENT_MODEL_PROPOSAL',
                    independent_human_confirmation=False,physical_owner_oracle_available=False))
            shared_by_id={record['support_id']:record for record in observations['shared_mesh_support_archive']}
            for absent in packet['no_direct_support']:
                predicted=shared_by_id[absent['support_id']]
                records.append(dict(id=ident,cluster_id=frame['cluster_id'],arm=arm,
                    support_id=absent['support_id'],predicted_role=predicted['role'],
                    evidence_dimension=0,raw_xy=None,object_segment_xyz_m=predicted['object_segment_xyz'],
                    physical_source_ids=[],no_direct_support_reason=absent['reason'],
                    object_other_occluder_background='UNKNOWN_WITHOUT_INDEPENDENT_PHYSICAL_LABEL',
                    actual_material_id='UNVERIFIED_PROXY_SUPPORT_ID_ONLY',
                    actual_point_or_virtual='AUTOMATIC_PROXY_SEGMENT; NO_DIRECT_MEASUREMENT_ADOPTED',
                    evidence_source='AUTOMATIC_RENDER_AND_IMAGE_ALGORITHM; NO_DIRECT_SUPPORT_IS_NOT_PHYSICAL_ABSENCE',
                    independent_human_confirmation=False,physical_owner_oracle_available=False))
        padded=[]
        for panel,title in panels:
            bar=np.full((48,panel.shape[1],3),30,np.uint8)
            cv2.putText(bar,title,(8,17),cv2.FONT_HERSHEY_SIMPLEX,.43,(235,235,235),1,cv2.LINE_AA)
            cv2.putText(bar,'Automatic proxy support; physical owner UNKNOWN',(8,36),cv2.FONT_HERSHEY_SIMPLEX,.42,(190,190,190),1,cv2.LINE_AA)
            padded.append(np.vstack([bar,panel]))
        filename=f'{order:02d}.jpg';cv2.imwrite(str(out/filename),np.hstack(padded),[cv2.IMWRITE_JPEG_QUALITY,88])
        cards.append(f'<figure><figcaption>{html.escape(ident)} — {html.escape(frame["cluster_id"])}</figcaption><img src="{filename}" loading="lazy"></figure>')
        bindings.append(dict(id=ident,production_frame_sha256=A.sha(path),image_sha256=frame['image_sha256'],panel_sha256=A.sha(out/filename)))
    record_path=out/'STRUCTURAL_REVIEW_RECORDS.jsonl.gz'
    if record_path.exists():
        if I.canonical(list(I.rows(record_path)))!=I.canonical(records):raise ValueError('Completed review records differ')
    else:I.save(record_path,records)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fixed24 automatic proxy structure review</title><style>body{font:15px sans-serif;background:#111;color:#eee}img{width:100%;height:auto}figure{margin:24px 0}figcaption{padding:8px}</style><h1>Fixed24 automatic proxy structure review</h1><p>No human-confirmed or independent physical-owner labels. Display padding is excluded from every production coordinate.</p>'+''.join(cards))
    receipt=dict(complete=True,frames=24,source=selected,bindings=bindings,measurement_records=len(records),
        evidence_dimension_counts=dict(Counter(str(record['evidence_dimension'])for record in records)),
        human_confirmation_count=0,independent_physical_owner_labels=0,
        physical_owner_accuracy='NOT_APPLICABLE_WITH_REASON_NO_INDEPENDENT_LABELS',
        all_production_support_records_retained=True,new_pose_updates=0,new_model_forwards=0,current_real_GT_reads=0)
    I.write(out/'REVIEW_RECEIPT.json',receipt);return dict(complete=True,frames=24,records=len(records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','render'])
    for name in ('inputs','production','selection','output'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();result=select(a.inputs,a.output)if a.stage=='select'else render(a.inputs,a.production,a.selection,a.output)
    print(I.canonical(result),flush=True)

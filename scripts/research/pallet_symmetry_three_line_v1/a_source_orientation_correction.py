"""Post-evaluation correctness repair: fixed source Y-up versus PnP Y-down basis.

No per-frame GT phase, weight, inference, or primary 2D metric change. Originals retained.
"""
import csv
import numpy as np
import env as E
from preflight import rotations

SOURCE_BASIS=np.diag([1.,-1.,-1.])  # proper Rx(pi), NOT a reflection

def source_rotation(R):
    return np.asarray(R,float)@SOURCE_BASIS

def angle(R,gt,order):
    return min(float(np.degrees(np.arccos(np.clip((np.trace(R.T@gt@q)-1)/2,-1,1)))) for q in rotations(order))

def main():
    geometry=dict(np.load(E.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'))
    index={str(v):i for i,v in enumerate(geometry['stems'])}
    side={r['id']:r for r in E.read(E.RAW/'A/rect_target_sidecar.json')['records']}
    old_summary=E.DOC/'A/history/POSE_SECONDARY_BEFORE_SOURCE_FRAME_CORRECTION.json'
    if not old_summary.exists():E.write(old_summary,E.read(E.DOC/'A/POSE_SECONDARY.json'))
    old_metrics=E.RAW/'A/history/POSE_FULL_METRICS_before_source_frame.json'
    if not old_metrics.exists():E.write(old_metrics,E.read(E.RAW/'A/POSE_FULL_METRICS.json'))
    metrics=E.read(old_metrics);summary=E.read(old_summary);changed=0;counts={}
    assert len(metrics['RECT_SYNTH_VAL'])==7
    assert all(len(rows)==4020 for rows in metrics['RECT_SYNTH_VAL'].values())
    for arm,rows in metrics['RECT_SYNTH_VAL'].items():
        p=E.read(E.RAW/f'A/pose_predictions/RECT_SYNTH_VAL/{arm}.json')['records'];corrected={}
        for r in rows:
            if not r['available']:continue
            fid=r['id'];R=source_rotation(p[fid]['R_physical'])
            corrected[fid]=dict(R_physical_source=R.tolist(),GT_input=False)
            r['equivalent_rotation_deg']=angle(R,geometry['R'][index[fid]],side[fid]['group_order']);changed+=1
        s=summary['summary']['RECT_SYNTH_VAL'][arm];valid=[r for r in rows if r['available']]
        s['equivalent_rotation_deg']=dict(median=float(np.median([r['equivalent_rotation_deg'] for r in valid])),P90=float(np.quantile([r['equivalent_rotation_deg'] for r in valid],.9)))
        s['pose_10cm_10deg_full_reference_rate']=sum(r['available'] and r['centroid_translation_cm']<=10 and r['equivalent_rotation_deg']<=10 for r in rows)/len(rows)
        E.write(E.RAW/f'A/pose_source_frame_corrected/{arm}.json',dict(records=corrected,prediction_basis_only=True,original_pose_predictions_preserved=True))
    counts['A_source_unique_frames']=7*4020;counts['A_available_orientations_corrected']=changed
    # Prove this repair changes no availability, identity, non-source value,
    # translation or body-IoU entry; not merely a declaration in the report.
    checked=0
    for split,arms in E.read(old_metrics).items():
        for arm,rows in arms.items():
            assert len(rows)==len(metrics[split][arm])
            for before,after in zip(rows,metrics[split][arm]):
                before=dict(before);after=dict(after)
                if split=='RECT_SYNTH_VAL':
                    before.pop('equivalent_rotation_deg',None);after.pop('equivalent_rotation_deg',None)
                assert before==after,(split,arm,before['id'])
                checked+=1
    counts['A_nonrotation_and_DEV_invariance_verified_frames']=checked
    summary['source_orientation_correction']='SOURCE_ORIENTATION_CORRECTION.json; fixed right multiplication Rx(pi), no GT-dependent orbit selection'
    E.write(E.DOC/'A/POSE_SECONDARY.json',summary);E.write(E.RAW/'A/POSE_FULL_METRICS.json',metrics)
    flat=[]
    for split,arms in metrics.items():
        for arm,rows in arms.items():
            for r in rows:flat.append(dict(split=split,arm=arm,id=r['id'],available=r['available'],centroid_translation_cm=r.get('centroid_translation_cm'),equivalent_rotation_deg=r.get('equivalent_rotation_deg'),body_extent_IoU=r.get('body_extent_IoU')))
    with (E.DOC/'A/pose_per_frame_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)
    # Same source-frame issue exists in historical B's synthetic rotation-only auxiliaries.
    # Preserve B originals verbatim; add a correction sidecar, leaving 2D/runtime/DEV untouched.
    bmetrics=E.read(E.RAW/'B/POSE_FULL_METRICS.json')['synth_val'];bs={};br={}
    assert len(bmetrics)==15 and all(len(rows)==512 for rows in bmetrics.values())
    for arm,rows in bmetrics.items():
        p=E.read(E.RAW/f'B/pose_predictions/synth_val/{arm}.json')['records']
        for r in rows:
            if r['available']:
                fid=r['id'];r['equivalent_rotation_deg']=angle(source_rotation(p[fid]['R_physical']),geometry['R'][index[fid]],side[fid]['group_order'])
        valid=[r for r in rows if r['available']];br[arm]=rows
        bs[arm]=dict(frames=len(rows),available=len(valid),equivalent_rotation_deg=dict(median=float(np.median([r['equivalent_rotation_deg'] for r in valid])),P90=float(np.quantile([r['equivalent_rotation_deg'] for r in valid],.9))))
    E.write(E.RAW/'B/POSE_SOURCE_ROTATION_CORRECTED.json',br)
    E.write(E.DOC/'B/POSE_SOURCE_ROTATION_CORRECTION.json',dict(status='CORRECTED_SECONDARY_ONLY',original=E.bound(E.DOC/'B/POSE_SECONDARY.json'),
      replaced_interpretation='Only synth_val equivalent_rotation_deg in the old pose auxiliary is invalid without this frame correction. Original numerical files preserved.',
      corrected_summary=bs,primary_2D_unchanged=True,DEV_pose_unchanged=True,centroid_and_IoU_unchanged=True,runtime_unchanged=True,new_inference=0,new_training=0))
    counts['B_source_unique_frames']=15*512
    E.write(E.DOC/'A/SOURCE_ORIENTATION_CORRECTION.json',dict(status='POST_EVALUATION_COORDINATE_CORRECTION',
      cause='PnP cuboid uses +Y down; renderer physical object convention uses +Y up. Historical b_pose.infer mapped width/depth but omitted the fixed proper source basis transform.',
      correction='R_source = R_previous * diag(1,-1,-1), for synthetic source only; determinant +1. Same fixed transform for every image/model, no GT phase selection.',
      evidence=[E.bound(E.ROOT/'scripts/research/pallet_translation_loss_v1/build_geometry_sidetable.py'),E.bound(E.ROOT/'scripts/research/pallet_point_line_v5/plpose_v5/geometry.py')],
      old_A_metrics=E.bound(old_metrics),old_A_summary=E.bound(old_summary),
      fixed_basis=SOURCE_BASIS.tolist(),**counts,neural_inference_reruns=0,training_reruns=0,
      centroid_body_IoU_and_primary_2D_unchanged=True,physical_C1_front_phase_still_not_recoverable=True,
      not_pre_registered_before_pose_results=True,not_a_model_improvement=True))
    print('SOURCE ORIENTATION CORRECTED',counts,flush=True)
if __name__=='__main__':main()

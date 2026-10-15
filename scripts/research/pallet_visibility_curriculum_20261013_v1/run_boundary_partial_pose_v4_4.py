"""CAL-first observation check and opt-in actual-case partial-boundary PnP."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse
from copy import deepcopy
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from . import common as C
from . import statewise_evaluation_v4 as V
from . import evaluation_interval_v4_2b as E
from . import statewise_uncertainty_calibration_v4 as UC
from . import boundary_partial_observations_v4_4 as O
from . import statewise_boundary_normal_solver_v4_4 as S
from . import statewise_uncertain_solver_v4_bias as OLD

ROOT=V.FIX/'boundary_recovery_v1'
OUT=ROOT/'partial_pose_v4_4'
TARGET='eval_pallet09:1778653661653195264'
MODEL=V.FIX/'calibration/final_adapter_CAL46/MODEL.json'


def put(path,value):
    if path.exists():raise FileExistsError('Preserve completed result: '+str(path))
    C.write(path,value)


def sources():
    return {str(Path(p).relative_to(C.ROOT)):C.sha(p) for p in
        (__file__,O.__file__,S.__file__,OLD.__file__,O.FITTER_SOURCE)}


def payload(row,model):
    rt=row['runtime'];g=rt['initial_geometry'];stem=row['id'].replace(':','__')
    image_path=C.ROOT/row['image'];assert C.sha(image_path)==row['image_sha256']
    image=cv2.imread(str(image_path));capture=ROOT/'image_evidence_v2'
    meta=C.read(capture/(stem+'.json'));mask_path=capture/(stem+'.npz')
    assert C.sha(mask_path)==meta['mask_file_sha256'] and meta['image_sha256']==row['image_sha256']
    mask=np.load(mask_path)['pallet_mask']
    correction_path=ROOT/'evaluation_v3'/f'{row["partition"]}_predictions'/(stem+'.json')
    correction=C.read(correction_path)
    evidence=O.observe_lower_boundary(image,mask,g,rt['states'],rt['state_observation']['records'],
        correction,model,rt['ot_domains']['nonself_prediction_scale_px'],image_sha256=row['image_sha256'],
        fitter_sha256=C.sha(O.FITTER_SOURCE),do_bootstrap=False)
    # Keep selected actual samples and the full unknown-split profile summaries.
    # Avoid storing the same branch observations once for every searched split.
    for fit in evidence['fits'].values():
        fit['split_profile']=[{k:p.get(k) for k in ('split_contour_index','split_contour_xy',
            'robust_Huber_cost','intersection_xy','intersection_sine','both_original_line_guards_pass',
            'finite_intersection','finite_observed_support_pass')} for p in fit['split_profile']]
    return evidence,correction,dict(actual_image_sha256=row['image_sha256'],
        mask_sha256=meta['mask_file_sha256'],correction_sha256=C.sha(correction_path))


def observe_cal():
    if OUT.exists():raise FileExistsError('Preserve previous experiment')
    rows,_,_=V._verify_prediction_seal(E.OUTPUT)
    rows=[r for r in rows if r['cohort']=='PRIMARY_NONSEVERE' and r['partition']=='CAL']
    assert len(rows)==46
    model=UC.load_uncertainty_model(MODEL);bindings=sources()
    protected=C.read(E.OUTPUT/'PROTOCOL.json')['protected_user_sha256']
    V._verify_sources(protected);OUT.mkdir(parents=True)
    protocol=dict(schema='CAL_FIRST_PARTIAL_BOUNDARY_OBSERVATION_V4_4',source_sha256=bindings,
        protected_user_sha256=protected,original_predictions_sha256=C.sha(E.OUTPUT/'PREDICTIONS.jsonl.gz'),
        CAL_model_sha256=C.sha(MODEL),CAL_model_refitted=False,observed_endpoint_recipe='frozen v4_3b image B endpoints',
        candidate_scope='B-fallback or U, shallow lower initial hull junction, two actual image B endpoints',
        split_search='all actual observed contour splits; no raw or painted junction anchor',
        partial_variance='2 * existing CAL fallback B effective variance at frozen nonS scale',
        variance_is_boundary_total_error_calibrated=False,states_BI_OT_S_Q_preserved=True,
        CAL_before_new_TEST_pose_experiment=True,TEST_is_reused_exploratory=True,
        boundary_factor_weight=1.,new_backbone_state_or_mask_forwards=0,new_pose_fits=0,GT_inputs=False)
    put(OUT/'PROTOCOL.json',protocol)
    values=[];started=time.monotonic();cv2.setNumThreads(1)
    for i,row in enumerate(rows):
        evidence,_,binding=payload(row,model)
        values.append(dict(id=row['id'],partition='CAL',evidence=evidence,bindings=binding))
        print(f'CAL {i+1}/46 accepted partial vertices={len(evidence["boundary_observations"])}',flush=True)
    C.save_rows(OUT/'CAL_BOUNDARY_PREDICTIONS.jsonl.gz',values)
    V._verify_sources(bindings);V._verify_sources(protected)
    put(OUT/'CAL_PREDICTION_SEAL.json',dict(predictions_sha256=C.sha(OUT/'CAL_BOUNDARY_PREDICTIONS.jsonl.gz'),
        protocol_sha256=C.sha(OUT/'PROTOCOL.json'),before_CAL_annotation_score=True,
        frames=46,accepted_partial_vertices=sum(len(v['evidence']['boundary_observations']) for v in values),
        elapsed_seconds=time.monotonic()-started,new_pose_fits=0,GT_inputs=False))


def score_cal():
    seal=C.read(OUT/'CAL_PREDICTION_SEAL.json')
    assert C.sha(OUT/'CAL_BOUNDARY_PREDICTIONS.jsonl.gz')==seal['predictions_sha256']
    label_path=UC.BASE/'OOF_pose_v1/ISOLATED_CAL_SCORING.json'
    labels=C.read(label_path)['rows'];original={r['id']:r for r in C.rows(E.OUTPUT/'PREDICTIONS.jsonl.gz')}
    results=[]
    for case in C.rows(OUT/'CAL_BOUNDARY_PREDICTIONS.jsonl.gz'):
        label=labels[case['id']];align=UC._reference_alignment(original[case['id']],label)
        for partial in case['evidence']['boundary_observations']:
            k=partial['native_corner'];mapped=label['metadata']['native_to_reference_corner_ids'][k]
            valid=label['target']['valid'][mapped];target=np.asarray(label['target']['gt'][mapped]) if valid else None
            item=dict(id=case['id'],native_corner=k,state=partial['state'],human_state=label['metadata']['human_states_native'][k],
                native_compatible=align['compatible'],valid_annotation=valid,lines=[])
            for line in partial['lines']:
                residual=float(np.asarray(line['normal_unit_xy'])@target+line['offset_px']) if valid else None
                item['lines'].append(dict(annotation_normal_residual_px=residual,
                    assumed_normal_std_px=float(np.sqrt(line['normal_variance_px2'])),
                    normal_residual_std_units=abs(residual)/np.sqrt(line['normal_variance_px2']) if valid else None))
            results.append(item)
    put(OUT/'CAL_POSTSEAL_CHECK.json',dict(label_sha256=C.sha(label_path),frames=46,partial_vertices=len(results),
        records=results,GT_inputs_to_predictions=False,new_variance_fit=False,normal_budget_is_borrowed_assumption=True))
    print(C.finite({'CAL_vertices':len(results),'records':results}))


def solve_target():
    # CAL prediction and diagnostic precede the new target pose experiment.
    cal=C.read(OUT/'CAL_POSTSEAL_CHECK.json');protocol=C.read(OUT/'PROTOCOL.json')
    V._verify_sources(protocol['source_sha256']);V._verify_sources(protocol['protected_user_sha256'])
    row=next(r for r in C.rows(E.OUTPUT/'PREDICTIONS.jsonl.gz') if r['id']==TARGET)
    rt=row['runtime'];g=rt['initial_geometry'];model=UC.load_uncertainty_model(MODEL)
    evidence,correction,binding=payload(row,model)
    fit=O.fixed_fit_measurements(rt['calibrated_visible_measurements'],correction)
    cfg=deepcopy(C.read(E.OUTPUT/'PROTOCOL.json')['solver_configuration'])
    args=(g['native_xyz'],g['K'],rt['states'],fit,rt['ot_domains']['domains'],g['image_hw'],
          dict(R=g['R_cf'],t=g['centroid'],physical_registration_Q=g['physical_registration_Q']))
    immutable=C.finite(dict(states=rt['states'],BI=fit,OT=rt['ot_domains'],Q=g['physical_registration_Q']))
    before=C.sha(O.FITTER_SOURCE);started=time.monotonic()
    put(OUT/'TARGET_OBSERVATION.json',dict(id=TARGET,evidence=evidence,bindings=binding,
        fixed_BI_fit_measurements=fit,model_sha256=C.sha(MODEL),CAL_check_sha256=C.sha(OUT/'CAL_POSTSEAL_CHECK.json'),
        before_pose_fit_and_new_target_annotation_score=True,marked_point_or_raw_junction_used=False))
    baseline=OLD.solve_uncertain_joint_ot_pnp(*args,config=cfg)
    zero=S.solve_boundary_normal_joint_ot_pnp(*args,config=cfg,
        boundary_observations=evidence['boundary_observations'],boundary_factor_weight=0.)
    assert C.finite(zero)==C.finite(baseline),'Zero weight must have exact parent parity'
    partial=S.solve_boundary_normal_joint_ot_pnp(*args,config=cfg,
        boundary_observations=evidence['boundary_observations'],boundary_factor_weight=1.)
    assert immutable==C.finite(dict(states=rt['states'],BI=fit,OT=rt['ot_domains'],Q=g['physical_registration_Q']))
    assert before==C.sha(O.FITTER_SOURCE)
    def projection(result):
        if not result['accepted']:return None
        return C.project(np.asarray(g['native_xyz']),result['R'],result['t'],g['K'])
    q0,q1=projection(baseline),projection(partial)
    delta=None
    if q0 is not None and q1 is not None:
        delta=dict(rotation_deg=float(np.degrees(Rotation.from_matrix(np.asarray(partial['R'])@np.asarray(baseline['R']).T).magnitude())),
            translation_cm=float(np.linalg.norm(np.asarray(partial['t'])-np.asarray(baseline['t']))*100),
            native3_projection_before=q0[3].tolist(),native3_projection_after=q1[3].tolist(),
            projection_change_px=float(np.linalg.norm(q1[3]-q0[3])))
    result=dict(id=TARGET,baseline=baseline,zero_weight_exact_parent=True,partial=partial,delta=delta,
        all_BI_fit_scales_states_OT_Q_preserved=True,original_S_RGB_queries=0,S_positions_from_PnP_only=True,
        new_pose_fits=3,source_sha256=protocol['source_sha256'],observation_sha256=C.sha(OUT/'TARGET_OBSERVATION.json'),
        target_annotations_read=False,new_boundary_variance_fit=False,TEST_is_reused_exploratory=True,
        partial_hard_support_canvas_geometry_changes_feasible_set=True,elapsed_seconds=time.monotonic()-started)
    put(OUT/'TARGET_POSE_RESULTS.json',result)
    V._verify_sources(protocol['source_sha256']);V._verify_sources(protocol['protected_user_sha256'])
    put(OUT/'TARGET_POSE_SEAL.json',dict(results_sha256=C.sha(OUT/'TARGET_POSE_RESULTS.json'),
        observation_sha256=C.sha(OUT/'TARGET_OBSERVATION.json'),target_labels_read=False,new_pose_fits=3))
    print(C.finite(dict(baseline_accepted=baseline['accepted'],partial_accepted=partial['accepted'],
        partial_observed_vertices=len(evidence['boundary_observations']),delta=delta)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('observe-cal','score-cal','solve-target'));a=p.parse_args()
    {'observe-cal':observe_cal,'score-cal':score_cal,'solve-target':solve_target}[a.stage]()


if __name__=='__main__':main()

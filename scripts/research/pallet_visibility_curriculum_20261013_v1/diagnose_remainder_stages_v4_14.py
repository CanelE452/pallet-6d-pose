"""Postseal cached-stage/observability diagnosis; no fits or new selectors."""
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np

from . import audit_remainder_pipeline_v4_14 as A
from . import score_remainder_derived_reference_v4_14 as REF
from . import evaluate_two_wd_families_v4_9 as W

OUTPUT=A.OUTPUT/'POSTSEAL_STAGE_OBSERVABILITY_V1.json'
STAGES=('INITIAL_N3_SUBPIX','FULL_P1_SEED','FULL_WD0_DATA_SELECTED','OLD_G10',
        'D_JOINT_RESELECT','FROZEN_WD_FACE_AWARE_PARTIAL_BI')


def summarize(rows,stage):
    return dict(frames=len(rows),translation_cm=A._distribution([r['metrics'][stage]['translation_cm']for r in rows]),
        rotation_deg=A._distribution([r['metrics'][stage]['rotation_deg']for r in rows]),
        translation_le5=sum(r['metrics'][stage]['translation_cm']<=5 for r in rows),
        successes=sum(r['metrics'][stage]['success_5cm_5deg']for r in rows))


def main():
    if OUTPUT.exists():raise FileExistsError('Preserve prior stage diagnosis')
    A.verify_source_closure()
    final=A.C.read(A.OUTPUT/'POSTSEAL_COMPARISON_V1.json')
    seal=A.C.read(A.POSE/'PREDICTION_SEAL.json');owned=A.C.read(REF.OUTPUT/'SCORE_SEAL.json')
    if seal['rows']!=143 or owned['prediction_seal_sha256']!=A.C.sha(A.POSE/'PREDICTION_SEAL.json'):
        raise ValueError('Complete143 +ownedpostseal gate required')
    annotations={r['id']:r for r in A.C.read(REF.OUTPUT/'PLAN.json')['annotations']}
    model=W.E.load_uncertainty_model(A.O.UNCERTAINTY_MODEL)
    rows=[]
    for record in final['records']:
        fid=record['id'];stem=fid.replace(':','__')
        prediction=A.C.read(A.POSE/'frames'/(stem+'.json'))
        original=A.C.read(A.O.OUTPUT/'frames'/(stem+'.json'))
        item=annotations[fid];path=A.C.ROOT/item['annotation']
        if A.C.sha(path)!=item['annotation_sha256']:raise ValueError('Fixed reference changed')
        reference=REF.reference_from_document(A.C.read(path))
        # Reporting helpers perform projection/readout only, not fitting. They
        # reproduce the saved accepted or explicit own fallback pose exactly.
        prior=W.V47._finish(original['_original_source'],original['frozen_BI_fit'],model,
            prediction['full_prior_seed'],original['u_region_observation']['u_regions'],
            'CACHED_FULL_P1_SEED_DIAGNOSTIC',original['partial_boundary_observation'])
        search=prediction['full_WD_search'];family=search['families'][search['selected_family_index']]
        data=W._family_finish(original,model,family,'CACHED_FULL_WD0_IMAGE_DATA_SELECTED',search['solution'])
        metrics={**record['metrics'], 'FULL_P1_SEED':A.independent_metric(prior['actual_pose'],reference),
                 'FULL_WD0_DATA_SELECTED':A.independent_metric(data['actual_pose'],reference)}
        d=prediction['partial_global_outline'];merged=d['merged_partial_observation']['boundary_observations']
        partial_ids={r['native_corner']for r in merged}; complete=[r for r in original['frozen_BI_fit']if r['native_corner']not in partial_ids]
        e=prediction['arms'][A.ARMS[-1]];Q=np.asarray(e['actual_pose']['physical_registration_Q'])
        pR=np.asarray(e['actual_pose']['R_physical']);initial=np.asarray(original['_original_source']['initial_geometry']['R_physical'])
        refR=np.asarray(reference['R']);dt=(np.asarray(e['actual_pose']['centroid'])-reference['t'])*100
        rows.append(dict(id=fid,session=record['session'],metrics=metrics,states=record['states'],
            archived_BI_count=len(original['frozen_BI_fit']),remaining_complete_2D_count=len(complete),
            complete_native_ids=[r['native_corner']for r in complete],
            partial_native_ids=sorted(partial_ids),partial_normal_rows=sum(len(r['lines'])for r in merged),
            final_BI_rank=e['solution'].get('rank'),material_ambiguity=e['ambiguity_audit']['material_ambiguity'],
            accepted=e['accepted'],OLD_G10_family=prediction['frozen_family'],
            data_only_WD_family=search['selected_family'],D_reselected_family=prediction['d_reselected_family'],
            full_P1_accepted=prior['accepted'],full_WD_data_selected_accepted=data['accepted'],
            final_centroid_error_xyz_cm=dt.tolist(),
            thickness_axis_camera_Y=dict(initial=float(initial[1,1]),final=float(pR[1,1]),reference=float(refR[1,1])),
            final_vs_initial_thickness_axis_cosine=float(initial[:,1]@pR[:,1]),
            final_vs_reference_thickness_axis_cosine=float(refR[:,1]@pR[:,1]),
            model_registration_Q=Q.tolist(),image_binding=A.binding(A.C.ROOT/original['image'])))
    groups={}
    for key,fn in (('session',lambda r:r['session']),('archived_BI_count',lambda r:str(r['archived_BI_count'])),
                   ('remaining_complete_2D_count',lambda r:str(r['remaining_complete_2D_count'])),
                   ('material_ambiguity',lambda r:str(r['material_ambiguity'])),
                   ('mask_reselected_data_family',lambda r:str(r['data_only_WD_family']!=r['OLD_G10_family']))):
        grouped=defaultdict(list)
        for row in rows:grouped[fn(row)].append(row)
        groups[key]={g:{s:summarize(values,s)for s in STAGES}for g,values in grouped.items()}
    summary={s:summarize(rows,s)for s in STAGES}
    report=dict(schema='POSTSEAL_REMAINDER143_CACHED_STAGE_OBSERVABILITY_DIAGNOSIS_V1',rows=143,
        summary=summary,groups=groups,records=rows,
        paired_final_minus_fullP1=A._paired(rows,'FULL_P1_SEED',A.ARMS[-1]),
        paired_final_minus_data_only_WD=A._paired(rows,'FULL_WD0_DATA_SELECTED',A.ARMS[-1]),
        source_binding=A.binding(__file__),full_prediction_seal=A.binding(A.POSE/'PREDICTION_SEAL.json'),
        owned_score_seal=A.binding(REF.OUTPUT/'SCORE_SEAL.json'),
        no_new_fits_forwards_parameter_choices_or_GT_family_routing=True,
        two_WD_data_selected_family_is_original_saved_image_objective_choice=True,
        thickness_axis_diagnostic_not_new_pose_constraint=True,
        rank_is_BI_or_partial_Jacobian_local_rank_not_unique_global_pose_certificate=True)
    A.C.write(OUTPUT,report)
    A.C.write(OUTPUT.with_name('POSTSEAL_STAGE_OBSERVABILITY_SEAL_V1.json'),dict(report_sha256=A.C.sha(OUTPUT),
        source_binding=A.binding(__file__),GT_decoded_after_full143_and_owned_score_seals=True,new_PnP_or_NN_calls=0))
    print(summary,flush=True)


if __name__=='__main__':main()

"""Small public P2 summaries from saved diagnostics, never new pose metrics."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
from scripts.research.pallet_risk_information_v1.common import read,write,sha


def csv_write(path,entries):
    columns=list(dict.fromkeys(k for row in entries for k in row))
    with path.open('x',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(entries)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--public-root',type=Path,required=True);p.add_argument('--summary-GO',action='store_true');a=p.parse_args()
    if not a.summary_GO:raise PermissionError('Root summary GO required')
    private,public=a.run_dir,a.public_root;query=read(private/'b_query_factorial_v1/B_QUERY_FACTORIAL_SUMMARY.json');bounds=read(private/'b_original_mask_bounds_v1/B_MASK_TRANSITION_BOUNDS.json')
    query_rows=[];mask_rows=[]
    for population,record in query['groups'].items():
        for family,variants in record['families'].items():
            for variant,data in variants.items():
                query_rows.append(dict(population=population,frames=record['frames'],B_input_frames=record['B_input_frames'],B_absent_frames=record['B_input_absent_frames'],family=family,query_variant=variant,
                    **{k:v for k,v in data.items()if k!='status_counts'},**{'geometry_status/'+k:v for k,v in data['status_counts'].items()}))
    for key,record in bounds['groups'].items():
        population,family,variant,kind=key.split('/');mask_rows.append(dict(population=population,family=family,query_variant=variant,mask_kind=kind,**record))
    csv_write(public/'B_QUERY_AGGREGATES.csv',query_rows);csv_write(public/'B_MASK_BOUNDS_AGGREGATES.csv',mask_rows)
    # These small original summaries contain no private locators/raw images/NN data.
    write(public/'B_QUERY_FACTORIAL_SUMMARY.json',query);write(public/'B_MASK_TRANSITION_BOUNDS.json',bounds)
    executions=[]
    for name in ('b_query_factorial_v1','b_original_masks_v1','b_original_mask_bounds_v1','procedural_shape_control_v1','procedural_shape_figures_v1','b_p2_figures_v1','b_p2_figures_v2','b_p2_figures_v3'):
        path=private/'execution'/name/'RECEIPT.json';record=read(path)
        executions.append(dict(name=name,original_receipt_sha256=sha(path),**{k:record[k]for k in('status','actual_exit_code','wall_seconds','child_CPU_user_seconds','child_CPU_system_seconds','child_peak_rss_kib','standalone_batch_wall','CUDA_activity_seconds','CUDA_note')}))
    write(public/'B_P2_EXECUTION.json',dict(schema='P2_SAVED_EXECUTION_SUMMARY_V1',batches=executions,
        standalone_parent_wall_sum_seconds=sum(r['wall_seconds']for r in executions),inner_timers_not_added=True,
        original_scene_replays=0,actual_new_procedural_software_renders=4,new_D1_samples=0,new_model_forwards=0,new_training=0,new_pose_solves=0,
        preserved_failure='b_p2_figures_v1 private RGB locator lacked actual _rgb suffix; exact source/logs/partial image preserved, then technical repair only',
        preserved_display_defect='b_p2_figures_v2 short geometric line display; source/images retained and v3 used identical records with infinite-line display',
        CUDA_activity='Not instrumented; no CUDA/model/solver API in these CPU-only sources; unknown timing not replaced by0'))
    inputs={}
    for relative in('b_query_factorial_v1/B_QUERY_FACTORIAL_SEAL.json','b_original_masks_v1/B_MASK_DIAGNOSTIC_SEAL.json','b_original_mask_bounds_v1/DIAGNOSTIC_SEAL.json',
        'procedural_shape_plan_v1/CONFIG.json','procedural_shape_plan_v1/LOCK.json','procedural_shape_control_v1/PROCEDURAL_RENDER_SEAL.json','B_SCENE_RENDER_PREFLIGHT.json',
        'independent_b_query_v1/VERIFICATION.json','B_P2_PIXEL_REVIEW.json','B_P2_PRIVATE_INPUT_PIXEL_REVIEW.json','B_P2_PRESENTATION_REPAIR_PARITY.json'):
        path=private/relative
        if not path.is_file():raise FileNotFoundError('Missing required real evidence: '+relative)
        inputs[relative]=sha(path)
    write(public/'B_DIAGNOSTIC_INPUTS.json',dict(schema='P2_DIAGNOSTIC_INPUT_AVAILABILITY_V1',source_sha256=sha(__file__),private_evidence_hashes=inputs,
        registered_frames=1003,B_input_frames=673,B_absent_frames=330,all_native_phase_family_variant_profiles=139976,
        original_masks_available=dict(visible=1003,amodal=1003,archived_SAM=1003),raw_original_mask_mode='L',raw_original_mask_format='PNG',
        missing_original_replay_inputs=['synth_data_scene_portable_stage2c2.blend','Authenticated prod40k generator/reset/compositor source/version','Complete source assets and original runnable Blender/bpy environment','Resolved publication rights for every reused original asset'],
        exact_original_scene_status='EXACT_SCENE_UNAVAILABLE',original_RGB_published=False,independent_physical_edge_owner='UNVERIFIED',renderer_encoding='UNVERIFIED_CONDITIONAL_0_255_ONLY',
        material_metadata='Original material variant/scene visibility metadata exist; material variant and object mask do not establish per-profile physical edge identity',
        typed_evaluator='Diagnostic contract and analytic counterexamples only; no automatic real supplier/production correction',
        old_roundtrip_failures_preserved=True,old_operating_defaults_unchanged=True,old_models_candidates_observations_references_unchanged=True))


if __name__=='__main__':main()

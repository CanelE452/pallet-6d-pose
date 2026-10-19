"""Final S3/S4 source, model and numeric-output closure for root scoring."""
from __future__ import annotations
import argparse
from pathlib import Path
from . import image_batch as I,material_assets as A


def seal(run):
    run=Path(run);producer=run/'s3_s4_production';post=run/'s3_s4_postprocess'
    finish=I.read(post/'FINISH_RECEIPT.json');full=I.read(producer/'FULL_RECEIPT.json')
    if not finish['complete']or finish['H1_fits']!=15 or not full['all1241_completed']or full['rows']!=1241:
        raise ValueError('Complete producer and all fifteen own H1 fits required')
    method=I.read(producer/'METHOD_LOCK_RESOURCE_LP4.json')
    if full['method_lock_sha256']!=A.sha(producer/'METHOD_LOCK_RESOURCE_LP4.json'):
        raise ValueError('Active scientific method lock differs')
    artifacts={}
    def bind(path,expected=None):
        path=Path(path);value=A.sha(path)
        if expected is not None and value!=expected:raise ValueError('Bound file changed: '+str(path))
        artifacts[str(path)]=value
    bind(producer/'FULL_RECEIPT.json',finish['producer_seal_sha256'])
    bind(post/'POSTPROCESS_SOURCE_LOCK.json',finish['postprocess_source_lock_sha256'])
    for path,expected in finish['sealed_artifact_paths'].items():bind(path,expected)
    for path,expected in method['sources'].items():bind(path,expected)
    for path,expected in method['asset_bindings'].items():bind(path,expected)
    source_lock=I.read(post/'POSTPROCESS_SOURCE_LOCK.json')
    for name,expected in source_lock['sources'].items():bind(Path(__file__).parent/name,expected)
    for name,expected in full['output_artifacts'].items():bind(producer/name,expected)
    row_archive=run/'s3_s4_archive_report';archive=I.read(row_archive/'ROW_ARCHIVE_SEAL.json')
    for name,expected in archive['output_artifacts'].items():bind(row_archive/name,expected)
    rows=list(I.rows(row_archive/'ALL_ROW_BINDINGS.jsonl.gz'))
    if len(rows)!=1241:raise ValueError('Full1241 row binding required')
    for row in rows:
        bind(producer/row['row_seal'],row['row_seal_sha256'])
        bind(producer/row['frame'],row['frame_sha256'])
    training=[]
    for arm in I.ARMS:
        manifest=I.read(run/'s3_s4_training'/arm/'TRAINING_MANIFEST.json')
        prediction=I.read(run/'s3_s4_predictions'/arm/'PREDICTION_SEAL.json')
        if not manifest['complete']or len(manifest['runs'])!=3 or not prediction['complete']or prediction['rows']!=238:
            raise ValueError('All five three-seed full238 predictions required')
        for row in manifest['runs']:
            checkpoint=Path(row['checkpoint'])
            bind(checkpoint,row['checkpoint_sha256'])
            bind(checkpoint.parent/'last.pt',row['last_checkpoint_sha256'])
            bind(checkpoint.parent/'HISTORY.json',row['history_sha256'])
            run_receipt=checkpoint.parent/'RUN_RECEIPT.json'
            if I.canonical(I.read(run_receipt))!=I.canonical(row):
                raise ValueError('H1 run receipt and manifest row differ')
            bind(run_receipt)
            bind(checkpoint.parent/'resume.pt')
            if not row['complete']or row['epochs']>50:raise ValueError('H1 completion or epoch limit differs')
            training.append((row['arm'],row['seed']))
        labeldir=run/'s3_s4_labels'/arm;labels=I.read(labeldir/'LABEL_SEAL.json')
        bind(labeldir/'LABEL_SEAL.json');bind(labeldir/'ACTION_TARGETS.jsonl.gz',labels['labels_sha256'])
        bind(labeldir/'TRAIN_ONLY_NORMALIZER.json',labels['normalizer_sha256'])
    if set(training)!={(arm,seed)for arm in I.ARMS for seed in(20261018,20261019,20261020)}:
        raise ValueError('Exactly the fifteen registered arm/seed fits required')
    fixed=[producer/'FULL_RECEIPT.json',producer/'METHOD_LOCK_RESOURCE_LP4.json',producer/'METHOD_LOCK.json',
        producer/'RESOURCE_AMENDMENT_LP4.json',producer/'PILOT_RECEIPT.json',
        producer/'EXTERNAL_FINALIZATION_TIME_PROVENANCE.json',producer/'MODEL_IDENTITY_INPUTS1241.jsonl.gz',
        producer/'ORIGINAL_NORMAL_RAW_SUPPORT_AUDIT.json',post/'FINISH_RECEIPT.json',post/'POSTPROCESS_SOURCE_LOCK.json',
        post/'VALIDATION_SELECTION_INPUTS.json',run/'s3_s4_postseal_analysis/ACTUAL_COST_INVENTORY.json',
        run/'s3_s4_postseal_analysis/SYNTHETIC_ASSET_POSITIVE_COVERAGE.json',
        run/'S3_S4_RENDER_THREAD_PARITY_LP4.json']
    for path in fixed:bind(path)
    for folder in (row_archive,run/'s3_s4_structure_review'):
        for path in folder.iterdir():
            if path.is_file()and path.suffix in('.json','.gz','.html','.jpg'):bind(path)
    for name in ('material_handoff.py','material_costs.py','material_label_coverage.py','material_asset_domains.py'):
        bind(Path(__file__).parent/name)
    receipt=dict(schema='S3_S4_FULL_AUTOMATIC_COMPARISON_HANDOFF_SEAL_V1',complete=True,
        producer_rows=1241,real_rows_per_arm=238,synthetic_rows_per_arm=1003,
        arms=list(I.ARMS),H1_fits=15,seeds=[20261018,20261019,20261020],
        all_five_full238_prediction_files_sealed_before_this_branch_reference_decode=True,
        validation_selection_inputs=I.read(post/'VALIDATION_SELECTION_INPUTS.json'),
        validation_joint_success_units='frame count out of the original235 validation IDs per seed',
        all_row_candidate_ancestry_validated=True,all_row_and_output_hashes_revalidated=True,
        scientific_source_and_asset_bytes_revalidated=True,all_best_last_resume_and_history_model_artifacts_bound=True,
        artifact_path_sha256=artifacts,producer_full_receipt_sha256=A.sha(producer/'FULL_RECEIPT.json'),
        method_lock_sha256=A.sha(producer/'METHOD_LOCK_RESOURCE_LP4.json'),
        physical_model_status='REGISTERED_STRUCTURED_PROXY; signed axes and real instance identity uncertified',
        all235_synthetic_validation_rows_use_missing_named_GLB_proxy=True,
        independent_physical_owner_labels=0,human_confirmations=0,
        physical_owner_accuracy='NOT_APPLICABLE_WITH_REASON_NO_INDEPENDENT_PHYSICAL_LABELS',
        renderer='Mesa llvmpipe software EGL; zero hardware GPU render jobs',
        image_representation='WHOLE_NATIVE_GRAYSCALE_DIS_FAST_RENDER_DEPTH_TO_PnP_REFINE_LM',
        this_branch_current_real_reference_reads=0,this_branch_PRIMARY_scoring_executed=False)
    I.write(run/'S3_S4_HANDOFF_SEAL.json',receipt)
    return dict(complete=True,H1_fits=15,artifacts_bound=len(artifacts),
        handoff_seal_sha256=A.sha(run/'S3_S4_HANDOFF_SEAL.json'),validation=receipt['validation_selection_inputs'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    print(I.canonical(seal(a.run)),flush=True)

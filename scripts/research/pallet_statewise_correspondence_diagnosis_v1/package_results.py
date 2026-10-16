"""Publish small sealed evidence without large raw observation/pose journals."""
from pathlib import Path
import argparse
import shutil

from .common import ARMS, read, rows, save_rows, sha, write


def _copy(source,target,manifest):
    source,target=Path(source),Path(target)
    if not source.is_file():raise FileNotFoundError(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(target)
    derived=False
    if source.name=='TEST_RESULTS.json':
        value=read(source)
        value['command'][0]='$TASK_PYTHON'
        value['public_derivation']=dict(change='local Python executable path replaced by reproduction variable',
                                       raw_private_parent_sha256=sha(source))
        write(target,value)
        derived=True
    else:
        shutil.copyfile(source,target)
    manifest[target.parent.name+'/'+target.name]=dict(bytes=target.stat().st_size,sha256=sha(target),source_sha256=sha(source))
    if derived:manifest[target.parent.name+'/'+target.name]['derived']=True


def compact_lineage(value):
    """Retain every owner/type transition and factor; omit repeated solver bank."""
    c=value['C_original_e'];d=value['D_post_e'];e=value['E_post_e_solve']
    observation=d['observation']
    return dict(schema='PUBLIC_COMPACT_LINEAGE_WITH_ORIGINAL_FULL_PARENT_V1',
        id=value['id'],partition=value['partition'],image_sha256=value['image_sha256'],
        original_N3=value['A_original_N3'],
        original_states=value['B_original_pre_e']['states'],
        original_mask_binding=value['B_original_pre_e']['mask_binding'],
        original_correction_binding=value['B_original_pre_e']['correction_binding'],
        original_e={k:c[k] for k in ('activation','actual_pose','boundary_observations','domains','registered_family','u_regions')},
        post_e=dict(native_vertex_mapping=d['native_vertex_mapping'],native_points=d['native_points'],
                    states=observation['states'],frozen_visible_measurements=observation['frozen_visible_measurements'],
                    mask_binding=d['mask_binding'],observation_binding=d['observation_binding']),
        audit=value['audit'],rays=value['rays'],semantic_owner_status=value['semantic_owner_status'],
        omission='repeated solver candidate banks, state feature arrays and dense outline samples remain in hash-bound private full lineage')


def package(folder,public):
    root,public=Path(folder),Path(public)
    if not (root/'INDEPENDENT_REAGGREGATION.json').exists():
        raise ValueError('Independent saved-Rt verification must finish before packaging')
    assert read(root/'INDEPENDENT_REAGGREGATION.json')['success_all_match']
    manifest={}
    sources={
        'metrics':root/('score_v2' if (root/'score_v2/SCORE_SEAL.json').exists() else 'score_v1'),
        'diagnostics':root/'diagnostics_v1',
        'oracle':root/'oracle_v1',
    }
    for label,source in sources.items():
        for path in sorted(source.iterdir()):
            if path.suffix not in ('.json','.csv','.md'):continue
            if path.name in ('FIXED_REFERENCES.json','SCORED_ROWS.json'):continue
            _copy(path,public/label/path.name,manifest)
    evidence=public/'evidence';evidence.mkdir(exist_ok=True)
    names=('START.json','ENVIRONMENT.json','SOURCE_AND_INPUT_MANIFEST.json','REPLAY_PROTOCOL.json','REFERENCE_SCORING_BINDING.json',
        'SOURCE_CONTRACT_AMENDMENT_V1.json','TYPED_METADATA_PACKAGING_FIX_V1.json',
        'PILOT_AUTHORIZATION.json','REOBS_PILOT_PARITY.json','TYPED_PILOT_PARITY_V1.json',
        'CONTINUITY_AUDIT.json','REPLAY_PARITY.json','PREDICTION_SEAL.json','CANDIDATE_SEAL.json',
        'INDEPENDENT_REAGGREGATION.json','INDEPENDENT_SUMMARY_CHECK.json','COST.json','FINAL_CONTRACT_CHECK.json','TEST_RESULTS.json',
        'PRE_SCORING_SOURCE_CHECK.json','POSTSEAL_SCORING_AUTHORIZATION.json','FINAL_SOURCE_TESTS.json')
    for name in names:_copy(root/name,evidence/name,manifest)
    for name in ('EXTRACTION_SEAL.json','OBSERVATION_TYPE_AND_OWNER_AUDIT.csv','RESOURCE_RECEIPT.json'):
        _copy(root/'e_trace_v2'/name,evidence/name,manifest)
    for name in ('CANDIDATE_MANIFEST.json','CANDIDATE_COVERAGE.json'):
        _copy(root/'original_candidate_archive_v1'/name,evidence/name,manifest)
    for name in ('RAW_START_SUPPLEMENT.json','SUPPLEMENT_SEAL.json'):
        _copy(root/'supplemental_cal_start_logging_v1'/name,evidence/name,manifest)
    _copy(root/'posefix_source_audit_v1/E0_ORIGINAL_STATUS_METADATA_AUDIT_V1.json',
          evidence/'E0_ORIGINAL_STATUS_METADATA_AUDIT_V1.json',manifest)
    _copy(root/'SCORE_V1_V2_METADATA_BRIDGE.json',evidence/'SCORE_V1_V2_METADATA_BRIDGE.json',manifest)
    _copy(root/'score_v1/SCORE_SEAL.json',evidence/'ORIGINAL_SCORE_V1_SEAL.json',manifest)
    _copy(root/'RESOURCE_LEDGER.jsonl',evidence/'RESOURCE_LEDGER.jsonl',manifest)
    _copy(root/'TEST_RESULTS_FINAL.log',evidence/'TEST_RESULTS_FINAL.log',manifest)
    _copy(root/'FINAL_SOURCE_TESTS.log',evidence/'FINAL_SOURCE_TESTS.log',manifest)
    if (root/'REPORT_DOCUMENT_QA_V2.json').exists():
        _copy(root/'REPORT_DOCUMENT_QA_V2.json',evidence/'REPORT_DOCUMENT_QA.json',manifest)
    save_rows(evidence/'OBSERVATION_LINEAGE_COMPACT.jsonl.gz',
              (compact_lineage(v) for v in rows(root/'e_trace_v2/OBSERVATION_LINEAGE.jsonl.gz')))
    def poses():
        for r in rows(root/'REPLAY_PREDICTIONS.jsonl.gz'):
            predictions={a:{k:v for k,v in r['predictions'][a].items() if k in
                    ('actual_pose','accepted','fallback','selected_family','technical_failure','timeout','audit',
                     'uses_e_initialization','elapsed_seconds','optimizer_calls','method')}
                    for a in ARMS}
            for arm in ARMS:
                predictions[arm]['uses_e_initialization']=arm!='TYPED_MULTI'
                predictions[arm]['uses_e_initialization_source']='DERIVED_METHOD_CONTRACT; raw sealed prediction preserved'
            yield dict(id=r['id'],partition=r['partition'],session=r['session'],cluster_id=r['cluster_id'],
                       predictions=predictions,GT_inputs=False)
    save_rows(evidence/'REPLAY_PREDICTIONS_COMPACT.jsonl.gz',poses())
    for path in evidence.glob('*COMPACT.jsonl.gz'):
        manifest['evidence/'+path.name]=dict(bytes=path.stat().st_size,sha256=sha(path),derived=True)
    write(public/'PUBLICATION_MANIFEST.json',dict(files=manifest,large_private_artifacts={
        'OBSERVATION_LINEAGE.jsonl.gz':sha(root/'e_trace_v2/OBSERVATION_LINEAGE.jsonl.gz'),
        'TYPED_PACKETS.jsonl.gz':sha(root/'e_trace_v2/TYPED_PACKETS.jsonl.gz'),
        'REPLAY_PREDICTIONS.jsonl.gz':sha(root/'REPLAY_PREDICTIONS.jsonl.gz'),
        'GENERATED_CANDIDATES.jsonl.gz':sha(root/'GENERATED_CANDIDATES.jsonl.gz')},
        compact_is_not_the_full_original=True,
        raw_journals_masks_weights_published=False,
        full_compact_distinguished_by_filenames=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--public',required=True)
    a=p.parse_args();package(a.folder,a.public)

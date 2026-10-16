"""Bind this fixed-pose analysis to the original 407de865 evidence."""
from collections import Counter
from pathlib import Path
import argparse
import subprocess
import time

from .common import read, rows, sha, write, timed_stage

RESULT='407de865445e4987614f50f893d2ccea78d507b6'
OLD_NAMESPACE='scripts/research/pallet_statewise_correspondence_diagnosis_v1/'
DOC='pallet_statewise_correspondence_diagnosis_v1'


def bind(source_root,parent,output):
    source_root,parent,output=map(Path,(source_root,parent,output))
    public=source_root/'_docs/experiments'/DOC
    with timed_stage(output,'input_source_reference_binding',new_pose_calls=0):
        old=read(parent/'SOURCE_AND_INPUT_MANIFEST.json')
        legacy={p:h for p,h in old['source_sha256'].items() if not p.startswith(OLD_NAMESPACE)}
        issues=[]
        for p,h in legacy.items():
            if sha(source_root/p)!=h:issues.append(p)
        assert not issues,('legacy numerical dependency mismatch',issues)
        # The final original publication contains later packaging corrections.
        # Bind its final artifact list as well as the pre-fit source snapshot.
        published=read(public/'PUBLICATION_CHECK.json')
        for p,value in published['files'].items():
            assert sha(source_root/p)==value['sha256'],p
        git_differences=subprocess.check_output(['git','diff','--name-only',RESULT,'HEAD','--',
            'scripts/research/pallet_statewise_correspondence_diagnosis_v1',
            '_docs/experiments/'+DOC],cwd=source_root,text=True).splitlines()
        assert not git_differences,git_differences
        ps=read(parent/'PREDICTION_SEAL.json');cs=read(parent/'CANDIDATE_SEAL.json')
        assert sha(parent/'REPLAY_PREDICTIONS.jsonl.gz')==ps['predictions_sha256']
        assert sha(parent/'GENERATED_CANDIDATES.jsonl.gz')==cs['generated_candidates_sha256']
        assert sha(parent/'CANDIDATE_SEAL.json')==ps['candidate_seal_sha256']
        es=read(parent/'e_trace_v2/EXTRACTION_SEAL.json')
        for p,b in es['files'].items():
            assert sha(parent/'e_trace_v2'/p)==b['sha256'],p
        os_=read(parent/'oracle_v1/ORACLE_SEAL.json')
        for p,h in os_['artifacts'].items():assert sha(parent/'oracle_v1'/p)==h,p
        ss=read(parent/'score_v2/SCORE_SEAL.json')
        for p,h in ss['artifacts'].items():assert sha(parent/'score_v2'/p)==h,p
        cm=read(parent/'original_candidate_archive_v1/CANDIDATE_MANIFEST.json')
        for p,h in cm['artifacts'].items():
            expected=h['sha256'] if isinstance(h,dict) else h
            assert sha(parent/'original_candidate_archive_v1'/p)==expected,p
        reference=read(parent/'REFERENCE_SCORING_BINDING.json')
        reference_path=parent.parent/'minimal-bi-ablation-20261016-v1/SCORING_REFERENCE_MANIFEST_V2.json'
        assert sha(reference_path)==reference['sha256']==os_['reference_manifest_sha256']==ss['reference_manifest_sha256']
        assert sha(parent/'PREDICTION_SEAL.json')==os_['prediction_seal_sha256']==ss['prediction_seal_sha256']
        assert sha(parent/'CANDIDATE_SEAL.json')==os_['candidate_seal_sha256']==ss['candidate_seal_sha256']
        ids=old['ordered_ids']
        assert ids==os_['ordered_ids']==ps['ordered_ids']==cs['ordered_ids']
        assert len(ids)==len(set(ids))==238
        partitions=Counter(f['partition'] for f in old['frames'])
        assert partitions=={'CAL':46,'TEST':49,'REMAINDER':143}
        annotation_checks=0
        for p,h in os_['annotation_sha256'].items():
            assert sha(source_root/p)==h,p
            annotation_checks+=1
        original_bindings=es['original_source_sha256']
        for p,h in original_bindings.items():assert sha(source_root/p)==h,p
        paths=['SOURCE_AND_INPUT_MANIFEST.json','PREDICTION_SEAL.json','CANDIDATE_SEAL.json',
            'REPLAY_PREDICTIONS.jsonl.gz','GENERATED_CANDIDATES.jsonl.gz','REPLAY_PARITY.json',
            'REFERENCE_SCORING_BINDING.json','e_trace_v2/EXTRACTION_SEAL.json',
            'e_trace_v2/TYPED_PACKETS.jsonl.gz','e_trace_v2/OBSERVATION_LINEAGE.jsonl.gz',
            'original_candidate_archive_v1/CANDIDATE_MANIFEST.json',
            'original_candidate_archive_v1/ORIGINAL_E_CANDIDATE_BANK.jsonl.gz',
            'supplemental_cal_start_logging_v1/SUPPLEMENT_SEAL.json',
            'supplemental_cal_start_logging_v1/RAW_START_SUPPLEMENT.json',
            'oracle_v1/ORACLE_ROWS.jsonl.gz','oracle_v1/ORACLE_SUMMARY.json',
            'oracle_v1/ORACLE_FRAME_TABLE.csv','oracle_v1/DIAGNOSTIC_CASE_STAGES.json',
            'score_v2/SCORE_SEAL.json','score_v2/ROW_METRICS.csv','score_v2/FIXED_REFERENCES.json']
        evidence={p:dict(alias='PRIVATE_PARENT/'+p,sha256=sha(parent/p),bytes=(parent/p).stat().st_size)
                  for p in paths}
        docs={p:sha(public/p) for p in ('REPORT_KO.md','ORACLE_LIMITS.md','OBSERVATION_SCHEMA_MAPPING.md',
               'PLAN_AND_CONTRACT.md','REPRODUCE.md','POSEFIX_TRANSFER_AUDIT.md')}
        receipt=dict(schema='SELECTION_SCORE_SOURCE_INPUT_BINDING_V1',source_result_commit=RESULT,
            execution_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source_root,text=True).strip(),
            original_published_artifacts_unchanged=len(published['files']),original_namespace_git_differences=git_differences,
            original_legacy_numerical_dependencies_checked=len(legacy),legacy_source_sha256=legacy,
            required_documents_sha256=docs,private_parent_artifacts=evidence,ordered_ids=ids,partitions=dict(partitions),
            original_observation_mask_sources_checked=len(original_bindings),annotations_sha_checked=annotation_checks,
            fixed_reference_manifest_sha256=reference['sha256'],new_data_or_reference_change=False,
            semantic_owner_status='SEMANTIC_OWNER_UNVERIFIED',source_and_input_parity=True,
            original_source_or_archive_rewritten=False,original_candidates_regenerated=False,
            reference_role='existing geometric reconstruction; posthoc fixed-candidate analysis only')
        write(output/'SOURCE_BINDINGS.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source-root','parent','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();result=bind(a.source_root,a.parent,a.output)
    print('SOURCE_BINDINGS PASS:',len(result['ordered_ids']),'IDs;',result['original_legacy_numerical_dependencies_checked'],'legacy source files')

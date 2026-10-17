"""Thin resumable postseal report/publication controller.  No git operations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .common import read,sha,write


def _verify_gallery(path,run,evaluation):
    value=read(path)
    if (value.get('explicit_postseal_GO') is not True or value.get('all_population_frames')!=238
            or value.get('PRIMARY_frames')!=192):
        raise ValueError('Full-population postseal gallery proof required')
    bindings=value.get('input_bindings',{})
    if sha(run/'PREDICTION_SEAL.json') not in bindings.values() or sha(evaluation/'EVALUATION_SEAL.json') not in bindings.values():
        raise ValueError('Gallery does not bind current predictions/evaluation')
    for output in value.get('outputs',[]):
        p=(path.parent/output['file']).resolve()
        if not p.is_relative_to(path.parent.resolve()) or sha(p)!=output['sha256']:
            raise ValueError('Gallery output bytes changed')
    numeric=value.get('public_numeric_export')
    if numeric:
        folder=Path(numeric['directory'])
        for name,digest in numeric['files'].items():
            target=(folder/name).resolve()
            if not target.is_relative_to(folder.resolve()) or sha(target)!=digest:
                raise ValueError('Public numeric gallery bytes changed')
        receipt=read(folder/'NUMERIC_GALLERY_RECEIPT.json')
        if (receipt.get('population_frames')!=238 or receipt.get('PRIMARY_frames')!=192
                or receipt.get('RGB_files_copied')!=0 or receipt.get('tail_values_removed')!=0
                or sha(evaluation/'EVALUATION_SEAL.json') not in receipt.get('input_bindings',{}).values()):
            raise ValueError('Public numeric gallery population/RGB/evaluation contract changed')
    return value


def _verify_report(path,run,evaluation,independent):
    from .report import _sealed_artifacts
    value=read(path)
    if (value.get('complete') is not True or value.get('explicit_root_report_GO') is not True
            or value.get('evaluation_seal_sha256')!=sha(evaluation/'EVALUATION_SEAL.json')
            or value.get('prediction_seal_sha256')!=sha(run/'PREDICTION_SEAL.json')
            or value.get('plan_sha256')!=sha(run/'PLAN_LOCK.json')
            or value.get('independent_receipt_sha256')!=sha(independent)):
        raise ValueError('Existing report is stale; preserve it and diagnose instead of rerunning')
    _sealed_artifacts(path.parent,value)
    synthetic=run/'synthetic_head_evaluation_v1/EVALUATION_SEAL.json'
    if value.get('synthetic_head_evaluation_seal_sha256')!=(sha(synthetic) if synthetic.is_file() else None):
        raise ValueError('Synthetic head validation changed after the existing report')
    return value


def _verify_publication(path,output,report_seal):
    value=read(path)
    if value.get('complete') is not True or Path(value['public_output']).resolve()!=output.resolve():
        raise ValueError('Incomplete/other publication receipt cannot be reused')
    manifest_path=output/'PUBLICATION_MANIFEST.json'
    if sha(manifest_path)!=value['publication_manifest_sha256']:
        raise ValueError('Public manifest bytes changed')
    manifest=read(manifest_path)
    if manifest.get('complete') is not True or manifest['private_report_seal_sha256']!=sha(report_seal):
        raise ValueError('Publication does not bind the current complete report')
    for derived in value['derived_files']:
        public=output/derived['public_file']
        if (not public.resolve().is_relative_to(output.resolve()) or sha(public)!=derived['public_sha256']
                or sha(derived['private_source'])!=derived['private_sha256']):
            raise ValueError('Publication derivative/private original changed')
    for name,digest in manifest['required_public_file_sha256'].items():
        if sha(output/name)!=digest:raise ValueError('Required public artifact changed: '+name)
    for name,digest in manifest.get('separately_generated_public_numeric_file_sha256',{}).items():
        if sha(output/name)!=digest:raise ValueError('Public numeric plot changed: '+name)
    return value


def run(run_dir,*,evaluation_dir=None,independent_receipt=None,gallery_receipt=None,
        report_dir=None,output_dir=None,source_root=None,cost_ledger=None,
        mechanism_dir=None,approved_assets_json=None,final_GO=False,
        final_receipt_path=None,publication_receipt_path=None):
    if final_GO is not True:raise PermissionError('Root explicit final GO is required before reading results')
    start=time.monotonic();run=Path(run_dir).resolve()
    source=Path(source_root or Path(__file__).resolve().parents[3]).resolve()
    evaluation=Path(evaluation_dir or run/'evaluation_v1').resolve()
    independent=Path(independent_receipt or run/'independent_v2/INDEPENDENT_CHECK.json').resolve()
    gallery=Path(gallery_receipt or run/'gallery_v3/GALLERY_RECEIPT.json').resolve()
    reportdir=Path(report_dir or run/'report_v1').resolve()
    output=Path(output_dir or source/'_docs/experiments/pallet_statewise_end_to_end_recovery_v1').resolve()
    receipt_path=Path(final_receipt_path).resolve() if final_receipt_path is not None else run/'FINALIZATION_RECEIPT.json'
    public_receipt=(Path(publication_receipt_path).resolve() if publication_receipt_path is not None
                    else run/'PUBLICATION_RECEIPT.json')
    fixed_bindings=dict(plan=sha(run/'PLAN_LOCK.json'),prediction=sha(run/'PREDICTION_SEAL.json'),
        evaluation=sha(evaluation/'EVALUATION_SEAL.json'),independent=sha(independent),gallery=sha(gallery),
        root_cost_ledger=sha(cost_ledger) if cost_ledger else None,
        mechanism={n:sha(Path(mechanism_dir)/n) for n in ('MECHANISM_DIAGNOSIS.json','MECHANISM_DIAGNOSIS_KO.md','MECHANISM_SEAL.json')} if mechanism_dir else None,
        approved_assets=sha(approved_assets_json) if approved_assets_json else None)
    synthetic_seal=run/'synthetic_head_evaluation_v1/EVALUATION_SEAL.json'
    fixed_bindings['synthetic_head_evaluation']=sha(synthetic_seal) if synthetic_seal.is_file() else None
    # Explicit proofs, not the presence of a folder or a log line, authorize
    # final reporting.  The report module separately verifies the full chain.
    check=read(independent)
    if check.get('status')!='PASS' or check.get('failure_count')!=0 or fixed_bindings['evaluation'] not in check.get('input_bindings',{}).values():
        raise ValueError('Current independent PASS required')
    _verify_gallery(gallery,run,evaluation)
    existing_final=read(receipt_path) if receipt_path.is_file() else None
    if existing_final and (existing_final.get('complete') is not True or existing_final['input_bindings']!=fixed_bindings):
        raise ValueError('Finalization inputs changed; preserve the old receipt instead of silently rerunning')
    if existing_final and existing_final.get('receipt_paths',{}).get('publication') not in (None,str(public_receipt)):
        raise ValueError('Existing finalization belongs to a different publication receipt path')
    reportseal=reportdir/'REPORT_SEAL.json';report_reused=reportseal.is_file()
    if report_reused:_verify_report(reportseal,run,evaluation,independent)
    else:
        from .report import run as make_report
        make_report(run,output_dir=reportdir,evaluation_dir=evaluation,independent_receipt=independent,
                    source_root=source,gallery_receipt=gallery,cost_ledger=cost_ledger,mechanism_dir=mechanism_dir,report_GO=True)
    publication_reused=public_receipt.is_file()
    if publication_reused:_verify_publication(public_receipt,output,reportseal)
    else:
        from .publish import run as prepare_publication
        prepare_publication(run,report_dir=reportdir,output_dir=output,source_root=source,
                            approved_assets_json=approved_assets_json,publish_GO=True,receipt_path=public_receipt)
    if existing_final:
        if (existing_final['report_seal_sha256']!=sha(reportseal)
                or existing_final['publication_receipt_sha256']!=sha(public_receipt)):
            raise ValueError('Finalization output proof changed')
        return existing_final
    result=dict(schema='ACTUAL_REPORT_PUBLICATION_FINALIZATION_V1',complete=True,explicit_root_final_GO=True,
        input_bindings=fixed_bindings,report_seal_sha256=sha(reportseal),
        publication_receipt_sha256=sha(public_receipt),public_manifest_sha256=sha(output/'PUBLICATION_MANIFEST.json'),
        receipt_paths=dict(finalization=str(receipt_path),publication=str(public_receipt)),
        private_report_dir=str(reportdir),public_output_dir=str(output),
        report_reused=report_reused,publication_reused=publication_reused,
        reporting_performed_or_verified=True,publication_prepared_and_verified=True,
        git_commit_push_performed=False,operating_default_changed=False,paper_changed=False,
        source_sha256=sha(__file__),elapsed_seconds=time.monotonic()-start,finalized_epoch=time.time())
    write(receipt_path,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    for name in ('evaluation-dir','independent-receipt','gallery-receipt','report-dir','output-dir','source-root','cost-ledger','mechanism-dir','approved-assets-json','final-receipt-path','publication-receipt-path'):
        p.add_argument('--'+name)
    p.add_argument('--final-GO',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.run_dir,evaluation_dir=a.evaluation_dir,independent_receipt=a.independent_receipt,
          gallery_receipt=a.gallery_receipt,report_dir=a.report_dir,output_dir=a.output_dir,
          source_root=a.source_root,cost_ledger=a.cost_ledger,mechanism_dir=a.mechanism_dir,
          approved_assets_json=a.approved_assets_json,final_GO=a.final_GO,
          final_receipt_path=a.final_receipt_path,publication_receipt_path=a.publication_receipt_path),ensure_ascii=False))


if __name__=='__main__':main()

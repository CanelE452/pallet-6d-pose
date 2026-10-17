"""Prepare small, auditable public derivatives; never run git or copy raw banks.

Private originals remain immutable.  Only an explicit numeric/prose allowlist
is transformed, with path aliases for presentation and both byte hashes in a
derivation receipt.  RGB/checkpoints/candidate pose journals are excluded.
Image publication additionally requires a separately supplied root approval
manifest; a successful renderer is not itself publication permission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .common import read, sha, write

REQUIRED = (
    'PLAN_LOCK.json','RUN_STATE.json','ARM_REGISTRY.json','SOURCE_BINDINGS.json',
    'LITERATURE_TO_IMPLEMENTATION.md','OBSERVATION_AND_OWNER_CONTRACT.md',
    'DATA_SPLITS_AND_LABEL_QUALITY.md','METHOD_A.md','METHOD_B.md','METHOD_L.md',
    'INPUT_REQUESTS.md','TRAINING_MANIFEST.json','PREDICTION_MANIFEST.json',
    'COMPARISON_ALL_ARMS.csv','PAIRED_TRANSITIONS.csv','CANDIDATE_STAGE_DIAGNOSIS.json',
    'COST_AND_EXECUTION.json','VALIDATION.json','FINAL_REPORT_KO.md',
    'FINAL_DECISION.json','REPRODUCE.md')
REPORT_FILES = (
    'RUN_STATE.json','SOURCE_BINDINGS.json','TRAINING_MANIFEST.json',
    'PREDICTION_MANIFEST.json','COST_AND_EXECUTION.json','VALIDATION.json',
    'FINAL_REPORT_KO.md','FINAL_DECISION.json','REPRESENTATIVE_CASES.json','AMBIGUITY_POSTHOC.json')
EVALUATION_FILES = (
    'COMPARISON_ALL_ARMS.csv','ROW_METRICS.csv','PAIRED_TRANSITIONS.csv',
    'CANDIDATE_STAGE_DIAGNOSIS.json','RESULTS.json')
PROTOCOL_FILES = (
    'PLAN_LOCK.json','ARM_REGISTRY.json','CONFIG_FROZEN.json','INPUT_CONTRACT_REPAIR_V1.json',
    'SOURCE_SYMMETRY_PROSE_CORRECTION.json','TECHNICAL_FEATURE_PACKAGING_AMENDMENT.json',
    'L_PRELABEL_SYMMETRY_DESCRIPTION_AMENDMENT.json',
    'RUNTIME_WORKER_SCHEDULING_AMENDMENT.json',
    'ARM_EXECUTION_STATUS.json','ENVIRONMENT.json',
    'FINAL_PRE_SCORING_TESTS.json','FINAL_PRE_SCORING_TESTS_REPAIRED.json',
    'POSTSEAL_TEST_ISOLATION_REPAIR.json','P6_POSTTRAIN_VALIDATION_AMENDMENT.json',
    'ROOT_ACTUAL_VISUAL_REVIEW.json','PUBLICATION_PROSE_REPAIR_V2.json')
FORBIDDEN_BASENAMES = {'PREDICTIONS.jsonl.gz','SCORED_ROWS.jsonl.gz','HEAD_FEATURES.jsonl.gz',
                       'SYNTH_GT_LABELS.jsonl.gz','TYPED_PACKETS.jsonl.gz','RESULT.jsonl.gz'}


def aliases_for(run, source_root, config):
    values={str(Path(run).resolve()):'$RUN_PRIVATE',
            str(Path(source_root).resolve()):'$SOURCE_ROOT'}
    for key,name in (('old_private','$OLD_STATE_PRIVATE'),):
        if config.get(key):values[str(Path(config[key]).resolve())]=name
    # Alias the remaining machine-specific prefixes without changing numeric
    # values, row IDs, relative source paths or file content in private storage.
    values[str(Path.home())]='$PRIVATE_HOME'
    return dict(sorted(values.items(),key=lambda p:-len(p[0])))


def redact_text(value, aliases):
    result=value
    for original,alias in sorted(aliases.items(),key=lambda p:-len(p[0])):
        result=result.replace(original,alias)
    # A receipt can refer to an old worktree or another account.  Only local
    # absolute prefixes are redacted; HTTP(S) citations remain untouched.
    result=re.sub(r'(?<![A-Za-z0-9:])/(?:home|tmp|mnt|opt)/[^\s"\'<>|,;\]\[(){}]+',
                  lambda m:'$LOCAL_PATH/'+m.group(0).rstrip('/').split('/')[-1],result)
    return result


def redact_value(value, aliases):
    if isinstance(value,str):return redact_text(value,aliases)
    if isinstance(value,list):return [redact_value(v,aliases) for v in value]
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            newkey=redact_text(key,aliases)
            if newkey in result:raise ValueError('Path redaction key collision')
            result[newkey]=redact_value(item,aliases)
        return result
    return value


def _write_unchanged_or_new(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes()!=data:
            raise FileExistsError('Preserve existing public artifact; use a new publication directory: '+str(path))
        return
    with path.open('xb') as f:f.write(data)


def _safe_target(output, relative):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Public target must stay below output namespace')
    result=output/rel
    if not result.resolve().is_relative_to(output.resolve()):raise ValueError('Public symlink path escape')
    return result


def _derive(source,target,aliases,max_bytes):
    source,target=Path(source),Path(target)
    if source.name in FORBIDDEN_BASENAMES or source.suffix not in ('.json','.csv','.md','.txt'):
        raise ValueError('Not an allowed small numeric/prose artifact: '+source.name)
    if source.stat().st_size>max_bytes:raise ValueError('Oversized public derivative needs a reviewed summary: '+source.name)
    original_digest=sha(source)
    if source.suffix=='.json':
        value=read(source)
        _reject_private_pose_payload(value)
        value=redact_value(value,aliases)
        data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    else:data=redact_text(source.read_text(),aliases).encode()
    _write_unchanged_or_new(target,data)
    return dict(private_source=str(source),private_sha256=original_digest,
                public_file=target.name,public_sha256=sha(target),public_bytes=len(data),
                operation='presentation-only local-path aliases; JSON formatting; numeric values/IDs unchanged')


def _reject_private_pose_payload(value):
    """Fail closed if an accounting receipt accidentally includes a pose bank."""
    if isinstance(value,dict):
        if {'R_physical','actual_pose','native_xyz','object_xyz','centroid'} & set(value):
            raise ValueError('Full pose/geometry payload belongs in private storage')
        for item in value.values():_reject_private_pose_payload(item)
    elif isinstance(value,list):
        for item in value:_reject_private_pose_payload(item)


def _approved_assets(manifest_path, output, max_bytes):
    if manifest_path is None:return []
    manifest=read(manifest_path)
    if manifest.get('root_publication_approval') is not True:
        raise PermissionError('Root image publication approval must be explicit')
    derived=[]
    for entry in manifest.get('assets',[]):
        if entry.get('public_permission_confirmed') is not True:
            raise PermissionError('Missing publication permission for a curated asset')
        source=Path(entry['source']);target=_safe_target(output,entry['target'])
        if (source.suffix.lower() not in ('.png','.jpg','.jpeg','.webp','.svg','.html')
                or sha(source)!=entry['sha256'] or source.stat().st_size>max_bytes):
            raise ValueError('Curated asset type/hash/size mismatch')
        data=source.read_bytes();_write_unchanged_or_new(target,data)
        derived.append(dict(private_source=str(source),private_sha256=entry['sha256'],
                            public_file=str(target.relative_to(output)),public_sha256=sha(target),
                            public_bytes=len(data),operation='exact root-approved asset copy',
                            permission_receipt_sha256=sha(manifest_path)))
    return derived


def _static_prose_derivatives(run,output,supplied,aliases,max_bytes):
    """Redact only this campaign's public prose, preserving exact originals.

    Scientific JSON lock bytes and Python sources are deliberately untouched.
    Already-authored new Markdown can contain local example paths. Its first
    version is copied to a private immutable snapshot before presentation-only
    redaction. Repeated execution accepts only that same original/derivative.
    """
    originals=run/'public_prose_originals_v1';result=[]
    for name in REQUIRED:
        if not name.endswith('.md') or name in supplied:continue
        public=output/name
        if public.stat().st_size>max_bytes:raise ValueError('Oversized static prose: '+name)
        private=originals/name
        if not private.exists():
            private.parent.mkdir(parents=True,exist_ok=True)
            with private.open('xb') as f:f.write(public.read_bytes())
        expected=redact_text(private.read_text(),aliases).encode()
        current=public.read_bytes()
        if current not in (private.read_bytes(),expected):
            raise ValueError('Concurrent/static documentation changed after its private snapshot: '+name)
        if current!=expected:
            # The exact current original was preserved above; scope is this
            # newly-created research namespace, never existing user documents.
            public.write_bytes(expected)
        result.append(dict(private_source=str(private),private_sha256=sha(private),public_file=name,
                           public_sha256=sha(public),public_bytes=len(expected),
                           operation='new-namespace prose path aliases only; exact original preserved privately'))
    return result


def run(run_dir, *, report_dir=None, output_dir=None, source_root=None,
        approved_assets_json=None, publish_GO=False,max_artifact_bytes=16*1024**2,
        max_total_bytes=64*1024**2,receipt_path=None):
    if publish_GO is not True:
        raise PermissionError('Root explicit safe-publication GO is required before reading report results')
    run=Path(run_dir).resolve()
    source_root=Path(source_root or Path(__file__).resolve().parents[3]).resolve()
    report=Path(report_dir or run/'report_v1').resolve()
    output=Path(output_dir or source_root/'_docs/experiments/pallet_statewise_end_to_end_recovery_v1').resolve()
    if output==run or output.is_relative_to(run):
        raise ValueError('Publication is a separate public namespace, not a private-original rewrite')
    report_seal=read(report/'REPORT_SEAL.json')
    if report_seal.get('complete') is not True or report_seal.get('explicit_root_report_GO') is not True:
        raise ValueError('A completed explicit-GO report is required')
    for name,digest in report_seal['artifacts'].items():
        path=(report/name).resolve()
        if not path.is_relative_to(report) or sha(path)!=digest:raise ValueError('Private report bytes changed')
    evaluation=Path(report_seal['evaluation_dir'])
    if (sha(evaluation/'EVALUATION_SEAL.json')!=report_seal['evaluation_seal_sha256']
            or sha(run/'PREDICTION_SEAL.json')!=report_seal['prediction_seal_sha256']
            or sha(run/'PLAN_LOCK.json')!=report_seal['plan_sha256']
            or sha(report_seal['independent_receipt'])!=report_seal['independent_receipt_sha256']):
        raise ValueError('Publication hash chain changed after reporting')
    # These imports contain no scoring calls; repeat the current seal/checker
    # chain to reject a stale/edited numerical derivative.
    from .report import verify_report_inputs, verify_synthetic_head_evaluation
    verify_report_inputs(run,evaluation,report_seal['independent_receipt'],report_GO=True)
    synthetic_evaluation=verify_synthetic_head_evaluation(run)
    synthetic_sha=synthetic_evaluation['seal_sha256'] if synthetic_evaluation else None
    if report_seal.get('synthetic_head_evaluation_seal_sha256')!=synthetic_sha:
        raise ValueError('Synthetic head evaluation changed after reporting')
    config=read(run/'CONFIG_FROZEN.json');aliases=aliases_for(run,source_root,config)
    sources=[(report/name,name) for name in REPORT_FILES]
    sources += [(report/name,name) for name in ('MECHANISM_DIAGNOSIS.json','MECHANISM_DIAGNOSIS_KO.md','MECHANISM_SEAL.json')
                if (report/name).is_file()]
    sources += [(report/name,name) for name in ('SYNTHETIC_LEARNED_EVALUATION.json','SYNTHETIC_LEARNED_EVALUATION_KO.md')
                if (report/name).is_file()]
    sources += [(report/name,name) for name in (
        'ROUNDING_DIRECTION_AUDIT.json','INDEPENDENT_CHECK_V1_FAILED.json','INDEPENDENT_CHECK_CURRENT.json',
        'PREDICTION_SCORE_AUDIT.json','POSITIVE_MASS_DIAGNOSIS.json','POSITIVE_MASS_DIAGNOSIS_KO.md',
        'POSITIVE_MASS_ROWS.csv','POSITIVE_MASS_SEAL.json','INDEPENDENT_MASS_COUNT_CHECK.json',
        'BOUNDARY_EXTREME_FORENSICS_PUBLIC.json','BOUNDARY_EXTREME_FORENSICS_KO.md',
        'BOUNDARY_EXTREME_FORENSIC_SEAL.json','CONTRACT_VALIDITY_NOTE.json','CONTRACT_VALIDITY_NOTE_KO.md',
        'CONTRACT_NOTE_SEAL.json','ROOT_RESULT_INTERPRETATION_KO.md') if (report/name).is_file()]
    sources += [(evaluation/name,name) for name in EVALUATION_FILES]
    sources += [(run/name,name) for name in PROTOCOL_FILES if (run/name).is_file()]
    supplied={name for _,name in sources}
    missing=[name for name in REQUIRED if name not in supplied and not (output/name).is_file()]
    if missing:
        raise RuntimeError('Publication waiting for required documentation; no files copied: '+', '.join(missing))
    output.mkdir(parents=True,exist_ok=True)
    derivatives=[]
    for source,name in sources:
        derivatives.append(_derive(source,_safe_target(output,name),aliases,max_artifact_bytes))
    derivatives+=_static_prose_derivatives(run,output,supplied,aliases,max_artifact_bytes)
    # A full alias map with original private prefixes is private; the public
    # manifest exposes symbolic names and both hashes, not the original paths.
    assets=_approved_assets(approved_assets_json,output,max_artifact_bytes)
    derivatives+=assets
    total=sum(v['public_bytes'] for v in derivatives)
    if total>max_total_bytes:raise ValueError('Public derivative size budget exceeded; do not stage it')
    missing=[name for name in REQUIRED if not (output/name).is_file()]
    static={name:sha(output/name) for name in REQUIRED if (output/name).is_file()}
    public_derivations=redact_value(derivatives,aliases)
    numeric_files={}
    for numeric in sorted(output.glob('metrics_gallery*')):
        if not numeric.is_dir() or not (numeric/'NUMERIC_GALLERY_RECEIPT.json').is_file():continue
        numeric_receipt=read(numeric/'NUMERIC_GALLERY_RECEIPT.json')
        if numeric_receipt.get('RGB_files_copied')!=0 or numeric_receipt.get('tail_values_removed')!=0:
            raise ValueError('Public numeric export cannot contain RGB or removed tails')
        for name in ('index.html','NUMERIC_GALLERY_RECEIPT.json'):
            numeric_files[numeric.name+'/'+name]=sha(numeric/name)
    manifest=dict(schema='SMALL_PUBLIC_DERIVATION_MANIFEST_V1',complete=not missing,
                  derived_files=public_derivations,required_artifact_status={name:'PRESENT' if name not in missing else 'MISSING' for name in REQUIRED},
                  required_public_file_sha256=static,public_derivative_bytes=total,
                  separately_generated_public_numeric_file_sha256=numeric_files,
                  private_report_seal_sha256=sha(report/'REPORT_SEAL.json'),
                  source_namespace='scripts/research/pallet_statewise_end_to_end_recovery_v1',
                  source_files={str(p.relative_to(source_root)):sha(p) for p in sorted((source_root/'scripts/research/pallet_statewise_end_to_end_recovery_v1').glob('*.py'))},
                  path_aliases=sorted(set(aliases.values())),
                  public_reproduction='source and small numeric results available; raw RGB, historical caches, checkpoints and full pose banks remain private',
                  no_git_mutation=True,raw_private_files_copied=False,
                  root_curated_assets=len(assets))
    _write_unchanged_or_new(output/'PUBLICATION_MANIFEST.json',
        (json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    private_receipt=Path(receipt_path).resolve() if receipt_path is not None else run/'PUBLICATION_RECEIPT.json'
    private_value=dict(schema='PRIVATE_PUBLICATION_DERIVATION_RECEIPT_V1',
          complete=not missing,public_output=str(output),aliases=aliases,derived_files=derivatives,
          publication_manifest_sha256=sha(output/'PUBLICATION_MANIFEST.json'),
          missing_required_artifacts=missing,no_git_mutation=True)
    if private_receipt.exists():
        if read(private_receipt)!=private_value:raise ValueError('Existing publication receipt has different derivations')
    else:write(private_receipt,private_value)
    if missing:
        raise RuntimeError('Public package is explicitly incomplete; missing: '+', '.join(missing))
    return dict(complete=True,public_output=str(output),public_derivative_bytes=total,
                manifest_sha256=sha(output/'PUBLICATION_MANIFEST.json'),git_performed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--report-dir');p.add_argument('--output-dir');p.add_argument('--source-root')
    p.add_argument('--receipt-path')
    p.add_argument('--approved-assets-json');p.add_argument('--publish-GO',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.run_dir,report_dir=a.report_dir,output_dir=a.output_dir,source_root=a.source_root,
          approved_assets_json=a.approved_assets_json,publish_GO=a.publish_GO,receipt_path=a.receipt_path),ensure_ascii=False))


if __name__=='__main__':main()

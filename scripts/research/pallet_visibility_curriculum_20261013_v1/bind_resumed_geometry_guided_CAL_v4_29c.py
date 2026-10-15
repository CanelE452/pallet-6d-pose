"""Add resume provenance to a NEW copy of sealed CAL metadata, without labels.

The canonical writer, its groups/report bytes and frozen noise plan/fitter are
unchanged. The noise prepare API accepts this new folder explicitly, so its
normal source/input union binds the verified resume receipt before fitting.
Only JSON metadata is decoded here. Large journals/spools are hashed as bytes.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
OBS=BASE/'geometry_guided_edge_observations_v4_29b'
CANONICAL=BASE/'geometry_guided_native_evidence_v4_29_checks'
OUTPUT=BASE/'geometry_guided_native_evidence_resume_bound_v4_29c'


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _read(path):
    with Path(path).open(encoding='utf-8')as stream:return json.load(stream)


def _write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def _relative(path,root):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def _binding(path,root):
    return dict(path=_relative(path,root),sha256=_sha(path))


def _verify_binding(value,path,root):
    if (Path(root)/value['path']).resolve()!=Path(path).resolve()or value['sha256']!=_sha(path):
        raise ValueError('Resume artifact binding changed: '+str(path))


def _verify_map(mapping,root):
    for path,digest in mapping.items():
        if _sha(Path(root)/path)!=digest:raise ValueError('Frozen resume source/input changed: '+path)


def _union(*maps):
    result={}
    for mapping in maps:
        for path,digest in mapping.items():
            if path in result and result[path]!=digest:raise ValueError('Source/input ancestry conflict: '+path)
            result[path]=digest
    return result


def verify_resume_closure(observation_folder=OBS,*,root=ROOT):
    """No journal row, image, model or reference value is decoded."""
    root=Path(root).resolve();folder=Path(observation_folder).resolve()
    paths={name:folder/name for name in ('PROTOCOL.json','PREPARATION_SEAL.json',
        'CAL_EXECUTION_PROTOCOL.json','CAL_OBSERVATION_SEAL.json','CAL_OBSERVATIONS.jsonl.gz',
        'CAL_SUMMARY.json','CAL_ROW_INDEX.json','CAL_RESUME_PREFLIGHT_SEAL_V1.json',
        'CAL_RESUME_EXECUTION_PROTOCOL_V1.json','CAL_RESUME_EXECUTION_SEAL_V1.json',
        'CAL_RESUME_COMPLETION_SUPPLEMENT_V1.json')}
    protocol=_read(paths['PROTOCOL.json']);prepared=_read(paths['PREPARATION_SEAL.json'])
    cal=_read(paths['CAL_OBSERVATION_SEAL.json']);pre=_read(paths['CAL_RESUME_PREFLIGHT_SEAL_V1.json'])
    execution=_read(paths['CAL_RESUME_EXECUTION_PROTOCOL_V1.json'])
    sealed=_read(paths['CAL_RESUME_EXECUTION_SEAL_V1.json']);supp=_read(paths['CAL_RESUME_COMPLETION_SUPPLEMENT_V1.json'])
    ids=cal['ordered_ids'];existing=pre['exact_existing_entries'];missing=pre['missing_CAL_ids'];new=sealed['new_row_entries']
    if (cal['rows']!=46 or len(ids)!=46 or len(set(ids))!=46
            or prepared['rows']!=238 or prepared['ordered_ids'][:46]!=ids
            or len(existing)!=32 or len(missing)!=14 or len(set(missing))!=14
            or set(existing)&set(missing)or set(existing)|set(missing)!=set(ids)
            or set(new)!=set(missing)or execution['missing_CAL_ids']!=missing
            or sealed['rows']!=14 or sealed['original_rows_reused']!=32 or sealed['final_CAL_rows']!=46
            or supp['rows']!=46 or supp['exact_preboot_rows']!=32 or supp['new_resumed_rows']!=14):
        raise ValueError('Exact disjoint32+14 ordered CAL46 resume inventory required')
    if (not cal['complete_CAL46_includes_exact32_preboot_plus14_resumed']
            or not sealed['original32_immutable']or not sealed['same_observe_one_NMS_TLS_guides_graph']
            or not supp['no_old_observation_remeasurement']or supp['producer_math_changes']!=0
            or execution['pixel_NMS_TLS_guides_graph_or_guard_changes']!=0):
        raise ValueError('Exact frozen worker and unchanged32 resume contract required')
    if (cal['protocol_sha256']!=_sha(paths['PROTOCOL.json'])
            or cal['preparation_seal_sha256']!=_sha(paths['PREPARATION_SEAL.json'])
            or prepared['protocol_sha256']!=_sha(paths['PROTOCOL.json'])
            or cal['execution_protocol_sha256']!=_sha(paths['CAL_EXECUTION_PROTOCOL.json'])):
        raise ValueError('Original preboot observation protocol/preparation/execution binding changed')
    for key,name in [('observations_sha256','CAL_OBSERVATIONS.jsonl.gz'),('summary_sha256','CAL_SUMMARY.json'),
            ('row_index_sha256','CAL_ROW_INDEX.json')]:
        if cal[key]!=_sha(paths[name]):raise ValueError('Final CAL payload changed: '+name)
    for value,name in [(cal['resume_execution_seal'],'CAL_RESUME_EXECUTION_SEAL_V1.json'),
            (cal['resume_preflight_seal'],'CAL_RESUME_PREFLIGHT_SEAL_V1.json'),
            (sealed['execution_protocol'],'CAL_RESUME_EXECUTION_PROTOCOL_V1.json'),
            (sealed['preflight'],'CAL_RESUME_PREFLIGHT_SEAL_V1.json'),
            (execution['preflight_seal'],'CAL_RESUME_PREFLIGHT_SEAL_V1.json'),
            (supp['resume_execution_seal'],'CAL_RESUME_EXECUTION_SEAL_V1.json'),
            (supp['CAL_observation_seal'],'CAL_OBSERVATION_SEAL.json'),
            (supp['original_protocol'],'PROTOCOL.json'),(supp['original_preparation'],'PREPARATION_SEAL.json'),
            (supp['original_execution'],'CAL_EXECUTION_PROTOCOL.json')]:
        _verify_binding(value,paths[name],root)
    source=pre['source_sha256']
    if any(value!=source for value in (cal['resume_source_sha256'],execution['source_sha256'],
            sealed['source_sha256'],supp['source_sha256'])):
        raise ValueError('Resume source maps must agree')
    for value in (pre['plan'],pre['restart_receipt']):
        path=root/value['path'];_verify_binding(value,path,root);paths['extra:'+value['path']]=path
    plan=_read(root/pre['plan']['path'])
    if plan['source_sha256']!=source or plan['restart_receipt']!=pre['restart_receipt']:
        raise ValueError('Resume plan source/restart ancestry changed')
    for value in (execution['plan'],sealed['plan']):
        if value!=pre['plan']:raise ValueError('Resume declaration must agree')
    # Retain explicit hashes of every original immutable file and both spool
    # sets. These are byte checks; no actual observation row is decoded.
    immutable=pre['immutable_file_sha256'];_verify_map(immutable,root)
    for ident,entry in {**existing,**new}.items():
        name=hashlib.sha256(ident.encode()).hexdigest()+'.jsonl.gz'
        path=folder/'row_spool'/name
        if entry['id']!=ident or entry['partition']!='CAL'or _sha(path)!=entry['sha256']:
            raise ValueError('Exact durable resumed/original CAL spool required')
        paths['spool:'+ident]=path
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify_map(protocol[key],root)
    source=_union(protocol['source_sha256'],source);_verify_map(source,root)
    bindings=_union(protocol['input_sha256'],immutable,{_relative(p,root):_sha(p)for p in paths.values()})
    return dict(schema='VERIFIED_EXACT32_PLUS14_CAL_RESUME_PROVENANCE_V29C',rows=46,original_rows=32,new_rows=14,
        ordered_ids=ids,source_sha256=source,input_sha256=bindings,
        protected_user_sha256=protocol['protected_user_sha256'],journal_rows_or_numeric_references_decoded=0,
        new_images_NN_measurement_pose_or_noise_fits=0)


def prepare_bound_metadata(canonical_metadata_folder=CANONICAL,output=OUTPUT,observation_folder=OBS,*,
        root=ROOT,root_metadata_GO=False):
    if root_metadata_GO is not True:raise PermissionError('Separate root metadata-copy GO required before IO')
    root=Path(root).resolve();canonical=Path(canonical_metadata_folder).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every bound metadata generation')
    closure=verify_resume_closure(observation_folder,root=root)
    prep=_read(canonical/'PREPARATION_SEAL.json');seal=_read(canonical/'CAL_PREFLIGHT_SEAL_V1.json')
    report=_read(canonical/'CAL_PREFLIGHT_V1.json')
    groups=canonical/'CAL_PREPARED_GROUPS.jsonl.gz'
    if (seal['preparation_sha256']!=_sha(canonical/'PREPARATION_SEAL.json')
            or seal['report_sha256']!=_sha(canonical/'CAL_PREFLIGHT_V1.json')
            or seal['prepared_groups_sha256']!=_sha(groups)
            or report['error_count']!=0 or report['successful_metadata_rows']!=46
            or prep['ids']!=closure['ordered_ids']):raise ValueError('Canonical successful ordered CAL46 metadata required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify_map(prep[key],root)
    if prep['protected_user_sha256']!=closure['protected_user_sha256']:raise ValueError('Protected map conflict')
    own=Path(__file__).resolve();own_source={_relative(own,root):_sha(own)}
    sources=_union(prep['source_sha256'],closure['source_sha256'],own_source)
    canonical_inputs={_relative(p,root):_sha(p)for p in
        (canonical/'PREPARATION_SEAL.json',canonical/'CAL_PREFLIGHT_SEAL_V1.json',canonical/'CAL_PREFLIGHT_V1.json',groups)}
    inputs=_union(prep['input_sha256'],closure['input_sha256'],canonical_inputs)
    output.mkdir(parents=True)
    for name in ('CAL_PREFLIGHT_V1.json','CAL_PREPARED_GROUPS.jsonl.gz'):
        shutil.copyfile(canonical/name,output/name)
        if _sha(canonical/name)!=_sha(output/name):raise ValueError('Canonical payload bytes must remain exact')
    receipt={**closure,'UTC':datetime.now(timezone.utc).isoformat(),
        'canonical_metadata_bindings':canonical_inputs,'source_sha256':sources,
        'canonical_groups_and_report_byte_identical':True,'no_policy_axis_q_covariance_or_population_change':True,
        'noise_fitter_API':'frozen prepare(prepared=this new folder); original plan/fitter/model output unchanged'}
    _write(output/'RESUME_SOURCE_CLOSURE_RECEIPT_V1.json',receipt)
    boundprep={**prep,'source_sha256':sources,'input_sha256':_union(inputs,
        {_relative(output/'RESUME_SOURCE_CLOSURE_RECEIPT_V1.json',root):_sha(output/'RESUME_SOURCE_CLOSURE_RECEIPT_V1.json')}),
        'resume_source_closure':_binding(output/'RESUME_SOURCE_CLOSURE_RECEIPT_V1.json',root),
        'canonical_metadata_unchanged':True,'new_numeric_label_reads':0}
    _write(output/'PREPARATION_SEAL.json',boundprep)
    _write(output/'CAL_PREFLIGHT_SEAL_V1.json',dict(preparation_sha256=_sha(output/'PREPARATION_SEAL.json'),
        report_sha256=_sha(output/'CAL_PREFLIGHT_V1.json'),prepared_groups_sha256=_sha(output/'CAL_PREPARED_GROUPS.jsonl.gz')))
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify_map(boundprep[key],root)
    return _binding(output/'CAL_PREFLIGHT_SEAL_V1.json',root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation-folder',type=Path,default=OBS)
    parser.add_argument('--canonical-metadata-folder',type=Path,default=CANONICAL)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--root-metadata-GO',action='store_true');args=parser.parse_args()
    print(prepare_bound_metadata(args.canonical_metadata_folder,args.output,args.observation_folder,
        root_metadata_GO=args.root_metadata_GO))

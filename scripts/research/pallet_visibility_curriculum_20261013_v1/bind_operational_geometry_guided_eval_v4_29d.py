"""Bind exact REST execution ancestry to a NEW canonical evaluation prepare.

Run the unchanged v29 prepare into UNBOUND first. This helper copies its
INPUTS bytes, preserves all inference settings, and adds verified operational
source/input ancestry before any predictions. No observation, model, label,
reference or prediction row is decoded here. Existing prepares are preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

from . import bind_resumed_geometry_guided_CAL_v4_29c as W

ROOT=W.ROOT
OBS=W.OBS
UNBOUND=W.BASE/'geometry_guided_edge_eval_unbound_preparation_v4_29d'
OUTPUT=W.BASE/'geometry_guided_edge_eval_v4_29'
CONTROLLER='scripts/research/pallet_visibility_curriculum_20261013_v1/evaluate_geometry_guided_edges_v4_29.py'
CONTROLLER_SHA256='5aff5081b54bee36966c5d2de5d6b140369c301a63d40b586d6232310433b1df'
ARMS=['SAME_FROZEN_E','GEOMETRY_EDGE_ALL_BI_NATIVE','GEOMETRY_EDGE_ALL_BI_CONTAINMENT',
      'GEOMETRY_EDGE_SOURCE_NATIVE','GEOMETRY_EDGE_SOURCE_CONTAINMENT']
EXEC_SCHEMA='EXACT_GRAPH_OPERATIONAL_REST_EXECUTION_SEAL_V29D'
COMPLETE_SCHEMA='EXACT_GRAPH_OPERATIONAL_ALL238_COMPLETION_SUPPLEMENT_V29D'
MAPS=('source_sha256','input_sha256','protected_user_sha256','original_immutable_file_sha256')


def verify_operational_closure(observation_folder=OBS,*,root=ROOT):
    root=Path(root).resolve();obs=Path(observation_folder).resolve()
    paths={name:obs/name for name in ('PROTOCOL.json','PREPARATION_SEAL.json',
        'OBSERVATION_SEAL.json','OBSERVATIONS.jsonl.gz','REST_OBSERVATION_SEAL.json',
        'REST_OBSERVATIONS.jsonl.gz','CAL_OBSERVATION_SEAL.json',
        'REST_OPERATIONAL_PREPARATION_SEAL_V1.json','REST_OPERATIONAL_EXECUTION_SEAL_V1.json',
        'REST_OPERATIONAL_COMPLETION_SUPPLEMENT_V1.json')}
    protocol=W._read(paths['PROTOCOL.json']);prep=W._read(paths['PREPARATION_SEAL.json'])
    final=W._read(paths['OBSERVATION_SEAL.json']);rest=W._read(paths['REST_OBSERVATION_SEAL.json'])
    execution=W._read(paths['REST_OPERATIONAL_EXECUTION_SEAL_V1.json'])
    complete=W._read(paths['REST_OPERATIONAL_COMPLETION_SUPPLEMENT_V1.json'])
    ids=final['ordered_ids'];cal=W._read(paths['CAL_OBSERVATION_SEAL.json'])
    if (execution['schema']!=EXEC_SCHEMA or complete['schema']!=COMPLETE_SCHEMA
            or execution['rows']!=192 or complete['rows']!=192 or complete['final_rows']!=238
            or final['rows']!=238 or len(ids)!=238 or len(set(ids))!=238
            or ids!=prep['ordered_ids'] or cal['rows']!=46 or cal['ordered_ids']!=ids[:46]
            or rest['rows']!=192 or rest['ordered_ids']!=ids[46:]):
        raise ValueError('Exact CAL46 plus REST192 ordered all238 operational closure required')
    for name in MAPS:
        if execution[name]!=complete[name]:raise ValueError('Operational completion/execution map conflict: '+name)
        W._verify_map(complete[name],root)
    if (protocol['protected_user_sha256']!=complete['protected_user_sha256']
            or final['source_sha256']!=complete['source_sha256']):
        raise ValueError('Exact final observation source/protected union required')
    for name in ('source_sha256','input_sha256','protected_user_sha256'):
        W._verify_map(protocol[name],root)
    W._union(protocol['source_sha256'],complete['source_sha256'])
    if not all(complete['source_sha256'].get(k)==v for k,v in protocol['source_sha256'].items()):
        raise ValueError('Operational source union must retain every original source')
    if (final['protocol_sha256']!=W._sha(paths['PROTOCOL.json'])
            or prep['protocol_sha256']!=W._sha(paths['PROTOCOL.json'])
            or final['preparation_seal_sha256']!=W._sha(paths['PREPARATION_SEAL.json'])
            or final['observations_sha256']!=W._sha(paths['OBSERVATIONS.jsonl.gz'])
            or rest['observations_sha256']!=W._sha(paths['REST_OBSERVATIONS.jsonl.gz'])
            or final['partition_seals']['CAL']!=W._sha(paths['CAL_OBSERVATION_SEAL.json'])
            or final['partition_seals']['REST']!=W._sha(paths['REST_OBSERVATION_SEAL.json'])):
        raise ValueError('Final all238 and partition payload bindings changed')
    for value,name in (
            (complete['execution_seal'],'REST_OPERATIONAL_EXECUTION_SEAL_V1.json'),
            (complete['final_all238_observation_seal'],'OBSERVATION_SEAL.json'),
            (complete['REST_observation_seal'],'REST_OBSERVATION_SEAL.json'),
            (execution['REST_observation_seal'],'REST_OBSERVATION_SEAL.json'),
            (complete['preparation_seal'],'REST_OPERATIONAL_PREPARATION_SEAL_V1.json'),
            (execution['preparation_seal'],'REST_OPERATIONAL_PREPARATION_SEAL_V1.json'),
            (final['operational_execution_seal'],'REST_OPERATIONAL_EXECUTION_SEAL_V1.json')):
        W._verify_binding(value,paths[name],root)
    if complete['actual_replay_proof']!=execution['actual_replay_proof']:
        raise ValueError('Actual replay proof must agree')
    proof=complete['actual_replay_proof'];proof_path=root/proof['path']
    W._verify_binding(proof,proof_path,root)
    proof_seal=W._read(proof_path);proof_result=proof_path.parent/'RESULT.json'
    if proof_seal['result_sha256']!=W._sha(proof_result):raise ValueError('Actual replay result changed')
    result=W._read(proof_result)
    if (result['failure_count']!=0 or len(result['cases'])!=2
            or not all(r['exact_original_optimized_stored'] for r in result['cases'])):
        raise ValueError('Both complete actual CAL graphs must exactly match before REST optimization')
    for case in result['cases']:
        methods=case['methods']
        if (len(methods)!=2 or [m['method'] for m in methods]!=['ORIGINAL','BOUNDED_HEAP_FRESH_MEMO']
                or len({m['canonical_graph_sha256'] for m in methods})!=1
                or not all(m['exact_full_graph_vs_stored'] and m['source_bytes_unchanged']
                    and m['producer_globals_unchanged'] and m['canonical_graph_sha256']==m['stored_graph_sha256']
                    and m['new_RGB_NMS_TLS_NN_reference_posefit_or_startsearch_calls']==0 for m in methods)):
            raise ValueError('Both complete actual CAL method records must prove stored graph equality')
    inputs=W._union(protocol['input_sha256'],complete['input_sha256'],
        complete['original_immutable_file_sha256'],{W._relative(p,root):W._sha(p) for p in paths.values()},
        {W._relative(proof_path,root):W._sha(proof_path),W._relative(proof_result,root):W._sha(proof_result)})
    return dict(ordered_ids=ids,rows=238,REST_rows=192,CAL_rows=46,
        source_sha256=complete['source_sha256'],input_sha256=inputs,
        protected_user_sha256=complete['protected_user_sha256'],
        original_immutable_file_sha256=complete['original_immutable_file_sha256'],
        completion_supplement=W._binding(paths['REST_OPERATIONAL_COMPLETION_SUPPLEMENT_V1.json'],root),
        observation_seal=W._binding(paths['OBSERVATION_SEAL.json'],root),
        actual_replay_proof=proof,all_observation_rows_and_numeric_references_decoded=0)


def prepare_bound_evaluation(unbound=UNBOUND,output=OUTPUT,observation_folder=OBS,*,root=ROOT,root_prepare_GO=False):
    if root_prepare_GO is not True:raise PermissionError('Root bound-evaluation prepare GO required before IO')
    root=Path(root).resolve();unbound=Path(unbound).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every existing evaluation preparation')
    protocol=W._read(unbound/'PROTOCOL.json');seal=W._read(unbound/'INPUT_SEAL.json')
    inputs_file=unbound/'INPUTS.jsonl.gz';ids=protocol['ids']
    if (seal['rows']!=238 or len(ids)!=238 or len(set(ids))!=238 or seal['ordered_ids']!=ids
            or seal['protocol_sha256']!=W._sha(unbound/'PROTOCOL.json')
            or seal['input_sha256']!=W._sha(inputs_file) or protocol['arms']!=ARMS
            or protocol['source_sha256'].get(CONTROLLER)!=CONTROLLER_SHA256
            or protocol['full_pose_TP_topnormal_prior_weights']!=0.
            or protocol['old_e_numeric_pose_added_as_prior_or_fallback'] is not False):
        raise ValueError('Unchanged frozen five-arm all238 prepare required')
    if any((unbound/name).exists() for name in ('PREDICTION_SEAL.json','PREDICTIONS.jsonl.gz','frames','SCORE_SEAL.json')):
        raise ValueError('Bind ancestry before any predictions or metrics')
    for name in ('source_sha256','input_sha256','protected_user_sha256'):W._verify_map(protocol[name],root)
    closure=verify_operational_closure(observation_folder,root=root)
    if ids!=closure['ordered_ids'] or protocol['protected_user_sha256']!=closure['protected_user_sha256']:
        raise ValueError('Exact unchanged population/protected contract required')
    own_paths=[Path(__file__).resolve(),Path(W.__file__).resolve(),
               Path(__file__).with_name('test_bind_operational_geometry_guided_eval_v4_29d.py').resolve()]
    sources=W._union(protocol['source_sha256'],closure['source_sha256'],
        {W._relative(p,root):W._sha(p) for p in own_paths})
    unbound_paths=[unbound/'PROTOCOL.json',unbound/'INPUT_SEAL.json',inputs_file]
    ancestry_inputs=W._union(protocol['input_sha256'],closure['input_sha256'],
        {W._relative(p,root):W._sha(p) for p in unbound_paths})
    output.mkdir(parents=True);shutil.copyfile(inputs_file,output/'INPUTS.jsonl.gz')
    if W._sha(output/'INPUTS.jsonl.gz')!=seal['input_sha256']:raise ValueError('Exact input bytes required')
    receipt={**closure,'schema':'EXACT_OPERATIONAL_ANCESTRY_BOUND_EVALUATION_PREPARE_V29D',
        'UTC':datetime.now(timezone.utc).isoformat(),'source_sha256':sources,
        'unbound_preparation_bindings':{W._relative(p,root):W._sha(p) for p in unbound_paths},
        'all238_input_bytes_unchanged':True,'inference_arms_parameters_points_states_noise_or_routing_changes':0,
        'new_RGB_NN_posefit_reference_or_prediction_row_decodes':0}
    receipt_path=output/'OPERATIONAL_SOURCE_CLOSURE_RECEIPT_V1.json';W._write(receipt_path,receipt)
    bound={**protocol,'source_sha256':sources,'input_sha256':W._union(ancestry_inputs,
        {W._relative(receipt_path,root):W._sha(receipt_path)}),
        'operational_source_closure':W._binding(receipt_path,root),
        'original_five_arm_inference_and_all238_input_bytes_unchanged':True}
    W._write(output/'PROTOCOL.json',bound)
    for relative,digest in sources.items():
        dest=output/'code_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/relative,dest)
        if W._sha(dest)!=digest:raise ValueError('Exact evaluation source snapshot required')
    for name in ('source_sha256','input_sha256','protected_user_sha256'):W._verify_map(bound[name],root)
    W._write(output/'INPUT_SEAL.json',{**seal,'protocol_sha256':W._sha(output/'PROTOCOL.json'),
        'operational_source_closure_sha256':W._sha(receipt_path)})
    return W._binding(output/'INPUT_SEAL.json',root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unbound',type=Path,default=UNBOUND);parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--observation-folder',type=Path,default=OBS);parser.add_argument('--root-prepare-GO',action='store_true')
    args=parser.parse_args()
    print(prepare_bound_evaluation(args.unbound,args.output,args.observation_folder,root_prepare_GO=args.root_prepare_GO))

"""Same v29 REST192 observations with proven-exact rank scheduling only.

The original29b protocol/preparation and completed CAL46 remain immutable.
The frozen29b spool loop and original V.observe_one code are privately reused;
only its P._rank_fresh implementation has bounded heap and rank-local memo.
No candidate/threshold/generation/math/NN/pose/label change is introduced.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import shutil
from types import FunctionType,SimpleNamespace

from . import observe_geometry_guided_edges_v4_29b as B
from . import exact_graph_operational_facade_v4_29 as A

C,V,OLD=B.C,B.V,B.OLD
OUTPUT=B.OUTPUT
PLAN=OUTPUT.parent/'geometry_guided_exact_REST_plan_v4_29d/PREDECLARATION.json'
PROOF=OUTPUT.parent/'exact_graph_actual_CAL_replay_v4_29'
METADATA=OUTPUT.parent/'geometry_guided_native_evidence_v4_29_checks'
PREPARATION='REST_OPERATIONAL_PREPARATION_SEAL_V1.json'
EXECUTION='REST_OPERATIONAL_EXECUTION_PROTOCOL_V1.json'
EXECUTION_SEAL='REST_OPERATIONAL_EXECUTION_SEAL_V1.json'
COMPLETION='REST_OPERATIONAL_COMPLETION_SUPPLEMENT_V1.json'
_WORKER=None


def _merge(*maps):
    result={}
    for mapping in maps:
        for relative,digest in mapping.items():
            if relative in result and result[relative]!=digest:raise ValueError('Frozen operational source/input conflict: '+relative)
            result[relative]=digest
    return result


def _verify(mapping):
    for relative,digest in mapping.items():
        if C.sha(C.ROOT/relative)!=digest:raise ValueError('Frozen operational binding changed: '+relative)


def _proof():
    seal=C.read(PROOF/'SEAL.json');result=C.read(PROOF/'RESULT.json');plan=C.read(PROOF/'PLAN.json')
    if (C.sha(PROOF/'RESULT.json')!=seal['result_sha256']or C.sha(PROOF/'PLAN.json')!=seal['plan_sha256']
            or result['failure_count']!=0 or len(result['cases'])!=2
            or not all(r['exact_original_optimized_stored']for r in result['cases'])
            or[result['cases'][i]['id']for i in range(2)]!=[r['id']for r in plan['cases']]):
        raise ValueError('Both fixed CAL entire-graph actual replays must be exact')
    prep=C.read(PROOF/'PREPARATION_SEAL.json')
    if C.sha(PROOF/'PREPARATION_SEAL.json')!=seal['preparation_seal_sha256']:raise ValueError('Actual replay source preparation changed')
    _verify(prep['source_sha256']);_verify(prep['input_sha256']);_verify(prep['protected_user_sha256'])
    return prep


def _sources():
    modules=(A,A.B,A.M,A.P,A.P.G,A.P.ORDER,B,V)
    result={str(Path(m.__file__).resolve().relative_to(C.ROOT)):C.sha(m.__file__)for m in modules}
    for path in(Path(__file__),Path(__file__).with_name('test_observe_geometry_guided_exact_REST_v4_29d.py')):
        result[str(path.resolve().relative_to(C.ROOT))]=C.sha(path)
    return result


def declare():
    if PLAN.exists():raise FileExistsError('Preserve operational declaration')
    proof=_proof();A.verify_frozen_sources();PLAN.parent.mkdir(parents=True,exist_ok=False)
    C.write(PLAN,dict(schema='EXACT_GRAPH_OPERATIONAL_REST_PREDECLARATION_V29D',
        UTC=datetime.now(timezone.utc).isoformat(),rows=192,final_rows=238,
        source_sha256=_merge(proof['source_sha256'],_sources()),actual_replay_proof=OLD._binding(PROOF/'SEAL.json'),
        canonical_output=str(OUTPUT.relative_to(C.ROOT)),original_worker='observe_geometry_guided_edges_v4_29.observe_one',
        worker_model='same original V._init_worker; capture existing _MODEL once per child',
        private_namespace_hook='P._rank_fresh only; fresh memo per rank, stable heap, bounded LRU',
        same_pixel_guides_NMS_TLS_all_alternatives_graph_math_thresholds_states_noise_and_generation=True,
        candidate_pruning_or_new_pose_NN_label_calls=0,new_CAL_remeasurement=0,
        original_CAL46_and_protocols_immutable=True,actual_observation_requires_separate_root_GO=True,
        runtime_generation='FRESH_GUIDED_NMS_TLS_PARTIAL_DIRECTED_GRAPH_WINNER_V29',
        source_binding='conservative inherited union plus operational sources; not every bound source executes in every frame',
        observer_CAL_metadata46_success_required_before_prepare=True,
        final_closure='EXECseal binds RESTseal; finalOBSseal binds EXECseal; COMPLETION binds finalOBSseal with no reverse cycle'))
    print(dict(plan=str(PLAN),sha256=C.sha(PLAN)),flush=True)


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output!=OUTPUT.resolve():raise ValueError('Keep canonical29b observation path')
    if(output/PREPARATION).exists():raise FileExistsError('Preserve operational preparation')
    plan=C.read(PLAN);proof=_proof();_verify(plan['source_sha256']);protocol,prep=B._load(output)
    OLD._verify_observation_seal(output,'CAL')
    meta=C.read(METADATA/'CAL_PREFLIGHT_V1.json');metaseal=C.read(METADATA/'CAL_PREFLIGHT_SEAL_V1.json')
    if(meta['rows']!=46 or meta['error_count']!=0 or meta['successful_metadata_rows']!=46
            or C.sha(METADATA/'CAL_PREFLIGHT_V1.json')!=metaseal['report_sha256']
            or C.sha(METADATA/'CAL_PREPARED_GROUPS.jsonl.gz')!=metaseal['prepared_groups_sha256']):
        raise ValueError('Unchanged canonical CAL46 metadata must first complete with zero failures')
    if C.sha(PROOF/'SEAL.json')!=plan['actual_replay_proof']['sha256']:raise ValueError('Declared actual replay proof changed')
    for name in('REST_EXECUTION_PROTOCOL.json','REST_OBSERVATION_SEAL.json','OBSERVATION_SEAL.json',EXECUTION,EXECUTION_SEAL,COMPLETION):
        if(output/name).exists():raise FileExistsError('Never replace an existing REST/final execution')
    immutable_names=['PROTOCOL.json','PREPARATION_SEAL.json','INPUTS.jsonl.gz','REUSE_SEAL.json',
        'CAL_EXECUTION_PROTOCOL.json','CAL_OBSERVATION_SEAL.json','CAL_OBSERVATIONS.jsonl.gz','CAL_ROW_INDEX.json','CAL_SUMMARY.json',
        'CAL_RESUME_PREFLIGHT_SEAL_V1.json','CAL_RESUME_EXECUTION_PROTOCOL_V1.json',
        'CAL_RESUME_EXECUTION_SEAL_V1.json','CAL_RESUME_COMPLETION_SUPPLEMENT_V1.json']
    immutable={str((output/name).relative_to(C.ROOT)):C.sha(output/name)for name in immutable_names}
    entries=C.read(output/'CAL_ROW_INDEX.json')['entries']
    if len(entries)!=46 or[r['id']for r in entries]!=prep['ordered_ids'][:46]:raise ValueError('All original ordered CAL46 required')
    immutable.update({str(B._row_path(output,r['id']).relative_to(C.ROOT)):r['sha256']for r in entries})
    immutable.update({str((METADATA/name).relative_to(C.ROOT)):C.sha(METADATA/name)for name in
        ('PREPARATION_SEAL.json','CAL_PREFLIGHT_V1.json','CAL_PREFLIGHT_SEAL_V1.json','CAL_PREPARED_GROUPS.jsonl.gz')})
    # Any additional source bytes of the CAL resume are binding ancestry, not
    # a change to the separate CAL noise population or fitted parameters.
    resume=C.read(output/'CAL_RESUME_COMPLETION_SUPPLEMENT_V1.json')
    sources=_merge(protocol['source_sha256'],proof['source_sha256'],plan['source_sha256'],resume['source_sha256'])
    inputs=_merge(protocol['input_sha256'],immutable)
    for path in(PLAN,PROOF/'PLAN.json',PROOF/'PREPARATION_SEAL.json',PROOF/'RESULT.json',PROOF/'SEAL.json'):
        inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    for mapping in(sources,inputs,protocol['protected_user_sha256']):_verify(mapping)
    C.write(output/PREPARATION,dict(schema='EXACT_GRAPH_OPERATIONAL_REST_PREPARATION_SEAL_V29D',
        rows=192,final_rows=238,ordered_REST_ids=prep['ordered_ids'][46:],source_sha256=sources,input_sha256=inputs,
        protected_user_sha256=protocol['protected_user_sha256'],original_immutable_file_sha256=immutable,
        plan=OLD._binding(PLAN),actual_replay_proof=OLD._binding(PROOF/'SEAL.json'),
        completed_before_new_REST_RGB_queries=True,original_CAL46_and_protocols_immutable=True))
    for relative,digest in sources.items():
        target=output/'exact_graph_operational_code_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Operational source snapshot mismatch')
    print(dict(prepared_REST=192,source_files=len(sources),CAL46_immutable=True),flush=True)


def _init_worker():
    global _WORKER
    V._init_worker();_WORKER=A.make_observe_one()


def _observe_one(task):
    if _WORKER is None:raise RuntimeError('Original worker initializer must run first')
    return _WORKER(task)


def make_partition_observer():
    """Frozen spool/queue loop with top-level picklable private workers."""
    namespace=dict(B.observe.__globals__)
    namespace['V']=SimpleNamespace(__file__=V.__file__,_init_worker=_init_worker,observe_one=_observe_one)
    observe=FunctionType(B.observe.__code__,namespace,B.observe.__name__,B.observe.__defaults__,B.observe.__closure__)
    observe.__kwdefaults__=deepcopy(B.observe.__kwdefaults__)
    return observe


def observe(output=OUTPUT,workers=3,*,root_actual_REST_GO=False):
    if not root_actual_REST_GO:raise ValueError('Separate root actual REST192 GO required')
    output=Path(output).resolve()
    if output!=OUTPUT.resolve():raise ValueError('Keep canonical29b observation path')
    prep=C.read(output/PREPARATION)
    for name in(EXECUTION,EXECUTION_SEAL,COMPLETION):
        if(output/name).exists():raise FileExistsError('Preserve every operational REST execution')
    for key in('source_sha256','input_sha256','protected_user_sha256','original_immutable_file_sha256'):_verify(prep[key])
    C.write(output/EXECUTION,dict(schema='EXACT_GRAPH_OPERATIONAL_REST_EXECUTION_PROTOCOL_V29D',
        UTC=datetime.now(timezone.utc).isoformat(),workers=workers,maximum_pending_tasks=workers,
        root_actual_REST_GO=True,source_sha256=prep['source_sha256'],preparation_seal=OLD._binding(output/PREPARATION),
        actual_replay_proof=prep['actual_replay_proof'],unchanged_observe_one_with_private_P_rank_hook=True))
    make_partition_observer()(output,'REST',workers)
    for key in('source_sha256','input_sha256','protected_user_sha256','original_immutable_file_sha256'):_verify(prep[key])
    inputs=dict(prep['input_sha256'])
    for name in(PREPARATION,EXECUTION,'REST_EXECUTION_PROTOCOL.json','REST_OBSERVATIONS.jsonl.gz',
                'REST_SUMMARY.json','REST_ROW_INDEX.json','REST_OBSERVATION_SEAL.json'):
        inputs[str((output/name).relative_to(C.ROOT))]=C.sha(output/name)
    C.write(output/EXECUTION_SEAL,dict(schema='EXACT_GRAPH_OPERATIONAL_REST_EXECUTION_SEAL_V29D',rows=192,
        source_sha256=prep['source_sha256'],input_sha256=inputs,protected_user_sha256=prep['protected_user_sha256'],
        original_immutable_file_sha256=prep['original_immutable_file_sha256'],
        preparation_seal=OLD._binding(output/PREPARATION),actual_replay_proof=prep['actual_replay_proof'],
        REST_observation_seal=OLD._binding(output/'REST_OBSERVATION_SEAL.json'),
        all_original_CAL46_immutable=True,all192_rows_and_abstentions_preserved=True,
        source_or_observation_generation_threshold_candidate_or_math_changes=0))
    print(dict(sealed_REST192=True,execution_seal_sha256=C.sha(output/EXECUTION_SEAL)),flush=True)


def finalize(output=OUTPUT):
    output=Path(output).resolve()
    if output!=OUTPUT.resolve():raise ValueError('Keep canonical29b observation path')
    if(output/COMPLETION).exists():raise FileExistsError('Preserve final operational completion')
    execution=C.read(output/EXECUTION_SEAL)
    for key in('source_sha256','input_sha256','protected_user_sha256','original_immutable_file_sha256'):_verify(execution[key])
    B.finalize(output)
    seal=C.read(output/'OBSERVATION_SEAL.json')
    seal.update(source_sha256=execution['source_sha256'],operational_execution_seal=OLD._binding(output/EXECUTION_SEAL),
        operational_exact_graph_source_union_is_additive=True,exact_original_rank_math_and_generation_preserved=True)
    C.write(output/'OBSERVATION_SEAL.json',seal)
    completion={key:deepcopy(execution[key])for key in('source_sha256','input_sha256','protected_user_sha256',
        'original_immutable_file_sha256','preparation_seal','actual_replay_proof','REST_observation_seal')}
    completion.update(schema='EXACT_GRAPH_OPERATIONAL_ALL238_COMPLETION_SUPPLEMENT_V29D',rows=192,final_rows=238,
        execution_seal=OLD._binding(output/EXECUTION_SEAL),final_all238_observation_seal=OLD._binding(output/'OBSERVATION_SEAL.json'),
        all_original_CAL46_immutable=True,no_parent_source_or_existing_observation_rewritten=True,
        completed_original_generation_with_only_exact_graph_scheduling_adapter=True)
    C.write(output/COMPLETION,completion)
    print(dict(final_all238=True,completion_sha256=C.sha(output/COMPLETION)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('declare','prepare','observe','finalize'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--root-actual-REST-GO',action='store_true');args=parser.parse_args()
    if args.stage=='declare':declare()
    elif args.stage=='prepare':prepare(args.output)
    elif args.stage=='observe':observe(args.output,args.workers,root_actual_REST_GO=args.root_actual_REST_GO)
    else:finalize(args.output)

"""Predeclare then replay two frozen CAL line graphs, no fresh RGB/pose/labels.

Only the existing target-mask NPZ is decoded, to reproduce exactly the old
global exterior and visibility context. Actual NMS/TLS arrays and pose guides
come from the completed row. Baseline and optimized cases run in sequential
subprocesses, each with a fresh process RSS peak and a fresh rank-local memo.
"""
from __future__ import annotations
import os
for _key in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import numpy as np

from . import common as C
from . import exact_graph_operational_facade_v4_29 as A

P,G,VIS=A.P,A.P.G,A.P.VIS
B=C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
OBS=B/'geometry_guided_edge_observations_v4_29b'
HOST=C.DOC/'publication_20261015_v1/HOST_RESTART_AND_PARTIAL_CAL.json'
OUTPUT=B/'exact_graph_actual_CAL_replay_v4_29'
FIELDS=('native_xyz','R_cf','centroid','physical_registration_Q','K','image_hw','initial_self_hidden')
MODULE='scripts.research.pallet_visibility_curriculum_20261013_v1.replay_exact_graph_actual_CAL_v4_29'


def _binding(path):return dict(path=str(Path(path).resolve().relative_to(C.ROOT)),sha256=C.sha(path))
def _verify(mapping):
    for relative,digest in mapping.items():
        if C.sha(C.ROOT/relative)!=digest:raise ValueError('Frozen replay binding changed: '+relative)
def _canonical_chunks(value):
    return json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(C.finite(value))
def _canonical_sha(value):
    digest=hashlib.sha256()
    for text in _canonical_chunks(value):digest.update(text.encode())
    return digest.hexdigest()


def choose_cases(host,protocol):
    """Metadata order only; no rows, image contents, errors or score selection."""
    partial=host['partial_observation'];completed=partial['completed_CAL_ids_original_order']
    ids=protocol['ids'][:46]
    if not ids or ids[0]not in completed:raise ValueError('Original first CAL must already be completed')
    first_wood=next(identifier for identifier in ids if identifier.startswith('wood_')and identifier in completed)
    byid={r['id']:r for r in partial['spools']}
    return [dict(label=label,**deepcopy(byid[identifier]))for label,identifier in
            (('FIRST_ORDERED_CAL',ids[0]),('FIRST_COMPLETED_WOOD_CAL_IN_ORIGINAL_ORDER',first_wood))]


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every replay preparation')
    A.verify_frozen_sources();host=C.read(HOST);protocol=C.read(OBS/'PROTOCOL.json')
    cases=choose_cases(host,protocol);wanted={r['id']:r for r in cases}
    remaining=set(wanted)
    # CAL-prefix streaming only: input metadata binds masks/first geometry;
    # no completed observation row or image/mask is decoded during prepare.
    stream=iter(C.rows(OBS/'INPUTS.jsonl.gz'))
    for _ in range(46):
        task=next(stream)
        if task['partition']!='CAL':raise ValueError('CAL prefix only')
        if task['id']in remaining:
            case=wanted[task['id']]
            case.update(mask_binding=task['mask_binding'],image_sha256=task['image_sha256'],
                initial_geometry_digest=G._digest({key:task['initial_geometry'][key]for key in FIELDS}),
                unchanged_observation_math_sha256=task['unchanged_observation_math_sha256'])
            remaining.remove(task['id'])
        if not remaining:break
    if remaining:raise ValueError('Both fixed CAL cases required')
    modules=(sys.modules[__name__],A,A.B,A.M,P,G,P.ORDER,VIS,G.NMS,C)
    sources={str(Path(m.__file__).resolve().relative_to(C.ROOT)):C.sha(m.__file__)for m in modules}
    for name in('test_exact_graph_operational_facade_v4_29.py','test_replay_exact_graph_actual_CAL_v4_29.py'):
        path=Path(__file__).with_name(name);sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    paths=[HOST,OBS/'PROTOCOL.json',OBS/'PREPARATION_SEAL.json',OBS/'INPUTS.jsonl.gz']
    for case in cases:paths.extend((C.ROOT/case['path'],C.ROOT/case['mask_binding']['path']))
    inputs={str(p.relative_to(C.ROOT)):C.sha(p)for p in paths}
    for case in cases:
        if C.sha(C.ROOT/case['path'])!=case['sha256']:raise ValueError('Frozen original completed row changed')
        if C.sha(C.ROOT/case['mask_binding']['path'])!=case['mask_binding']['sha256']:raise ValueError('Bound old target mask changed')
    protected=protocol['protected_user_sha256']
    for mapping in(sources,inputs,protected):_verify(mapping)
    output.mkdir(parents=True)
    plan=dict(schema='TWO_FIXED_CAL_EXACT_GRAPH_REPLAY_PLAN_V1',UTC=datetime.now(timezone.utc).isoformat(),
        cases=cases,methods=['ORIGINAL','BOUNDED_HEAP_FRESH_MEMO'],configuration=protocol['configuration'],
        selection='original first CAL and original-order first completed wood; metadata only, no error/cost routing',
        comparison='entire canonical decorated joint_graph including all alternatives/diagnostics, against stored original and each other',
        guides='reconstruct saved guide_geometries in original insertion order; require exact regenerated guide_bank before ranking',
        target_mask='existing immutable NPZ pallet_mask_clean; same exterior and visibility contexts; no RGB decode/remap or NMS/TLS fit',
        execution='sequential isolated subprocess: original then optimized per case; rank time and ru_maxrss',
        cache_limits=dict(local_term_entries=4096,intersection_entries=4096),
        parent_source_or_module_globals_mutated=False,new_RGB_NN_labels_posefit_or_startsearch=0,
        original_observation_rows_or_resume29c_modified=False,actual_replay_requires_separate_root_GO=True,
        noREST_optimized_adoption_before_valid_smoke=True)
    C.write(output/'PLAN.json',plan)
    C.write(output/'PREPARATION_SEAL.json',dict(plan_sha256=C.sha(output/'PLAN.json'),source_sha256=sources,
        input_sha256=inputs,protected_user_sha256=protected,completed_observation_rows_decoded=0,
        actual_mask_or_RGB_decodes=0,before_replay_execution=True))
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Replay source snapshot mismatch')
    print(dict(prepared=2,ids=[r['id']for r in cases],candidate_counts=[r['summary']['actual_chain_hypotheses']for r in cases],
        source_sha256=sources,plan_sha256=C.sha(output/'PLAN.json')),flush=True)


def restore_pose_guides(row):
    saved=row['observation']['guide_geometries'];states=row['states'];guides=[]
    if G._digest(P._geometry_record(row['initial_geometry'],states))!=G._digest(saved['ORIGINAL_INITIAL']):
        raise ValueError('Original stored guide geometry differs')
    for name,record in saved.items():
        if name=='ORIGINAL_INITIAL':continue
        geometry={key:deepcopy(record[key])for key in FIELDS}
        geometry['initial_projection_xy']=[None if geometry['initial_self_hidden'][k]or states[k]=='S'
            else deepcopy(record['nonS_model_projection_xy'][str(k)])for k in range(8)]
        if G._digest(P._geometry_record(geometry,states))!=G._digest(record):raise ValueError('Stored guide reconstruction changed')
        guides.append(dict(guide_id=name,source_kind='ORIGINAL_TASK_COARSE'if name.endswith('/OWN_INITIAL_COARSE')
            else'UNIFORM_INITIAL_PROPOSAL_BANK',geometry=geometry))
    return guides


def decorate_graph(graph,chain_observation,cfg):
    """Exactly the frozen producer's post-rank support audit and ref metadata."""
    P._joint_support_audit(graph,cfg)
    refs={}
    for ri,ray in enumerate(chain_observation['edge_records']):
        for ci,line in enumerate(ray.get('chains',[])):
            refs[str(ri)+':'+str(ci)]=dict(ray_index=ri,chain_index=ci,native_corner=ray['native_corner'],
                native_neighbor=ray['native_neighbor'],guide_reference=line['guide_reference'],
                guide_reference_id=line['guide_reference_id'],chain_id=line['chain_id'])
    for node in graph['line_candidates']:
        node.pop('line_record',None)
        node['chain_references']=[refs[key]for key in node['provenance']]
    return graph


def worker(output,case_index,method):
    if method not in('ORIGINAL','BOUNDED_HEAP_FRESH_MEMO'):raise ValueError('Exact predeclared replay method required')
    output=Path(output).resolve();plan=C.read(output/'PLAN.json');seal=C.read(output/'PREPARATION_SEAL.json')
    if C.sha(output/'PLAN.json')!=seal['plan_sha256']:raise ValueError('Exact predeclared plan required')
    for mapping in(seal['source_sha256'],seal['input_sha256'],seal['protected_user_sha256']):_verify(mapping)
    case=plan['cases'][case_index];folder=output/('case'+str(case_index))/method;folder.mkdir(parents=True)
    before_sources={str(Path(m.__file__).resolve()):C.sha(m.__file__)for m in(P,G,P.ORDER,A.B,A.M)}
    before_globals={k:id(v)for k,v in P.__dict__.items()}
    started=time.monotonic();rows=iter(C.rows(C.ROOT/case['path']));row=next(rows,None)
    if row is None or next(rows,None)is not None:
        raise ValueError('One complete frozen spool row required')
    if row['id']!=case['id']or row['partition']!='CAL'or _canonical_sha(row)!=case['canonical_row_sha256']:
        raise ValueError('Exact single frozen CAL observation required')
    if row['unchanged_observation_math_sha256']!=case['unchanged_observation_math_sha256']:raise ValueError('Original input archive changed')
    if G._digest({key:row['initial_geometry'][key]for key in FIELDS})!=case['initial_geometry_digest']:raise ValueError('First geometry changed')
    guides=restore_pose_guides(row);geometry=row['initial_geometry'];observation=row['observation']
    binding=dict(sha256=row['image_sha256'],image_hw=geometry['image_hw'])
    regenerated=G.prepare_joint_geometry_guides(geometry,row['states'],row['visible_measurements'],
        image_binding=binding,configuration=plan['configuration'],pose_guides=guides)
    if _canonical_sha(regenerated)!=_canonical_sha(observation['guide_bank']):raise ValueError('Exact saved original pose-guide bank required')
    with np.load(C.ROOT/case['mask_binding']['path'],allow_pickle=False)as data:mask=data['pallet_mask_clean'].copy()
    if VIS.mask_pixel_sha256(mask)!=observation['target_mask_pixel_sha256']:raise ValueError('Actual old target-mask pixels changed')
    context=VIS.prepare_edge_visibility(mask,geometry,row['states'],row['visible_measurements'],image_binding=binding,
        occluder_mask=None,occluder_binding=None)
    diagnostics=[];rank=P._rank_fresh if method=='ORIGINAL'else A.make_rank_adapter(diagnostics=diagnostics)
    rank_start=time.monotonic();graph=rank(observation['chain_observation'],geometry,row['states'],row['visible_measurements'],
        image_binding=binding,configuration=plan['configuration'],pose_guides=guides,pallet_mask=mask,visibility_context=context)
    rank_seconds=time.monotonic()-rank_start;rank_peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cfg,_=G._config(plan['configuration']);decorate_graph(graph,observation['chain_observation'],cfg)
    actual_sha=_canonical_sha(graph);stored_sha=_canonical_sha(observation['joint_graph'])
    with gzip.open(folder/'CANONICAL_GRAPH.json.gz','wt',compresslevel=3)as stream:
        for text in _canonical_chunks(graph):stream.write(text)
    report=dict(id=row['id'],method=method,rank_seconds=rank_seconds,rank_peak_RSS_kib=rank_peak_RSS_kib,
        total_seconds=time.monotonic()-started,canonical_graph_sha256=actual_sha,stored_graph_sha256=stored_sha,
        exact_full_graph_vs_stored=actual_sha==stored_sha,full_graph_includes_alternatives_and_diagnostics=True,
        actual_candidates=len(graph['line_candidates']),joint_states_evaluated=graph['joint_states_evaluated'],
        selected_line_names=graph['selected_line_names'],operational_diagnostics=diagnostics,
        mask_decodes=1,new_RGB_NMS_TLS_NN_reference_posefit_or_startsearch_calls=0,
        saved_pose_guides_restored_without_new_search=True,
        source_bytes_unchanged=all(C.sha(p)==sha for p,sha in before_sources.items()),
        producer_globals_unchanged=before_globals=={k:id(v)for k,v in P.__dict__.items()},
        frozen_row_sha256_after=C.sha(C.ROOT/case['path']))
    C.write(folder/'RESULT.json',report)
    print(dict(id=row['id'],method=method,seconds=rank_seconds,RSS_KiB=rank_peak_RSS_kib,exact=actual_sha==stored_sha),flush=True)
    if not report['exact_full_graph_vs_stored']or not report['source_bytes_unchanged']or not report['producer_globals_unchanged']:
        raise SystemExit(1)


def run(output=OUTPUT,after_root_replay_go=False):
    if not after_root_replay_go:raise ValueError('Separate root actual-replay GO required')
    output=Path(output).resolve();plan=C.read(output/'PLAN.json');seal=C.read(output/'PREPARATION_SEAL.json')
    if(output/'EXECUTION_PROTOCOL.json').exists():raise FileExistsError('Preserve all actual replay attempts')
    if C.sha(output/'PLAN.json')!=seal['plan_sha256']:raise ValueError('Frozen replay plan required')
    C.write(output/'EXECUTION_PROTOCOL.json',dict(UTC=datetime.now(timezone.utc).isoformat(),
        explicit_root_replay_GO_recorded_by_invocation=True,parallel_workers=1,sequential_cases_and_methods=True,
        plan_sha256=seal['plan_sha256'],source_sha256=seal['source_sha256']))
    results=[]
    for index,_ in enumerate(plan['cases']):
        pair=[]
        for method in plan['methods']:
            completed=subprocess.run([sys.executable,'-m',MODULE,'worker','--output',str(output),
                '--case-index',str(index),'--method',method],check=False)
            if completed.returncode:raise RuntimeError('Actual replay subprocess failed; preserved all outputs')
            pair.append(C.read(output/('case'+str(index))/method/'RESULT.json'))
        equal=pair[0]['canonical_graph_sha256']==pair[1]['canonical_graph_sha256']
        results.append(dict(id=pair[0]['id'],exact_original_optimized_stored=equal and all(r['exact_full_graph_vs_stored']for r in pair),
            methods=pair,speed_ratio_original_over_optimized=pair[0]['rank_seconds']/pair[1]['rank_seconds'],
            peak_RSS_ratio_original_over_optimized=pair[0]['rank_peak_RSS_kib']/pair[1]['rank_peak_RSS_kib']))
    for mapping in(seal['source_sha256'],seal['input_sha256'],seal['protected_user_sha256']):_verify(mapping)
    report=dict(cases=results,failure_count=sum(not r['exact_original_optimized_stored']for r in results),
        entire_canonical_graph_equality=True,no_original_source_or_archive_edits=True,new_RGB_NN_labels_posefits=0,
        two_CAL_smoke_not_population_or_accuracy_claim=True,REST_optimization_not_installed=True)
    C.write(output/'RESULT.json',report);C.write(output/'SEAL.json',dict(result_sha256=C.sha(output/'RESULT.json'),
        plan_sha256=C.sha(output/'PLAN.json'),preparation_seal_sha256=C.sha(output/'PREPARATION_SEAL.json')))
    print(dict(cases=2,failure_count=report['failure_count'],results=[{k:r[k]for k in('id','exact_original_optimized_stored',
        'speed_ratio_original_over_optimized','peak_RSS_ratio_original_over_optimized')}for r in results]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','run','worker'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--after-root-replay-go',action='store_true')
    parser.add_argument('--case-index',type=int);parser.add_argument('--method',choices=('ORIGINAL','BOUNDED_HEAP_FRESH_MEMO'))
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output)
    elif args.stage=='run':run(args.output,args.after_root_replay_go)
    else:worker(args.output,args.case_index,args.method)

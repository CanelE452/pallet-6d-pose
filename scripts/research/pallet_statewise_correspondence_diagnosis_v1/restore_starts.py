"""Fill one missing prelabel raw-start log using its exact original generator.

Only the original _initial_poses is called. No minimize/least_squares, model,
new seed, altered start, reference, scoring, or complete pose replay occurs.
The original optimizer results remain immutable cached observations.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import inspect
import time

from .common import digest, finite, read, rows, sha, write, record_stage
from .typed_replay import _module, original_stage_inputs

IDENT = 'eval_cad:1778653003088339968'


def supplied_arguments(packet, family_name):
    solutions, explicit = original_stage_inputs(packet, family_name)
    coarse = packet['original_e_stage1_family_seed_bank'][family_name]['family']['coarse_pose']
    supplied = [dict(R=deepcopy(coarse['R']),t=deepcopy(coarse['t']),name='REGISTERED_COARSE_PRIOR')]
    supplied += deepcopy(explicit)
    for j, solution in enumerate(solutions):
        if solution.get('accepted'):
            supplied.append(dict(R=deepcopy(solution['R']),t=deepcopy(solution['t']),name=f'SAVED_{j}_WINNER'))
        supplied.extend(dict(R=deepcopy(s['R']),t=deepcopy(s['t']),name=f'SAVED_{j}_SAMPLE_{i}')
                        for i,s in enumerate(solution.get('feasible_pose_samples',[])))
    return supplied


def restore(*, folder, output):
    folder,output=Path(folder),Path(output)
    if output.exists():raise FileExistsError('Preserve supplemental logging attempts')
    started,cpu=time.monotonic(),time.process_time()
    extraction=read(folder/'e_trace_v2/EXTRACTION_SEAL.json')
    source=folder/'e_trace_v2/TYPED_PACKETS.jsonl.gz'
    if sha(source)!=extraction['files'][source.name]['sha256']:raise ValueError('Frozen original packet changed')
    packets=[p for p in rows(source) if p['id']==IDENT]
    if len(packets)!=1 or packets[0]['partition']!='CAL':raise ValueError('Only one fixed CAL raw-log gap allowed')
    packet=packets[0];spec=packet['solver_input']
    if spec['outline_observation']['supported']:raise ValueError('Only original unsupported-outline path')
    slot_path=folder/'slots/000_TYPED_MULTI.json';slot=read(slot_path)
    if slot['id']!=IDENT or slot['result'].get('technical_failure'):raise ValueError('Original same-input optimizer cache required')
    backend=_module('statewise_partial_BI_outline_solver_v4_13');V=_module('joint_ot_pnp')
    ledger=slot['result']['generated_candidates'];families={}
    for name,wanted in (('ORIGINAL_WD',51),('SWAPPED_WD',50)):
        family=packet['original_e_stage1_family_seed_bank'][name]['family']
        parsed=backend._prepare(family['native_xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],
            spec['image_hw'],family['coarse_pose'],spec['config'],spec['boundary_observations'],spec['u_regions'])
        xyz,K,_,measured,_,_,cfg,*_=parsed
        supplied=supplied_arguments(packet,name)
        inputs=finite(dict(xyz=xyz,K=K,measurements=measured,supplied_initializations=supplied,config=cfg))
        starts,search=V._initial_poses(xyz,K,measured,supplied,cfg)
        logged=[c['original_endpoint_record']['name'] for c in ledger if c['stage']=='G13' and c['family']==name
                and c['kind']=='partial_BI_seed_fit']
        old=next(f for f in slot['result']['stage_results']['G13']['families'] if f['name']==name)['solution']['search']
        if len(starts)!=wanted or [s['name'] for s in starts]!=logged:
            raise ValueError('Exact original initialization count/name/order parity failed: '+name)
        for key,value in search.items():
            if old.get(key)!=value:raise ValueError('Exact original generator search metadata differs: '+name+'/'+key)
        families[name]=dict(starts=finite(starts),search=search,exact_original_names_and_order=logged,
            initialization_count_parity=True,name_order_parity=True,search_metadata_parity=True,
            original_generator_inputs=inputs,input_sha256=digest(inputs),
            repeated_generator_calls=1,
            repeated_P3P_calls=search['p3p_triples_searched'],
            repeated_SQPNP_calls=int(len(measured)>=3),
            P3P_call_count_evidence='joint_ot_pnp.py increments searched once immediately before its only solveP3P call per nondegenerate triple',
            SQPNP_call_count_evidence='One unconditional try solvePnPGeneric when len(ids)>=3',
            cached_original_optimizer_results_reused=True)
    output.mkdir(parents=True)
    write(output/'RAW_START_SUPPLEMENT.json',dict(schema='ONE_CAL_EXACT_ORIGINAL_RAW_START_LOG_SUPPLEMENT_V1',
        id=IDENT,partition='CAL',stage='G13',families=families,GT_inputs=False,before_new_scoring=True,
        new_top_level_pose_slots=0,new_full_replays=0,new_minimize_calls=0,new_least_squares_calls=0,
        repeated_initial_generator_calls=2,repeated_P3P_calls=sum(f['repeated_P3P_calls'] for f in families.values()),
        repeated_SQPNP_calls=sum(f['repeated_SQPNP_calls'] for f in families.values()),
        new_start_policy_or_seed_or_budget=False,all_other_original_238_inputs_unchanged=True))
    write(output/'SUPPLEMENT_SEAL.json',dict(schema='ONE_CAL_RAW_START_LOG_SUPPLEMENT_SEAL_V1',
        id=IDENT,raw_start_supplement_sha256=sha(output/'RAW_START_SUPPLEMENT.json'),
        extraction_seal_sha256=sha(folder/'e_trace_v2/EXTRACTION_SEAL.json'),
        typed_packet_file_sha256=sha(source),typed_multi_slot_sha256=sha(slot_path),
        packet_sha256=digest(packet),source_sha256={'restore_starts.py':sha(__file__),
            'joint_ot_pnp.py':sha(V.__file__),'statewise_partial_BI_outline_solver_v4_13.py':sha(backend.__file__),
            'typed_replay.py':sha(Path(__file__).with_name('typed_replay.py'))},
        original_generator_function_sha256=digest(inspect.getsource(V._initial_poses)),
        GT_inputs=False,before_new_scoring=True,new_pose_fit_calls=0,new_top_level_slots=0,
        repeated_initial_generator_calls=2,repeated_P3P_calls=20,repeated_SQPNP_calls=2,
        elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu))
    record_stage(folder,'one_CAL_missing_raw_start_logging_repair',started,
        worker_cpu_seconds=time.process_time()-cpu,new_pose_fit_calls=0,new_top_level_slots=0,
        repeated_initial_generator_calls=2,repeated_P3P_calls=20,repeated_SQPNP_calls=2)
    return read(output/'SUPPLEMENT_SEAL.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',required=True);p.add_argument('--output',required=True)
    print(restore(**vars(p.parse_args())))

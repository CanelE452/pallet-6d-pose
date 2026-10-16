"""Sealed, reference-only candidate-stage diagnosis; never a pose selector.

The historical WD/P1 generation bank is incomplete. Positive existence is
observable, but absence in a recorded subset is never a global proposal
limit. `generated` means logged e-refinement endpoints, not all historical
PnP proposals or optimizer internal iterates. Reference evaluation is a
callback and is not imported/opened by this module.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse
import csv
import time
import numpy as np

from .candidates import pose_record
from .typed_replay import original_stage_inputs, _module

DIAGNOSTIC_IDS = ('wood_night_01:029428', 'plastic_day_01:010995',
                  'plastic_day_01:011640', 'wood_day_01:005249',
                  'wood_night_01:031426', 'wood_night_01:031344')
STAGES = ('generated', 'after_support', 'feasible', 'family', 'refinement', 'selected')


def consumer_coverage(original, replay, packet, *, parity_passed, initial_start_supplement=None):
    """Exact e-consumer scope, excluding WD/P1 proposals it never read.

    Reconstruct only archived arguments; this calls no PnP/optimizer. The one
    partial-only CAL producer's fresh raw P3P starts are deliberately unknown.
    """
    V = _module('joint_ot_pnp')
    cache = not any(b.get('explicit_complete_BI_likelihood_revocation') is True
                    for b in packet['solver_input']['boundary_observations'])
    incoming, audit = [], {}
    for stage in ('G10','G13'):
        audit[stage]=[]
        source_stage='G10' if stage=='G13' and cache else stage
        for family in original['stages'][stage]['families']:
            name,schema=family['name'],family['solver_schema']
            outline='outline_solver' in schema
            partial=schema=='statewise_partial_BI_u_solver_v4_13'
            active_partial=partial and family.get('original_search') is not None
            endpoints=[c for c in replay.get('generated_candidates',[]) if c['stage']==source_stage
                       and c['family']==name and (c['kind']=='outline_consider' if outline
                            else c['kind']=='partial_joint_fit')]
            expected=sum('error' not in r['original_record'] for r in family['original_outcomes'])
            numeric_complete=bool(parity_passed and (len(endpoints)==expected if outline or active_partial else True))
            starts=[];restored_raw=False;delegated_saved=False
            if outline or active_partial:
                solutions, supplied=original_stage_inputs(packet,name)
                saved=[]
                for j,solution in enumerate(solutions):
                    if solution.get('accepted'):
                        saved.append(dict(R=solution['R'],t=solution['t'],name=f'SAVED_{j}_WINNER'))
                    saved.extend(dict(R=s['R'],t=s['t'],name=f'SAVED_{j}_SAMPLE_{i}')
                                 for i,s in enumerate(solution.get('feasible_pose_samples',[])))
                if outline:
                    starts=saved+supplied
                else:
                    # Actual P13 supplied arguments; fresh P3P roots that came
                    # before these arguments in _initial_poses are not guessed.
                    coarse=packet['original_e_stage1_family_seed_bank'][name]['family']['coarse_pose']
                    starts=[dict(R=coarse['R'],t=coarse['t'],name='REGISTERED_COARSE_PRIOR')]+supplied+saved
                    if initial_start_supplement is not None and initial_start_supplement['id']==packet['id']:
                        restored=initial_start_supplement['families'][name]
                        if not all(restored[k] for k in ('initialization_count_parity','name_order_parity','search_metadata_parity')):
                            raise ValueError('Exact original initialization-generator supplement parity required')
                        if len(restored['starts'])!=family['original_search']['initialization_count']:
                            raise ValueError('Original raw initialization count changed')
                        starts=deepcopy(restored['starts']);restored_raw=True
            elif not outline and not active_partial:
                # Returning a saved accepted solution makes that exact pose
                # an incoming/output candidate, not a new optimizer endpoint.
                solutions,_=original_stage_inputs(packet,name)
                if solutions[0].get('accepted'):
                    starts=[dict(R=solutions[0]['R'],t=solutions[0]['t'],name='DELEGATED_FIRST_SAVED_SOLUTION')]
                    delegated_saved=True
            unique=[];scale=max(float(np.linalg.norm(np.ptp(family['native_xyz'],axis=0))),1e-3)
            for index,start in enumerate(starts):
                representative=next((old for old in unique if V._same_pose(
                    (np.asarray(start['R']),np.asarray(start['t'])),
                    (np.asarray(old['R']),np.asarray(old['t'])),scale)),None) if outline else None
                kept=(representative is None) if outline else True if restored_raw else None
                if outline and kept:unique.append(start)
                incoming.append(dict(stage=stage,family=name,kind=('delegated_saved_solver_output' if delegated_saved else
                    'restored_original_initial_generator_output' if restored_raw else 'incoming_archived_start_before_dedup'),
                    R=deepcopy(start['R']),t=deepcopy(start['t']),
                    original_endpoint_record=dict(initialization=start['name'],original_input_index=index,
                        original_dedup_kept=kept,
                        original_dedup_representative=None if representative is None else representative['name'],
                        actual_optimizer_start_membership_known=outline or restored_raw,
                        delegated_accepted_cached_pose_without_new_optimizer=delegated_saved,
                        source='Original accepted winner/complete feasible sample/explicit supplied argument; no new proposal')))
            audit[stage].append(dict(family=name,schema=schema,
                source_stage_alias=source_stage,original_consumer_numeric_endpoint_count=expected if outline or active_partial else 0,
                replay_logged_numeric_endpoint_count=len(endpoints),numeric_endpoints_complete=numeric_complete,
                explicit_archived_incoming_start_count=len(starts),
                incoming_starts_before_dedup_complete=bool(parity_passed and (not active_partial or restored_raw)),
                fresh_P3P_raw_starts_unknown=active_partial and not restored_raw,
                fresh_P3P_generator_output_restored=restored_raw,
                original_raw_initialization_count=(family.get('original_search') or {}).get('initialization_count'),
                original_backend_delegated_or_admission_return=not outline and not active_partial,
                historical_WD_P1_rejected_proposals_consumed=False))
    complete=all(a['numeric_endpoints_complete'] and a['incoming_starts_before_dedup_complete']
                 for stage in audit.values() for a in stage)
    return dict(original_e_consumer_generated_bank_complete=complete,
        original_e_consumer_numeric_endpoints_complete=all(a['numeric_endpoints_complete'] for s in audit.values() for a in s),
        historical_WD_P1_full_generation_complete=False,
        historical_rejected_WD_P1_outside_original_e_consumer=True,
        stages=audit,source='Original G10 lines197-203/G13 lines221-227 read accepted+feasible+explicit starts, never parent candidates',
        incoming_candidates=incoming)


def pose_from_native(R, t, family):
    """Center and physical registration are from this candidate's own family."""
    try:
        record = pose_record(dict(R=R, t=t), family)
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return dict(available=False, reason='original_logged_endpoint_has_no_finite_registered_pose')
    if record is None:
        return dict(available=False, reason='original_logged_endpoint_Rt_missing')
    return dict(available=True, R_cf=record['R_cf'], centroid=record['centroid'],
                R_physical=record['R_physical'],
                physical_registration_Q=deepcopy(family['physical_registration_Q']),
                cf_extents=deepcopy(family['cf_extents']))


def _summary(candidates):
    success = [c['candidate_id'] for c in candidates if c['metric']['success_le5'] is True]
    available = [c for c in candidates if c['metric']['pose_available']]
    resolved = [c for c in candidates if c['metric']['success_le5'] is not None]
    return dict(candidate_count=len(candidates), pose_available_count=len(available),
                reference_joint_resolved_count=len(resolved), joint_success_exists=bool(success),
                joint_success_candidate_ids=success,
                translation_min_cm=min((c['metric']['translation_cm'] for c in available
                    if c['metric']['translation_cm'] is not None), default=None),
                rotation_min_deg=min((c['metric']['rotation_deg'] for c in available
                    if c['metric']['rotation_deg'] is not None), default=None),
                separate_T_R_minima_do_not_form_a_virtual_joint_pose=True)


def single_fallback_audit(result, e0_pose):
    """Prediction-only check, separate from the oracle and physical references."""
    actual = result.get('actual_pose') or {}
    available = actual.get('available') is True
    diffs = {}
    if available:
        for key in ('R_cf', 'R_physical', 'centroid', 'physical_registration_Q', 'cf_extents'):
            if key in actual and key in e0_pose:
                diffs[key] = float(np.max(np.abs(np.asarray(actual[key])-e0_pose[key])))
    equal = bool(available and all(v <= 1e-9 for v in diffs.values()) and
                 {'R_cf', 'R_physical', 'centroid'} <= set(diffs))
    optimizer_endpoints = [c for c in result.get('generated_candidates', [])
        if c['kind'] != 'outline_consider' or
        c['original_endpoint_record'].get('optimizer_success') is not None]
    optimizer_count = result.get('optimizer_calls', len(optimizer_endpoints))
    return dict(pose_available=available, accepted=result.get('accepted'), fallback=result.get('fallback'),
        optimizer_endpoint_count=optimizer_count, no_optimizer_endpoint=optimizer_count == 0,
        actual_pose_matches_E0=equal, pose_component_atol=1e-9, actual_pose_maxabs_difference=diffs,
        fallback_matches_own_E0=(equal if result.get('fallback') is True else None),
        delegated_without_new_fit=result.get('audit', {}).get('delegated_without_new_single_fit'),
        strict_single_local_interpretation_blocked=bool(result.get('fallback') is True and not equal),
        fallback_source=actual.get('source'),
        note='No new bound or fallback policy is introduced; original backend fallback can differ from its E0 optimizer start.')


def analyze_frame(original, replay, packet, metric_for_pose, support_for_candidate=None,
                  *, replay_parity_passed=True, initial_start_supplement=None):
    """Return stage oracles without changing any prediction or candidate order.

    `replay` is the TYPED_MULTI result, not an E-after reobservation. Optional
    support callback evaluates the *original* visible/partial support
    predicates and returns {passed:bool|None,...}; it does not refit anything.
    Missing support evidence remains unknown and is never assumed feasible.
    """
    if original['id'] != packet['id'] or replay['id'] != packet['id']:
        raise ValueError('Exact original/replay/packet ID binding required')
    family_bank = {f['name']: f for f in original['stages']['G13']['families']}
    selected_family = original['original_e']['frozen_family']
    pools = {name: [] for name in STAGES}
    stage_pools = {stage: {name: [] for name in ('generated','after_support','feasible','selected')}
                   for stage in ('G10','G13')}
    records, support_unknown, refinement_pairs = [], [], []
    coverage=(consumer_coverage(original,replay,packet,parity_passed=replay_parity_passed,
        initial_start_supplement=initial_start_supplement) if 'original_e_stage1_family_seed_bank' in packet else
        dict(original_e_consumer_generated_bank_complete=False,original_e_consumer_numeric_endpoints_complete=False,
             historical_WD_P1_full_generation_complete=False,incoming_candidates=[],reason='no_bound_original_start_packet'))

    def add(pool, cid, pose, **fields):
        entry = dict(candidate_id=cid, pose=pose, metric=metric_for_pose(pose), **fields)
        pools[pool].append(entry)
        return entry

    # G10 and G13 have different observation/objective packages. Their oracle
    # ladders are separate; no union is called a pruning transition.
    for stage in ('G10','G13'):
        for family in original['stages'][stage]['families']:
            for index, sample in enumerate(family['feasible_samples']):
                cid = f'ORIGINAL_{stage}/{family["name"]}/feasible/{index}'
                pose = dict(available=True, **sample['pose'])
                entry = dict(candidate_id=cid,pose=pose,metric=metric_for_pose(pose),family=family['name'],
                             source='original_untouched_full_feasible_pose_samples')
                stage_pools[stage]['feasible'].append(entry)
            if family['name'] == original['stages'][stage]['selected_family']:
                pose=family['published_actual_pose']
                stage_pools[stage]['selected'].append(dict(candidate_id=f'ORIGINAL_{stage}/selected',
                    pose=pose,metric=metric_for_pose(pose),family=family['name']))
    pools['feasible']=stage_pools['G13']['feasible']
    pools['family']=[entry for entry in pools['feasible'] if entry['family']==selected_family]

    # Exact endpoint ledger. It includes duplicates before original feasible
    # dedup, so counts are endpoint counts; success existence is unaffected.
    generated_inputs=(list(replay.get('generated_candidates', []))+coverage['incoming_candidates']) if replay_parity_passed else []
    for index, candidate in enumerate(generated_inputs):
        family = family_bank[candidate['family']]
        cid = f'REPLAY_ENDPOINT/{index}'
        pose = pose_from_native(candidate['R'], candidate['t'], family)
        record = candidate['original_endpoint_record']
        entry = dict(candidate_id=cid, pose=pose, metric=metric_for_pose(pose),
                     family=candidate['family'],stage=candidate['stage'],
                     endpoint_kind=candidate['kind'],original_endpoint_record=record)
        stage_pools[candidate['stage']]['generated'].append(entry)
        support = (dict(passed=False, reason='logged_endpoint_pose_unavailable') if not pose['available'] else
                   support_for_candidate(candidate, packet) if support_for_candidate else
                   dict(passed=None, reason='separate_original_support_predicate_not_stored'))
        entry['support_audit'] = support
        if support['passed'] is True:
            stage_pools[candidate['stage']]['after_support'].append(entry)
        records.append(entry)
    cache_reuse=not any(b.get('explicit_complete_BI_likelihood_revocation') is True
                        for b in packet.get('solver_input',{}).get('boundary_observations',[]))
    if cache_reuse:
        # Exact original d control reused G10, not a fresh G13 optimizer.
        stage_pools['G13']['generated']=stage_pools['G10']['generated']
        stage_pools['G13']['after_support']=stage_pools['G10']['after_support']
    pools['generated']=stage_pools['G13']['generated']
    pools['after_support']=stage_pools['G13']['after_support']
    support_unknown=[c['candidate_id'] for c in pools['generated'] if c['support_audit']['passed'] is None]

    # The intermediate selected G10 family pose and final same-family d pose
    # remain distinct diagnostics. The `family` pool above is all candidates
    # available *inside* the family that G10 froze, before its final d choice.
    chosen = family_bank[selected_family]
    refined_pose = chosen['published_actual_pose']
    add('refinement', 'ORIGINAL_G13/'+selected_family+'/selected_or_own_fallback', refined_pose,
        family=selected_family, source='original_final_same_family_refinement')
    add('selected', 'ORIGINAL_E0_FINAL', original['original_e']['actual_pose'],
        family=selected_family, source='original_frozen_e_output')

    lookup = {}
    for entry in records:
        record = entry['original_endpoint_record']
        name = record.get('initialization')
        if entry['endpoint_kind'] != 'outline_consider' or not isinstance(name, str):
            continue
        key = (entry['stage'], entry['family'], record.get('branch'))
        if name.endswith('_UNREFINED'):
            lookup[key+(name[:-len('_UNREFINED')],)] = entry
        else:
            before = lookup.get(key+(name,))
            if before is not None:
                refinement_pairs.append(dict(stage=entry['stage'], family=entry['family'],
                    branch=record.get('branch'), initialization=name,
                    before_candidate_id=before['candidate_id'], after_candidate_id=entry['candidate_id'],
                    before_metric=before['metric'], after_metric=entry['metric'],
                    joint_success_to_failure=(before['metric']['success_le5'] is True and
                                              entry['metric']['success_le5'] is False)))

    summaries = {name: _summary(pools[name]) for name in STAGES}
    final_success = summaries['selected']['joint_success_exists']
    flags = [] if coverage['original_e_consumer_generated_bank_complete'] else ['CANDIDATE_RECORD_INCOMPLETE']
    if (coverage['original_e_consumer_generated_bank_complete'] and not summaries['generated']['joint_success_exists']
            and not final_success):
        flags.append('OBSERVATION_OR_PROPOSAL_LIMIT')
    if summaries['feasible']['joint_success_exists'] and not final_success:
        flags.append('SELECTION_LOSS')
    observed_generation_to_feasible_loss = bool(summaries['generated']['joint_success_exists'] and
                                                not summaries['feasible']['joint_success_exists'])
    if observed_generation_to_feasible_loss and not final_success:
        # An existential observed loss, not a claim about an unknown full bank.
        flags.append('OBSERVED_REFINEMENT_OR_PRUNING_LOSS')
    pruned_success = [c['candidate_id'] for c in pools['generated']
        if c['metric']['success_le5'] is True and
        (c['original_endpoint_record'].get('joint_feasible') is False or
         (c['endpoint_kind'] != 'partial_BI_seed_fit' and
          c['original_endpoint_record'].get('feasible') is False) or
         c['original_endpoint_record'].get('original_dedup_kept') is False)]
    if pruned_success and not summaries['feasible']['joint_success_exists'] and not final_success:
        flags.append('PRUNING_LOSS')
    if any(p['joint_success_to_failure'] for p in refinement_pairs):
        flags.append('REFINEMENT_LOSS')
    if not final_success:
        earliest = ('family_selection' if summaries['feasible']['joint_success_exists'] and
                    not summaries['family']['joint_success_exists'] else
                    'within_family_final_selection' if summaries['family']['joint_success_exists'] else
                    'logged_generated_to_retained_feasible' if summaries['generated']['joint_success_exists'] else
                    'confirmed_original_consumer_observation_or_proposal_limit' if coverage['original_e_consumer_generated_bank_complete'] else
                    'unknown_due_to_incomplete_original_consumer_generation')
    else:
        earliest = None
    physical_rotation = pools['selected'][0]['metric'].get('rotation_deg')
    unresolved = pools['selected'][0]['metric']['success_le5'] is None
    registration = dict(fixed_existing_reference_only=True, symmetry='physical registered rectangular C2; no C4',
        original_annotation_physical_axis_truth_independently_confirmed=False,
        semantic_owner_status=packet.get('source_group_semantic_owner_status', 'SEMANTIC_OWNER_UNVERIFIED'),
        reference_rotation_unresolved=unresolved,
        large_physical_rotation_over45_diagnostic=physical_rotation is not None and physical_rotation > 45,
        flag='REFERENCE_OR_REGISTRATION_UNRESOLVED' if unresolved else None)
    if unresolved:
        flags.append('REFERENCE_OR_REGISTRATION_UNRESOLVED')
    return dict(id=packet['id'], partition=packet['partition'], session=packet['session'],
        stages=summaries, failure_flags=flags, earliest_observed_failure_stage=earliest,
        final_joint_success=final_success,
        final_joint_reference_resolved=pools['selected'][0]['metric']['success_le5'] is not None,
        original_stage_oracles={stage:{name:_summary(value) for name,value in stage_pools[stage].items()}
                                for stage in ('G10','G13')},
        generated_scope='Actual original-e incoming archived starts (before outline dedup), restored one-CAL initialization-generator output and evaluated refinement endpoints; final G13 or exact G10 cache separately; historical WD/P1 rejected proposals and optimizer internal iterates are outside this consumer bank',
        no_typed_exact_G10_cache_reuse=cache_reuse,
        G10_to_G13_NOT_ISOLATED='Complete/legacy G10 package differs from typed-partial G13 package; outer G10 family selection and inner G13 refinement are separate stages',
        entire_historical_generated_bank_complete=False,
        original_consumer_generated_bank_complete=coverage['original_e_consumer_generated_bank_complete'],
        original_consumer_numeric_endpoints_complete=coverage['original_e_consumer_numeric_endpoints_complete'],
        original_consumer_coverage={k:v for k,v in coverage.items() if k!='incoming_candidates'},
        historical_rejected_WD_P1_outside_original_e_consumer=True,
        generated_scope_is_actual_consumer_initializations_and_evaluated_endpoints=True,
        replay_parity_passed=bool(replay_parity_passed),
        replay_endpoints_withheld_on_parity_failure=not replay_parity_passed,
        original_final_feasible_sample_lists_preserved_complete=True,
        support_unknown_candidate_ids=support_unknown,
        after_support_is_recomputed_audit_not_original_sequential_filter=True,
        no_OBSERVATION_OR_PROPOSAL_LIMIT_claim_from_absent_records=True,
        proposal_limit_is_fixed_original_consumer_scope_not_global_physical_impossibility=True,
        reference_and_registration_audit=registration,
        refinement_pairs=refinement_pairs,
        pruned_joint_success_candidate_ids=pruned_success,
        rejected_good_endpoint_count=len(pruned_success),
        observed_generation_to_feasible_loss=observed_generation_to_feasible_loss,
        pruning_flag_scope='Final output fails; a same-final-stage joint-success endpoint was rejected by original guards and no joint-success candidate remains in that stage retained feasible bank. Rejection of one good candidate alone is only a separate endpoint diagnostic.',
        refinement_flag_scope='Candidate-level same-stage/start damage; can exist on a finally successful frame. Aggregate final-failure-only counts are separate.',
        diagnostic_details=(dict(pools=pools,
            original_G10_selected_family=original['stages']['G10']['selected_family'],
            original_family_audits={stage:[{key:family[key] for key in ('name','native_xyz',
                'cf_extents','physical_registration_Q','accepted','solver_schema','solver_status',
                'original_objective','original_rank','original_ambiguity','original_search')}
                for family in original['stages'][stage]['families']] for stage in ('G10','G13')},
            original_hierarchical_correspondence=original['original_e'].get('hierarchy'))
                            if packet['id'] in DIAGNOSTIC_IDS else None))


def aggregate(rows):
    """Fixed denominators; no success/availability filtering or hidden retries."""
    populations = {}
    for name, count in (('CAL',46), ('TEST',49), ('REMAINDER',143), ('PRIMARY',192)):
        selected = [r for r in rows if (r['partition'] != 'CAL' if name == 'PRIMARY' else r['partition'] == name)]
        if len(selected) != count:
            raise ValueError('Exact '+name+' denominator required')
        populations[name] = dict(denominator=count,
            final_success_n=sum(r['final_joint_success'] for r in selected),
            final_failure_n=sum(not r['final_joint_success'] for r in selected),
            final_reference_unresolved_n=sum(not r['final_joint_reference_resolved'] for r in selected),
            stages={stage: dict(joint_success_oracle=sum(r['stages'][stage]['joint_success_exists'] for r in selected),
                candidate_available_frames=sum(r['stages'][stage]['candidate_count'] > 0 for r in selected),
                no_candidates_frames=sum(r['stages'][stage]['candidate_count'] == 0 for r in selected),
                after_support_unknown_frames=sum(bool(r['support_unknown_candidate_ids']) for r in selected)
                    if stage == 'after_support' else None) for stage in STAGES},
            failure_flags=dict(Counter(flag for r in selected for flag in r['failure_flags'])),
            failure_flags_denominator_scope='All frames, including final successes; coverage and candidate-level REFINE flags may occur on successful frames',
            failure_flags_final_failure_only=dict(Counter(flag for r in selected
                if not r['final_joint_success'] for flag in r['failure_flags'])),
            failure_flags_final_success_only=dict(Counter(flag for r in selected
                if r['final_joint_success'] for flag in r['failure_flags'])),
            rejected_good_endpoint_diagnostic=dict(
                frames=sum(r['rejected_good_endpoint_count']>0 for r in selected),
                endpoint_count=sum(r['rejected_good_endpoint_count'] for r in selected),
                final_success_frames=sum(r['rejected_good_endpoint_count']>0 and r['final_joint_success'] for r in selected),
                not_automatically_an_overall_pruning_loss=True),
            earliest_observed_failure=dict(Counter(str(r['earliest_observed_failure_stage']) for r in selected)),
            original_stage_oracles={stage:{name:dict(
                joint_success_oracle=sum(r['original_stage_oracles'][stage][name]['joint_success_exists'] for r in selected),
                candidate_available_frames=sum(r['original_stage_oracles'][stage][name]['candidate_count']>0 for r in selected))
                for name in ('generated','after_support','feasible','selected')} for stage in ('G10','G13')},
            original_consumer_generated_complete_frames=sum(r['original_consumer_generated_bank_complete'] for r in selected),
            original_consumer_generated_unknown_frames=sum(not r['original_consumer_generated_bank_complete'] for r in selected),
            original_consumer_generated_bank_complete=all(r['original_consumer_generated_bank_complete'] for r in selected),
            original_bank_incomplete=not all(r['original_consumer_generated_bank_complete'] for r in selected),
            entire_historical_generated_bank_complete=False,
            oracles_are_fixed_reference_candidate_upper_bounds_not_physical_truth=True)
    return populations


def run(*, folder, source_root, lineage_folder, original_candidate_folder, reference_manifest,
        expected_prediction_seal_sha256, expected_candidate_seal_sha256, candidate_file,
        expected_reference_manifest_sha256, parity_receipt, output, root_scoring_GO=False):
    """Explicit postseal entry point. A missing/failed parity never fills gaps."""
    if root_scoring_GO is not True:
        raise PermissionError('Root postseal oracle GO required before any IO')
    from . import analysis as A
    from .common import save_rows
    from .continuity import support_for_candidate
    started, cpu = time.monotonic(), time.process_time()
    output, folder, lineage_folder, original_candidate_folder = map(Path,
        (output, folder, lineage_folder, original_candidate_folder))
    if output.exists():
        raise FileExistsError('Preserve every previous oracle attempt')
    # All prediction and provenance gates precede the first numeric reference.
    predicted, _, candidate_seal = A.prediction_gate(folder, expected_prediction_seal_sha256,
                                       expected_candidate_seal_sha256, candidate_file)
    ids = [r['id'] for r in predicted]
    extraction = A.read(lineage_folder/'EXTRACTION_SEAL.json')
    manifest = A.read(original_candidate_folder/'CANDIDATE_MANIFEST.json')
    original_path = original_candidate_folder/'ORIGINAL_E_CANDIDATE_BANK.jsonl.gz'
    typed_path = lineage_folder/'TYPED_PACKETS.jsonl.gz'
    if (extraction['ordered_ids'] != ids or manifest['ordered_ids'] != ids or
            A.sha(typed_path) != extraction['files'][typed_path.name]['sha256'] or
            A.sha(original_path) != manifest['artifacts'][original_path.name] or
            A.sha(reference_manifest) != expected_reference_manifest_sha256):
        raise ValueError('Original prediction/lineage/candidate/reference binding changed')
    parity = A.read(parity_receipt)
    supplement_folder=folder/'supplemental_cal_start_logging_v1'
    supplement_seal_path=supplement_folder/'SUPPLEMENT_SEAL.json'
    supplement_seal=A.read(supplement_seal_path)
    supplement_path=supplement_folder/'RAW_START_SUPPLEMENT.json'
    if (A.sha(supplement_seal_path)!=candidate_seal['initial_start_supplement_seal_sha256'] or
        A.sha(supplement_path)!=supplement_seal['raw_start_supplement_sha256'] or
        supplement_seal['typed_packet_file_sha256']!=A.sha(typed_path) or
        supplement_seal['GT_inputs'] is not False or supplement_seal['before_new_scoring'] is not True or
        A.sha(parity_receipt)!=candidate_seal['strict_replay_parity_sha256'] or
        A.sha(original_candidate_folder/'CANDIDATE_MANIFEST.json')!=candidate_seal['original_candidate_manifest_sha256']):
        raise ValueError('Original-consumer supplemental start/parity seal changed')
    supplement=A.read(supplement_path)
    if parity['ordered_ids'] != ids or parity['rows'] != 238 or parity['atol'] != 1e-9 or parity['rtol'] != 0.:
        raise ValueError('Full fixed238 strict original replay parity receipt required')
    parity_by_id = {r['id']: r['passed'] for r in parity['result']}
    if set(parity_by_id) != set(ids):
        raise ValueError('Every replay parity outcome must remain explicit')
    originals, packets = A.journal(original_path), A.journal(typed_path)
    if [r['id'] for r in originals] != ids or [r['id'] for r in packets] != ids:
        raise ValueError('Exact ordered238 original candidate and typed packets required')
    refs, reference_bindings, annotations = A.fixed_references(source_root, reference_manifest, predicted)
    results, fallbacks = [], []
    for source, pred, packet in zip(originals, predicted, packets):
        ident = packet['id']
        replay = dict(pred['predictions']['TYPED_MULTI'])
        if replay.get('technical_failure'):
            replay['id'] = ident
        reference = refs[packet['partition']][ident]
        results.append(analyze_frame(source, replay, packet, lambda pose: A.metric(pose, reference),
            support_for_candidate, replay_parity_passed=parity_by_id[ident],initial_start_supplement=supplement))
        fallbacks.append(dict(id=ident, partition=packet['partition'], session=packet['session'],
            arms={arm: single_fallback_audit(pred['predictions'][arm], packet['e0_pose'])
                  for arm in ('TYPED_SINGLE', 'REOBS_SINGLE')}))
    output.mkdir(parents=True)
    save_rows(output/'ORACLE_ROWS.jsonl.gz', results)
    A.write(output/'ORACLE_SUMMARY.json', dict(schema='SEALED_ORIGINAL_E_CANDIDATE_STAGE_ORACLE_V1',
        populations=aggregate(results), stage_definitions=dict(
            generated='Actual incoming archived starts before outline dedup, original evaluated refinement endpoints, and the exact one-CAL restored initialization-generator output; final G13 or exact G10 cache; G10/G13 ladders separate; rejected historical WD/P1 proposals were never consumed by e',
            after_support='Posthoc original visible geometry/canvas and finite partial-line support predicates on those same logged endpoints; not a newly executed sequential pruning policy',
            feasible='All retained original G13-stage feasible samples across both registered families, with original G10 cache reuse where applicable',
            family='Those original G13 feasible samples inside the exact original G10-frozen family',
            refinement='Original d selected pose or explicit own fallback in that frozen family',
            selected='Original frozen e output including every fallback'),
        original_bank_incomplete=not all(r['original_consumer_generated_bank_complete'] for r in results),
        entire_historical_generated_bank_complete=False,
        historical_rejected_WD_P1_outside_original_e_consumer=True,
        original_consumer_complete_frames=sum(r['original_consumer_generated_bank_complete'] for r in results),
        no_cross_model_or_SINGLE_union=True,
        no_new_training_or_seed_or_posefit=True,
        reference_kind='Existing reconstructed pose/C2 reference; not independent physical truth',
        diagnostic_ids=list(DIAGNOSTIC_IDS), diagnostic_ids_present=[i for i in DIAGNOSTIC_IDS if i in ids]))
    A.write(output/'SINGLE_FALLBACK_AUDIT.json', dict(rows=fallbacks, reference_reads_for_this_audit=0,
        all238_retained=True, component_atol=1e-9,
        counts={arm:dict(fallbacks=sum(r['arms'][arm]['fallback'] is True for r in fallbacks),
            fallback_matches_E0=sum(r['arms'][arm]['fallback_matches_own_E0'] is True for r in fallbacks),
            fallback_differs_from_E0=sum(r['arms'][arm]['fallback_matches_own_E0'] is False for r in fallbacks),
            no_optimizer_endpoint=sum(r['arms'][arm]['no_optimizer_endpoint'] for r in fallbacks))
            for arm in ('TYPED_SINGLE', 'REOBS_SINGLE')}))
    diagnostic = [r for r in results if r['id'] in DIAGNOSTIC_IDS]
    A.write(output/'DIAGNOSTIC_CASE_STAGES.json', dict(rows=diagnostic, required_ids=list(DIAGNOSTIC_IDS),
        missing_ids=[i for i in DIAGNOSTIC_IDS if i not in ids]))
    fields = ['id','partition','session']+[stage+'_joint_success_exists' for stage in STAGES]+[
        'earliest_observed_failure_stage','failure_flags','original_consumer_generated_bank_complete','historical_generated_bank_complete']
    with (output/'ORACLE_FRAME_TABLE.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in results:
            writer.writerow(dict(id=row['id'],partition=row['partition'],session=row['session'],
                **{stage+'_joint_success_exists':row['stages'][stage]['joint_success_exists'] for stage in STAGES},
                earliest_observed_failure_stage=row['earliest_observed_failure_stage'],
                failure_flags='|'.join(row['failure_flags']),
                original_consumer_generated_bank_complete=row['original_consumer_generated_bank_complete'],
                historical_generated_bank_complete=False))
    A.write(output/'ORACLE_SEAL.json', dict(schema='SEALED_ORIGINAL_E_ORACLE_RECEIPT_V1',rows=238,
        ordered_ids=ids,prediction_seal_sha256=expected_prediction_seal_sha256,
        candidate_seal_sha256=expected_candidate_seal_sha256,
        original_candidate_manifest_sha256=A.sha(original_candidate_folder/'CANDIDATE_MANIFEST.json'),
        extraction_seal_sha256=A.sha(lineage_folder/'EXTRACTION_SEAL.json'),
        parity_receipt_sha256=A.sha(parity_receipt), reference_bindings=reference_bindings,
        initial_start_supplement_seal_sha256=A.sha(supplement_seal_path),
        reference_manifest_sha256=expected_reference_manifest_sha256,annotation_sha256=annotations,
        source_sha256={'oracle.py':A.sha(__file__),'analysis.py':A.sha(A.__file__),
                      'continuity.py':A.sha(Path(__file__).with_name('continuity.py')),
                      'restore_starts.py':A.sha(Path(__file__).with_name('restore_starts.py'))},
        actual_GT_used_for_selection_or_routes=False,new_model_forwards=0,new_pose_solves=0,
        elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
        artifacts={p.name:A.sha(p) for p in output.iterdir() if p.is_file()}))
    return dict(populations=aggregate(results), rows=238, output=str(output))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('folder','source-root','lineage-folder','original-candidate-folder','reference-manifest',
                 'candidate-file','parity-receipt','output'):
        parser.add_argument('--'+name,required=True)
    for name in ('prediction-seal-sha256','candidate-seal-sha256','reference-manifest-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--root-scoring-go',action='store_true')
    args=vars(parser.parse_args())
    for name in ('prediction_seal_sha256','candidate_seal_sha256','reference_manifest_sha256'):
        args['expected_'+name]=args.pop(name)
    args['root_scoring_GO']=args.pop('root_scoring_go')
    print(run(**args))

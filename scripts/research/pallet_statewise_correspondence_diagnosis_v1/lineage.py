"""Stream the fixed 238 pre-e / post-e observation lineage, without fitting.

The immutable e_trace display classifier is retained as an audit comparator.
Actual complete factors exclude *all* partial boundary IDs, including legacy
fallback-B.  Replay packets retain the exact original consumer arguments;
the e pose is a separate optimizer start, never a replacement coarse prior.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from scripts.research.pallet_guarded_bi_final_v1 import e_inputs as EI
from scripts.research.pallet_posefix_gap_audit_v1 import e_trace as ET

SCHEMA = 'STATEWISE_CORRESPONDENCE_LINEAGE_V1'
PACKET_SCHEMA = 'ORIGINAL_E_TYPED_REPLAY_PACKET_V1'
PARTITIONS = {'CAL': 46, 'TEST': 49, 'REMAINDER': 143}
SEMANTIC_OWNER = 'SEMANTIC_OWNER_UNVERIFIED'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return ET.canonical(value)


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def source_kind(measurement, record=None):
    """A stored 'image_measurement' flag does not make fallback an RGB fit."""
    if measurement.get('source') == 'VISIBLE_BACKBONE_FALLBACK':
        return 'inherited_predictor_point'
    if measurement.get('source') in ('B_TARGET_BACKGROUND_LINES', 'I_INTERNAL_RGB_JUNCTION',
                                      'B_ACTUAL_BOUNDARY_RECOVERY'):
        if record is not None and record.get('corrected') is not True:
            return 'unknown'
        return 'complete_image_point'
    return 'unknown'


def line_source_group(image_sha256, ray):
    """Exact shared sample group, invariant to endpoint/sample order reversal.

    This diagnoses duplicated evidence; it does not estimate a correlation
    coefficient or authenticate an image line's semantic physical owner.
    """
    points, weights = ray.get('observed_points'), ray.get('observation_confidence')
    if not isinstance(points, list) or not isinstance(weights, list) or len(points) != len(weights):
        return None
    pairs = sorted((list(p), w) for p, w in zip(points, weights))
    return digest(dict(image_sha256=image_sha256, exact_observed_samples_and_confidences=pairs))


def consumer_activation(visible, boundary, states, solution=None):
    """Mirror P13._joint_objective_pair's exclusion IDs, not display flags."""
    by_id = {m['native_corner']: m for m in visible}
    require(len(by_id) == len(visible), 'Unique original visible IDs required')
    partial = {b['native_corner'] for b in boundary}
    require(len(partial) == len(boundary), 'Unique original partial IDs required')
    require(partial <= set(by_id), 'Partial needs its archived visible seed')
    require(all(states[k] in ('B', 'I') for k in by_id), 'S/O/T/U cannot be point factors')
    typed = {b['native_corner'] for b in boundary
             if b.get('explicit_complete_BI_likelihood_revocation') is True}
    legacy = partial - typed
    require(all(states[k] == 'B' and by_id[k].get('source') == 'VISIBLE_BACKBONE_FALLBACK'
                for k in legacy), 'Legacy partial must preserve fallback-B provenance')
    components = (solution or {}).get('objective_components') or {}
    reported = components.get('removed_complete_BI_2D_likelihood_ids',
                              components.get('removed_raw_B_2D_likelihood_ids'))
    component_parity = None if reported is None else set(reported) == partial
    require(component_parity is not False, 'Stored winning objective removal IDs disagree with consumer')
    return dict(complete_ids=sorted(set(by_id)-partial), all_partial_ids=sorted(partial),
                typed_partial_ids=sorted(typed), legacy_partial_ids=sorted(legacy),
                legacy_e_trace_complete_ids=sorted(set(by_id)-typed),
                legacy_e_trace_false_complete_ids=sorted(legacy),
                selected_objective_component_parity=component_parity,
                final_pose_is_solver_accepted=(solution or {}).get('accepted'),
                configured_factors_are_not_fallback_solution_success=True,
                consumer='statewise_partial_BI_u_solver_v4_13._joint_objective_pair; legacy parent equivalent',
                S_image_terms=0, O_T_U_point_terms=0)


def truncation_union_branches(domains):
    """Exact fixed union choices from joint_ot_pnp._inputs / G13 search."""
    choices=[]
    for d in domains:
        if d['state']!='T':continue
        side=d.get('outside_canvas_side')
        if side in ('unknown','any'):side=None
        choices.append(list(dict.fromkeys(side))if isinstance(side,(tuple,list))else
                       [side]if side else['left','right','top','bottom'])
    count=1
    for c in choices:count*=len(c)
    return dict(T_domain_count=len(choices),choices=choices,branch_count=count)


def active_factor_view(packet):
    """Small audit-only view: display/readout poisoning cannot change factors."""
    s, a = packet['solver_input'], packet['activation']
    complete = {m['native_corner']: m for m in s['visible']}
    return dict(points=[{k: complete[i][k] for k in
                        ('native_corner', 'xy', 'covariance_px2', 'fit_covariance_px2',
                         'fit_reliability_weight') if k in complete[i]}
                       for i in a['complete_ids']],
                partial=[dict(native_corner=b['native_corner'], state=b['state'],
                              fit_reliability_weight=b['fit_reliability_weight'],
                              lines=[{k: line[k] for k in ('normal_unit_xy', 'offset_px',
                                     'normal_variance_px2', 'support_tangent_interval_px',
                                     'support_extrapolation_px') if k in line} for line in b['lines']])
                         for b in s['boundary_observations']],
                domains=s['domains'], u_regions=s['u_regions'],
                outline_observation=s['outline_observation'], config=s['config'],
                geometry={k:s[k] for k in ('xyz', 'K', 'states', 'image_hw', 'coarse_pose')})


def build_packet(row, e_frame, bindings):
    """Pure extraction; keep original e owner hypothesis and prior unchanged."""
    ident = row['id']
    require(e_frame['id'] == ident, 'Exact ordered E frame required')
    old = row['finish_input']['_original_source']
    boundary = row['finish_input']['partial_boundary_observation']['boundary_observations']
    require(boundary == row['boundary_typed'], 'Original e boundary_typed binding differs')
    require({b['native_corner'] for b in boundary} ==
            {b['native_corner'] for b in row['boundary_all']}, 'Later tangent archive changed boundary IDs')
    require(row['spec']['states'] == old['states'], 'Original authoritative states changed')
    require(row['spec']['visible'] == row['finish_input']['frozen_BI_fit'], 'Original fit archive differs')
    e0 = e_frame['arms']['E_FROZEN']['coarse_pose']
    ET.pose_check(row['baseline']['actual_pose'], e0, ident+'/E0 numeric bridge')
    family = row['family']
    require(family['name'] == row['fixed_family'] == row['baseline']['WD_family'], 'Original selected family differs')
    solver = {k: deepcopy(row['spec'][k]) for k in
              ('xyz', 'K', 'states', 'visible', 'domains', 'image_hw', 'coarse_pose', 'config')}
    solver.update(boundary_observations=deepcopy(boundary), u_regions=deepcopy(row['u_regions']),
                  outline_observation=deepcopy(row['outline']))
    activation = consumer_activation(solver['visible'], boundary, solver['states'], row['baseline']['solution'])
    packet = dict(schema=PACKET_SCHEMA, id=ident, partition=row['partition'], session=row['session'],
        image=row['image'], image_sha256=row['image_sha256'], solver_input=solver,
        activation=activation, registered_family=deepcopy(family), e0_pose=deepcopy(e0),
        original_e_stage1_family_seed_bank=deepcopy(row['stage1_family_seed_bank']),
        original_e_legacy_boundary_observations=deepcopy(row['boundary_original_legacy']),
        original_e_selected_solution_schema=row['baseline']['solution']['schema'],
        original_e_accepted=row['baseline']['accepted'],
        original_T_union=truncation_union_branches(solver['domains']),
        original_e_hierarchical_correspondence_audit=deepcopy(row['baseline'].get('hierarchical_correspondence_audit')),
        source_group_semantic_owner_status=SEMANTIC_OWNER,
        provenance=dict(**deepcopy(bindings), original_boundary_field='finish_input.partial_boundary_observation.boundary_observations',
            archive_boundary_all_not_used_as_original_e_factor=True,
            source_prior_not_replaced_by_E0=True, E0_is_single_optimizer_start_only=True,
            original_e_native_channels_not_permuted=True, GT_inputs=False,
            no_new_fit_or_NN_or_PnP=True, original_states_weights_covariances_and_losses_preserved=True))
    packet['active_factor_view_sha256'] = digest(active_factor_view(packet))
    return packet


def _corrections(root, ids, source_bindings):
    locations = []
    for part in ('CAL', 'TEST'):
        # e consumed the later corner-reach recovery, not evaluation_v3.FULL.
        # The original v4_15 protocol pins these exact V47 journals/seals.
        folder = root/ET.BOUNDARY/'shared_edge_partial_b_eval_v4_7'
        seal = folder/(part+'_PREDICTION_SEAL.json'); path = folder/(part+'_PREDICTIONS.jsonl.gz')
        expected = read(seal)['predictions_sha256']
        require(sha(folder/'PROTOCOL.json')==read(seal)['protocol_sha256'],'Original V47 protocol changed')
        source_bindings[str((folder/'PROTOCOL.json').relative_to(root))]=sha(folder/'PROTOCOL.json')
        locations.append((path, expected, 'correction', seal))
    folder = root/ET.VIS/'remainder_after_test49_v1/fresh_observations_v4_14'
    seal = folder/'OBSERVATION_SEAL.json'; path = folder/'OBSERVATIONS.jsonl.gz'
    locations.append((path, read(seal)['observations_sha256'], 'correction', seal))
    result = {}
    for path, expected, field, seal in locations:
        require(sha(path) == expected, 'Original RGB correction seal changed')
        source_bindings[str(path.relative_to(root))] = expected
        source_bindings[str(seal.relative_to(root))] = sha(seal)
        with gzip.open(path, 'rt') as stream:
            for number, line in enumerate(stream, 1):
                r = json.loads(line)
                if r['id'] not in ids:
                    continue
                require(r['id'] not in result, 'Duplicate original correction')
                result[r['id']] = dict(correction=r['results']['FULL'] if field == 'results.FULL' else r['correction'],
                    image_sha256=r['image_sha256'], binding=dict(path=str(path.relative_to(root)),
                    sha256=expected, source_field=field, source_line_number=number, input_row_sha256=digest(r)))
    require(set(result) == ids, 'Complete238 original RGB corrections required')
    return result


def _final_index(root, ids, source_bindings):
    folder = root/ET.FINAL; seal = read(folder/'PREDICTION_SEAL.json')
    require(seal['rows'] == 238 and sha(folder/'PREDICTIONS.jsonl.gz') == seal['predictions_sha256'], 'Original e238 seal mismatch')
    source_bindings[str((folder/'PREDICTION_SEAL.json').relative_to(root))] = sha(folder/'PREDICTION_SEAL.json')
    source_bindings[str((folder/'PREDICTIONS.jsonl.gz').relative_to(root))] = seal['predictions_sha256']
    result = {}
    with gzip.open(folder/'PREDICTIONS.jsonl.gz', 'rt') as stream:
        for number, line in enumerate(stream, 1):
            r = EI.select_fields(line, {'id','image_sha256','states','arms'}, raw_fields={'arms'})
            a = EI.select_fields(r['arms'], {'SAME_FROZEN_E'})['SAME_FROZEN_E']
            require(r['id'] not in result, 'Duplicate original e output')
            result[r['id']] = dict(baseline_sha256=digest(a), actual_pose=a['actual_pose'],
                image_sha256=r['image_sha256'], states=r['states'], source_line_number=number)
    require(set(result) == ids, 'Original e output population differs')
    return result


def _rays(correction, boundary, image_hash):
    rays = []
    for record in correction['records']:
        for ray in record.get('rays', []):
            rays.append(dict(origin='original_pre_e_BI_recovery', native_corner=record.get('native_corner'), ray=deepcopy(ray)))
    for b in boundary:
        for line in b.get('lines', []):
            ray = line.get('provenance', {}).get('line_record')
            if ray is not None:
                rays.append(dict(origin='original_e_typed_partial', native_corner=b['native_corner'], ray=deepcopy(ray)))
    for r in rays:
        r['exact_sample_source_group'] = line_source_group(image_hash, r['ray'])
        r['semantic_owner_status'] = SEMANTIC_OWNER
    return rays


def audit_row(row, packet, e_frame, observation, solution, correction):
    """Separate e-owner continuation from raw-source permutation lineage."""
    old = row['finish_input']['_original_source']; states = old['states']; newstates = observation['states']
    a = packet['activation']; partial = set(a['all_partial_ids']); typed = set(a['typed_partial_ids'])
    mapping = e_frame['arms']['E_FROZEN']['native_vertex_mapping']; perm = mapping['target_to_source_native_ids']
    inverse = mapping['source_to_target_native_ids']; oldby = {m['native_corner']:m for m in row['spec']['visible']}
    newby = {m['native_corner']:m for m in observation['frozen_visible_measurements']}
    oldrec = {r['native_corner']:r for r in correction['records']}
    oldgeo = old['initial_geometry']; xyz0 = np.asarray(oldgeo['native_xyz']); Q0 = np.asarray(oldgeo['R_cf']).T@np.asarray(oldgeo['R_physical'])
    xyz1 = np.asarray(row['family']['native_xyz']); Q1 = np.asarray(row['family']['physical_registration_Q'])
    rows = []
    for k in range(8):
        before = oldby.get(k); after = newby.get(k); raw_target = inverse[k]
        kind = 'partial_line' if k in partial else source_kind(before,oldrec.get(k)) if before else 'model_projection' if states[k]=='S' else 'bounded_region' if states[k] in ('O','T','U') else 'unknown'
        rows.append(dict(id=row['id'],partition=row['partition'],session=row['session'],native_corner=k,
            original_state=states[k],post_e_same_owner_state=newstates[k],
            original_observation_type=kind,original_full_active=k in a['complete_ids'],
            explicit_typed_partial=k in typed,legacy_fallback_partial=k in set(a['legacy_partial_ids']),
            old_e_trace_false_full=k in set(a['legacy_partial_ids']),
            original_source=before.get('source')if before else None,
            raw_predictor_fallback=bool(before and before.get('source')=='VISIBLE_BACKBONE_FALLBACK'),
            post_e_full_active=after is not None,post_e_source=after.get('source')if after else None,
            e_owner_partial_to_post_full=k in partial and after is not None,
            e_owner_typed_to_post_full=k in typed and after is not None,
            e_owner_legacy_to_post_full=k in set(a['legacy_partial_ids'])and after is not None,
            post_e_injected_N3_source_id=perm[k],N3_channel_target_after_reindex=raw_target,
            N3_channel_state_changed_after_reindex=states[k]!=newstates[raw_target],
            e_owner_state_changed=states[k]!=newstates[k],
            N3_to_e_same_channel_physical_owner_changed=not np.allclose(xyz0[k]@Q0,xyz1[k]@Q1,atol=1e-8,rtol=0),
            model_registration_bijection_unique=mapping['bijection_unique'],
            semantic_owner_status=SEMANTIC_OWNER,post_e_is_new_observation_package=True,
            direct_reuse_of_original_partial_samples=False))
    return rows


def _json_writer(stack, path):
    raw = stack.enter_context(Path(path).open('xb'))
    return stack.enter_context(gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0))


def _emit(stream, value):
    stream.write((json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())


def prepare(source_root, guarded_dir, output_dir):
    """Write a new private extraction. No previous output may be replaced."""
    started, cpu = time.monotonic(), time.process_time()
    root, guarded, output = Path(source_root), Path(guarded_dir), Path(output_dir)
    require(not output.exists(), 'Preserve previous lineage extraction')
    edoc = read(guarded/'e_inputs/INPUTS.json'); frames = edoc['frames']; ids = [f['id']for f in frames]
    require(len(ids)==len(set(ids))==238 and Counter(f['partition']for f in frames)==Counter(PARTITIONS), 'Explicit complete238 E input population required')
    eseal = read(guarded/'e_inputs/INPUT_SEAL.json')
    require(eseal['ordered_ids']==ids and sha(guarded/'e_inputs/INPUTS.json')==eseal['inputs_sha256'], 'E input seal changed')
    bindings = {}; corrections = _corrections(root,set(ids),bindings); final = _final_index(root,set(ids),bindings)
    ap = root/ET.ARCHIVE; aseal = read(ap/'INPUT_SEAL.json')
    for filename,key in [('INPUTS.jsonl.gz','input_sha256'),('PROTOCOL.json','protocol_sha256')]:
        require(sha(ap/filename)==aseal[key], 'Original typed archive seal changed: '+filename)
        bindings[str((ap/filename).relative_to(root))]=aseal[key]
    bindings[str((ap/'INPUT_SEAL.json').relative_to(root))]=sha(ap/'INPUT_SEAL.json')
    require(aseal['rows']==238,'Original archive must contain238')
    postseal = read(guarded/'e_run/OBSERVATION_SEAL.json'); require(postseal['rows']==238,'Own E post observation238 required')
    post_predictions = {}
    predpath=guarded/'PREDICTIONS.jsonl.gz'; predseal=read(guarded/'PREDICTION_SEAL.json')
    require(sha(predpath)==predseal['predictions_sha256'],'Guarded prediction seal changed')
    with gzip.open(predpath,'rt')as stream:
        for line in stream:
            r=EI.select_fields(line,{'id','predictions'},raw_fields={'predictions'})
            post_predictions[r['id']]=EI.select_fields(r['predictions'],{'E0','E1','EA','EB'})
    require(set(post_predictions)==set(ids),'Post-E prediction population differs')
    output.mkdir(parents=True); totals=Counter(); old_sources=Counter(); transition_sources=Counter(); post_bindings={}; seen=[]
    with ExitStack()as stack:
        packets=_json_writer(stack,output/'TYPED_PACKETS.jsonl.gz'); traces=_json_writer(stack,output/'E_TRACE.jsonl.gz'); lineage=_json_writer(stack,output/'OBSERVATION_LINEAGE.jsonl.gz')
        csvfile=stack.enter_context((output/'OBSERVATION_TYPE_AND_OWNER_AUDIT.csv').open('x',newline='')); writer=None
        with gzip.open(ap/'INPUTS.jsonl.gz','rt')as stream:
            for i,line in enumerate(stream):
                r=json.loads(line);f=frames[i];ident=r['id'];require(ident==f['id'],'Original archive ordered IDs differ');seen.append(ident)
                require(digest(r['baseline'])==final[ident]['baseline_sha256'],'Original e exact baseline bridge differs')
                require(r['image_sha256']==final[ident]['image_sha256']==f['image_sha256'],'RGB identity differs')
                require(r['spec']['states']==final[ident]['states'],'Original final authoritative states differ')
                correction=corrections[ident]; require(correction['image_sha256']==r['image_sha256'],'Original correction RGB differs')
                by={m['native_corner']:m for m in correction['correction']['visible_measurements']}
                for m in r['finish_input']['frozen_BI_fit']:
                    require(m['xy']==by[m['native_corner']]['xy']and m['source']==by[m['native_corner']]['source'],'Original correction xy/source binding differs')
                opath=guarded/'e_run/observations'/f'{i:03d}_E_FROZEN.json'; spath=guarded/'e_run/poses'/f'{i:03d}_E_FROZEN.json'
                require(sha(opath)==postseal['observation_sha256'][opath.name],'Post-E observation seal changed')
                o,s=read(opath),read(spath);require(o['id']==s['id']==ident,'Own post-E identity differs')
                require(digest(o['result'])==o['result_sha256']and digest(s['result'])==s['result_sha256'],'Post-E result bytes differ')
                obs=o['result']['observation'];sol=s['result'];mpath=guarded/'e_run'/o['result']['mask_binding']['path']
                require(sha(mpath)==o['result']['mask_binding']['sha256'],'Own post-E mask changed')
                mask=r.get('mask_binding');require(mask is not None and sha(root/mask['path'])==mask['sha256'],'Original mask binding missing/changed')
                bindings[mask['path']]=mask['sha256']
                packet=build_packet(r,f,dict(archive_path=str(ET.ARCHIVE/'INPUTS.jsonl.gz'),archive_sha256=aseal['input_sha256'],archive_line_number=i+1,input_row_sha256=digest(r),
                    original_mask_binding=deepcopy(mask),original_correction_binding=correction['binding'],source_e_arm='SAME_FROZEN_E',
                    original_e_final_path=str(ET.FINAL/'PREDICTIONS.jsonl.gz'),original_e_final_sha256=bindings[str(ET.FINAL/'PREDICTIONS.jsonl.gz')]))
                audit=audit_row(r,packet,f,obs,sol,correction['correction'])
                rays=_rays(correction['correction'],packet['solver_input']['boundary_observations'],r['image_sha256'])
                raw_before=r['finish_input']['_original_source'];a=packet['activation']
                trace=dict(schema=SCHEMA,id=ident,partition=r['partition'],session=r['session'],image_sha256=r['image_sha256'],initial_points=deepcopy(raw_before['original_raw_points']),initial_pose=deepcopy(raw_before['original_coarse_pose']),
                    states=deepcopy(raw_before['states']),state_records=deepcopy(raw_before['state_observation']['records']),native_xyz=deepcopy(raw_before['initial_geometry']['native_xyz']),K=deepcopy(r['spec']['K']),raw_hw=deepcopy(r['spec']['image_hw']),
                    measurements=deepcopy(r['spec']['visible']),consumer_authoritative_activation=deepcopy(a),final_complete_measurements=[deepcopy(m)for m in r['spec']['visible']if m['native_corner']in a['complete_ids']],
                    archived_visible_observations=deepcopy(r['baseline']['solution'].get('visible_observations',[])),rays=rays,regions=dict(OT=deepcopy(r['spec']['domains']),U=deepcopy(r['u_regions']),partial_BI=deepcopy(packet['solver_input']['boundary_observations'])),
                    final_pose=deepcopy(r['baseline']['actual_pose']),mask_binding=deepcopy(mask),selected_candidate_audit=dict(accepted=r['baseline']['accepted'],WD_family=r['fixed_family'],method=r['baseline']['method'],solution_schema=r['baseline']['solution']['schema'],hierarchy=deepcopy(r['baseline'].get('hierarchical_correspondence_audit')),objective_components=deepcopy(r['baseline']['solution'].get('objective_components'))),
                    provenance=dict(**deepcopy(packet['provenance']),post_e_E_FROZEN_reobservations_used=False,legacy_e_trace_classifier_retained_in_activation=True,original_e_pose_numeric_bridge_max_abs=ET.pose_check(r['baseline']['actual_pose'],f['arms']['E_FROZEN']['coarse_pose'],ident)))
                record=dict(schema=SCHEMA,id=ident,partition=r['partition'],image_sha256=r['image_sha256'],
                    A_original_N3=dict(points=deepcopy(raw_before['original_raw_points']),pose=deepcopy(raw_before['original_coarse_pose']),geometry=deepcopy(raw_before['initial_geometry'])),
                    B_original_pre_e=dict(states=deepcopy(raw_before['states']),state_observation=deepcopy(raw_before['state_observation']),mask_binding=deepcopy(mask),correction_binding=deepcopy(correction['binding']),correction=deepcopy(correction['correction'])),
                    C_original_e=dict(activation=deepcopy(a),registered_family=deepcopy(r['family']),boundary_observations=deepcopy(packet['solver_input']['boundary_observations']),domains=deepcopy(r['spec']['domains']),u_regions=deepcopy(r['u_regions']),outline=deepcopy(r['outline']),actual_pose=deepcopy(r['baseline']['actual_pose'])),
                    D_post_e=dict(native_vertex_mapping=deepcopy(f['arms']['E_FROZEN']['native_vertex_mapping']),native_points=deepcopy(f['arms']['E_FROZEN']['native_points']),observation=obs,observation_binding=dict(path=str(opath.relative_to(guarded)),sha256=sha(opath)),mask_binding=deepcopy(o['result']['mask_binding'])),
                    E_post_e_solve=dict(solution=sol,predictions=post_predictions[ident],solution_binding=dict(path=str(spath.relative_to(guarded)),sha256=sha(spath))),
                    audit=audit,rays=rays,source_groups_are_dependency_diagnostics_not_independence_proofs=True,semantic_owner_status=SEMANTIC_OWNER)
                _emit(packets,packet);_emit(traces,trace);_emit(lineage,record)
                if writer is None:writer=csv.DictWriter(csvfile,fieldnames=list(audit[0]));writer.writeheader()
                writer.writerows(audit)
                totals.update(rows=1,visible_archives=len(r['spec']['visible']),complete=len(a['complete_ids']),typed_partial=len(a['typed_partial_ids']),legacy_partial=len(a['legacy_partial_ids']),OT_domains=len(r['spec']['domains']),U_regions=len(r['u_regions']))
                totals['rows_with_partial']+=bool(a['all_partial_ids']);totals['outline_supported']+=r['outline'].get('supported')is True
                totals['T_union_branch_count_max']=max(totals['T_union_branch_count_max'],packet['original_T_union']['branch_count'])
                totals['rows_with_multiple_T_union_branches']+=packet['original_T_union']['branch_count']>1
                totals['T_domains']+=packet['original_T_union']['T_domain_count']
                totals['selected_objective_component_parity_verified']+=a['selected_objective_component_parity']is True
                totals['selected_objective_component_absent']+=a['selected_objective_component_parity']is None
                for x in audit:
                    for key in ('old_e_trace_false_full','e_owner_partial_to_post_full','e_owner_typed_to_post_full','e_owner_legacy_to_post_full','e_owner_state_changed','N3_channel_state_changed_after_reindex','N3_to_e_same_channel_physical_owner_changed'):
                        totals[key]+=int(x[key])
                    if x['e_owner_partial_to_post_full']:transition_sources[x['post_e_source']]+=1
                old_sources.update(m['source']for m in r['spec']['visible']);totals['post_visible_backbone_fallback']+=sum(m['source']=='VISIBLE_BACKBONE_FALLBACK'for m in obs['frozen_visible_measurements'])
                groups=Counter(x['exact_sample_source_group']for x in rays if x['exact_sample_source_group']is not None)
                totals['exact_sample_groups_repeated']+=sum(n>1 for n in groups.values())
                post_bindings[ident]=dict(observation_sha256=sha(opath),solution_sha256=sha(spath),mask_sha256=sha(mpath))
    require(seen==ids,'Complete ordered238 extraction required')
    for path,h in bindings.items():require(sha(root/path)==h,'Original source changed during extraction')
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p))for p in output.iterdir()if p.is_file()}
    seal=dict(schema=SCHEMA,rows=238,ordered_ids=ids,partitions=PARTITIONS,totals=dict(totals),original_measurement_sources=dict(old_sources),partial_to_post_full_sources=dict(transition_sources),
        original_source_sha256=bindings,post_e_bindings=post_bindings,guarded_inputs_sha256=sha(guarded/'e_inputs/INPUTS.json'),guarded_prediction_sha256=sha(predpath),files=files,
        S_projection_only=True,semantic_owner_status=SEMANTIC_OWNER,new_NN_calls=0,new_PnP_calls=0,optimizer_updates=0,GT_or_reference_numeric_opens=0,
        wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256=sha(__file__),all_238_ids_explicit_from_guarded_inputs=True)
    code_root=Path(__file__).resolve().parents[3]
    source_names=('pallet_posefix_gap_audit_v1/e_trace.py','pallet_guarded_bi_final_v1/e_inputs.py',
        'pallet_guarded_bi_final_v1/guard.py',
        'pallet_visibility_curriculum_20261013_v1/statewise_partial_BI_u_solver_v4_13.py',
        'pallet_visibility_curriculum_20261013_v1/statewise_partial_BI_outline_solver_v4_13.py',
        'pallet_visibility_curriculum_20261013_v1/statewise_two_wd_family_v4_9.py',
        'pallet_visibility_curriculum_20261013_v1/evaluate_global_outline_pose_v4_10.py',
        'pallet_visibility_curriculum_20261013_v1/evaluate_face_partial_BI_outline_v4_13d.py',
        'pallet_visibility_curriculum_20261013_v1/evaluate_fixed_wd_archived_tangent_v4_15.py')
    seal['consumer_and_extractor_source_sha256']={str(Path('scripts/research')/s):sha(code_root/'scripts/research'/s)for s in source_names}
    require(seal['peak_rss_kib']<4*1024*1024,'Extraction exceeded4GiB memory bound')
    (output/'EXTRACTION_SEAL.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return seal


def iter_typed_packets(output_dir):
    folder=Path(output_dir);seal=read(folder/'EXTRACTION_SEAL.json');path=folder/'TYPED_PACKETS.jsonl.gz'
    require(sha(path)==seal['files'][path.name]['sha256'],'Typed packet seal changed')
    with gzip.open(path,'rt')as stream:
        for i,line in enumerate(stream):
            value=json.loads(line);require(value['id']==seal['ordered_ids'][i],'Typed packet order changed')
            yield value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--guarded-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    result=prepare(a.source_root,a.guarded_dir,a.output_dir)
    print(json.dumps({k:result[k]for k in ('rows','totals','wall_seconds','process_cpu_seconds','peak_rss_kib')},ensure_ascii=False))

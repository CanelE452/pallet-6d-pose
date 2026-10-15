"""Pure uniform V32 raw-archive to selected-observation adapter.

The raw extra queries stay immutable in the input/archive. Only F32 admitted
original/measured queries enter the graph, which receives no external pose
guide. No legacy selected graph, stored acceptance or guide bank is reused.
Source/initial-input ancestry and durable inventory seals are the caller's
responsibility. This function reads no files, RGB, references or pose banks.
"""
from collections import Counter
from copy import deepcopy
import time

import numpy as np

from . import initial_only_observed_ray_association_v4_32 as F
from . import rerank_observed_bi_edge_population_v4_30 as PARENT

C, R = PARENT.C, PARENT.D.R
RAW_OBSERVATION_GENERATION = F.P.OBSERVATION_GENERATION
SELECTION_PROFILE = F.SELECTION_PROFILE


def rerank_cached_row(row, configuration, mask):
    """Same arithmetic on CAL/TEST/REMAINDER, all zero rows retained."""
    if row['partition'] not in ('CAL', 'TEST', 'REMAINDER'):
        raise ValueError('Declared inventory partition required before payload')
    observation, geometry, states = row['observation'], row['initial_geometry'], row['states']
    if observation['observation_generation'] != RAW_OBSERVATION_GENERATION:
        raise ValueError('Original immutable raw V29 archive generation required')
    if F.P.VIS.mask_pixel_sha256(mask) != observation['target_mask_pixel_sha256']:
        raise ValueError('Frozen same-image target mask pixels required')
    binding = dict(sha256=row['image_sha256'], image_hw=geometry['image_hw'])
    context = F.P.VIS.prepare_edge_visibility(mask, geometry, states, row['visible_measurements'],
        image_binding=binding, occluder_mask=None, occluder_binding=None)
    before = R._canonical_sha(observation['chain_observation'])
    started = time.monotonic()
    ranked = F.rank_initial_only_observed_ray_graph(observation['chain_observation'], geometry,
        states, row['visible_measurements'], image_binding=binding, configuration=configuration,
        pose_guides=None, pallet_mask=mask, visibility_context=context)
    seconds = time.monotonic()-started
    graph = ranked['graph']
    cfg, _ = F.G._config(configuration)
    R.decorate_graph(graph, observation['chain_observation'], cfg)
    refs = PARENT.rebuild_selected_references(graph, observation['chain_observation'],
        states, geometry['initial_self_hidden'])
    visibility = []
    for ref in refs:
        ray = observation['chain_observation']['edge_records'][ref['ray_index']]
        chain = ray['chains'][ref['chain_index']]
        actual = np.asarray(chain['observed_points'])[np.asarray(chain['inlier_mask'], bool)]
        visibility.append(context.classify_expected_edge(ref['native_corner'], ref['native_neighbor'],
            observed_chain_points=actual, mask_context_kind=ray.get('mask_context_kind')))
    if R._canonical_sha(observation['chain_observation']) != before:
        raise ValueError('All raw alternatives, query identities and actual arrays are immutable')
    if ranked['external_pose_guides_consumed'] != 0:
        raise ValueError('No external pose guide influence permitted')
    # Full raw/archive containers remain source evidence; the numerical worker
    # should receive only the authenticated typed selected bundle. Old guide
    # fields below are preserved archive metadata, never current scoring input.
    new_observation = {**observation, 'joint_graph': graph, 'selected_chain_refs': refs,
        'edge_visibility': visibility, 'selected_physical_edges': len(refs),
        'parent_legacy_selected_physical_edges': observation['selected_physical_edges'],
        'raw_observation_generation': RAW_OBSERVATION_GENERATION,
        'selection_profile': SELECTION_PROFILE, 'cached_admission_generation': F.OBSERVATION_GENERATION,
        'association_recipe': deepcopy(F.RECIPE),
        'association_configuration': deepcopy(observation['chain_observation']['configuration']),
        'association_ledger': ranked['node_association_ledger'], 'association_raw_archive_sha256': before,
        'current_graph_geometry_source': 'ORIGINAL_INITIAL_ONLY',
        'external_pose_guides_consumed': 0,
        'preserved_old_guide_metadata_used_as_current_scoring_input': False,
        'old_V29_V30_V31_noise_validated_for_this_selection_population': False}
    summary = {**row['summary'], 'selected_physical_edges': len(refs),
        'selected_partial_edges': sum(ref['proposal_grade'] == F.P.PARTIAL_GRADE for ref in refs),
        'zero_selected_edges': not refs}
    new_row = {**row, 'observation': new_observation, 'summary': summary,
        'selection_profile': SELECTION_PROFILE, 'cached_actual_measurements_unchanged': True,
        'new_RGB_NMS_TLS_labels_posefit': 0}
    return new_row, dict(id=row['id'], partition=row['partition'], rank_seconds=seconds,
        selected=len(refs), zero_selected=not refs,
        checked=ranked['candidate_nodes_checked'], rejected=ranked['rejected_node_count'],
        rejection_reasons=dict(Counter(r['reason'] for r in ranked['node_association_ledger'] if not r['passes'])),
        unique_admitted_line_candidates=graph['actual_unique_line_candidates'],
        external_pose_guides_consumed=0, new_precise_KPs_or_PnP_or_labels=0)

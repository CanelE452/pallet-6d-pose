"""CAL46 metadata-only V30 association-selected actual-edge/tangent population.

The new producer's CAL prefix and immutable v25 physics inputs are joined.
Both source policies are planned at zero CAL variance before labels, with the
same actual graph winner and unchanged original observation archives. No RGB,
reference, objective, optimizer or primary-row decode is performed here.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
from itertools import zip_longest

from . import common as C
from . import rerank_observed_bi_edge_population_v4_30 as RERANK
from . import geometry_guided_edge_problem_v4_30 as G
from . import geometry_guided_edge_authentication_v4_30 as AUTH
from . import prepare_prepartial_RGB_evidence_v4_26 as PREFIX

B=PREFIX.B
TASKS=B/'coarse_prior_containment_eval_v4_25'
OBS=B/'geometry_guided_edge_association_observations_v4_30'
OUTPUT=B/'geometry_guided_native_evidence_v4_30_checks'
DECLARATION=B/'observed_bi_associated_edges_plan_v4_30/PREDECLARATION.json'
DECLARATION_SHA256='1598f5ac0a4308ce758f792656f05bfba7f351adcfd2fc1b4000f564a33d22b1'


def native_baseline(task,family):
    spec=task['spec']
    original=G.N.prepare_native_edge_problem(family['native_xyz'],spec['K'],spec['states'],spec['visible'],
        spec['domains'],spec['image_hw'],family['coarse_pose'],config={**spec['config'],'prior_weight':0.},
        boundary_observations=task['boundary_merged'],u_regions=task['u_regions'],
        edge_bundle=dict(edges=[],replaces_normal_factor_refs=[]))
    if not np.allclose(original.Q,family['physical_registration_Q'],atol=1e-12,rtol=0):
        raise ValueError('Trial native basis and physical Q must remain paired')
    return G.BASELINE.prepare_prepartial_native_problem(original,visible_measurements=spec['visible'],
                                                        initial_self_hidden=task['initial_self_hidden'])


def validate_join(task,row):
    if task['partition']!='CAL' or row['partition']!='CAL' or task['id']!=row['id']:
        raise ValueError('Exact ordered CAL46-only join required')
    return validate_runtime_join(task,row)


def validate_runtime_join(task,row):
    spec=task['spec'];geometry=row['initial_geometry'];family=task['families']['ORIGINAL_WD']
    for m in [*spec['visible'],*row['visible_measurements']]:
        k=G._id(m['native_corner'])
        if task['initial_self_hidden'][k]or task['states'][k]not in('B','I'):
            raise ValueError('NonS BI gate before observed coordinate/source comparisons')
    if (row['id']!=task['id']or row['states']!=task['states']or spec['states']!=task['states']
            or row['visible_measurements']!=spec['visible']or row['u_regions']!=task['u_regions']
            or row['domains']!=spec['domains']or row['image_sha256']!=task['image_sha256']
            or geometry['native_xyz']!=family['native_xyz']or geometry['physical_registration_Q']!=family['physical_registration_Q']
            or geometry['K']!=spec['K']or geometry['image_hw']!=spec['image_hw']
            or geometry['initial_self_hidden']!=task['initial_self_hidden']
            or geometry['initial_projection_xy']!=task['complete_BI_evidence']['initial_projection_xy']):
        raise ValueError('Original image/camera/geometry/channel/S/U/q/source/domain input changed')
    rotation=np.asarray(family['coarse_pose']['R']);translation=np.asarray(family['coarse_pose']['t'])
    center=rotation@np.asarray(family['native_xyz']).mean(axis=0)+translation
    if (not np.allclose(geometry['R_cf'],rotation,atol=1e-12,rtol=0)
            or not np.allclose(geometry['centroid'],center,atol=1e-12,rtol=0)):
        raise ValueError('Exact first native coarse rotation/model-center guide required')
    return geometry


def prepare_one(task,row):
    geometry=validate_join(task,row)
    if row.get('selection_profile')!=G.SELECTION_PROFILE or row['observation'].get('selection_profile')!=G.SELECTION_PROFILE:
        raise ValueError('Exact V30 selected observation/profile required')
    family=task['families']['ORIGINAL_WD'];base=native_baseline(task,family)
    bundle=AUTH.authenticate_geometry_guided_edges(base,row['observation'],
        image_binding=dict(sha256=task['image_sha256'],image_hw=task['spec']['image_hw']),initial_geometry=geometry,
        normal_variance_component_px2=0.,angular_variance_component_sine2=0.,
        noise_provenance=dict(fit_partition='CAL',observation_family=G.OBSERVATION_FAMILY,selection_profile=G.SELECTION_PROFILE,
                              mode='ZERO_COMPONENT_PREFIT_METADATA_NOT_FITTED'),calibration_preparation=True)
    profiles={};metadata={}
    for policy in G.SOURCE_POLICIES:
        plan=G.plan_geometry_guided_replacement(base,bundle,source_policy=policy)
        groups=[];tangents=[]
        for group in plan['active_groups']:
            component=deepcopy(group);component['mixture_component_prior']=1.
            groups.append(dict(physical_native_edge=group['physical_native_edge'],
                reliability_weight=group['fit_reliability_weight'],components=[component],
                selected_chain_ref=deepcopy(group['selected_chain_ref']),activation_owners=deepcopy(group['activation_owners'])))
        for record in plan['corner_plans']:
            if record['mode']!='ONE_SELECTED_LINE_ARCHIVED_Q_TANGENT':continue
            archive=record['source_archive']
            tangents.append(dict(kind='ARCHIVED_Q_SELECTED_MEASURED_LINE_TANGENT',native_corner=record['native_corner'],
                state=record['state'],source=archive['original_measurement_source'],archived_q_xy=archive['original_observed_xy'],
                selected_line_tangent_unit_xy=record['selected_line_tangent_unit_xy'],
                effective_fit_covariance_px2=archive['original_effective_fit_covariance_px2'],
                fit_reliability_weight=archive['original_fit_reliability_weight'],selected_line_ref=record['selected_line_ref']))
        profiles[policy]=dict(groups=groups,tangent_records=tangents,runtime_plan_digest=G._digest(plan),selection_profile=G.SELECTION_PROFILE)
        metadata[policy]=dict(groups=len(groups),tangents=len(tangents),owners=len(plan['new_owner_ids']),
            modes=dict(Counter(r['mode']for r in plan['corner_plans'])),ledger=plan['ledger'])
    prepared=dict(id=task['id'],partition='CAL',session=task['session'],states=task['states'],initial_geometry=geometry,
        observation_generation=G.OBSERVATION_FAMILY,raw_observation_generation=AUTH.PRODUCER_GENERATION,
        noise_population=G.POPULATION,selection_profile=G.SELECTION_PROFILE,profiles=profiles)
    return prepared,dict(id=task['id'],session=task['session'],profiles=metadata,
        authenticated_groups=len(bundle['groups']),skipped=bundle['skipped'],
        cross_native_overlap_withheld=bundle['cross_native_support_overlap_withheld_component_ids'])


def main(output=OUTPUT,*,root_metadata_GO=False):
    if root_metadata_GO is not True:raise PermissionError('Root actual CAL metadata GO before IO')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every CAL metadata generation')
    if C.sha(DECLARATION)!=DECLARATION_SHA256:raise ValueError('Exact central V30 declaration required')
    central=C.read(DECLARATION)
    for key in('source_sha256','input_sha256','protected_user_sha256'):PREFIX._verify(central[key])
    protocol=C.read(TASKS/'PROTOCOL.json');taskseal=C.read(TASKS/'INPUT_SEAL.json')
    obsseal=C.read(OBS/'CAL_OBSERVATION_SEAL.json');journal=OBS/'CAL_OBSERVATIONS.jsonl.gz'
    obs_protocol=C.read(OBS/'PROTOCOL.json');obs_preparation=C.read(OBS/'PREPARATION_SEAL.json')
    ids=obsseal['ordered_ids']
    if len(ids)!=46 or ids!=protocol['ids'][:46]or len(set(ids))!=46:
        raise ValueError('Original complete CAL46 prefix required')
    if C.sha(TASKS/'INPUTS.jsonl.gz')!=taskseal['input_sha256']:
        raise ValueError('Frozen238 physics inputs required')
    if C.sha(journal)!=obsseal['observations_sha256']:
        raise ValueError('New actual CAL observation journal seal required')
    if (obsseal['protocol_sha256']!=C.sha(OBS/'PROTOCOL.json')
            or obsseal['preparation_seal_sha256']!=C.sha(OBS/'PREPARATION_SEAL.json')
            or obs_preparation['protocol_sha256']!=C.sha(OBS/'PROTOCOL.json')):
        raise ValueError('Observation CAL/prepare/protocol source-input closure required')
    for name in ('source_sha256','input_sha256','protected_user_sha256'):PREFIX._verify(obs_protocol[name])
    source=dict(protocol['source_sha256'])
    for module in (AUTH,G,G.BASELINE,G.N,G.J,G.P,AUTH.RELATION,AUTH.OBS,C,PREFIX,RERANK):
        path=Path(module.__file__).resolve();source[str(path.relative_to(C.ROOT))]=C.sha(path)
    source[str(Path(__file__).resolve().relative_to(C.ROOT))]=C.sha(__file__)
    inputs={str(p.relative_to(C.ROOT)):C.sha(p)for p in (DECLARATION,TASKS/'PROTOCOL.json',TASKS/'INPUT_SEAL.json',
        TASKS/'INPUTS.jsonl.gz',OBS/'CAL_OBSERVATION_SEAL.json',journal,OBS/'PROTOCOL.json',OBS/'PREPARATION_SEAL.json')}
    protected=dict(protocol['protected_user_sha256'])
    RERANK.bind_central_maps(source,inputs,protected,central)
    # Include the actual observation producer's own frozen source union.
    for relative,digest in obs_protocol['source_sha256'].items():
        if relative in source and source[relative]!=digest:raise ValueError('Frozen observation source ancestry conflict')
        source[relative]=digest
    for mapping in (source,inputs,protected):PREFIX._verify(mapping)
    output.mkdir(parents=True)
    C.write(output/'PREPARATION_SEAL.json',dict(schema='ZERO_COMPONENT_CAL46_V30_ASSOCIATION_METADATA_PREPARATION',
        ids=ids,source_sha256=source,input_sha256=inputs,protected_user_sha256=protected,
        before_CAL_reference_decode=True,CAL_labels_read=False,TEST_REMAINDER_rows_decoded=0,
        central_declaration_sha256=DECLARATION_SHA256))
    for relative,digest in source.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot mismatch')
    records=[];errors=[];observed_ids=[]
    def prepared_rows():
        for task,row in zip_longest(PREFIX._cal_task_prefix(TASKS/'INPUTS.jsonl.gz',ids),C.rows(journal)):
            if task is None or row is None:raise ValueError('Exact CAL-only observation row count required')
            observed_ids.append(row['id'])
            try:
                item,record=prepare_one(task,row);records.append(record);yield item
            except Exception as error:errors.append(dict(id=task['id'],exception=type(error).__name__,reason=str(error)))
    C.save_rows(output/'CAL_PREPARED_GROUPS.jsonl.gz',prepared_rows())
    if observed_ids!=ids:raise ValueError('Exact CAL-only observation order required')
    if len(records)+len(errors)!=46:raise ValueError('Keep all CAL inventory including failures')
    report=dict(rows=46,successful_metadata_rows=len(records),error_count=len(errors),errors=errors,records=records,
        policy_totals={p:{key:sum(r['profiles'][p][key]for r in records)for key in ('groups','tangents','owners')}for p in G.SOURCE_POLICIES},
        observation_generation=G.OBSERVATION_FAMILY,raw_observation_generation=AUTH.PRODUCER_GENERATION,
        selection_profile=G.SELECTION_PROFILE,noise_population=G.POPULATION,
        CAL_reference_decodes=0,primary_rows_decoded=0,pose_objective_or_fit_calls=0,new_image_queries=0,
        selected_graph_population_and_measured_tangent_axes_frozen_before_labels=True)
    C.write(output/'CAL_PREFLIGHT_V1.json',report)
    for mapping in (source,inputs,protected):PREFIX._verify(mapping)
    C.write(output/'CAL_PREFLIGHT_SEAL_V1.json',dict(preparation_sha256=C.sha(output/'PREPARATION_SEAL.json'),
        report_sha256=C.sha(output/'CAL_PREFLIGHT_V1.json'),prepared_groups_sha256=C.sha(output/'CAL_PREPARED_GROUPS.jsonl.gz')))
    print(dict(CAL46=len(records),errors=len(errors),policy_totals=report['policy_totals'],CAL_labels=0),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--root-metadata-GO',action='store_true');args=parser.parse_args()
    main(args.output,root_metadata_GO=args.root_metadata_GO)

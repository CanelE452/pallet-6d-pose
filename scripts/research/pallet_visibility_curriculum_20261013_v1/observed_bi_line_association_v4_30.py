"""Opt-in NEW observed-B/I association population for cached actual TLS lines.

Only candidate admission changes. Raw observations retain every chain and
original index. The old heap, memo, score arithmetic and graph ordering remain
unchanged. This new population needs its own CAL/noise validation; V29 models
must not be relabelled or treated as validated for it.
"""
from __future__ import annotations
import ast
from copy import deepcopy
from types import FunctionType,SimpleNamespace
import numpy as np

from . import image_edge_observations_v4_29 as P
from . import bounded_geometry_graph_rank_v4_29 as B
from . import geometry_graph_exact_memo_v4_29 as M

G=P.G
SELECTION_PROFILE='ALL_ELIGIBLE_NAMED_BI_ENDPOINTS_TO_ACTUAL_INFINITE_TLS_WITHIN_EXISTING_NORMAL_REACH_V30'
OBSERVATION_GENERATION='CACHED_FULL_PROVENANCE_V29_WITH_OBSERVED_BI_ASSOCIATION_SELECTION_V30'
RECIPE=dict(selection_profile=SELECTION_PROFILE,observation_generation=OBSERVATION_GENERATION,
    parent_raw_observation_generation=P.OBSERVATION_GENERATION,
    gate='all original eligible nonS B/I endpoints of named edge within cfg.normal_search_radius_px of infinite actual TLS line',
    threshold='existing configured normal search reach43px; no tuned constant',
    minimum_eligible_BI_endpoints=1,both_BI_endpoints_checked=True,
    Euclidean_corner_cap_or_along_line_cap=False,old_V29_noise_validated_for_new_population=False,
    raw_chain_archive_and_original_indices_unchanged=True,new_point_or_pose_adoption=False,
    next_scope='separate CAL46 population/noise and uniform evaluation required before method adoption')


def assess_observed_BI_line_association(node,points,states,initial_self_hidden,cfg):
    """State gates precede per-endpoint coordinates and measured-line fields."""
    edge=tuple(node['native_edge'])
    if (len(edge)!=2 or any(isinstance(k,(bool,np.bool_))or not isinstance(k,(int,np.integer))or not 0<=k<8 for k in edge)
            or edge[0]==edge[1]):raise ValueError('Two strict named native endpoints required')
    if any(initial_self_hidden[k]or states[k]=='S'for k in edge):
        return dict(passes=False,native_edge=list(edge),reason='initial_S_gate_before_line_or_q_fields',endpoints=[])
    eligible=[int(k)for k in edge if states[k]in('B','I')and k in points]
    if not eligible:return dict(passes=False,native_edge=list(edge),reason='no_eligible_original_BI_endpoint',endpoints=[])
    limit=float(cfg['normal_search_radius_px'])
    if not np.isfinite(limit)or limit<=0:raise ValueError('Existing positive configured normal reach required')
    n=np.asarray(node['normal'],float);offset=float(node['offset'])
    if (n.shape!=(2,)or not np.isfinite([*n,offset]).all()
            or not np.isclose(n@n,1.,atol=1e-10,rtol=0)):raise ValueError('Fresh-node normalized actual TLS required')
    endpoints=[]
    for k in eligible:
        record=points[k]['record'];q=np.asarray(record['xy'],float)
        if q.shape!=(2,)or not np.isfinite(q).all():raise ValueError('Frozen eligible B/I q must be finite')
        signed=float(n@q+offset)
        endpoints.append(dict(native_corner=k,state=states[k],original_q_xy=q.tolist(),
            original_measurement_source=record['source'],signed_distance_to_infinite_actual_TLS_px=signed,
            absolute_distance_px=abs(signed),passes=abs(signed)<=limit))
    passes=all(e['passes']for e in endpoints)
    return dict(passes=passes,native_edge=list(edge),configured_normal_reach_px=limit,
        eligible_BI_endpoint_count=len(eligible),endpoints=endpoints,
        reason='all_original_BI_normal_associations_pass'if passes else'original_BI_endpoint_outside_existing_normal_reach',
        unknown_S_U_endpoint_coordinates_read=0,along_line_or_Euclidean_distance_not_gated=True)


def _filtered_rank_code():
    """Add one continue after frozen _node, plus the exact old heap adapter."""
    sites=[]
    def parse(source,*args,**kwargs):
        tree=ast.parse(source,*args,**kwargs)
        if (tree.body and isinstance(tree.body[0],ast.FunctionDef)
                and tree.body[0].name=='rank_joint_geometry_line_graph'):
            for loop in ast.walk(tree):
                if not isinstance(loop,ast.For):continue
                for i,statement in enumerate(loop.body):
                    if (isinstance(statement,ast.Assign)and isinstance(statement.value,ast.Call)
                            and isinstance(statement.value.func,ast.Name)and statement.value.func.id=='_node'):
                        guard=ast.parse('if not _association_pass(node,points,states,hidden,cfg):\n    continue\n').body[0]
                        loop.body.insert(i+1,guard);sites.append(True);break
            if len(sites)!=1:raise ValueError('Exactly one fresh-node association site required')
        return tree
    # B._rank_code still pins the original parent SHA and verifies its exact
    # cache/beam blocks. Only this private parser facade sees the extra guard.
    namespace=dict(B._rank_code.__globals__)
    namespace['ast']=SimpleNamespace(**{**vars(ast),'parse':parse})
    code=FunctionType(B._rank_code.__code__,namespace)()
    if len(sites)!=1:raise ValueError('Frozen rank AST association insertion unavailable')
    return code


def rank_observed_BI_associated_graph(chain_observation,geometry,states,visible_measurements,*,
        image_binding,configuration=None,pose_guides=None,pallet_mask=None,visibility_context=None,
        maximum_cache_entries=4096,intersection_cache_limit=4096):
    ledger=[]
    def check(node,points,current_states,hidden,cfg):
        verdict=assess_observed_BI_line_association(node,points,current_states,hidden,cfg)
        ledger.append(dict(line_name=node['name'],node_provenance=list(node['provenance']),
            source_native_corner=node['native_corner'],source_native_neighbor=node['native_neighbor'],**verdict))
        return verdict['passes']
    namespace=dict(G.rank_joint_geometry_line_graph.__globals__)
    namespace.update(_association_pass=check,_node=P._fresh_node,_score_graph=M.make_ordered_scorer(maximum_cache_entries=maximum_cache_entries),
        _bounded_beam_step=B.bounded_beam_step,_new_intersection_cache=lambda:B.BoundedIntersectionCache(intersection_cache_limit))
    rank=FunctionType(_filtered_rank_code(),namespace)
    rank.__kwdefaults__=deepcopy(G.rank_joint_geometry_line_graph.__kwdefaults__)
    graph=rank(chain_observation,geometry,states,visible_measurements,image_binding=image_binding,
        configuration=configuration,pose_guides=pose_guides,pallet_mask=pallet_mask,visibility_context=visibility_context)
    graph['joint_alternatives']=[a for a in graph['joint_alternatives']if not a['ordered_support_unavailable']]
    return dict(selection_profile=SELECTION_PROFILE,observation_generation=OBSERVATION_GENERATION,
        parent_raw_observation_generation=P.OBSERVATION_GENERATION,graph=graph,node_association_ledger=ledger,
        rejected_node_count=sum(not r['passes']for r in ledger),candidate_nodes_checked=len(ledger),
        raw_archive_unchanged=True,original_chain_indices_and_node_provenance_preserved=True,
        old_V29_noise_validated_for_this_population=False,new_CAL46_noise_and_uniform_validation_required=True,
        new_RGB_NMS_TLS_posefit_labels_or_precise_KPs=0)

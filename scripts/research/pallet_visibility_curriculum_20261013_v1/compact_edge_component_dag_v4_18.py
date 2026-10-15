"""Diagnostic compact NMS component graph; no path enumeration or selection.

Raw actual components and all admissible directed links are sufficient to
represent subset paths. Distinct overlapping parallel components remain
alternatives. A whole connected group is only an optional line family when
its actual-point union passes the same TLS/support gates without dropping a
component. This is a research representation, not a production observer.
"""
from collections import defaultdict,deque
import numpy as np
from . import boundary_refine as F


def graph_summary(links,isolated_nodes=()):
    edges={(int(r['components'][0]),int(r['components'][1])) for r in links}
    nodes=set(map(int,isolated_nodes))|{v for pair in edges for v in pair}
    successors={v:set() for v in nodes};predecessors={v:set() for v in nodes}
    for a,b in edges:successors[a].add(b);predecessors[b].add(a)
    indegree={v:len(predecessors[v]) for v in nodes};queue=deque(sorted(v for v in nodes if not indegree[v]));order=[]
    while queue:
        node=queue.popleft();order.append(node)
        for after in sorted(successors[node]):
            indegree[after]-=1
            if not indegree[after]:queue.append(after)
    if len(order)!=len(nodes):raise ValueError('Actual along-edge component links must be acyclic')
    descendants={v:set() for v in nodes};paths={}
    for node in reversed(order):
        paths[node]=sum(paths[v] for v in successors[node]) if successors[node] else 1
        for after in successors[node]:descendants[node].add(after);descendants[node]|=descendants[after]
    reduced=[]
    for a,b in sorted(edges):
        if not any(b==other or b in descendants[other] for other in successors[a] if other!=b):
            reduced.append((a,b))
    reduced_successors={v:set() for v in nodes}
    for a,b in reduced:reduced_successors[a].add(b)
    reduced_paths={}
    for node in reversed(order):
        reduced_paths[node]=sum(reduced_paths[v] for v in reduced_successors[node]) if reduced_successors[node] else 1
    roots=[v for v in order if not predecessors[v]]
    unseen=set(nodes);groups=[]
    while unseen:
        seed=min(unseen);stack=[seed];group=set()
        while stack:
            node=stack.pop()
            if node in group:continue
            group.add(node);stack.extend(successors[node]|predecessors[node])
        unseen-=group;groups.append(sorted(group))
    return dict(nodes=sorted(nodes),edges=[list(v) for v in sorted(edges)],topological_order=order,
        connected_groups=groups,transitive_reduction_edges=[list(v) for v in reduced],
        all_start_terminal_path_count_upper_bound=str(sum(paths[v] for v in nodes)),
        nontrivial_all_start_path_count_upper_bound=str(sum(paths[v] for v in nodes if successors[v])),
        root_terminal_path_count_upper_bound=str(sum(paths[v] for v in roots)),
        transitive_reduced_root_path_count=str(sum(reduced_paths[v] for v in roots)),
        path_counts_are_not_independent_physical_alternatives=True,path_sets_enumerated=0)


def compact_actual_component_groups(xy,observed,valid,component,direction,seed_normal,cfg,*,
                                      confidence,positions,links):
    observed=np.asarray(observed,float);xy=np.asarray(xy,float);valid=np.asarray(valid,bool)
    component=np.asarray(component,int);confidence=np.asarray(confidence,float);positions=np.asarray(positions,float)
    if (observed.shape!=xy.shape or observed.ndim!=2 or observed.shape[1]!=2
            or any(v.shape!=(len(observed),) for v in (valid,component,confidence,positions))
            or not np.isfinite(observed).all() or not np.isfinite(confidence).all()):
        raise ValueError('Bound actual NMS points/components/confidence required')
    groups={int(label):np.flatnonzero(valid&(component==label)) for label in np.unique(component[valid]) if label>0}
    graph=graph_summary(links,groups)
    raw=[dict(component=label,actual_image_lattice_xy=xy[idx].tolist(),actual_NMS_points_xy=observed[idx].tolist(),
        sample_confidence=confidence[idx].tolist(),sample_position=positions[idx].tolist()) for label,idx in groups.items()]
    families=[];unresolved=[]
    line_cfg=dict(along_interval=[0.,1.],huber_delta_px=cfg['maximum_line_rms_px'],
        maximum_angle_change_deg=cfg['maximum_direction_change_deg'],minimum_inlier_count=cfg['minimum_inlier_count'],
        minimum_inlier_fraction=cfg['minimum_inlier_fraction'],minimum_inlier_span_fraction=cfg['minimum_inlier_span_fraction'],
        maximum_line_rms_px=cfg['maximum_line_rms_px'])
    for labels in graph['connected_groups']:
        intervals={label:(float((observed[groups[label]]@direction).min()),float((observed[groups[label]]@direction).max())) for label in labels}
        overlapping=[]
        for i,a in enumerate(labels):
            for b in labels[i+1:]:
                overlap=min(intervals[a][1],intervals[b][1])-max(intervals[a][0],intervals[b][0])
                if overlap>1e-9:overlapping.append(dict(components=[a,b],tangent_overlap_px=overlap))
        indices=np.concatenate([groups[label] for label in labels])
        if overlapping:
            unresolved.append(dict(components=labels,reason='overlapping_actual_components_remain_parallel_alternatives',
                                   overlaps=overlapping));continue
        fitted,diag=F._fit_line(observed[indices],confidence[indices],positions[indices],
                               np.asarray(seed_normal),len(indices),line_cfg)
        if fitted is None:
            unresolved.append(dict(components=labels,reason=diag['reason'],union_TLS_audit=diag));continue
        good=np.asarray(diag['inlier_mask'],bool)
        per_component={label:float(np.mean(good[np.isin(indices,groups[label])])) for label in labels}
        if min(per_component.values())<cfg['minimum_inlier_fraction']:
            unresolved.append(dict(components=labels,reason='union_TLS_would_discard_a_raw_component',
                                   per_component_inlier_fraction=per_component,union_TLS_audit=diag));continue
        families.append(dict(components=labels,actual_point_count=len(indices),
            per_component_inlier_fraction=per_component,union_TLS=diag,
            invented_bridge_points=0,path_multiplicity_not_used_as_evidence_weight=True,
            native_edge_identity_not_certified=True))
    return dict(schema='COMPACT_ACTUAL_NMS_COMPONENT_DAG_DIAGNOSTIC_V4_18',graph=graph,
        raw_components=raw,whole_component_line_families=families,unresolved_groups=unresolved,
        raw_components_discarded=0,best_RMS_or_GT_family_selection=False,production_observer_changed=False,
        new_CAL_model_required_before_any_changed_representation_is_adopted=True)

"""Opt-in exact repeated-term memoization of the frozen ordered line scorer.

Create one scorer per rank invocation. This module does not install a hook or
change a producer. Cache eviction only causes the same frozen arithmetic to
run again. Guide means, reductions and final addition order remain unchanged.
"""
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np

from . import joint_geometry_guided_lines_v4_28 as G
from . import ordered_geometry_support_v4_28b as ORDER


G_SOURCE_SHA256='c6ebc3b515a25eabdf6d6d100c620d7c6d52305821d201edfac3c949a2bf4a57'
ORDER_SOURCE_SHA256='2dedbc9ce15fffdec708da01af4e75750ef8f1b0bb6d06e30c54c228097095fd'
RECIPE=dict(scope='opt-in exact arithmetic memo; not installed in actual observation',
    context='one rank, one frozen points/states/guide-bank/configuration identity',
    keys='corner/axis and selected existing node names in original order; ordered pair names',
    computation='first evaluation uses frozen numpy operations and reduction order',
    cache='separate bounded LRU caches; eviction recomputes, no candidate pruning',
    new_measurements_models_thresholds_or_scores=False)


class _LRU:
    def __init__(self,limit):
        self.limit=limit
        self.values=OrderedDict()
        self.hits=0
        self.misses=0

    def get(self,key,calculate):
        if key in self.values:
            self.hits+=1
            value=self.values[key]
            self.values.move_to_end(key)
            return value
        self.misses+=1
        value=calculate()
        self.values[key]=value
        while len(self.values)>self.limit:
            OrderedDict.__delitem__(self.values,next(iter(self.values)))
        return value


def _corner(k,incident,points,states,guide_bank,cfg,graph_cfg):
    """The frozen G._score_graph per-corner block, in its original order."""
    margin=float(cfg['normal_search_radius_px'])
    role_cost=(1-max(n['actual_outline_fraction']*(n['global_external_context']['external_support_fraction']
        if n['global_external_context']['available'] else .5) for n in incident)) if states[k]=='B' else 0.
    intersections=[]
    for a,b in combinations(incident,2):
        key=tuple(sorted([a['name'],b['name']]))
        cache=guide_bank['_cached_intersections']
        if key not in cache:
            cache[key]=G._intersection((a['normal'],a['offset']), (b['normal'],b['offset']), cfg['minimum_intersection_sine'])
        p,sine=cache[key]
        if p is not None:intersections.append(p)
    scatter=0.;anchor_cost=0.;junction=None
    if intersections:
        center=np.median(intersections,axis=0)
        scatter=float(np.mean(np.sum((np.asarray(intersections)-center)**2,axis=1))/margin**2)
        anchor_cost=float(np.sum((center-np.asarray(points[k]['record']['xy']))**2)/margin**2)
        junction=dict(native_corner=k,state=states[k],line_pair_intersections_xy=np.asarray(intersections).tolist(),
            intersection_scatter_px2=scatter*margin**2,
            observed_BI_to_intersection_shift_px=float(np.sqrt(anchor_cost)*margin),
            old_12px_corner_cap_would_pass=bool(np.sqrt(anchor_cost)*margin<=cfg['maximum_corner_shift_px']),
            full_2D_corner_measurement=False)
    return role_cost+scatter+graph_cfg['B_or_I_anchor_displacement_weight']*anchor_cost,junction


def _axis(group,guide_bank):
    """The frozen per-axis SVD/ray/concurrency block, without rearrangement."""
    lines=np.array([n['normal']+[n['offset']] for n in group],float)
    h,w=guide_bank['image_binding']['image_hw']
    transform=np.array([[w,0,w/2],[0,h,h/2],[0,0,1.]])
    normalized=lines@transform
    normalized/=np.linalg.norm(normalized[:,:2],axis=1)[:,None]
    _,_,vectors=np.linalg.svd(normalized,full_matrices=True)
    vp=transform@vectors[-1]
    ray=np.linalg.solve(guide_bank['_cached_K'],vp)
    measured=ray/np.linalg.norm(ray) if np.linalg.norm(ray)>1e-12 else None
    residual=[]
    for n in group:
        try:residual.append(float(np.asarray(n['normal'])@G.vanishing_direction_at(vp,n['support_anchor_xy']))**2)
        except ValueError:pass
    concurrency=float(np.mean(residual)) if residual and len(group)>=3 else None
    return measured,concurrency


def _ordered_pair(k,a,b,points,guide_bank,cfg):
    """The frozen ORDER._ordered_score pair's angle and two support checks."""
    key=tuple(sorted([a['name'],b['name']]))
    cache=guide_bank['_cached_intersections']
    if key not in cache:
        cache[key]=G._intersection((a['normal'],a['offset']),(b['normal'],b['offset']),cfg['minimum_intersection_sine'])
    p,sine=cache[key]
    angle=dict(native_corner=k,line_names=[a['name'],b['name']],
        sine=sine,minimum_sine=cfg['minimum_intersection_sine'],passes=p is not None)
    checks=[]
    if p is not None:
        for node in (a,b):
            mate=next(i for i in node['native_edge']if i!=k)
            checks.append(ORDER._support(k,mate,p,node,points,guide_bank,cfg))
    return angle,checks


class _OrderedScorer:
    def __init__(self,limit):
        self._context=None
        self._state_values=None
        self._node_ids={}
        self.corner=_LRU(limit)
        self.axis=_LRU(limit)
        self.ordered_pair=_LRU(limit)

    def _bind(self,selected,points,states,guide_bank,cfg,graph_cfg):
        context=(points,states,guide_bank,cfg,graph_cfg)
        if self._context is None:
            # Strong references prevent recycled object IDs from passing.
            self._context=context
            self._state_values=tuple(states)
        if (not all(a is b for a,b in zip(context,self._context))
                or tuple(states)!=self._state_values):
            raise AssertionError('A memo scorer belongs to one immutable rank context; create a new instance')
        for node in selected:
            if node is None:continue
            name=node['name']
            if name in self._node_ids and self._node_ids[name]!=id(node):
                raise AssertionError('Existing node names must keep the same immutable node identity')
            self._node_ids[name]=id(node)

    def __call__(self,selected,points,states,guide_bank,cfg,graph_cfg):
        self._bind(selected,points,states,guide_bank,cfg,graph_cfg)
        image_energy=sum(n['image_quality_energy']if n is not None else graph_cfg['missing_edge_cost']for n in selected)
        nodes=[n for n in selected if n is not None]
        incident_terms=[];junctions=[]
        for k in sorted(points):
            if states[k]not in('B','I'):continue
            incident=[n for n in nodes if k in n['native_edge']]
            if not incident:continue
            key=(k,tuple(n['name']for n in incident))
            term,junction=self.corner.get(key,lambda:_corner(k,incident,points,states,guide_bank,cfg,graph_cfg))
            incident_terms.append(term)
            if junction is not None:junctions.append(deepcopy(junction))
        # Keep the frozen node traversal, normalized means and dict order.
        guide_costs={}
        for node in nodes:
            for name,value in guide_bank['_cached_node_guide_costs'][node['name']].items():
                guide_costs.setdefault(name,{})[node['name']]=value
        means={name:float(np.mean(list(values.values())))for name,values in guide_costs.items()if values}
        weak=min(means.values())if means else 0.
        concurrency=[];measured_axis_rays={}
        axis_edges=guide_bank['_cached_edge_axes']
        for axis in range(3):
            group=[n for n in nodes if axis_edges.get(tuple(n['native_edge']))==axis]
            if len(group)<2:continue
            key=(axis,tuple(n['name']for n in group))
            ray,value=self.axis.get(key,lambda:_axis(group,guide_bank))
            if ray is not None:measured_axis_rays[axis]=ray
            if value is not None:concurrency.append(value)
        orthogonality=[float(a@b)**2 for a,b in combinations(measured_axis_rays.values(),2)]
        orthogonality_cost=float(np.mean(orthogonality))if orthogonality else 0.
        cost=(graph_cfg['image_quality_weight']*image_energy
            +graph_cfg['incident_graph_weight']*sum(incident_terms)
            +graph_cfg['shared_axis_concurrency_weight']*(sum(concurrency)+orthogonality_cost)
            +graph_cfg['coarse_guide_weight']*weak)
        score=dict(total_engineering_energy=float(cost),image_quality_energy=float(image_energy),
            incident_graph_energy=float(sum(incident_terms)),shared_axis_concurrency_energy=float(sum(concurrency)),
            image_measured_axis_orthogonality_energy=orthogonality_cost,
            independently_measured_axis_ray_count=len(measured_axis_rays),
            weak_normalized_coarse_guide_energy=weak,normalized_guide_means=means,
            guide_term_count=1 if means else 0,junctions=junctions)
        checks=[];angle_checks=[]
        for k in sorted(points):
            if states[k]not in('B','I'):continue
            incident=[n for n in nodes if k in n['native_edge']]
            for a,b in combinations(incident,2):
                key=(k,a['name'],b['name'])
                angle,support=self.ordered_pair.get(key,lambda:_ordered_pair(k,a,b,points,guide_bank,cfg))
                angle_checks.append(deepcopy(angle));checks.extend(deepcopy(support))
        failed=[c for c in checks if not c['passes']]
        score.update(parent_v28_engineering_energy=score['total_engineering_energy'],
            ordered_support_unavailable=bool(failed),ordered_support_checks=checks,
            unchanged_intersection_angle_checks=angle_checks,
            directed_support_is_separate_from_angle_conditioning=True)
        if failed:score['total_engineering_energy']=float('inf')
        return score

    def cache_info(self):
        return {name:dict(entries=len(cache.values),limit=cache.limit,hits=cache.hits,misses=cache.misses)
            for name,cache in [('corner',self.corner),('axis',self.axis),('ordered_pair',self.ordered_pair)]}


def make_ordered_scorer(*,maximum_cache_entries=4096):
    """Return a new rank-local scorer compatible with the score_graph hook."""
    if (isinstance(maximum_cache_entries,bool)or not isinstance(maximum_cache_entries,int)
            or maximum_cache_entries<1):raise ValueError('Positive integer cache bound required')
    for module,expected in ((G,G_SOURCE_SHA256),(ORDER,ORDER_SOURCE_SHA256)):
        if sha256(Path(module.__file__).read_bytes()).hexdigest()!=expected:
            raise ValueError('Exact frozen scorer source required')
    return _OrderedScorer(maximum_cache_entries)

"""Opt-in, source-only exact-order beam memory adapter for frozen v29.

No producer or running job imports this module. A private copy of the frozen
rank function replaces materialized expansion sorting by stable nsmallest and
replaces the disposable intersection dictionary by a bounded LRU. Every path
is still scored, with the same scorer and floating arithmetic order. This
changes memory retention, not the approximate beam budget or its ranking.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
import hashlib
import heapq
import inspect
from pathlib import Path
from types import FunctionType

from . import joint_geometry_guided_lines_v4_28 as G
from . import image_edge_observations_v4_29 as PRODUCER

PARENT_SOURCE_SHA256='c6ebc3b515a25eabdf6d6d100c620d7c6d52305821d201edfac3c949a2bf4a57'
ADAPTER_RECIPE=dict(parent_source_sha256=PARENT_SOURCE_SHA256,
    scope='opt-in source-only prototype; not connected to the current observation job',
    expansion='all paths scored in original generation order; stable nsmallest with unchanged key',
    intersection_cache='bounded LRU; pure deterministic intersections recomputed after eviction',
    axis_SVD_or_incident_arithmetic_changed=False,
    observation_threshold_beam_width_candidate_generation_or_scoring_changed=False)


class BoundedIntersectionCache(OrderedDict):
    """Only disposable deterministic values; cache ordering is never emitted."""
    def __init__(self,maximum_entries):
        super().__init__()
        self.maximum_entries=int(maximum_entries)
        if self.maximum_entries<1:raise ValueError('Positive bounded cache size required')

    def __getitem__(self,key):
        value=super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self,key,value):
        super().__setitem__(key,value)
        self.move_to_end(key)
        while len(self)>self.maximum_entries:
            # MutableMapping.popitem may route through our touching getter.
            # Evict directly so a removed key is never moved to the LRU end.
            OrderedDict.__delitem__(self,next(iter(self)))


def _rank_key(pair):
    return (pair[1]['total_engineering_energy'],
        tuple(n['name']if n is not None else'~'for n in pair[0]))


def bounded_beam_step(beam,choices,score,maximum_states,explored,truncated):
    """Same stable sorted prefix without retaining losing score dictionaries."""
    maximum_states=int(maximum_states)
    if maximum_states<1:raise ValueError('Positive existing beam budget required')
    expanded_count=len(beam)*(len(choices)+1)
    def expansions():
        for selected,_ in beam:
            for node in [*choices,None]:
                path=selected+[node]
                yield path,score(path)
    retained=heapq.nsmallest(maximum_states,expansions(),key=_rank_key)
    return retained,explored+expanded_count,truncated or expanded_count>maximum_states


def _rank_code():
    """Fail closed unless exactly the two intended frozen sites are present."""
    if hashlib.sha256(Path(G.__file__).read_bytes()).hexdigest()!=PARENT_SOURCE_SHA256:
        raise ValueError('Exact frozen parent source required for private adapter')
    tree=ast.parse(inspect.getsource(G.rank_joint_geometry_line_graph))
    function=tree.body[0]
    cache_sites=0;beam_sites=0
    for index,node in enumerate(function.body):
        if isinstance(node,ast.Assign)and len(node.targets)==1:
            target=node.targets[0]
            if (isinstance(target,ast.Subscript)and isinstance(target.value,ast.Name)
                    and target.value.id=='guides'and isinstance(target.slice,ast.Constant)
                    and target.slice.value=='_cached_intersections'):
                if not isinstance(node.value,ast.Dict)or node.value.keys:
                    raise ValueError('Disposable parent intersection cache structure changed')
                node.value=ast.Call(func=ast.Name(id='_new_intersection_cache',ctx=ast.Load()),args=[],keywords=[])
                cache_sites+=1
        if (isinstance(node,ast.For)and isinstance(node.target,ast.Name)and node.target.id=='edge'
                and isinstance(node.iter,ast.Call)and isinstance(node.iter.func,ast.Name)
                and node.iter.func.id=='sorted'):
            expected=ast.parse('''for edge in sorted(groups):
    expanded = []
    for selected,_ in beam:
        for node in [*groups[edge],None]:
            path=selected+[node]
            score=_score_graph(path,points,states,guides,cfg,graph_cfg)
            expanded.append((path,score));explored+=1
    expanded.sort(key=lambda pair:(pair[1]['total_engineering_energy'],
        tuple(n['name'] if n is not None else '~' for n in pair[0])))
    if len(expanded)>graph_cfg['maximum_beam_states']:truncated=True
    beam=expanded[:int(graph_cfg['maximum_beam_states'])]
''').body[0]
            if ast.dump(node,include_attributes=False)!=ast.dump(expected,include_attributes=False):
                raise ValueError('Exact parent beam block changed')
            replacement=ast.parse('''beam,explored,truncated=_bounded_beam_step(
    beam,groups[edge],lambda path:_score_graph(path,points,states,guides,cfg,graph_cfg),
    graph_cfg['maximum_beam_states'],explored,truncated)
''').body
            node.body=replacement;beam_sites+=1
    if(cache_sites,beam_sites)!=(1,1):raise ValueError('Exactly two intended memory sites required')
    ast.fix_missing_locations(tree)
    module_code=compile(tree,str(Path(__file__).resolve()),'exec')
    return next(item for item in module_code.co_consts if inspect.iscode(item))


def rank_joint_geometry_line_graph_bounded(*args,intersection_cache_limit=4096,
        score_graph=None,node_factory=None,**kwargs):
    namespace=dict(G.rank_joint_geometry_line_graph.__globals__)
    namespace.update(_bounded_beam_step=bounded_beam_step,
        _new_intersection_cache=lambda:BoundedIntersectionCache(intersection_cache_limit))
    if score_graph is not None:namespace['_score_graph']=score_graph
    if node_factory is not None:namespace['_node']=node_factory
    rank=FunctionType(_rank_code(),namespace)
    return rank(*args,**kwargs)


def rank_fresh_v29_bounded(*args,intersection_cache_limit=4096,**kwargs):
    """The existing fresh producer scorer/node facade, without producer mutation."""
    graph=rank_joint_geometry_line_graph_bounded(*args,
        intersection_cache_limit=intersection_cache_limit,
        score_graph=PRODUCER.ORDER._ordered_score,node_factory=PRODUCER._fresh_node,**kwargs)
    graph['joint_alternatives']=[a for a in graph['joint_alternatives']if not a['ordered_support_unavailable']]
    return graph

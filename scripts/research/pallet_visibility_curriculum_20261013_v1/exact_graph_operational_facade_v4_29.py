"""Opt-in private v29 rank hook: identical arithmetic with bounded retention.

No module globals are patched. Each rank call gets a new frame-local memo.
The original v29 producer/observer and existing archives stay untouched.
Actual installation requires the separate saved-CAL replay verification.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import FunctionType,SimpleNamespace
import time

from . import image_edge_observations_v4_29 as P
from . import bounded_geometry_graph_rank_v4_29 as B
from . import geometry_graph_exact_memo_v4_29 as M

FROZEN_SOURCES={P:'b58870ff70831e55ba6ee45078b98e76f1efca4d467a30479da949ba784af787',
 B:'9ddc506ebda74707fed16ac755ec5a7234c4299889915c212da44cea44b5eb9a',
 M:'b694270aa587dd9cf23b54b4fe8827e42e8e45d05b939190c45f9066de625bc2'}


def verify_frozen_sources():
    for module,expected in FROZEN_SOURCES.items():
        if sha256(Path(module.__file__).read_bytes()).hexdigest()!=expected:
            raise ValueError('Exact frozen source required: '+module.__name__)


def make_rank_adapter(*,maximum_cache_entries=4096,intersection_cache_limit=4096,diagnostics=None):
    verify_frozen_sources()
    def rank(*args,**kwargs):
        memo=M.make_ordered_scorer(maximum_cache_entries=maximum_cache_entries)
        started=time.monotonic()
        graph=B.rank_joint_geometry_line_graph_bounded(*args,**kwargs,
            intersection_cache_limit=intersection_cache_limit,score_graph=memo,node_factory=P._fresh_node)
        graph['joint_alternatives']=[a for a in graph['joint_alternatives']if not a['ordered_support_unavailable']]
        if diagnostics is not None:
            diagnostics.append(dict(rank_seconds=time.monotonic()-started,cache_info=deepcopy(memo.cache_info())))
        return graph
    return rank


def make_observer(*,maximum_cache_entries=4096,intersection_cache_limit=4096,diagnostics=None):
    """Clone only producer function globals; reuse its exact fresh RGB kernel."""
    rank=make_rank_adapter(maximum_cache_entries=maximum_cache_entries,
        intersection_cache_limit=intersection_cache_limit,diagnostics=diagnostics)
    parent=P.observe_joint_geometry_edge_evidence
    namespace=dict(parent.__globals__);namespace['_rank_fresh']=rank
    observer=FunctionType(parent.__code__,namespace,parent.__name__,parent.__defaults__,parent.__closure__)
    observer.__kwdefaults__=deepcopy(parent.__kwdefaults__)
    observer.__annotations__=dict(parent.__annotations__)
    return observer


def make_observe_one(*,maximum_cache_entries=4096,intersection_cache_limit=4096,diagnostics=None):
    """Clone frozen V.observe_one after its original worker initializer runs.

    The private namespace captures V._MODEL at construction, so create this
    once inside each initialized worker. Its original input/image/mask checks,
    guide selection, fresh measurements, row schema and summary are unchanged.
    """
    from . import observe_geometry_guided_edges_v4_29 as V
    if sha256(Path(V.__file__).read_bytes()).hexdigest()!='dbc039badc396ebf755b839a162a521a7613764b9018bce6abe34e89b9206228':
        raise ValueError('Exact original v29 observe_one required')
    producer=SimpleNamespace(**P.__dict__)
    producer.observe_joint_geometry_edge_evidence=make_observer(maximum_cache_entries=maximum_cache_entries,
        intersection_cache_limit=intersection_cache_limit,diagnostics=diagnostics)
    parent=V.observe_one;namespace=dict(parent.__globals__);namespace['P']=producer
    observe=FunctionType(parent.__code__,namespace,parent.__name__,parent.__defaults__,parent.__closure__)
    observe.__kwdefaults__=deepcopy(parent.__kwdefaults__)
    return observe

"""Reusable deployment lifecycle for the unchanged frozen Pipeline.

Keep one instance and call predict(image,K,physical_WHD,metadata=None) repeatedly.
After close, another instance may be created in the same process. The facade
adds only validated legacy package reachability; inherited prediction,
observation selection, pose solving and output coordinates are unchanged.
Source/baseline/checkpoints/calibration are explicit deployment dependencies.
Evaluation cohort, stored observations and truth caches are not API inputs.
"""
from __future__ import annotations

from pathlib import Path
import sys

from . import common as C
from .pipeline import Pipeline as FrozenPipeline


class Pipeline(FrozenPipeline):
    """One active model lifecycle per process; sequential lifecycles are supported."""
    _active_lifecycle = False

    def __init__(self,args=None,calibration=None):
        if args is None:
            args=C.parser(__doc__).parse_args([])
        C.require(args.source_root and args.baseline_root,'explicit source-root/baseline-root required')
        C.require(not Pipeline._active_lifecycle,'close the active deployment Pipeline before constructing another')
        import scripts.research as research
        self._before_deployment_namespace=list(research.__path__)
        self._deployment_outer=C.legacy_context(args)
        self._deployment_outer_active=False
        self._deployment_parent_ready=False
        self._deployment_cleanup_done=False
        Pipeline._active_lifecycle=True
        try:
            # original_context validates the actual baseline root/immutable a22
            # code bindings. A cached resolver may no longer append its path.
            self._deployment_outer.__enter__()
            self._deployment_outer_active=True
            baseline_research=(Path(args.baseline_root).resolve()/'scripts/research')
            C.require(baseline_research.is_dir(),'validated baseline research package missing')
            if str(baseline_research) not in research.__path__:
                research.__path__=list(research.__path__)+[str(baseline_research)]
            super().__init__(args=args,calibration=calibration)
            self._deployment_parent_ready=True
        except BaseException:
            self._finish_deployment_outer(*sys.exc_info())
            raise

    def _finish_deployment_outer(self,*exception):
        if self._deployment_cleanup_done:
            return
        self._deployment_cleanup_done=True
        import scripts.research as research
        try:
            if self._deployment_outer_active:
                self._deployment_outer_active=False
                self._deployment_outer.__exit__(*(exception or (None,None,None)))
        finally:
            research.__path__=self._before_deployment_namespace
            Pipeline._active_lifecycle=False

    def close(self):
        try:
            if self._deployment_parent_ready:
                self._deployment_parent_ready=False
                super().close()
        finally:
            self._finish_deployment_outer()

"""CPU-only stub check of deployment lifecycle; no model, image or geometry run."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from . import common as C
from . import deployment as D
from . import pipeline as P


def run(args):
    import scripts.research as research
    C.require(args.source_root and args.baseline_root,'explicit deployment roots required')
    saved=list(research.__path__)
    baseline=str(Path(args.baseline_root).resolve()/'scripts/research')
    C.require(Path(baseline).is_dir(),'actual existing baseline namespace missing')
    counts=dict(parent_init=0,parent_close=0,outer_enter=0,outer_exit=0)
    fail={'parent':False,'outer':False}
    original=(C.legacy_context,P.Pipeline.__init__,P.Pipeline.close)
    @contextmanager
    def stub_outer(_):
        counts['outer_enter']+=1
        before=list(research.__path__)
        try:
            if fail['outer']:
                raise RuntimeError('STUB_OUTER_GUARD_FAILURE')
            # Simulate a cached resolver: it does NOT re-append baseline path.
            yield (None,None,None)
        finally:
            research.__path__=before
            counts['outer_exit']+=1
    def stub_init(self,args=None,calibration=None):
        C.require(baseline in research.__path__,'facade did not keep validated baseline reachable')
        counts['parent_init']+=1
        if fail['parent']:
            raise RuntimeError('STUB_PARENT_INITIALIZATION_FAILURE')
        self.closed=False
    def stub_close(self):
        C.require(baseline in research.__path__,'baseline removed before parent close')
        counts['parent_close']+=1;self.closed=True
    C.legacy_context,P.Pipeline.__init__,P.Pipeline.close=stub_outer,stub_init,stub_close
    checks=[]
    try:
        research.__path__=[p for p in saved if str(Path(p).resolve())!=baseline]
        starting=list(research.__path__)
        previous=None
        for i in range(2):
            instance=D.Pipeline(args)
            C.require(baseline in research.__path__,'namespace inaccessible during lifetime')
            if previous is not None:
                previous.close()
                C.require(D.Pipeline._active_lifecycle and baseline in research.__path__,
                          'closing an already-closed earlier instance damaged the active lifecycle')
            instance.close();instance.close()
            C.require(research.__path__==starting and not D.Pipeline._active_lifecycle,'sequential close did not restore state')
            checks.append(dict(name=f'sequential_lifecycle_{i+1}_idempotent_close',passed=True))
            previous=instance
        fail['parent']=True
        try:D.Pipeline(args)
        except RuntimeError as error:C.require(str(error)=='STUB_PARENT_INITIALIZATION_FAILURE','wrong parent rejection')
        else:raise AssertionError('parent failure was not propagated')
        C.require(research.__path__==starting and not D.Pipeline._active_lifecycle,'parent exception did not restore state')
        checks.append(dict(name='failed_parent_initialization_restores_namespace_once',passed=True))
        fail['parent']=False;fail['outer']=True
        try:D.Pipeline(args)
        except RuntimeError as error:C.require(str(error)=='STUB_OUTER_GUARD_FAILURE','wrong outer rejection')
        else:raise AssertionError('outer failure was not propagated')
        C.require(research.__path__==starting and not D.Pipeline._active_lifecycle,'outer exception did not restore state')
        checks.append(dict(name='failed_outer_guard_restores_namespace',passed=True))
        C.require(all(getattr(D.Pipeline,name) is getattr(P.Pipeline,name) for name in
                      ('predict','capture','outcomes','registry_metadata','fixed_result','packet')),
                  'facade changed scientific methods')
        checks.append(dict(name='prediction_capture_solver_assembly_methods_identical_inheritance',passed=True))
        C.require(counts==dict(parent_init=3,parent_close=2,outer_enter=4,outer_exit=4),'lifecycle stub entry/exit count differs')
    finally:
        C.legacy_context,P.Pipeline.__init__,P.Pipeline.close=original
        research.__path__=saved
        D.Pipeline._active_lifecycle=False
    payload=dict(schema='validated_boundary_deployment_lifecycle_checks_v2',passed=True,checks=checks,counts=counts,
        deployment_code=C.binding(D.__file__),frozen_pipeline=C.binding(P.__file__),checks_code=C.binding(__file__),
        scope='CPU factory/context stubs and exact inherited-function identity; actual model construction not performed',
        actual_models_loaded=0,detector_forwards=0,head_forwards=0,PnP_calls=0,ray_calls=0,GT_reads=0,
        actual_baseline_source_code_validation_replayed=False)
    C.write_new(C.output_path(args,'DEPLOYMENT_CHECKS.json'),payload)
    print('BOUNDARY_DEPLOYMENT_CHECKS',len(checks),'PASS',flush=True)


if __name__=='__main__':
    parser=C.parser(__doc__)
    parser.set_defaults(output=str(C.DOC))
    run(parser.parse_args())

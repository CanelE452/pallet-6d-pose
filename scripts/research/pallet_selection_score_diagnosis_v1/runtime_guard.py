"""Process/thread-local call audit; no solver or model function is patched."""
from collections import Counter
from contextlib import AbstractContextManager
import sys


class ForbiddenScientificCall(RuntimeError):
    pass


class FixedPoseOnly(AbstractContextManager):
    """Reject forbidden entries before their function bodies execute.

    Fixed projection, Rodrigues/rotation conversion, residual evaluation and
    gradients at a fixed pose remain allowed. Each analysis worker installs its
    own profiler. Existing models/optimizers and other processes are untouched.
    """
    cv_names={'solvePnP','solvePnPGeneric','solvePnPRansac','solvePnPRefineLM',
              'solvePnPRefineVVS','solveP3P','solveP3PRansac'}
    optimize_names={'minimize','minimize_scalar','least_squares','leastsq',
                    '_minimize_slsqp','fmin_slsqp','differential_evolution',
                    'basinhopping','dual_annealing'}
    generator_names={'_initial_poses','_initializations','_initial_guesses',
                     'solve_two_wd_families'}

    def __init__(self):
        self.blocked_attempts=Counter();self.profiled_python_calls=0

    def _reject(self,category,name):
        self.blocked_attempts[category+':'+name]+=1
        raise ForbiddenScientificCall('fixed-pose analysis forbids '+category+':'+name)

    def _profile(self,frame,event,arg):
        if event=='call':
            self.profiled_python_calls+=1
            name=frame.f_code.co_name;file=frame.f_code.co_filename.replace('\\','/')
            if '/scipy/optimize/' in file and name in self.optimize_names:
                self._reject('numeric_optimization',name)
            if '/pallet_visibility_curriculum_20261013_v1/' in file and name in self.generator_names:
                self._reject('candidate_generation',name)
            if '/pallet_statewise_correspondence_diagnosis_v1/typed_replay.py' in file and name in {'replay','replay_multi','replay_single'}:
                self._reject('pose_replay',name)
            if '/torch/nn/modules/module.py' in file and name=='_call_impl':
                self._reject('neural_forward',name)
            if '/torch/optim/' in file and name=='step':
                self._reject('training_update',name)
        elif event=='c_call':
            name=getattr(arg,'__name__','')
            if name in self.cv_names:self._reject('PnP',name)

    def __enter__(self):
        self.previous=sys.getprofile()
        if self.previous is not None:
            raise RuntimeError('Existing profiler must be preserved; use a dedicated worker')
        sys.setprofile(self._profile)
        return self

    def __exit__(self,*args):
        sys.setprofile(self.previous)
        return False

    def receipt(self):
        return dict(scope='this analysis process/thread only',
                    profiled_python_calls=self.profiled_python_calls,
                    forbidden_blocked_attempts=dict(self.blocked_attempts),
                    forbidden_function_bodies_executed=0,
                    no_solver_or_model_monkeypatch=True)

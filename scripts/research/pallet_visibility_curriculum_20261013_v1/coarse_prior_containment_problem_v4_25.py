"""One-sided global containment, with native and three priors kept separate.

The observed foreground is conditionally a subset of the projected cuboid.
No model-to-observed outline equality or full-mask numeric tie break remains.
"""
from copy import deepcopy
import math
import numpy as np

from . import joint_ot_pnp as G
from . import statewise_partial_BI_outline_solver_v4_13 as CONFIG
from . import global_foreground_containment_v4_25 as FACTOR
from . import bounded_geometric_priors_problem_v4_22 as OLD

REGULARIZER_KINDS=(*OLD.REGULARIZER_KINDS,'CAL_COARSE_POSE_PRIOR')


class ContinuousContainmentProblem:
    def __init__(self,base,observation,configuration):
        self.base=base;self.containment_observation=observation
        self.containment_configuration=CONFIG._outline_config(configuration)
        self.containment_supported=bool(observation.get('supported'))
        self.containment_weight=self.containment_configuration['outline_factor_weight']if self.containment_supported else 0.

    def __getattr__(self,name):return getattr(self.base,name)

    def containment_value_and_gradient(self,p):
        if not self.containment_supported or self.containment_weight==0:
            return 0.,np.zeros(6),dict(valid_projection=True,enabled=False,unsupported_or_zero_weight=True)
        value,gradient,audit=FACTOR.outline_value_and_gradient(p,self.xyz,self.K,self.scale,
            self.containment_observation,self.containment_configuration)
        if not audit.get('valid_projection')or not np.isfinite(value)or value<0 or not np.isfinite(gradient).all():
            raise ValueError('Finite valid conditional foreground-containment factor required')
        return float(value),gradient,audit

    def _ledger(self,regularized,shape,audit):
        return dict(native_plus_coarse_regularized_objective=float(regularized),
            global_containment_cost=float(shape),global_containment_weight=float(self.containment_weight),
            weighted_containment_objective=float(self.containment_weight*shape),
            total_optimization_objective=float(regularized+self.containment_weight*shape),
            continuous_containment_gradient_active=bool(self.containment_supported and self.containment_weight>0),
            containment_count_in_optimizer=int(self.containment_supported and self.containment_weight>0),
            containment_not_in_native_costs_or_three_model_regularizers=True,
            old_bidirectional_full_outline_factor_used=False,global_containment_audit=deepcopy(audit))

    def objective(self,p):
        value,gradient,parts=self.base.objective(p)
        shape,sg,audit=self.containment_value_and_gradient(p)
        return value+self.containment_weight*shape,gradient+self.containment_weight*sg,{**parts,**self._ledger(value,shape,audit)}

    def evaluate(self,R,t):
        checked=self.base.evaluate(R,t)
        if checked.get('objective')is None:return checked
        try:
            p=G._pose_vector(np.asarray(R),np.asarray(t),self.scale)
            shape,_,audit=self.containment_value_and_gradient(p)
        except(ValueError,np.linalg.LinAlgError,FloatingPointError)as error:
            return {**checked,'accepted':False,'objective':None,'reason':str(error),
                'containment_numeric_validity_failed_not_new_physical_guard':True}
        value=float(checked['objective'])
        return {**checked,**self._ledger(value,shape,audit),'objective':value+self.containment_weight*shape}


def select_candidates(states,candidates,*,containment_weight=0.,locked_family=None):
    candidates=list(candidates);weight=float(containment_weight)
    if not math.isfinite(weight)or weight<0:raise ValueError('Fixed nonnegative containment weight required')
    if locked_family is not None:
        if locked_family not in('ORIGINAL_WD','SWAPPED_WD'):raise ValueError('Stored GT-free e family label required')
        candidates=[r for r in candidates if r['family']==locked_family]
    authenticated=OLD.AUTH.select_hierarchical_native_candidates(states,candidates)
    rows=[]
    for record in authenticated['candidates']:
        source=candidates[record['index']];priors=source['regularizer_costs']
        if [r['kind']for r in priors]!=list(REGULARIZER_KINDS):raise ValueError('Exactly O, S and one separate CAL coarse prior required')
        values=[float(r['weighted_cost'])for r in priors]
        shape=float(source['global_containment_cost'])
        if any(not math.isfinite(v)or v<0 for v in(*values,shape)):raise ValueError('Finite nonnegative separate costs required')
        regularizer=math.fsum(values);regularized=record['native_energy']+regularizer
        if source['global_mask_cost']!=shape:raise ValueError('Legacy AUTH field must bind this containment scalar only')
        rows.append(dict(index=record['index'],name=record['name'],family=record['family'],
            native_energy=record['native_energy'],regularizer_costs=deepcopy(priors),regularizer_energy=regularizer,
            regularized_native_energy=regularized,global_containment_cost=shape,
            weighted_containment_objective=weight*shape,total_energy=regularized+weight*shape))
    selected=None
    if rows:
        minimum=min(r['total_energy']for r in rows)
        ties=[r for r in rows if abs(r['total_energy']-minimum)<=32*np.finfo(float).eps*max(1.,abs(minimum),abs(r['total_energy']))]
        selected=min(ties,key=lambda r:r['index'])
    ambiguity=deepcopy(authenticated['ambiguity'])
    ambiguity.update(coarse_regularizer_or_containment_does_not_prove_unique_pose=True)
    return dict(schema='AUTHENTICATED_NATIVE_THREE_PRIORS_CONTAINMENT_SELECTION_V25',
        selected_name=None if selected is None else selected['name'],selected_family=None if selected is None else selected['family'],
        candidates=rows,native_authentication_audit=OLD._compact_authentication(authenticated),
        native_Pareto_is_audit_not_regularizer_prefilter=True,regularizer_kinds=list(REGULARIZER_KINDS),
        containment_weight=weight,old_full_outline_factor_or_tie_break_used=False,
        predicted_e_family_constraint=locked_family,family_filter_before_authentication=True,
        e_numeric_pose_used=False,old_best_pose_fallback=False,global_uniqueness_proved=False,ambiguity=ambiguity)

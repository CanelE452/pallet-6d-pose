"""Restore the frozen whole-outline factor inside continuous optimization.

The base owns native observations, bounded O/S priors and every hard guard.
Outline is a separate factor, included once in the optimizer total and the
combined selector total. No pixels, raw keypoints or reference are queried.
"""
from copy import deepcopy
import numpy as np

from . import joint_ot_pnp as G
from . import statewise_partial_BI_outline_solver_v4_13 as OUTLINE


class ContinuousOutlineProblem:
    def __init__(self,base,outline,configuration):
        self.base=base;self.outline=outline
        self.outline_configuration=OUTLINE._outline_config(configuration)
        self.mask_supported=bool(outline.get('supported'))
        self.mask_weight=self.outline_configuration['outline_factor_weight']if self.mask_supported else 0.

    def __getattr__(self,name):return getattr(self.base,name)

    def mask_value_and_gradient(self,p):
        if not self.mask_supported or self.mask_weight==0:
            return 0.,np.zeros(6),dict(valid_projection=True,enabled=False,mask_factor_unavailable_or_zero_weight=True)
        value,gradient,audit=OUTLINE.outline_value_and_gradient(p,self.xyz,self.K,self.scale,
            self.outline,self.outline_configuration)
        if not audit.get('valid_projection')or not np.isfinite(value)or value<0 or not np.isfinite(gradient).all():
            raise ValueError('Finite valid existing whole-outline factor required')
        return float(value),gradient,audit

    def _ledger(self,base_total,mask_value,mask_audit):
        return dict(native_plus_coarse_regularized_objective=float(base_total),
            global_mask_cost=float(mask_value),global_mask_weight=float(self.mask_weight),
            weighted_mask_objective=float(self.mask_weight*mask_value),
            total_optimization_objective=float(base_total+self.mask_weight*mask_value),
            continuous_mask_gradient_active=bool(self.mask_supported and self.mask_weight>0),
            mask_count_in_optimizer=1 if self.mask_supported and self.mask_weight>0 else 0,
            mask_not_in_native_costs_or_O_S_regularizers=True,
            global_mask_audit=deepcopy(mask_audit))

    def objective(self,p):
        value,gradient,parts=self.base.objective(p)
        mask,mg,ma=self.mask_value_and_gradient(p)
        return value+self.mask_weight*mask,gradient+self.mask_weight*mg,{**parts,**self._ledger(value,mask,ma)}

    def evaluate(self,R,t):
        checked=self.base.evaluate(R,t)
        if checked.get('objective')is None:return checked
        try:
            p=G._pose_vector(np.asarray(R),np.asarray(t),self.scale)
            mask,_,ma=self.mask_value_and_gradient(p)
        except(ValueError,np.linalg.LinAlgError,FloatingPointError)as error:
            return {**checked,'accepted':False,'objective':None,'reason':str(error),
                'outline_factor_numeric_validity_failed_not_new_physical_guard':True}
        value=float(checked['objective'])
        return {**checked,**self._ledger(value,mask,ma),'objective':value+self.mask_weight*mask}

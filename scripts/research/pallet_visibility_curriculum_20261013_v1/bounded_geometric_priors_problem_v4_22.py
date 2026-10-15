"""Dependent bounded O/S priors around the unchanged native observation problem.

Native cost authentication remains separate from these model regularizers.
There is no S/O point correspondence, image query, label, file access or new
hard guard. The caller supplies sealed CAL parameters and authoritative states.
"""
from copy import deepcopy
import math
import hashlib
import json
import numpy as np

from . import joint_ot_pnp as G
from . import bounded_o_localization_prior_v4_21 as O
from . import bounded_coarse_visibility_prior_v4_20 as S
from . import hierarchical_native_candidate_score_v4_17 as AUTH

REGULARIZER_KINDS=('O_COARSE_LOCALIZATION_PRIOR','S_COARSE_VISIBILITY_PRIOR')


class BoundedGeometricPriorsProblem:
    def __init__(self,base,o_regions,o_parameters,s_prepared=None):
        self.base=base
        o_regions=list(o_regions)
        # State/S gates precede copying the prior's localization fields.
        for region in o_regions:
            k=region.get('native_corner')
            if isinstance(k,bool)or not isinstance(k,(int,np.integer))or not 0<=k<8 or base.states[k]!='O':
                raise ValueError('Only authoritative non-S O localization priors allowed')
        self.o_regions=deepcopy(list(o_regions));self.o_parameters=dict(o_parameters)
        if set(self.o_parameters)!={'epsilon','huber_scale'}:
            raise ValueError('Only sealed epsilon and existing Huber scale expected')
        self.s_prepared=deepcopy(s_prepared)
        if s_prepared is not None:
            if not np.array_equal(base.xyz,np.asarray(s_prepared['native_xyz'])):
                raise ValueError('Trial S geometry must be rebound to this exact native family')
            for record in s_prepared['eligible_initial_S']:
                if base.states[record['native_corner']]!='S':
                    raise ValueError('Original eligible S ID must remain fixed in every family')

    def __getattr__(self,name):
        return getattr(self.base,name)

    def _priors(self,p):
        ov,og,oa=O.bounded_o_localization_value_and_gradient(p,self.xyz,self.K,self.scale,
            self.o_regions,**self.o_parameters)
        if self.s_prepared is None:
            sv,sg,sa=0.,np.zeros(6),dict(active=False,S_model_prior_term=False,S_image_or_2D_terms=0)
        else:
            sv,sg,sa=S.coarse_visibility_prior_value_and_gradient(p,self.xyz,self.scale,self.s_prepared)
        values=(float(ov),float(sv))
        if any(not math.isfinite(v)or v<0 for v in values):
            raise ValueError('Finite nonnegative bounded regularizers required')
        return ov+sv,og+sg,dict(regularizer_costs=[dict(kind=k,weighted_cost=v)for k,v in zip(REGULARIZER_KINDS,values)],
            O_coarse_localization_prior_audit=oa,S_coarse_visibility_prior_audit=sa,
            S_model_prior_term=self.s_prepared is not None,S_image_or_2D_terms=0,O_point_terms=0,
            priors_are_correlated_coarse_regularizers_not_independent_image_information=True,
            new_hard_constraints=0)

    def objective(self,p):
        native,ng,parts=self.base.objective(p)
        prior,pg,audit=self._priors(p)
        return native+prior,ng+pg,{**parts,**audit,
            'native_observation_objective':float(native),'regularizer_objective':float(prior),
            'total_regularized_objective':float(native+prior),'native_costs_exclude_regularizers':True}

    def evaluate(self,R,t):
        checked=self.base.evaluate(R,t)
        if checked.get('objective')is None:return checked
        try:
            p=G._pose_vector(np.asarray(R),np.asarray(t),self.scale)
            prior,_,audit=self._priors(p)
        except(ValueError,np.linalg.LinAlgError,FloatingPointError)as error:
            return dict(accepted=False,objective=None,native_costs=[],reason=str(error))
        native=float(checked['objective'])
        return {**checked,**audit,'objective':native+prior,
            'native_observation_objective':native,'regularizer_objective':float(prior),
            'total_regularized_objective':native+prior,'native_costs_exclude_regularizers':True}


def _compact_authentication(authenticated):
    result={k:v for k,v in authenticated.items()if k!='pareto_domination'}
    result['pareto_domination']=[dict(index=r['index'],dominated_by_count=len(r['dominated_by']),
        dominated_by_indices_sha256=hashlib.sha256(json.dumps(r['dominated_by'],separators=(',',':')).encode()).hexdigest())
        for r in authenticated['pareto_domination']]
    result['full_domination_sets_computed_then_compacted']=True
    return result


def _select_from_authenticated(candidates,authenticated,*,mask_weight=0.,mask_ties_only=True):
    weight=float(mask_weight)
    if not math.isfinite(weight)or weight<0:raise ValueError('Fixed nonnegative mask weight required')
    rows=[]
    for record in authenticated['candidates']:
        source=candidates[record['index']];priors=source['regularizer_costs']
        if [p['kind']for p in priors]!=list(REGULARIZER_KINDS):
            raise ValueError('Explicit O and S regularizers, separate from native evidence, required')
        values=[float(p['weighted_cost'])for p in priors]
        if any(not math.isfinite(v)or v<0 for v in values):raise ValueError('Invalid regularizer energy')
        regularizer=math.fsum(values);regularized=record['native_energy']+regularizer
        mask=record['global_mask_cost']
        if not mask_ties_only and mask is None:continue
        total=regularized if mask_ties_only else regularized+weight*mask
        rows.append(dict(index=record['index'],name=record['name'],family=record['family'],
            native_energy=record['native_energy'],regularizer_costs=deepcopy(priors),
            regularizer_energy=regularizer,regularized_native_energy=regularized,
            global_mask_cost=mask,total_energy=total))
    selected=None;ties=[]
    if rows:
        minimum=min(r['total_energy']for r in rows)
        ties=[r for r in rows if abs(r['total_energy']-minimum)<=32*np.finfo(float).eps*max(1.,abs(minimum),abs(r['total_energy']))]
        selected=min(ties,key=lambda r:((float('inf')if r['global_mask_cost']is None else r['global_mask_cost'])if mask_ties_only else 0.,r['index']))
    ambiguity=deepcopy(authenticated['ambiguity'])
    ambiguity.update(regularizer_or_mask_selection_proves_uniqueness=False,
        prior_may_resolve_representative_but_not_native_information=True)
    return dict(schema='AUTHENTICATED_NATIVE_PLUS_BOUNDED_MODEL_REGULARIZERS_SELECTION_V4_22',
        selected_name=None if selected is None else selected['name'],
        selected_family=None if selected is None else selected['family'],candidates=rows,
        native_authentication_audit=_compact_authentication(authenticated),
        native_Pareto_is_audit_not_regularizer_prefilter=True,
        native_costs_exclude_regularizers=True,regularizer_kinds=list(REGULARIZER_KINDS),
        mask_weight=weight,mask_only_machine_numeric_regularized_energy_ties=bool(mask_ties_only),
        ambiguity=ambiguity,global_uniqueness_proved=False,
        S_image_terms=0,S_model_prior_term_separately_audited=True,
        no_GT_ID_or_partition_cost_routing=True,old_best_pose_fallback=False)


def select_regularized_native_candidates(states,candidates,*,mask_weight=0.,mask_ties_only=True):
    """Authenticate native evidence before adding separate model priors.

    Native Pareto is an audit, not a prefilter. Priors never enter the native
    BI/normal/tangent/RGB-edge evidence vocabulary or bypass its validation.
    """
    candidates=list(candidates)
    authenticated=AUTH.select_hierarchical_native_candidates(states,candidates)
    return _select_from_authenticated(candidates,authenticated,mask_weight=mask_weight,mask_ties_only=mask_ties_only)


def select_regularized_candidate_pair(states,candidates,*,mask_weight):
    """Same fully authenticated bank, two selectors, one domination pass."""
    candidates=list(candidates)
    authenticated=AUTH.select_hierarchical_native_candidates(states,candidates)
    return (_select_from_authenticated(candidates,authenticated,mask_ties_only=True),
            _select_from_authenticated(candidates,authenticated,mask_weight=mask_weight,mask_ties_only=False))

"""Conservative replacement of archived partial lines by actual NMS mixtures.

Only named edges already used by an archived partial-normal factor are
eligible. All new actual-pixel alternatives survive; shared physical edges
are one factor. Other old normals, complete B/I predictions, archived T4 and
every O/T/U/visibility guard remain. No new full keypoint is manufactured.
"""
from __future__ import annotations

from copy import copy, deepcopy
import hashlib
import numpy as np

from . import native_edge_problem_v4_17 as N
from . import mixture_native_edge_factor_v4_18 as M
from . import image_supported_edge_chains_v4_17c as OBS
from .interior_observation import NEIGHBORS

NOISE_OBSERVATION_FAMILY='UNIFORM_ACTUAL_IMAGE_NMS17C_PARTIAL_NATIVE_EDGE_MIXTURES'


def authenticate_image_chain_groups(base, observation, *, initial_self_hidden,
        image_binding, normal_variance_component_px2, angular_variance_component_sine2,
        noise_provenance, normal_angle_correlation=0.):
    """Authenticate immutable actual-pixel chains, before any pose scoring."""
    hidden = np.asarray(initial_self_hidden, bool)
    if hidden.shape != (8,) or list(hidden) != [s=='S' for s in base.states]:
        raise ValueError('Permanent initial-S gate required')
    binding=observation['image_binding']
    if (binding.get('sha256')!=image_binding.get('sha256')
            or tuple(binding.get('image_hw',()))!=tuple(image_binding.get('image_hw',()))
            or tuple(binding.get('image_hw',()))!=tuple(base.image_hw)
            or ('image_pixel_sha256'in image_binding and binding.get('image_pixel_sha256')!=image_binding['image_pixel_sha256'])):
        raise ValueError('New NMS archive must bind this exact RGB/canvas')
    if (noise_provenance.get('fit_partition') != 'CAL'
            or noise_provenance.get('observation_family') != NOISE_OBSERVATION_FAMILY):
        raise ValueError('Explicit separate CAL model for this new observer required')
    vn, va = float(normal_variance_component_px2), float(angular_variance_component_sine2)
    rho = float(normal_angle_correlation)
    if not np.isfinite([vn,va,rho]).all() or min(vn,va)<0 or abs(rho)>=1.-1e-8:
        raise ValueError('Nonnegative CAL noise components and valid correlation required')
    cfg = OBS.DEFAULT_CONFIGURATION; old = {}
    for boundary in base.boundary:
        k=boundary['native_corner']
        for index,line in enumerate(boundary['lines']):
            mate=line.get('provenance',{}).get('native_neighbor')
            if (not isinstance(mate,(int,np.integer)) or isinstance(mate,(bool,np.bool_))
                    or mate not in NEIGHBORS[k] or hidden[mate]):
                continue
            old.setdefault(tuple(sorted((k,int(mate)))),[]).append(dict(native_corner=k,
                line_index=index, weight=boundary['fit_reliability_weight']))
    grouped={}; skipped=[]
    for ray in observation['edge_records']:
        k,mate=ray['native_corner'],ray['native_neighbor']
        if (not isinstance(k,(int,np.integer)) or not isinstance(mate,(int,np.integer))
                or isinstance(k,(bool,np.bool_)) or isinstance(mate,(bool,np.bool_))
                or not 0<=k<8 or not 0<=mate<8):
            raise ValueError('Valid native endpoint IDs required')
        # This gate must precede copying or reading any point/line field.
        if hidden[k] or hidden[mate] or base.states[k] not in ('B','I'):
            skipped.append(dict(native_corner=k,native_neighbor=mate,reason='initial_S_or_non_BI'))
            continue
        pair=tuple(sorted((int(k),int(mate))))
        if mate not in NEIGHBORS[k]:raise ValueError('Registered incident native edge required')
        if pair not in old or not ray.get('supported'):
            continue
        if ray.get('measurement_source')!='ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT':
            raise ValueError('Actual image-lattice NMS source required')
        if not ray.get('response_is_not_window_clipped'):
            raise ValueError('Guide-window endpoint samples cannot become actual NMS evidence')
        for index,chain in enumerate(ray['chains']):
            if (chain.get('native_corner')!=k or chain.get('native_neighbor')!=mate
                    or sorted(chain.get('native_edge',()))!=list(pair)):
                raise ValueError('Actual NMS chain native identity must match its ray')
            xy=np.asarray(chain['observed_points'],float);mask=np.asarray(chain['inlier_mask'],bool)
            if (xy.ndim!=2 or xy.shape[1]!=2 or mask.shape!=(len(xy),) or not np.isfinite(xy).all()):
                raise ValueError('Finite actual observed NMS pixels and bound inlier mask required')
            actual=xy[mask];normal=np.asarray(chain['normal'],float);offset=float(chain['offset'])
            quality=np.asarray([chain['inlier_fraction'],chain['inlier_span_fraction'],chain['inlier_rms_px']],float)
            if (not np.isfinite(quality).all() or quality[2]<0
                    or len(actual)<cfg['minimum_inlier_count'] or chain['inlier_count']!=len(actual)
                    or chain['inlier_fraction']<cfg['minimum_inlier_fraction']
                    or chain['inlier_span_fraction']<cfg['minimum_inlier_span_fraction']
                    or chain['inlier_rms_px']>cfg['maximum_line_rms_px']
                    or chain.get('invented_bridge_points')!=0):
                raise ValueError('Original NMS actual support/count/fraction/span/RMS guards required')
            if (normal.shape!=(2,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal),1.,atol=1e-7,rtol=0)
                    or not np.isfinite(offset) or np.min(actual)<0
                    or np.any(actual[:,0]>=base.image_hw[1]) or np.any(actual[:,1]>=base.image_hw[0])):
                raise ValueError('Bound finite unit TLS line and actual canvas points required')
            center=actual.mean(axis=0);tangent=np.array([-normal[1],normal[0]])
            # The archived fit uses IRLS weights. Its fitted line need not
            # pass through the *unweighted* inlier centroid. Keep the fitted
            # coefficients, and project that statistic onto their line.
            actual_rms=float(np.sqrt(np.mean((actual@normal+offset)**2)))
            if not np.isclose(actual_rms,chain['inlier_rms_px'],atol=1e-7,rtol=1e-7):
                raise ValueError('Actual inlier RMS must bind the archived weighted TLS line')
            anchor=center-normal*(normal@center+offset)
            span=float(np.ptp(actual@tangent))
            if span<=0:raise ValueError('Nondegenerate actual inlier span required')
            floor_n=max(1.,float(chain['inlier_rms_px']))
            floor_sine=float(np.sin(np.arctan2(np.sqrt(2.)*floor_n,span)))
            component=dict(native_corner=int(k),native_neighbor=int(mate),normal_unit_xy=normal.tolist(),
                offset_px=offset,support_anchor_xy=anchor.tolist(),normal_variance_px2=vn+floor_n**2,
                angular_variance_rad2=va+floor_sine**2,normal_angle_correlation=rho,
                actual_support_tangent_interval_px=[float((actual@tangent).min()),float((actual@tangent).max())],
                support_extrapolation_px=float(cfg['along_start_px']+cfg['boundary_context_px']),
                actual_inlier_span_px=span,normal_floor_px=floor_n,angular_sine_floor=floor_sine,
                mask_context_kind=chain['mask_context_kind'],source_ray_native_corner=int(k),
                source_chain_index=index,noise_provenance=deepcopy(noise_provenance),
                actual_unweighted_inlier_centroid_xy=center.tolist(),
                support_anchor_is_projected_inlier_statistic_not_actual_pixel_or_precise_KP=True)
            # Identical actual pixels can be found from either directed end.
            ordered=np.array(sorted(map(tuple,np.round(actual,6))))
            token=hashlib.sha256(ordered.tobytes()).hexdigest()
            grouped.setdefault(pair,{})[token]=component
    groups=[];refs=[]
    for pair,components in sorted(grouped.items()):
        replacements=[dict(native_corner=r['native_corner'],line_index=r['line_index'])for r in old[pair]]
        anchor=min(r['native_corner']for r in old[pair]);mate=next(k for k in pair if k!=anchor)
        values=[]
        for token,component in sorted(components.items()):
            values.append({**component,'native_corner':anchor,'native_neighbor':mate,'actual_inlier_pixel_digest':token})
        groups.append(dict(native_corner=anchor,native_neighbor=mate,components=values,
            reliability_weight=min(r['weight']for r in old[pair]),replaces_normal_factor_refs=replacements,
            component_priors='uniform over deduplicated actual-pixel alternatives'))
        refs.extend(replacements)
    return dict(groups=groups,replaces_normal_factor_refs=refs,skipped=skipped,
        new_full_BI_point_revocations=0,new_full_2D_KPs=0,old_q_covariances_reliabilities_preserved=True,
        S_local_queries=0,U_point_queries=0,new_image_queries=0,all_parallel_alternatives_retained=True)


class ImageChainMixtureProblem:
    def __init__(self, base, group_bundle):
        if base.edges:raise ValueError('Prepare old-normal base without old projected-edge replacements')
        self.base=copy(base);self.groups=[]
        for group in group_bundle['groups']:
            k,mate=group['native_corner'],group['native_neighbor']
            if (not isinstance(k,(int,np.integer)) or not isinstance(mate,(int,np.integer))
                    or isinstance(k,(bool,np.bool_)) or isinstance(mate,(bool,np.bool_))
                    or not 0<=k<8 or not 0<=mate<8):
                raise ValueError('Valid native endpoint IDs required')
            if base.states[k]not in('B','I')or base.states[mate]=='S':
                raise ValueError('Only permanent nonS BI anchors and nonS model mates allowed')
            self.groups.append(deepcopy(group))
        refs={(r['native_corner'],r['line_index'])for r in group_bundle['replaces_normal_factor_refs']}
        expected=[]
        bycorner={b['native_corner']:b for b in base.boundary}
        for group in self.groups:
            weights=[]
            for ref in group['replaces_normal_factor_refs']:
                k,index=ref['native_corner'],ref['line_index']
                b=bycorner[k];line=b['lines'][index]
                if {k,line['provenance']['native_neighbor']}!={group['native_corner'],group['native_neighbor']}:
                    raise ValueError('Only identical old native physical edges may be replaced')
                weights.append(b['fit_reliability_weight']);expected.append((k,index))
            if not weights or group['reliability_weight']!=min(weights):
                raise ValueError('Old shared-edge reliability must be applied once without strengthening')
        if len(expected)!=len(set(expected))or set(expected)!=refs:
            raise ValueError('Old normal references must be replaced exactly once')
        self.base.remaining_normals=[]
        for b in base.remaining_normals:
            keep=[line for index,line in enumerate(b['lines'])if(b['native_corner'],index)not in refs]
            if keep:self.base.remaining_normals.append({**b,'lines':keep})
        self.replaced_refs=refs

    def __getattr__(self,name):return getattr(self.base,name)

    def objective(self,p):
        value,gradient,parts=self.base.objective(p);audits=[]
        for group in self.groups:
            extra,derivative,audit=M.mixture_edge_value_and_gradient(p,self.xyz,self.K,self.scale,
                group['components'],reliability_weight=group['reliability_weight'],huber_scale=self.cfg['huber_scale'])
            value+=extra;gradient+=derivative;audits.append(audit)
            parts['native_costs'].append(dict(kind='PROJECTED_NATIVE_EDGE',native_corner=group['native_corner'],
                native_neighbor=group['native_neighbor'],weighted_robust_cost=extra))
        parts.update(image_chain_mixture_audit=audits,replaced_old_normal_factor_count=len(self.replaced_refs),
            old_guide_conditioned_angle_factors_used=False,all_new_image_alternatives_retained=True,
            new_full_keypoints=0,full_BI_likelihood_revocations=0)
        return value,gradient,parts

    def evaluate(self,R,t):
        checked=self.base.evaluate(R,t)
        if checked.get('objective')is None:return checked
        try:
            p=N.V._pose_vector(np.asarray(R),np.asarray(t),self.scale)
            value,_,parts=self.objective(p)
        except(ValueError,np.linalg.LinAlgError,FloatingPointError)as error:
            return dict(accepted=False,objective=None,native_costs=[],reason=str(error))
        return {**checked,**parts,'objective':float(value),'accepted':bool(checked['accepted']and np.isfinite(value))}

"""Actual COMMON/BI observation-policy intervention, before any pose solving.

Both policies consume the same role-independent RGB/mask cue pool at every
automatic visible anchor. The condition changes adoption/recovery preferences,
not a condition bit passed to a frozen score network. No pose/reference loader.
The points remain virtual native targets; material correspondence is unverified.
"""
from copy import deepcopy
from itertools import product
import numpy as np
import cv2
from scripts.research.pallet_bi_conditioned_observation_v1 import observer as O
from scripts.research.pallet_visibility_curriculum_20261013_v1.interior_observation import NEIGHBORS
from .s1_union import payload_sha, finite

POLICY = dict(schema='F_WHOLE_OBSERVATION_POLICY_V1',
    evidence='same two cue types RGB/mask at same native-neighbor rays, tight/full automatic SAM masks',
    original='same saved N0, fixed automatic S/B/I/O/T/U, raw original RGB/K/Q',
    search=deepcopy(O.CONFIGURATION), new_state_or_mask_forwards=0,
    maximum_rays_per_visible_anchor=12, source_dedup='actual identical inlier pixel samples, all aliases retained',
    COMMON='best shared normalized support/consistency/uncertainty among both cues and masks; RGB tie first',
    BI='B tight-mask then full-mask then RGB; I RGB then tight-mask then full-mask; same admissibility',
    full_point='two distinct incident neighbors and original common intersection/covariance gates',
    partial='no valid 2D intersection but supported actual line -> one normal only, complete2D explicitly revoked',
    no_new_support='keep original N0 detector point with declared fallback noise; no invented image junction',
    covariance='same conditional formula; own CAL pooled floor by frozen S2 choice CP; B/I floors not inherited across changed q',
    downstream='identical original31feature summaries and matched full hierarchical normal-only producer; B/I summary remains',
    physical_support_certified=False, archived_partial_xy_used_as_seed_or_prior=False,
    partial_tangent_constraints=0, common_fallback_conditional_variance_px2=6.5,
    tie='quality, RGB-before-mask, tight-before-full, numerical xy/sourceID; no ID/reference-dependent selection')

def line_quality(ray):
    cfg=O.CONFIGURATION
    # Same dimensionless score for every actual cue. Validity is established
    # by the unchanged v2 fitter before this ranking; no post-score cutoff.
    rms=ray.get('inlier_rms_px',cfg['maximum_line_rms_px'])
    n=max(1,int(ray.get('samples',cfg['along_samples'])))
    count=int(ray.get('inlier_count',ray.get('transition_sample_count',0)))
    return float(rms)/cfg['maximum_line_rms_px']+1-min(1,count/n)

def candidate_quality(candidate,rays):
    ls=[rays[x]for x in candidate['source_line_ids']]
    cov=np.asarray(candidate['covariance_px2'])
    return float(np.trace(cov)/(2*O.CONFIGURATION['detector_sigma_px']**2)
        +np.mean([line_quality(r)for r in ls])+1-candidate['intersection_sine'])

def priority(lines,state,arm):
    if arm=='COMMON':return 0
    if arm!='BI':raise ValueError('COMMON or BI policy required')
    if all(r['cue']=='I'for r in lines):return 0 if state=='I' else 2
    if all(r['cue']=='B'for r in lines):
        tight=all('tight'in r['mask_sources']for r in lines)
        return (0 if tight else 1)if state=='B'else (1 if tight else 2)
    return 3

def tie_lines(lines):
    return tuple((0 if r['cue']=='I'else 1,0 if 'tight'in r['mask_sources']else 1,r['source_line_id'])for r in lines)

def cue_pool(frame,observation,masks):
    if observation['id']!=frame['id']:raise ValueError('Input identity mismatch')
    image=cv2.imread(frame['image']);states=observation['states'];g=observation['runtime']['initial_geometry']
    if image is None or list(image.shape[:2])!=frame['raw_hw']:raise ValueError('Original RGB required')
    q=np.asarray(frame['initial_points'],float);p=np.asarray(g['initial_projection_xy'],float)
    hidden=np.asarray(g['initial_self_hidden'],bool)
    if any((s=='S')!=hidden[k]for k,s in enumerate(states)):raise ValueError('Fixed S gate required')
    lab=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32),(3,3),.8)
    gradients=cv2.Sobel(lab,cv2.CV_32F,1,0,ksize=3)/8.,cv2.Sobel(lab,cv2.CV_32F,0,1,ksize=3)/8.
    result=[]
    for k,state in enumerate(states):
        if state not in ('B','I'):
            result.append(dict(native_corner=k,state=state,queried=False,rays=[],candidates=[]));continue
        rays=[];aliases=[]
        for name,key in [('tight','pallet_mask_tight'),('full','pallet_mask_clean')]:
            mask=np.asarray(masks[key],float)
            if mask.shape!=image.shape[:2]or not np.isfinite(mask).all():raise ValueError('Bound mask required')
            for mate in NEIGHBORS[k]:
                if hidden[mate]or states[mate]=='S':continue
                sampling,_=O._sampling_trace(image,mask,lab,gradients,q,k,mate,p[mate]-p[k],p.mean(0))
                for cue in ('B','I'):
                    ray=O._ray(image,mask,lab,gradients,q,k,mate,p,cue,sampling)
                    ray['source_line_id']=name+':'+ray['source_line_id'];ray['mask_sources']=[name]
                    if not ray.get('supported'):aliases.append(ray);continue
                    same=next((r for r in rays if r['native_neighbor']==mate and r['cue']==cue
                              and r['physical_source_id']==ray['physical_source_id']),None)
                    if same is None:rays.append(ray)
                    else:
                        same['mask_sources']=sorted(set(same['mask_sources']+[name]));aliases.append(dict(ray,alias_of=same['source_line_id']))
        candidates,rejected=O._full_candidates(k,q[k],rays,image.shape[:2])
        repeated=[c for c in candidates if len(set(c['physical_source_ids']))<2]
        candidates=[c for c in candidates if len(set(c['physical_source_ids']))==2]
        rejected.extend(dict(source_line_ids=c['source_line_ids'],reason='one_actual_pixel_source_cannot_create_two_independent_directions')for c in repeated)
        result.append(dict(native_corner=k,state=state,queried=True,q0=q[k].tolist(),rays=rays,
                           aliases_and_unsupported=aliases,candidates=candidates,rejected_pairs=rejected))
    return finite(dict(id=frame['id'],states=list(states),corners=result,policy=POLICY,
                       image_sha256=frame['image_sha256'],GT_inputs=False))

def choose(pool,frame,arm):
    if arm not in ('COMMON','BI'):raise ValueError('Unregistered policy')
    full=[];partial=[];choices=[]
    for corner in pool['corners']:
        if not corner['queried']:continue
        k=corner['native_corner'];state=corner['state'];q=np.asarray(corner['q0']);rays={r['source_line_id']:r for r in corner['rays']}
        candidates=corner['candidates']
        if candidates:
            def key(c):
                ls=[rays[x]for x in c['source_line_ids']]
                return priority(ls,state,arm),candidate_quality(c,rays),tie_lines(ls),tuple(c['applied_xy'])
            c=min(candidates,key=key);cov=deepcopy(c['covariance_px2'])
            record=dict(native_corner=k,state=state,xy=deepcopy(c['applied_xy']),covariance_px2=cov,
                        fit_covariance_px2=deepcopy(cov),fit_reliability_weight=1.,protection_px=O.CONFIGURATION['visible_protection_px'],
                        source='F_SHARED_ACTUAL_INTERSECTION',physical_source_ids=deepcopy(c['physical_source_ids']),
                        provenance=dict(candidate_id=c['candidate_id'],q0=q.tolist(),policy=arm))
            full.append(record);choices.append(dict(native_corner=k,dimension=2,new_cue_dimension=2,selected=deepcopy(c),quality=key(c)[1]));continue
        variance=POLICY['common_fallback_conditional_variance_px2']
        cov=(np.eye(2)*variance).tolist()
        record=dict(native_corner=k,state=state,xy=q.tolist(),covariance_px2=cov,fit_covariance_px2=deepcopy(cov),
                    fit_reliability_weight=1.,protection_px=O.CONFIGURATION['visible_protection_px'],
                    source='VISIBLE_BACKBONE_FALLBACK',physical_source_ids=['N0_POINT:'+str(k)],provenance=dict(q0=q.tolist(),policy=arm))
        full.append(record)
        supported=list(rays.values())
        if not supported:
            choices.append(dict(native_corner=k,dimension=2,new_cue_dimension=0,selected='N0_KEEP',reason='no_admissible_actual_line'));continue
        ray=min(supported,key=lambda r:(priority([r],state,arm),line_quality(r),tie_lines([r])))
        n=np.asarray(ray['normal']);offset=float(ray['offset']);display=q-n*(n@q+offset)
        archive=dict(original_measurement_source=record['source'],original_observed_xy=q.tolist(),
            original_observed_covariance_px2=deepcopy(cov),original_effective_fit_covariance_px2=deepcopy(cov),
            original_fit_reliability_weight=1.,revocation_reason='no_unique_face_supported_junction',
            original_raw_xy=q.tolist(),normal_projected_display_xy=display.tolist(),display_coordinate_is_full_2D_measurement=False,
            archived_xy_used_as_seed_or_prior=False,physical_target='VIRTUAL_NATIVE_CORNER_CONDITIONAL_GUIDE_NOT_CERTIFIED_MATERIAL_JUNCTION')
        line=dict(normal_unit_xy=n.tolist(),offset_px=offset,normal_variance_px2=float(n@np.asarray(cov)@n),
            support_tangent_interval_px=deepcopy(ray['actual_inlier_tangent_interval_px']),support_extrapolation_px=0.,
            physical_source_ids=[ray['physical_source_id']],provenance=dict(line_record=deepcopy(ray),support_interval_is_archive_only=True))
        partial.append(dict(native_corner=k,state=state,lines=[line],image_binding=dict(sha256=frame['image_sha256'],image_hw=frame['raw_hw']),
            fit_reliability_weight=1.,replaces_complete_BI_likelihood=True,explicit_complete_BI_likelihood_revocation=True,
            association_provenance=archive))
        choices.append(dict(native_corner=k,dimension=1,new_cue_dimension=1,selected=ray['source_line_id'],quality=line_quality(ray),reason='one_supported_direction_no_full_junction'))
    return finite(dict(states=deepcopy(pool['states']),full_points=full,partial_lines=partial,
        selected_candidates=choices,policy=arm,cue_pool_sha256=payload_sha(pool),GT_inputs=False))

def intervention_table():
    return dict(schema='F_CALLABLE_INTERVENTION_TABLE_V1',policies=[
        dict(item=1,original='visible_bi_correction_v2._line / correct_visible_bi: B mask hull vs I RGB incident',
             shared='cue_pool: all legal nonSincident neighbors, both cues and same fit gates',COMMON='choose common normalized cue cost',BI='choose role priority then same cost'),
        dict(item=2,original='observe_remainder_rgb_v4_14.recover_stages: preserve corrections then B-only tight/full/reach',
             shared='cue_pool computes tight/full for all B/I; same original N0 fallback',COMMON='joint tight/full minimum cost',BI='B tight/full/RGB; I RGB/tight/full'),
        dict(item=3,original='face_aware_partial_bi_observations_v4_13d._select_supported_line: role preference',
             shared='choose: two supported independent neighbors gives 2D; no full junction + line gives1D; no new support keeps N0; typed archive validator unchanged',
             COMMON='shared line quality',BI='role preference among same qualified lines'),
        dict(item=4,original='statewise_uncertainty_calibration_v4.apply_calibrated_measurements role floor',
             shared='own CAL-only pooled zero-centered floor using calibration.fit; S2 preregistered LOGO selected CP for both',COMMON='same pooled recipe',BI='same pooled recipe, this item tied'),
        dict(item=5,original='original owner_model /31features/H1 B/I summaries',shared='kept in both groups',COMMON='same callable',BI='same callable',claim='not removal of every downstream role input')],
        actual_callables=['f_observer.cue_pool','f_observer.choose','calibration.apply_selection','fullnormal_backend','heads'],
        policy=POLICY)

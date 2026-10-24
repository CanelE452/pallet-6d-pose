#!/usr/bin/env python3
"""DRAFT reference-visibility math; generated arrays only, no data admission.

The reference target/identity/r/uncertainty are GT-only. Query movement never
changes that reference's D/E state. Crop evidence availability and native query
mask support are separate variables. No old M4/radius2 label is reinterpreted.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np

MODEL_INPUT_KEYS=frozenset(('rgb','initial_query_xy','target_box_xyxy'))
BASE_STATES=frozenset(('D_teacher','S','F','U'))
DATA_ADMISSION='HOLD_UNADMITTED_DESIGN_ONLY'


def require(condition,message):
    if not condition:raise ValueError(message)


@dataclass(frozen=True)
class ReferenceGT:
    """Immutable identity metadata, never a model-input record.

    uncertainty_bbox is a required, GT-only half-open continuous XYXY region
    including BOTH coordinate uncertainty and relevant visual-cue uncertainty.
    Its magnitude is not inferred, defaulted, or selected by this module.
    """
    target_instance_id: str
    reference_id: str
    reference_xy: tuple
    uncertainty_bbox: tuple
    original_state: str='U'
    reference_provenance: str='UNADMITTED'
    original_D_teacher_evidence: str=''
    evidence_tier: str='UNADMITTED_CONDITIONAL_TEACHER'
    semantic_keypoint_id: object=None
    canonical8_mapping_evidence: str=''
    semantic_self_evidence_kind: str=''
    semantic_self_evidence: str=''
    source_name: str='UNADMITTED'

    def __post_init__(self):
        require(bool(self.target_instance_id) and bool(self.reference_id),'Explicit target/reference identity required')
        require(self.original_state in BASE_STATES,'No E/O/S inference from mask absence; only registered base states')
        require(np.asarray(self.reference_xy).shape==(2,) and np.asarray(self.uncertainty_bbox).shape==(4,),
                'Explicit reference xy and uncertainty bbox required')
        if self.semantic_keypoint_id is not None:
            require(bool(self.canonical8_mapping_evidence),'Semantic slot identity requires independently admitted correspondence evidence')
            require(self.source_name!='R2B3_AI_ANCHOR','Existing R2B3 AI anchor canonical8 identity must remain UNKNOWN')


@dataclass(frozen=True)
class OccluderGT:
    """GT-only pasted layer; no alpha/mask/identity certificate enters a model."""
    alpha: object
    instance_id: str
    front_of_target_evidence: str=''
    provenance: str='UNADMITTED_CONTROLLED_LAYER'


def validate_model_inputs(inputs):
    """Strict field/type gate. Does not prove independent input provenance."""
    require(isinstance(inputs,dict) and set(inputs)==MODEL_INPUT_KEYS,
            'Only RGB + initial query q + target box are permitted; r/state/GT/offset/identity metadata forbidden')
    rgb=np.asarray(inputs['rgb'])
    require(rgb.ndim==3 and rgb.shape[2]==3 and rgb.dtype==np.uint8,'uint8 RGB HWC required')
    q=np.asarray(inputs['initial_query_xy'],np.float64);box=np.asarray(inputs['target_box_xyxy'],np.float64)
    require(q.shape==(2,) and np.isfinite(q).all(),'Finite initial input q required')
    require(box.shape==(4,) and np.isfinite(box).all() and box[2]>box[0] and box[3]>box[1],'Input-only nondegenerate target box required')
    return True


def reference_geometry(reference,raw_hw):
    """Conservative native-cell support of the declared uncertainty region.

    Native pixel j represents [j-.5,j+.5). Include every cell with positive-area
    intersection with the half-open uncertainty rectangle; never shrink it to
    its center pixel. Unobservable cells remain counted, preventing a falsely
    certified full cover at an image edge.
    """
    h,w=map(int,raw_hw);require(h>0 and w>0,'Positive original H/W required')
    r=np.asarray(reference.reference_xy,np.float64);b=np.asarray(reference.uncertainty_bbox,np.float64)
    result=dict(raw_hw=[h,w],reference_frame='UNKNOWN',region_status='INVALID',region_bounds=None,
                region_cells=0,region_inframe_cells=0,region_outside_cells=0)
    if not np.isfinite(r).all():return dict(result,reason='NONFINITE_REFERENCE')
    if not(0<=r[0]<w and 0<=r[1]<h):return dict(result,reference_frame='OUT_OF_FRAME',reason='REFERENCE_OUT_OF_FRAME')
    result['reference_frame']='IN_FRAME'
    if not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1]:return dict(result,reason='INVALID_UNCERTAINTY_REGION')
    if not(b[0]<=r[0]<b[2] and b[1]<=r[1]<b[3]):return dict(result,reason='UNCERTAINTY_MUST_CONTAIN_REFERENCE')
    x0,y0=map(int,np.floor(b[:2]+.5));x1,y1=map(int,np.ceil(b[2:]+.5))
    total=(x1-x0)*(y1-y0);ix0,iy0=max(0,x0),max(0,y0);ix1,iy1=min(w,x1),min(h,y1)
    inside=max(0,ix1-ix0)*max(0,iy1-iy0)
    return dict(result,region_status='FULL_IN_FRAME' if total==inside else 'PARTLY_OUT_OF_FRAME',
        region_bounds=[x0,y0,x1,y1],clipped_region_bounds=[ix0,iy0,ix1,iy1],
        region_cells=total,region_inframe_cells=inside,region_outside_cells=total-inside,
        reason='VALID_REGION' if total==inside else 'UNOBSERVABLE_UNCERTAINTY_CELLS')


def controlled_reference_state(reference,raw_hw,occluder=None):
    """No q argument: controlled D/E is a function of the fixed GT reference."""
    g=reference_geometry(reference,raw_hw)
    base=dict(target_instance_id=reference.target_instance_id,reference_id=reference.reference_id,
        semantic_keypoint_id=reference.semantic_keypoint_id,canonical8_correspondence_known=reference.semantic_keypoint_id is not None,
        original_state=reference.original_state,evidence_tier=reference.evidence_tier,geometry=g,
        source_admission=DATA_ADMISSION,semantic_gold=False,training_authorized=False)
    def answer(state,reason,**extra):return dict(base,reference_state=state,reason=reason,**extra)
    if g['reference_frame']=='OUT_OF_FRAME':return answer('F','REFERENCE_OUT_OF_FRAME; not external, self, or physical absence')
    if g['region_status']!='FULL_IN_FRAME':return answer('U',g['reason'])
    h,w=map(int,raw_hw);x0,y0,x1,y1=g['region_bounds'];total=g['region_cells']
    if occluder is None:
        clear,opaque,partial=total,0,0
    else:
        alpha=np.asarray(occluder.alpha)
        require(alpha.shape==(h,w) and np.isfinite(alpha).all() and (alpha>=0).all() and (alpha<=1).all(),
                'Native alpha in [0,1] required; no geometric/mask0 visibility inference')
        region=alpha[y0:y1,x0:x1]
        clear=int((region==0).sum());opaque=int((region==1).sum());partial=int(((region>0)&(region<1)).sum())
    counts=dict(region_cells=total,clear_cells=clear,opaque_cells=opaque,fractional_alpha_cells=partial)
    if reference.original_state=='F':return answer('F','EXPLICIT_ORIGINAL_F_REFERENCE',coverage=counts)
    if reference.original_state=='U':return answer('U','ORIGINAL_REFERENCE_STATE_UNRESOLVED',coverage=counts)
    if reference.original_state=='S':
        certified=reference.semantic_self_evidence_kind=='INDEPENDENT_SEMANTIC_SELF' and bool(reference.semantic_self_evidence)
        if not certified:return answer('U','NO_INDEPENDENT_SEMANTIC_SELF_EVIDENCE',coverage=counts)
        if clear!=total:return answer('U','ADDITIONAL_LAYER_OVER_SELF_REFERENCE_REQUIRES_NEW_CAUSE_REVIEW',coverage=counts)
        return answer('S','PRESERVED_INDEPENDENT_SEMANTIC_SELF_EVIDENCE',coverage=counts)
    if not reference.original_D_teacher_evidence:
        return answer('U','NO_ORIGINAL_D_TEACHER_EVIDENCE; coordinates or M4 alone are insufficient',coverage=counts)
    if clear==total:return answer('D_teacher','REFERENCE_UNCERTAINTY_REGION_COMPLETELY_UNCOVERED',coverage=counts)
    if partial or opaque!=total:return answer('U','PARTIAL_COVER_OR_PARTIAL_OPACITY',coverage=counts)
    distinct=bool(occluder.instance_id) and occluder.instance_id!=reference.target_instance_id
    if not distinct or not occluder.front_of_target_evidence:
        return answer('U','NO_DISTINCT_FOREGROUND_INSTANCE_CERTIFICATE',coverage=counts)
    return answer('E_teacher','DISTINCT_OPAQUE_FOREGROUND_COVERS_ENTIRE_REFERENCE_UNCERTAINTY_REGION',coverage=counts,
                  controlled_external_provenance=occluder.provenance)


def crop_availability(reference,initial_query_xy,crop_side,raw_hw):
    """GT-only availability audit; q crop geometry uses no r correction.

    FULL means the entire declared uncertainty rectangle is contained in the
    input crop's actually observed image footprint. PARTIAL/MISS are not E/O.
    The future input-only L recipe is deliberately not selected here.
    """
    q=np.asarray(initial_query_xy,np.float64);h,w=map(int,raw_hw)
    require(q.shape==(2,),'q xy shape required')
    base=dict(availability='INVALID',query_frame='UNKNOWN',location_teacher_flag='INVALID',
              reference_geometry=reference_geometry(reference,raw_hw),crop_bounds=None,observed_crop_bounds=None)
    if not np.isfinite(q).all():return dict(base,reason='NONFINITE_INITIAL_QUERY')
    query_frame='IN_FRAME' if 0<=q[0]<w and 0<=q[1]<h else 'OUT_OF_FRAME'
    base['query_frame']=query_frame
    if not math.isfinite(float(crop_side)) or crop_side<=0:return dict(base,reason='INVALID_INPUT_ONLY_CROP_SIDE')
    if base['reference_geometry']['region_status']=='INVALID':return dict(base,reason='INVALID_REFERENCE_REGION')
    b=np.asarray(reference.uncertainty_bbox,np.float64);half=float(crop_side)/2
    crop=np.array([q[0]-half,q[1]-half,q[0]+half,q[1]+half])
    observed=np.array([max(crop[0],-.5),max(crop[1],-.5),min(crop[2],w-.5),min(crop[3],h-.5)])
    iw=max(0.,min(b[2],observed[2])-max(b[0],observed[0]));ih=max(0.,min(b[3],observed[3])-max(b[1],observed[1]))
    full=(observed[2]>observed[0] and observed[3]>observed[1] and
          observed[0]<=b[0] and observed[1]<=b[1] and observed[2]>=b[2] and observed[3]>=b[3])
    state='FULL' if full else 'PARTIAL' if iw>0 and ih>0 else 'MISS'
    return dict(base,availability=state,location_teacher_flag=state,reason='GT_ONLY_CROP_EVIDENCE_AVAILABILITY',
        crop_bounds=crop.tolist(),observed_crop_bounds=observed.tolist(),
        uncertainty_area=float((b[2]-b[0])*(b[3]-b[1])),observed_uncertainty_intersection_area=iw*ih,
        note='MISS is a location/evidence teacher flag, never semantic E/O or physical absence. Query out-of-frame is a separate flag.')


def assess_query(reference,initial_query_xy,crop_side,raw_hw,occluder=None):
    semantic=controlled_reference_state(reference,raw_hw,occluder)
    availability=crop_availability(reference,initial_query_xy,crop_side,raw_hw)
    state=semantic['reference_state']
    target=state if availability['availability']=='FULL' and state in ('D_teacher','E_teacher') else 'U'
    return dict(reference=semantic,availability=availability,training_target=target,
        training_target_scope='Proposed conditional D/E target only when the entire uncertainty region is observable in the crop; source admission remains HOLD.',
        underlying_reference_state_unchanged_by_query=True,training_authorized=False,
        semantic_self_is_separate=state=='S',location_flag_is_not_semantic_absence=True)


def native_support_audit(visible_mask,initial_query_xy):
    """Separate GT-mask diagnostic; never alters reference labels or primary M4."""
    a=np.asarray(visible_mask);require(a.ndim==2 and a.dtype==bool,'Bool native visible-mask audit only')
    q=np.asarray(initial_query_xy,np.float64);h,w=a.shape
    if q.shape!=(2,) or not np.isfinite(q).all():return dict(valid=False,exact=None,radius2=None)
    x,y=map(int,np.floor(q+.5))
    valid=0<=q[0]<w and 0<=q[1]<h and 0<=x<w and 0<=y<h
    if not valid:return dict(valid=False,exact=None,radius2=None)
    return dict(valid=True,exact=bool(a[y,x]),radius2=bool(a[max(0,y-2):min(h,y+3),max(0,x-2):min(w,x+3)].any()),
                role='Separate existing native exact/fixed5x5 support audit; never the reference-state label')


def generated_tests(output):
    """Pure generated arrays; no RGB files, masks, weights or datasets opened."""
    path=Path(output);require(not path.exists(),'Preserve previous pure-test report')
    hw=(11,13)
    r=ReferenceGT('target-A','ref-A',(4.,4.),(2.5,2.5,5.5,5.5),original_state='D_teacher',
        original_D_teacher_evidence='GENERATED_TOY_FIXED_REFERENCE_D',reference_provenance='GENERATED_ARRAY_ONLY')
    zero=np.zeros(hw,np.float32);full=zero.copy();full[3:6,3:6]=1
    layer=OccluderGT(full,'other-B','GENERATED_TOY_DISTINCT_FRONT_LAYER')
    visible=np.zeros(hw,bool);visible[:,:5]=True
    # A background q can be arbitrarily displaced while reference D stays fixed.
    d=assess_query(r,(7.,4.),12,hw)
    require(d['reference']['reference_state']=='D_teacher' and d['training_target']=='D_teacher','Background query must not relabel D as O/E')
    require(native_support_audit(visible,(7.,4.))['exact'] is False,'Toy q must really be background')
    # E reference while an adjacent q and its radius2 see another target surface.
    after=np.ones(hw,bool);after[3:6,3:6]=False
    e=assess_query(r,(7.,4.),12,hw,layer)
    require(e['reference']['reference_state']=='E_teacher' and e['training_target']=='E_teacher','Visible q must not erase reference E')
    require(native_support_audit(after,(7.,4.))['exact'] and native_support_audit(after,(7.,4.))['radius2'],'Toy q/near support must both be visible')
    cases=0
    for x in range(-2,16):
        for y in range(-2,14):
            for occluder,expected in ((None,'D_teacher'),(layer,'E_teacher')):
                result=assess_query(r,(x+.2,y+.1),8,hw,occluder)
                require(result['reference']['reference_state']==expected,'Reference state must be invariant to all q jitter')
                if result['availability']['availability'] in ('MISS','PARTIAL','INVALID'):
                    require(result['training_target']=='U','Insufficient crop evidence must not be supervised as D/E')
                cases+=1
    partial=zero.copy();partial[4,4]=1
    require(controlled_reference_state(r,hw,OccluderGT(partial,'other-B','front'))['reference_state']=='U','Partial coverage must be U')
    fractional=full*.5
    require(controlled_reference_state(r,hw,OccluderGT(fractional,'other-B','front'))['reference_state']=='U','Partial opacity must be U')
    query_only=zero.copy();query_only[4,7]=1
    require(controlled_reference_state(r,hw,OccluderGT(query_only,'other-B','front'))['reference_state']=='D_teacher','Covering q alone cannot change reference D')
    require(controlled_reference_state(r,hw,OccluderGT(full,'target-A','front'))['reference_state']=='U','Same-target cover is not external/S')
    require(controlled_reference_state(r,hw,OccluderGT(full,'other-B',''))['reference_state']=='U','Front-layer evidence is mandatory')
    unverified=ReferenceGT('target-A','ref-A',(4.,4.),(2.5,2.5,5.5,5.5),original_state='D_teacher')
    require(controlled_reference_state(unverified,hw,layer)['reference_state']=='U','Coordinates/mask support cannot manufacture D/E')
    # Availability and frame conditions remain distinct from semantic state.
    for q,L,state in (((4.,4.),8,'FULL'),((7.,4.),6,'PARTIAL'),((10.,4.),2,'MISS')):
        result=assess_query(r,q,L,hw)
        require(result['availability']['availability']==state and result['reference']['reference_state']=='D_teacher','FULL/PARTIAL/MISS separate from D')
    invalid=assess_query(r,(float('nan'),4),8,hw)
    require(invalid['availability']['availability']=='INVALID' and invalid['training_target']=='U' and invalid['reference']['reference_state']=='D_teacher','Invalid q does not change reference D')
    out=assess_query(r,(30.,4.),2,hw)
    require(out['availability']['query_frame']=='OUT_OF_FRAME' and out['availability']['availability']=='MISS' and out['reference']['reference_state']=='D_teacher','Out-of-frame q is not reference F/E')
    fr=ReferenceGT('target-A','ref-out',(-2.,4.),(-3.,3.,-1.,5.),original_state='U')
    require(controlled_reference_state(fr,hw)['reference_state']=='F','Out-of-frame reference must remain separate')
    sr=ReferenceGT('target-A','ref-S',(4.,4.),(3.5,3.5,4.5,4.5),original_state='S')
    require(controlled_reference_state(sr,hw)['reference_state']=='U','Mask0/declared S without independent evidence is not self truth')
    certified=ReferenceGT('target-A','ref-S',(4.,4.),(3.5,3.5,4.5,4.5),original_state='S',
        semantic_self_evidence_kind='INDEPENDENT_SEMANTIC_SELF',semantic_self_evidence='GENERATED_TOY_SEMANTIC_SELF_FACT')
    require(controlled_reference_state(certified,hw)['reference_state']=='S','Explicit independent self evidence is separate')
    # AI local anchors cannot masquerade as canonical8 keypoint truth.
    try:ReferenceGT('A','r',(4,4),(3,3,5,5),semantic_keypoint_id=0,canonical8_mapping_evidence='claimed',source_name='R2B3_AI_ANCHOR')
    except ValueError:pass
    else:raise AssertionError('R2B3 canonical8 claim must fail')
    inputs=dict(rgb=np.zeros((11,13,3),np.uint8),initial_query_xy=np.array([7.,4.]),target_box_xyxy=np.array([0.,0.,10.,10.]))
    require(validate_model_inputs(inputs),'Minimal input whitelist')
    forbidden=('reference_xy','uncertainty_bbox','reference_state','semantic_GT','occluder_mask','occluder_alpha','generated_offset',
               'reference_id','target_instance_id','semantic_keypoint_id','depth','pose','GT')
    for key in forbidden:
        try:validate_model_inputs(dict(inputs,**{key:0}))
        except ValueError:pass
        else:raise AssertionError('Forbidden GT/reference/offset model input escaped: '+key)
    # No q argument can enter the state function by API design.
    try:controlled_reference_state(r,hw,q=(0,0))
    except TypeError:pass
    else:raise AssertionError('Reference state function must not consume q')
    report=dict(schema='V69_REFERENCE_CONTRACT_GENERATED_ARRAY_TESTS_V1',PASS=True,
        code=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        reference_state_invariance_cases=cases,background_q_D_preserved=True,visible_q_nearV_reference_E_preserved=True,
        partial_coverage_opacity_U=True,query_only_occlusion_negative_control=True,missing_D_or_distinct_front_evidence_U=True,
        FULL_PARTIAL_MISS_INVALID_separate=True,query_outframe_not_reference_F=True,reference_outframe_F_separate=True,
        self_requires_independent_semantic_evidence=True,R2B3_canonical8_claim_rejected=True,
        forbidden_model_input_fields_rejected=list(forbidden),reference_state_API_has_no_q=True,
        numeric_jitter_recipe_selected=False,uncertainty_radius_selected=False,actual_semantic_gold_admitted=0,
        source_admission=DATA_ADMISSION,source_RGB_masks_features_weights_reads=0,current_RGB_GT_scores_inputs_read=False,
        composite_images_generated=0,model_forward_calls=0,optimizer_updates=0,PnP_calls=0,
        toy_numbers_are_not_a_source_jitter_or_crop_recipe=True)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(PASS=True,report=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--test-generated',type=Path,required=True)
    generated_tests(ap.parse_args().test_generated)

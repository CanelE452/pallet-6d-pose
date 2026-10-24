"""Integrated draft: root GT-only decisions + V2 geometry + V1 input materializer.

Pure array/callback API; no source RGB/mask/NPZ/checkpoint or model loader.
Actual construction needs a later root entry. Generated fixtures may call it now.
"""
from pathlib import Path
import hashlib
import importlib.util

import numpy as np

V2_CODE = Path(__file__).resolve().parent.parent/'v2_geometry_mass_proposal/geometry_mass_v2.py'
V2_SHA = '61e61356ea16287c41dff09c739c05c07ea5941a9539a9e1af4b83c8f97530a4'
if hashlib.sha256(V2_CODE.read_bytes()).hexdigest() != V2_SHA:
    raise ValueError('Bound V2 proposal changed')
spec = importlib.util.spec_from_file_location('v3_bound_geometry_mass', V2_CODE)
v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
v1 = v2.v1
VARIANTS = v1.VARIANTS


def partition(split):
    aliases = {'TRAIN':'TRAIN', 'TRAIN_LAB':'TRAIN',
               'VALIDATION':'VALIDATION', 'VAL':'VALIDATION', 'VAL_HALL':'VALIDATION'}
    if split not in aliases:
        raise ValueError('Unknown original partition')
    return aliases[split]


def checked_row(row):
    """Validate a normalized copy; never alter original split/q/L/slot."""
    result = dict(row); result['split'] = partition(row['split'])
    v1.check_input(result)
    return result


def decision_identity(row, decision):
    """Root ledger is GT-only; matching verifies identity, never chooses q."""
    assert decision['query_key'] == [row['input_id'], row['native_slot_logical_only']]
    assert decision['input_id'] == row['input_id']
    assert decision['native_slot'] == row['native_slot_logical_only']
    assert decision['frame_index'] == row['frame_index']
    assert decision['split'] == row['split']
    assert decision['image_sha256'] == row['image_sha256']
    assert decision['original_q'] == row['query_xy'] and decision['original_L'] == row['L']
    assert decision['before_aux_state'] in ('D','U')
    assert decision['native_expected_part_state'] == 'UNKNOWN' and decision['human_gold'] is False


def pixel_edge_box_cells(box, hw):
    """Half-open continuous bbox in native pixel-centre coordinates.

    Root bbox edges are q+(edge96/96-.5)*L. For each axis, all native nearest
    cells touched by [lo,hi) are floor(lo+.5):ceil(hi+.5), without clipping.
    This honours an exact right/bottom half-pixel edge without an extra cell.
    """
    if box is None or np.asarray(box).shape != (4,) or not np.isfinite(box).all():
        return None
    x0,y0,x1,y1=map(float,box);h,w=hw
    if x0>=x1 or y0>=y1 or x0<-.5 or y0<-.5 or x1>w-.5 or y1>h-.5:
        return None
    left,top=np.floor(np.array([x0,y0])+.5).astype(int)
    right,bottom=np.ceil(np.array([x1,y1])+.5).astype(int)
    if not (0<=left<right<=w and 0<=top<bottom<=h):
        return None
    out=np.zeros((h,w),bool);out[top:bottom,left:right]=True
    return out


def controlled_protected_support(row, decision):
    """Geometry for controlled edits, not a new before-label admission rule."""
    if decision['before_aux_state']!='D' or not decision['paired_construction_eligible']:
        return None,'ROOT_NOT_ELIGIBLE_D_BEFORE_U'
    q,L=row['query_xy'],row['L']
    if q is None or L is None or np.asarray(q).shape!=(2,) or not np.isfinite(q).all() or not np.isfinite(L) or L<=0:
        return None,'MISSING_FINITE_ORIGINAL_QUERY_U'
    support=pixel_edge_box_cells(decision['support_bbox_xyxy'],row['raw_hw'])
    uncertainty=pixel_edge_box_cells(decision['uncertainty_bbox_xyxy'],row['raw_hw'])
    if support is None or uncertainty is None:
        return None,'UNUSABLE_NATIVE_SUPPORT_OR_UNCERTAINTY_U'
    return support|uncertainty,'ROOT_D_WITH_USABLE_GEOMETRY'


def choose_one_donor(row, donor_metadata):
    """Input-key minSHA assignment, same split AND physical object family."""
    return v1.choose_one_donor_metadata(checked_row(row),donor_metadata)


def prepare_occurrences(native_rgb,row,decision,donor_rgb,donor_visible,donor_meta,crop):
    """All3 variants retained; V2 helpers are actually wired into this API.

    Root D is the sole before-state authority. V2 geometry/mass failures only
    make the two controlled after variants U; they never move q or relabel D.
    """
    normalized=checked_row(row);decision_identity(row,decision)
    if native_rgb.dtype!=np.uint8 or native_rgb.shape!=(*row['raw_hw'],3):
        raise ValueError('Native target RGB shape/dtype')
    root_before_label=0 if decision['before_aux_state']=='D' else -1
    protected,reason=controlled_protected_support(row,decision)
    cutout=None;pair=None;scale_facts={};mass_facts={}
    donor_ok=(donor_meta is not None and donor_meta.get('split')==normalized['split']
              and donor_meta.get('image_sha256')!=row['image_sha256']
              and donor_meta.get('foreign_opaque_cutout_admitted') is True
              and v1.donor_family_allowed(donor_meta.get('object_id'),normalized['split']))
    if protected is not None:
        if not donor_ok:
            reason='ASSIGNED_DONOR_UNUSABLE_OR_PARTITION_NOT_ADMITTED_U'
        else:
            cutout,scale_facts=v2.support_scaled_cutout(donor_rgb,donor_visible,protected,row['L'])
            if cutout is None:
                reason=scale_facts['reason']
            else:
                pair,mass_facts=v2.core_aligned_pair(normalized,cutout,protected)
                reason='CONTROLLED_PAIR_READY'if pair is not None else mass_facts['reason']
    images=[native_rgb.copy(),native_rgb.copy(),native_rgb.copy()]
    candidate_labels=[root_before_label,-1,-1]
    reasons=['ROOT_D_BEFORE_CONDITIONAL_AUX'if root_before_label==0 else'ROOT_U_PRESERVED',reason,reason]
    certificate={'pair_ready':pair is not None,'scale':scale_facts,'mass':mass_facts,
                 'assigned_donor_id':None if donor_meta is None else donor_meta.get('id'),
                 'donor_count':0 if donor_meta is None else 1,'query_reselection':False}
    if pair is not None:
        (cxy,ca),(sxy,sa)=pair
        images[1]=v1.compose(native_rgb,cutout,cxy,ca);images[2]=v1.compose(native_rgb,cutout,sxy,sa)
        assert ca[protected].all()and not sa[protected].any()
        assert np.array_equal(images[2][protected],native_rgb[protected])
        assert ca.sum()==sa.sum()==cutout['alpha'].sum()
        passed,checkmass=v2.mass_certificate(normalized,ca,sa);assert passed
        candidate_labels=[0,1,0]
        reasons=['ROOT_D_BEFORE_CONDITIONAL_AUX','CONTROLLED_COVER_E_AUX','CONTROLLED_SHAM_D_AUX']
        certificate.update(cover_xy=list(cxy),sham_xy=list(sxy),same_donor_RGBA_transform=True,
                           protected_all_alpha1_cover=True,protected_all_alpha0_sham=True,
                           same_global_alpha_cells=int(ca.sum()),mass_recheck=checkmass,
                           donor_transform={'source_bbox_xyxy':cutout['donor_native_bbox_xyxy'],
                                            'resized_hw':cutout['resized_hw'],'scale':cutout['scale']})
    occurrences=[]
    finite=(row['query_xy']is not None and row['L']is not None
            and np.isfinite(row['query_xy']).all()and np.isfinite(row['L'])and row['L']>0)
    for variant,rgb,label,why in zip(VARIANTS,images,candidate_labels,reasons):
        if finite:
            rgb96,valid96=crop(rgb,np.asarray(row['query_xy']),row['L'],96)
        else:
            rgb96=np.zeros((3,96,96),np.uint8);valid96=np.zeros((1,96,96),bool)
        oid=f"{row['input_id']}::slot{row['native_slot_logical_only']}::{variant}"
        original=dict(row);original.update(occurrence_id=oid,variant=variant)
        occurrences.append({'input_only':original,'full_rgb':rgb,'rgb96':rgb96,'crop_valid96':valid96,
          'gt_only':{'candidate_aux_hidden_label':label,'known_candidate_label':label>=0,
                     'root_before_aux_state':decision['before_aux_state'],'root_query_key':decision['query_key'],
                     'reason':why,'native_expected_part_state':'UNKNOWN','human_gold':False,
                     'training_admitted':False,'operator_query_available':None}})
    assert len(occurrences)==3
    return {'occurrences':occurrences,'gt_edit_certificate_only':certificate,
            'pair_ready':pair is not None,'all_original_rows_retained':True,
            'native_gold_or_fit_admission_created':False}


def materialize_row(native_rgb,row,decision,donor_rgb,donor_visible,donor_meta,crop,
                    before_bound_roi,extract_after_full_rgb):
    """Integrated image edits -> matched RGB96 AND fresh after-full-image P3.

    Only the three head tensors leave the materializer. r/support/labels/donor
    and variant metadata never enter the extractor or head tensor dictionary.
    """
    result=prepare_occurrences(native_rgb,row,decision,donor_rgb,donor_visible,donor_meta,crop)
    for occurrence in result['occurrences']:
        m=v1.materialize_head_inputs(occurrence,before_bound_roi,extract_after_full_rgb)
        occurrence.update(m)
        occurrence['gt_only']['operator_query_available']=m['input_accounting_only']['operator_query_available']
    return result


def all_row_occurrence_plan(rows,decisions):
    """No filtering/nearest association: exact key join, all rows x3 variants."""
    keys=[(r['input_id'],r['native_slot_logical_only'])for r in rows]
    if len(set(keys))!=len(keys)or [r['slot_index']for r in rows]!=list(range(len(rows))):
        raise ValueError('Unique original order required')
    decisions_by_key={tuple(d['query_key']):d for d in decisions}
    if len(decisions_by_key)!=len(decisions)or set(decisions_by_key)!=set(keys):
        raise ValueError('Root ledger must preserve every original row/key')
    result=[]
    for row,key in zip(rows,keys):
        checked_row(row);decision_identity(row,decisions_by_key[key])
        for variant in VARIANTS:
            result.append({'slot_index':row['slot_index'],'query_key':list(key),'variant':variant,
                           'original_split':row['split'],'original_query_xy':row['query_xy'],
                           'original_L':row['L'],'root_before_aux_state_GT_ONLY':decisions_by_key[key]['before_aux_state']})
    return result

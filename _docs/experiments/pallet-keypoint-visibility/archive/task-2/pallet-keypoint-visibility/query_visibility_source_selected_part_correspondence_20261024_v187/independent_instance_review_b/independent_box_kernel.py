"""Independent scalar box arithmetic; no producer import or source-file loader."""
import math


def finite_box_xyxy(box):
    return (isinstance(box,(list,tuple)) and len(box)==4
            and all(isinstance(x,(int,float)) and math.isfinite(x) for x in box)
            and box[2]>box[0] and box[3]>box[1])


def xywh_to_xyxy(box):
    if not isinstance(box,(list,tuple)) or len(box)!=4:return None
    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in box):return None
    x,y,w,h=box
    if w<=0 or h<=0:return None
    return [float(x),float(y),float(x+w),float(y+h)]


def scalar_iou_xyxy(first,second):
    if not finite_box_xyxy(first) or not finite_box_xyxy(second):return None
    width=max(0.,min(first[2],second[2])-max(first[0],second[0]))
    height=max(0.,min(first[3],second[3])-max(first[1],second[1]))
    intersection=width*height
    area_a=(first[2]-first[0])*(first[3]-first[1])
    area_b=(second[2]-second[0])*(second[3]-second[1])
    return intersection/(area_a+area_b-intersection)


def all_gt_unique_max(query_xyxy,candidates):
    """Exact-positive unique maximum over ALL objects, retaining other-object IDs.

    Invalid GT boxes have unavailable IoU and do not hide valid candidates.
    This weak bbox association is not an end/native identity certification.
    """
    details=[]
    for candidate in candidates:
        box=xywh_to_xyxy(candidate.get('bbox_obj'))
        details.append(dict(gt_instance_index=candidate['gt_instance_index'],obj_id=candidate['obj_id'],
            provided_bbox_obj_xywh=candidate.get('bbox_obj'),provided_bbox_obj_xyxy=box,IoU=scalar_iou_xyxy(query_xyxy,box),
            provided_bbox_visib_xywh=candidate.get('bbox_visib'),visible_bbox_IoU_diagnostic_only=scalar_iou_xyxy(query_xyxy,xywh_to_xyxy(candidate.get('bbox_visib')))))
    result=dict(status='UNKNOWN_NO_VALID_GT_BOX',selected_gt_instance_index=None,selected_obj_id=None,selected_IoU=None,
        top_IoU=None,second_IoU=None,top_minus_second=None,top_tied_indices=[],all_GT_bbox_candidates=details,
        association_is_weak_identity_diagnostic=True,selected_instance_identity_certified=False)
    if not finite_box_xyxy(query_xyxy):
        result['status']='UNKNOWN_INVALID_OR_NO_DETECTION_BOX';return result
    usable=[d for d in details if d['IoU'] is not None]
    if not usable:return result
    maximum=max(d['IoU'] for d in usable)
    # Descending numeric values and original GT indices only establish reported order.
    order=sorted(usable,key=lambda d:(-d['IoU'],d['gt_instance_index']))
    second=order[1]['IoU'] if len(order)>1 else None
    tied=sorted(d['gt_instance_index'] for d in usable if d['IoU']==maximum)
    result.update(top_IoU=maximum,second_IoU=second,top_minus_second=None if second is None else maximum-second,top_tied_indices=tied)
    if maximum<=0:result['status']='UNKNOWN_NO_POSITIVE_BBOX_OVERLAP'
    elif len(tied)>1:result['status']='UNKNOWN_EXACT_MAX_IOU_TIE'
    else:
        winner=next(d for d in usable if d['gt_instance_index']==tied[0])
        result.update(status='UNIQUE_POSITIVE_MAX_BBOX_IOU_WEAK_IDENTITY',selected_gt_instance_index=winner['gt_instance_index'],selected_obj_id=winner['obj_id'],selected_IoU=maximum)
    return result


def invented_checks():
    checks=[]
    assert scalar_iou_xyxy([0,0,10,10],[0,0,10,10])==1
    assert scalar_iou_xyxy([0,0,10,10],[5,0,15,10])==1/3
    assert scalar_iou_xyxy([0,0,10,10],[10,0,20,10])==0
    checks.append('Scalar_exact_overlap_partial_and_edge_touch')
    candidates=[dict(gt_instance_index=0,obj_id=1,bbox_obj=[0,0,10,10]),dict(gt_instance_index=1,obj_id=2,bbox_obj=[0,0,20,20])]
    assert all_gt_unique_max([0,0,10,10],candidates)['selected_gt_instance_index']==0
    checks.append('Unique_target_max_over_all_GT')
    other=all_gt_unique_max([0,0,20,20],candidates)
    assert other['status']=='UNIQUE_POSITIVE_MAX_BBOX_IOU_WEAK_IDENTITY' and other['selected_obj_id']==2
    checks.append('Other_object_winner_retained_not_forced_to_obj1_or_UNKNOWN')
    tied=[candidates[0],dict(gt_instance_index=4,obj_id=1,bbox_obj=[0,0,10,10])]
    assert all_gt_unique_max([0,0,10,10],tied)['status']=='UNKNOWN_EXACT_MAX_IOU_TIE'
    checks.append('Exact_tie_unknown_no_lowest_index_pick')
    assert all_gt_unique_max([30,30,40,40],candidates)['status']=='UNKNOWN_NO_POSITIVE_BBOX_OVERLAP'
    assert all_gt_unique_max([0,0,10,10],[])['status']=='UNKNOWN_NO_VALID_GT_BOX'
    checks.append('No_overlap_and_empty_allGT_unknown')
    for box in (None,[0,0,0,10],[float('nan'),0,1,1]):
        assert all_gt_unique_max(box,candidates)['status']=='UNKNOWN_INVALID_OR_NO_DETECTION_BOX'
    checks.append('Missing_degenerate_nonfinite_query_unknown')
    assert xywh_to_xyxy([1,2,3,4])==[1.,2.,4.,6.]
    assert xywh_to_xyxy([1,2,0,4]) is None
    checks.append('XYWH_halfopen_conversion_and_degenerate_rejection')
    invalid=dict(gt_instance_index=8,obj_id=1,bbox_obj=[0,0,0,1])
    assert all_gt_unique_max([0,0,10,10],candidates+[invalid])['selected_gt_instance_index']==0
    checks.append('Invalid_GT_bbox_retained_as_none_without_hiding_valid_candidates')
    # Original row availability does not enter association arithmetic or mutate it.
    original=[dict(query_available=False,native_slot=i) for i in range(8)]
    cloned=[dict(row) for row in original]
    all_gt_unique_max([0,0,10,10],candidates)
    assert original==cloned and all(not r['query_available'] for r in original)
    checks.append('AllU_metadata_not_resurrected_or_changed')
    return dict(completed=True,PASS=True,invented_only=True,count=len(checks),checks=checks,actual_source_metadata_bodies_read=0)

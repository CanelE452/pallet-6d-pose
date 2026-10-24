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


def all_gt_unique_max(query_xyxy,candidates,target_obj_id=1):
    """Proposed exact-positive unique maximum over ALL objects, never target-only.

    The final adapter will bind this to the frozen producer's metadata policy.
    This function does not certify instance identity or train any labels.
    """
    if not finite_box_xyxy(query_xyxy):
        return dict(state='UNKNOWN',reason='INVALID_QUERY_BOX',selected_gt_index=None,all_ious=[])
    values=[]
    for candidate in candidates:
        value=scalar_iou_xyxy(query_xyxy,candidate['box_xyxy'])
        values.append(value)
    if any(value is None for value in values):
        return dict(state='UNKNOWN',reason='INVALID_GT_BOX',selected_gt_index=None,all_ious=values)
    if not values or max(values)<=0:
        return dict(state='UNKNOWN',reason='NO_POSITIVE_OVERLAP',selected_gt_index=None,all_ious=values)
    best=max(values);winners=[i for i,value in enumerate(values) if value==best]
    if len(winners)!=1:
        return dict(state='UNKNOWN',reason='EXACT_MAX_TIE',selected_gt_index=None,all_ious=values)
    winner=winners[0]
    if candidates[winner]['obj_id']!=target_obj_id:
        return dict(state='UNKNOWN',reason='OTHER_OBJECT_UNIQUE_MAX',selected_gt_index=None,all_ious=values,
                    winning_other_gt_index=candidates[winner]['gt_index'])
    return dict(state='SUPPORTED_METADATA_ASSOCIATION',reason='TARGET_OBJECT_UNIQUE_POSITIVE_MAX',
                selected_gt_index=candidates[winner]['gt_index'],all_ious=values)


def invented_checks():
    checks=[]
    assert scalar_iou_xyxy([0,0,10,10],[0,0,10,10])==1
    assert scalar_iou_xyxy([0,0,10,10],[5,0,15,10])==1/3
    assert scalar_iou_xyxy([0,0,10,10],[10,0,20,10])==0
    checks.append('Scalar_exact_overlap_partial_and_edge_touch')
    candidates=[dict(gt_index=0,obj_id=1,box_xyxy=[0,0,10,10]),dict(gt_index=1,obj_id=2,box_xyxy=[0,0,20,20])]
    assert all_gt_unique_max([0,0,10,10],candidates)['selected_gt_index']==0
    checks.append('Unique_target_max_over_all_GT')
    assert all_gt_unique_max([0,0,20,20],candidates)['reason']=='OTHER_OBJECT_UNIQUE_MAX'
    checks.append('Other_object_max_not_silently_target_only')
    tied=[candidates[0],dict(gt_index=4,obj_id=1,box_xyxy=[0,0,10,10])]
    assert all_gt_unique_max([0,0,10,10],tied)['reason']=='EXACT_MAX_TIE'
    checks.append('Exact_tie_unknown_no_lowest_index_pick')
    assert all_gt_unique_max([30,30,40,40],candidates)['reason']=='NO_POSITIVE_OVERLAP'
    assert all_gt_unique_max([0,0,10,10],[])['reason']=='NO_POSITIVE_OVERLAP'
    checks.append('No_overlap_and_empty_allGT_unknown')
    for box in (None,[0,0,0,10],[float('nan'),0,1,1]):
        assert all_gt_unique_max(box,candidates)['reason']=='INVALID_QUERY_BOX'
    checks.append('Missing_degenerate_nonfinite_query_unknown')
    assert xywh_to_xyxy([1,2,3,4])==[1.,2.,4.,6.]
    assert xywh_to_xyxy([1,2,0,4]) is None
    checks.append('XYWH_halfopen_conversion_and_degenerate_rejection')
    # Original row availability does not enter association arithmetic or mutate it.
    original=[dict(query_available=False,native_slot=i) for i in range(8)]
    cloned=[dict(row) for row in original]
    all_gt_unique_max([0,0,10,10],candidates)
    assert original==cloned and all(not r['query_available'] for r in original)
    checks.append('AllU_metadata_not_resurrected_or_changed')
    return dict(completed=True,PASS=True,invented_only=True,count=len(checks),checks=checks,actual_source_metadata_bodies_read=0)

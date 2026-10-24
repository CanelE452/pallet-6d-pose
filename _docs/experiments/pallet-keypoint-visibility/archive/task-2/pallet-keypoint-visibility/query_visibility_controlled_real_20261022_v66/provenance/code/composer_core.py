"""Deterministic opaque real-cutout geometry. No dataset/model/pose imports."""
import hashlib
import math
import cv2
import numpy as np

SEED=20261022
SCALES=(.25,.40,.60)


def rank_key(*parts):
    return hashlib.sha256(':'.join(map(str,(SEED,*parts))).encode()).hexdigest()


def query_pixels(points,hw):
    q=np.asarray(points,np.float64);h,w=hw
    if q.shape!=(8,2):raise ValueError('Exactly unchanged query8 is required')
    finite=np.isfinite(q).all(1)&~(q==-1).all(1)
    p=np.floor(np.where(np.isfinite(q),q,-1e8)+.5).astype(np.int64)
    valid=finite&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
    valid&=(p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)
    return p,valid


def strict_square(mask,px,py,radius=2):
    h,w=mask.shape
    if px-radius<0 or py-radius<0 or px+radius>=w or py+radius>=h:return False
    return bool(mask[py-radius:py+radius+1,px-radius:px+radius+1].all())


def erode_core(mask,radius=2):
    return cv2.erode(np.asarray(mask,np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8),
                     borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)


def resize_visible_cutout(rgb,visible,longest_side):
    """RGB linear / hard support nearest; exact row-major deepest-mask anchor."""
    rgb,visible=np.asarray(rgb),np.asarray(visible,bool)
    if rgb.dtype!=np.uint8 or rgb.shape!=(*visible.shape,3):raise ValueError('uint8 RGB/mask shapes differ')
    yy,xx=np.nonzero(visible)
    if not len(xx):return None
    bbox=(int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1)
    x0,y0,x1,y1=bbox;crop_rgb=rgb[y0:y1,x0:x1];crop_mask=visible[y0:y1,x0:x1]
    longest_side=int(longest_side)
    if longest_side<9:raise ValueError('Registered minimum longest-side is9')
    ratio=longest_side/max(x1-x0,y1-y0)
    width=max(1,int(math.floor((x1-x0)*ratio+.5)))
    height=max(1,int(math.floor((y1-y0)*ratio+.5)))
    resized_rgb=cv2.resize(crop_rgb,(width,height),interpolation=cv2.INTER_LINEAR)
    resized_mask=cv2.resize(crop_mask.astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST).astype(bool)
    padded=np.pad(resized_mask.astype(np.uint8),1)
    distance=cv2.distanceTransform(padded,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    if not resized_mask.any():return None
    my,mx=np.unravel_index(int(np.argmax(distance)),distance.shape)
    return dict(rgb=resized_rgb,mask=resized_mask,bbox=list(bbox),resized_hw=[height,width],
                source_crop_hw=[y1-y0,x1-x0],longest_side=longest_side,scale_factor=float(ratio),
                donor_anchor_xy=[int(mx),int(my)],donor_anchor_distance=float(distance[my,mx]))


def paste_support(hw,cutout,anchor_xy):
    h,w=hw;px,py=map(int,anchor_xy);mx,my=cutout['donor_anchor_xy']
    left,top=px-mx,py-my;ch,cw=cutout['mask'].shape
    x0,y0=max(0,left),max(0,top);x1,y1=min(w,left+cw),min(h,top+ch)
    mask=np.zeros((h,w),bool);rgb=np.zeros((h,w,3),np.uint8)
    if x1>x0 and y1>y0:
        sx,sy=x0-left,y0-top
        mask[y0:y1,x0:x1]=cutout['mask'][sy:sy+y1-y0,sx:sx+x1-x0]
        rgb[y0:y1,x0:x1]=cutout['rgb'][sy:sy+y1-y0,sx:sx+x1-x0]
    return mask,rgb,dict(translation_xy=[left,top],destination_clip_xyxy=[x0,y0,x1,y1],
                          source_clip_xyxy=[x0-left,y0-top,x1-left,y1-top])


def candidate_checks(visible,full,paste,points,anchor_index):
    """The preregistered accept tests; no score/model feedback."""
    p,valid=query_pixels(points,visible.shape);k=int(anchor_index);px,py=p[k]
    if not valid[k]or not strict_square(visible&paste,int(px),int(py)):
        return False,'ANCHOR_NOT_STRICT_NEW_E',{}
    original_count=int(visible.sum());covered=int((visible&paste).sum());remaining=original_count-covered
    coverage=covered/max(original_count,1)
    if coverage<.03 or coverage>.60:return False,'COVERAGE_OUTSIDE_3_TO_60_PERCENT',{'covered_fraction':coverage}
    if remaining<25:return False,'FEWER_THAN25_VISIBLE_TARGET_PIXELS_LEFT',{}
    visible_controls=[j for j,(x,y)in enumerate(p)if j!=k and valid[j]and visible[y,x]and not paste[y,x]]
    if not visible_controls:return False,'NO_OTHER_UNPASTED_VISIBLE_QUERY',{}
    outside_controls=[j for j,(x,y)in enumerate(p)if valid[j]and not full[y,x]and not paste[y,x]]
    if not outside_controls:return False,'NO_UNTOUCHED_OUTSIDE_QUERY',{}
    return True,'ACCEPTED',dict(original_visible_pixels=original_count,newly_covered_visible_pixels=covered,
        remaining_visible_pixels=remaining,covered_fraction=coverage,anchor_query=k,
        visible_control_queries=visible_controls,outside_control_queries=outside_controls,
        anchor_strict5x5_new_E=True,anchor_pixel_xy=[int(px),int(py)])


def compose_rgb(target_rgb,donor_canvas,paste):
    output=np.array(target_rgb,copy=True);output[paste]=donor_canvas[paste]
    if not np.array_equal(output[~paste],target_rgb[~paste])or not np.array_equal(output[paste],donor_canvas[paste]):
        raise AssertionError('Binary paste RGB invariants failed')
    return output

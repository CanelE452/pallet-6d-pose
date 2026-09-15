"""Separate prediction-only RGB/pose inputs from supervised source targets."""
import numpy as np
import cv2
import torch
from env import *
from posefix_contract_math import axis_aligned_crop_matrix,transform_points

class SourceRGB:
    def __init__(self):
        self.data=dataset()
    def item(self,row,check_hash=False):
        d=self.data;a=d.arrays;r=d.source['records'][int(d.indices[row])]
        assert r['source_kind']=='synthetic'
        image=cv2.imread(r['image']);assert image is not None
        if check_hash:assert sha(r['image'])==r['image_sha256']
        assert list(image.shape[:2])==r['prepared_shape_hw'],'Do not reflect-pad a prepared image again'
        gain,offset=old('features').canvas_affine(r['prepared_shape_hw'],a['input_shape'][row])
        points=(a['points'][row]-offset)/gain
        box=((a['boxes'][row].reshape(2,2)-offset)/gain).reshape(4)
        matrix=axis_aligned_crop_matrix(box)
        crop=cv2.warpAffine(image,matrix[:2],(288,384),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        rgb=(crop[:,:,::-1].astype(np.float32)-np.array([123.68,116.78,103.94],np.float32)).transpose(2,0,1)
        valid=a['point_valid'][row].astype(bool)&np.isfinite(points).all(-1)
        cp=transform_points(np.where(valid[:,None],points,0),matrix).astype(np.float32)
        gt=(a['gt_points'][row]-offset)/gain;gv=a['gt_valid'][row].astype(bool)&np.isfinite(gt).all(-1);gv[8]=False
        target=transform_points(np.where(gv[:,None],gt,0),matrix).astype(np.float32)
        support=gv&(target>=0).all(-1)&(target[:,0]<288)&(target[:,1]<384)
        return dict(rgb=rgb,points=cp,valid=valid,target=target,target_valid=support,row=np.int64(row),original_points=points,matrix=matrix,original_gt_valid=gv,source_id=r['id'],source_partition=r['partition'])
    def batch(self,rows,device='cuda'):
        rows=[self.item(int(r)) for r in rows]
        return {k:torch.as_tensor(np.stack([r[k] for r in rows]),device=device) for k in ('rgb','points','valid','target','target_valid')}

def prepare():
    src=SourceRGB();items=[src.item(int(i),True) for i in src.data.train_rows[:8]]
    for x in items:assert x['source_partition']=='train'
    np.savez(RAW/'actual_source8.npz',**{k:np.stack([r[k] for r in items]) for k in ('rgb','points','valid','target','target_valid','original_points','matrix')})
    write(DOC/'SOURCE8_ADAPTER.json',dict(complete=True,rows=[int(x['row']) for x in items],ids=[x['source_id'] for x in items],partition='synthetic_train',input_pose='frozen_R0_cache_not_GT',input_box='frozen_R0_cache_not_GT',RGB=True,second_reflect_pad=False,warp_border='CONSTANT_ZERO_official_crop',GT_stream_separate=True,target_support=[int(x['target_valid'].sum()) for x in items],all_GT_support=[int(x['original_gt_valid'].sum()) for x in items],arrays=bound(RAW/'actual_source8.npz')))
    return items

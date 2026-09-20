"""2x field-of-view at fixed pixel scale; parent experiment remains immutable."""
import argparse
import copy
import time
from contextlib import contextmanager
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch.nn import functional as TF

from . import dino_localization as D
from . import dino_wide_model as W
from . import recovery_pose as RP

C=D.C;P=D.P;N=D.N
PHASE='dino_wide';DOC=D.R.DOC/PHASE;RAW=D.R.RAW/PHASE
ORIGINAL_PREPARE=N.C.prepare_input


def warp(image,matrix):
    crop=cv2.warpAffine(image,matrix[:2],(W.WIDTH,W.HEIGHT),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    return (crop[:,:,::-1].astype(np.float32)-N.C.MEAN).transpose(2,0,1)


def prepare_input(image,prediction):
    x=ORIGINAL_PREPARE(image,prediction)
    if x is None:return None
    matrix=W.widen_matrix(x['matrix'])
    x.update(rgb=warp(image,matrix),matrix=matrix,points=x['points']+np.array([144,192],np.float32))
    return x


def source_item(source,row):
    x=source.item(row);original_mask=x['target_valid'].copy()
    image=cv2.imread(str(C.ROOT/x['image_binding']['path']));assert image is not None
    matrix=W.widen_matrix(x['matrix']);valid=x['original_gt_valid'].copy();valid[8]=False
    target=N.C.transform_points(np.where(valid[:,None],x['original_gt'],0),matrix).astype(np.float32)
    valid&=(target>=0).all(-1)&(target[:,0]<W.WIDTH)&(target[:,1]<W.HEIGHT)
    assert not (original_mask&~valid).any()
    x.update(rgb=warp(image,matrix),matrix=matrix,points=x['points']+np.array([144,192],np.float32),
        target=target,target_valid=valid,old_target_valid=original_mask)
    return x


def real_item(row):
    C.verify(row['image']);image=cv2.imread(str(C.ROOT/row['image']['path']));assert image is not None
    old=P.make_item(row);x=prepare_input(image,row['raw'])
    labels,_=RP.paired_labels(row)
    valid=np.asarray(labels['REF'].split(),float)[5:].reshape(9,3)[:,2]==2
    target=N.C.transform_points(np.asarray(P.top(row['refined'])['keypoints_xy']),x['matrix']).astype(np.float32)
    valid&=np.isfinite(target).all(-1)&(target>=0).all(-1)&(target[:,0]<W.WIDTH)&(target[:,1]<W.HEIGHT);valid[8]=False
    assert not (old['target_valid']&~valid).any()
    x.update(id=row['id'],target=target,target_valid=valid,old_target_valid=old['target_valid'])
    return x


@torch.no_grad()
def extract(model,items):
    rgb=np.stack([x['rgb'] for x in items])+N.C.MEAN[None,:,None,None]
    x=torch.as_tensor(rgb,device='cuda')/255
    x=TF.interpolate(x,size=(784,588),mode='bilinear',align_corners=False)
    mean=torch.tensor([.485,.456,.406],device=x.device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=x.device)[None,:,None,None]
    z=model.forward_features((x-mean)/std)['x_norm_patchtokens']
    z=z.reshape(len(items),56,42,384).permute(0,3,1,2).contiguous()
    assert torch.isfinite(z).all()
    return z.cpu().numpy().astype(np.float16)


@contextmanager
def scope():
    with patch.object(D,'PHASE',PHASE),patch.object(D,'DOC',DOC),patch.object(D,'RAW',RAW),\
         patch.object(D,'M',W),patch.object(D,'extract',extract),patch.object(N.C,'prepare_input',prepare_input):
        yield


def verify():
    with scope():return D.verify()


def prepare():
    parent=D.verify();p=copy.deepcopy(parent)
    p.update(objective='Remove an observed crop reachability barrier without reducing native-image pixel density; learn actual large spatial corrections.',
        backbone='Same frozen DINOv2 weights. Same R0 predicted box. Shift old affine center by144,192, retaining identical linear scale; crop768x576,resize784x588,features384x56x42. Prepared source RGB is not padded a second time.',
        head='Same parameter names/shapes/initialization as parent. Double spatial dimensions through all stages:56x42->112x84->192x144; physical output stride4 and sigma12crop_px unchanged. No displacement cap, crop/hypothesis choice or new rejection rule.',
        targets='Same existing GT/pseudo coordinates and trust definitions, transformed into the larger crop. Recompute only in-crop support; previously cropped-out already trusted points may become supervised. Count this difference explicitly;no new annotation or pseudo relabeling.',
        controls='Same1412source images,64source held,217real train+32pseudo probe,1000steps,seeds,sample order,corruption,optimizer,decoder5x5. Crop context and spatial tensor sizes change together;extra accessible supervision is reported. Not a backbone or loss change.',
        rationale='Posthoc parent diagnostic found29/64far-spatial corners cannot be reached within10px in old output rectangle, although all64GT points are in raw images. This is repeatedDEV hypothesis generation,not an independent test.',
        factor_selection='Exactly2x FOV in each axis, fixed to retain original pixel scale and integral patch grids; no factor sweep or GT-based per-image crop.',
        decoder='Same5x5 local expectation around global maximum,now192x144grid; input/output stride4 retained. Center/box/confidence stay original.',
        sources=parent['sources']+[C.bound(x) for x in [__file__,W.__file__,C.HERE/'test_dino_wide_model.py',
            D.DOC/'PROTOCOL.json',D.DOC/'RESULTS.json',D.DOC/'COMPLETION_AUDIT.json']])
    C.freeze(DOC/'BACKBONE.json',C.read(D.DOC/'BACKBONE.json'))
    C.freeze(DOC/'PROTOCOL.json',p)
    D.R.evaluation_protocol(PHASE,D.ARMS,p['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('WIDE_PROTOCOL_LOCKED','same scale,2x FOV',flush=True)


def cache():
    p=verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    backbone,_=D.A.load();source=P.SourceData()
    pool={r['id']:r for r in C.read(D.R.BASE_RAW/'PSEUDO_ACCEPTED.json') if r['kind']=='PLASTIC'}
    entries=[dict(id=r['id'],key='S'+str(r['row']),row=r['row'],domain='source',train=r['partition']=='train') for r in p['source_records']]
    entries += [dict(id=r['id'],key='R'+str(i),domain='real',train=r['train']) for i,r in enumerate(p['real_records'])]
    start=time.monotonic();records=[];counts={d:dict(old=0,new=0) for d in ['source_train','source_heldout','real_train','real_heldout']}
    # B1 caps attention memory at4x token count; no dependence on targets.
    for i,r in enumerate(entries):
        x=source_item(source,r['row']) if r['domain']=='source' else real_item(pool[r['id']])
        path=RAW/'cache'/(r['key']+'.npz')
        if not path.exists():
            feature=extract(backbone,[x])[0];path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:np.savez(f,feature=feature,old_target_valid=x['old_target_valid'],
                protocol_sha256=np.array(C.sha(DOC/'PROTOCOL.json')),**D.item_arrays(x))
        with np.load(path) as z:
            assert str(z['protocol_sha256'])==C.sha(DOC/'PROTOCOL.json') and z['feature'].shape==(384,56,42)
            assert np.isfinite(z['feature']).all()
            for k,v in D.item_arrays(x).items():np.testing.assert_array_equal(z[k],v)
            np.testing.assert_array_equal(z['old_target_valid'],x['old_target_valid'])
        group=r['domain']+('_train' if r['train'] else '_heldout')
        counts[group]['old']+=int(x['old_target_valid'].sum());counts[group]['new']+=int(x['target_valid'].sum())
        records.append(dict(**r,cache=C.bound(path)))
        if (i+1)%100==0:print('WIDE_CACHE',i+1,'/',len(entries),round(time.monotonic()-start,1),N.E.gpu(),flush=True)
    C.freeze(DOC/'CACHE_COMPLETE.json',dict(records=records,seconds=time.monotonic()-start,
        protocol=C.bound(DOC/'PROTOCOL.json'),backbone=C.bound(DOC/'BACKBONE.json'),supervision_counts=counts,real_eval_features_used=False))
    print('WIDE_CACHE_COMPLETE',len(records),counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','cache','train','infer','report']);p.add_argument('arm',nargs='?',choices=D.ARMS);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='cache':cache()
    else:
        with scope():
            if a.action=='train':D.train(a.arm)
            elif a.action=='infer':D.infer()
            else:D.report()

"""Standalone deployment adapter. No target, symmetry branch or reference input."""
import copy
import numpy as np,torch,cv2
from .a_common import context,POSE
from .a_data import affine,network_bank
from .geometry import build_bank,permute_bank
from .scorer import decode_bank

@torch.no_grad()
def predict_captured_joint(head,captured,dimensions,order,K,raw_hw,norm,*,readout='J',arm='GEO',frame_id='runtime'):
    """Preserve detector instance/box/confidence/missing/centre; score raw YOLO once.

    dimensions is canonical [W,D,H], K original pixel pinhole intrinsics.
    Return the full detector candidates plus the actual final F(q) pose.
    """
    before=captured['candidates'];index=captured['selected_index'];after=copy.deepcopy(before)
    if index is None:return dict(candidates=after,selected_index=None,head_used=False),dict(available=False),dict(actions=1,NoOp=True)
    qraw=np.array(before[index]['keypoints_xy'],float);valid=np.isfinite(qraw).all(-1)&~(qraw==-1).all(-1)
    xyz=np.asarray(dimensions)[[0,2,1]]
    try:bank=build_bank(qraw,K,xyz,raw_hw,valid)
    except (cv2.error,ValueError,FloatingPointError):bank=dict(points=qraw[None].copy(),hypotheses=['NoOp'],details=[],reason='INITIALIZATION_FAILED')
    if arm=='PERM':bank=permute_bank(bank,frame_id)
    elif arm!='GEO':raise ValueError('Arm must be GEO or PERM')
    if len(bank['points'])==1:return dict(candidates=after,selected_index=index,head_used=False),POSE.infer(qraw,K,xyz,False),dict(actions=1,NoOp=True)
    scale,offset=affine(captured['canvas_shape'],captured['input_shape']);offset=offset+captured['added_border']*scale
    qnet=(qraw*scale+offset).astype('float32');qnet[~valid]=qraw[~valid]
    box=(np.array(before[index]['box_xyxy']).reshape(2,2)*scale+offset).reshape(4).astype('float32')
    device=next(head.parameters()).device;b={k:torch.as_tensor(v,device=device)[None] for k,v in dict(points=qnet,boxes=box,point_valid=valid,input_shape=captured['input_shape']).items()}
    b.update(p3=captured['p3'].to(device),p4=captured['p4'].to(device))
    ctx=torch.as_tensor(context(np.asarray(dimensions)[None],[order],norm,False),device=device)
    qb=network_bank(bank,dict(scale=scale,offset=offset,point_valid=valid),qnet)
    out=head.forward_bank(*(b[k] for k in ['p3','p4','points','boxes','point_valid','input_shape']),context=ctx,candidate_points=torch.as_tensor(qb[None],device=device),action_valid=torch.ones((1,len(qb)),dtype=torch.bool,device=device))
    _,indices=decode_bank(out,readout);ix=indices[0].cpu().numpy();support=out['point_support'][0].cpu().numpy()
    if readout=='J':q=bank['points'][int(ix)].copy()
    else:
        q=qraw.copy()
        for i,j in enumerate(ix):q[i]=bank['points'][int(j),i]
    if readout=='I':q[:8][~support]=qraw[:8][~support]
    q[8:]=qraw[8:]
    after[index]['keypoints_xy']=q
    for i,(a,c) in enumerate(zip(before,after)):
        for k in a:
            if k!='keypoints_xy':assert np.array_equal(np.asarray(a[k]),np.asarray(c[k])),k
        if i!=index:assert np.array_equal(a['keypoints_xy'],c['keypoints_xy'],equal_nan=True)
    assert np.array_equal(qraw[8:],q[8:],equal_nan=True)
    assert np.array_equal(qraw[~valid],q[~valid],equal_nan=True)
    return dict(candidates=after,selected_index=index,head_used=True),POSE.infer(q,K,xyz,False),dict(actions=len(qb),selected_index=int(ix) if readout=='J' else ix.tolist(),NoOp=bool(np.all(ix==0)),generating_hypothesis=bank['hypotheses'][int(ix)] if readout=='J' else None)

"""GT-free inference and detection-contract proof. No phase-target calls here."""
import copy,math
import numpy as np
import torch
import dcp_env as E
from refiner import model,forward,decode,context,specification
from point_inference import replace_selected

def load_head(arm,seed):
    path=E.C.BRAW/f'runs/seed{seed}/last.pt' if arm=='OLD_P' else E.RAW/f'runs/{arm}_seed{seed}/last.pt'
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['complete'] and ck['step']==6000 and ck['baseline_checkpoint_sha256']==E.R0_SHA
    head=model(arm,ck['config']).cuda().eval();head.load_state_dict(ck['model_state_dict']);head.requires_grad_(False)
    return head,path

def preservation(before,after,selected):
    assert len(before)==len(after)
    for i,(a,b) in enumerate(zip(before,after)):
        assert set(a)==set(b)
        for key in a:
            if key!='keypoints_xy':assert np.array_equal(np.asarray(a[key]),np.asarray(b[key])),key
        if i==selected:assert np.array_equal(np.asarray(a['keypoints_xy'])[8],np.asarray(b['keypoints_xy'])[8])
        else:assert np.array_equal(np.asarray(a['keypoints_xy']),np.asarray(b['keypoints_xy']))
    expected=int(np.argmax([r['score'] for r in after])) if after else None
    assert expected==selected

def serial(candidates):
    return [{k:v.tolist() if hasattr(v,'tolist') else v for k,v in c.items()} for c in candidates]

def registry_input(object_type):
    registry=E.read(E.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json')['objects']
    groups=E.read(E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']
    r=next((r for r in registry if r['object_type']==object_type),None)
    g=next((g for g in groups if g['object_type']==object_type),None)
    if r is None or g is None:raise ValueError('Unknown deployment object type: no GT geometry fallback')
    d=r['physical_dimensions_m'];return np.array([d['x'],d['z'],d['y']]),g['group_order']

@torch.no_grad()
def predict_captured(head,arm,captured,dimensions,order,T,rule,raw_hw,norm,diagnostic_context=None):
    features=E.old('features');inputs=features.branch_inputs(captured);selected=captured['selected_index']
    before=captured['candidates']
    dim,_=specification(arm)
    z=None
    if dim:
        z=context(np.array(dimensions)[None],[order],norm,dim==8) if diagnostic_context is None else np.asarray(diagnostic_context)
        if z.shape!=(1,dim) or not np.isfinite(z).all():raise ValueError('Required geometry context is invalid, including on empty detections')
    if inputs is None:return dict(candidates=serial(before),selected_index=None,head_used=False),None
    b={k:torch.as_tensor(inputs[k],device='cuda')[None] for k in ['points','boxes','point_valid','input_shape']}
    b.update(p3=captured['p3'],p4=captured['p4'])
    if dim:
        b['context']=torch.as_tensor(z,device='cuda',dtype=torch.float32)
    o=forward(head,b);fraction=rule['max_move_image_diagonal_fraction'];cap=None if fraction is None else fraction*math.hypot(*raw_hw)*inputs['gain']
    q=decode(o,T,rule['lam'],cap)[0].cpu().numpy()
    after=replace_selected(before,selected,inputs['points'],q,inputs['gain'],rule['lam']);preservation(before,after,selected)
    return dict(candidates=serial(after),selected_index=selected,head_used=True),dict(logits=o['logits'][0].cpu().numpy(),support=o['point_support'][0].cpu().numpy(),points=q)

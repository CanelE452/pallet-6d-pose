"""Prediction-only expert routing contract, no severity/session or reference inputs."""
import numpy as np
from . import common as C
from . import features as F
from . import models as M

PER=['bbox_conf',*[f'kpt_conf{k}' for k in range(9)],'kpt_conf_mean','kpt_conf_min','kpt_conf_P10','kpt_conf_P50']
RES=[n for n in F.names() if n.startswith('residual_')]
CROSS=['bbox_IoU','center_dx_norm','center_dy_norm','size_dw_norm','size_dh_norm',*[f'kpt{k}_L2_bboxnorm' for k in range(9)],
       'kpt_L2_mean_bboxnorm','kpt_L2_max_bboxnorm',*[f'diff_{n}' for n in PER],
       'selected_WD_disagreement','centroid_disagreement_m','centroid_disagreement_objectnorm','rotation_disagreement_deg','selector_margin_difference']

def contract():
    C.freeze(C.sdoc(4)/'ROUTER_FEATURE_CONTRACT.json',dict(created_at=C.now(),per_expert=PER+['selector_margin']+RES,
        cross_model=CROSS,context='frozen S1 pose-head GAP448 if available',GT_input=False,severity_input=False,session_input=False,
        missing='zero-vector per missing expert; numerical nonfinite replaced0; no extra confidence threshold',
        occlusion=dict(seed=20260926,policy='Original S1 plan size/fill/coverage/paired-placement constraints loaded as function AST; synthetic supervised0..7, P8 ignored',
            scheduled_probability=.5,coordinate_space='640 letterbox policy canvas; accepted rectangle mapped back to original prepared RGB with integer bounds',
            no_error_based_placement=True,skipped_policy_occurrences='retain identical clean image and record not-applied; do not force application or resample'),
        train=dict(architecture='input->64->GELU->32->GELU->1',label='lower exact synthetic ADDnorm; |diff|<=1e-9 tie S0',seed=42,lr=.001,weight_decay=.0001,batch=256,max_epoch=30,patience=5,val='expert-selection accuracy'),
        real_features_locked_before_stage4_scoring=True,dual_expert_pilot_not_Jetson_deployment=True))

def make(p0,p1,g0,g1,ctx,choice0,choice1,margin0,margin1,dims):
    features=[];allnames=F.names();rows=[]
    for p,g,ch,margin in [(p0,g0,choice0,margin0),(p1,g1,choice1,margin1)]:
        c=C.selected(p);idx=C.HYP.index(ch) if ch in C.HYP else -1
        v=np.array(g['features'][idx]) if g['valid'] and idx>=0 else np.zeros(len(allnames))
        features.extend(v[[allnames.index(n) for n in PER]]);features.append(margin)
        features.extend(v[[allnames.index(n) for n in RES]]);rows.append((c,v))
    a,b=rows[0][0],rows[1][0]
    if a is None or b is None:cross=np.zeros(len(CROSS))
    else:
        aa=np.array(a['box_xyxy']);bb=np.array(b['box_xyxy']);wh0=np.maximum(aa[2:]-aa[:2],1e-6);wh1=np.maximum(bb[2:]-bb[:2],1e-6);diag=max(np.linalg.norm(wh0),1e-6)
        inter=np.prod(np.maximum(0,np.minimum(aa[2:],bb[2:])-np.maximum(aa[:2],bb[:2])));iou=inter/max(np.prod(wh0)+np.prod(wh1)-inter,1e-6)
        centers=((bb[:2]+bb[2:])-(aa[:2]+aa[2:]))/2/diag;size=(wh1-wh0)/diag
        kd=np.linalg.norm(np.array(a['keypoints_xy'])-np.array(b['keypoints_xy']),axis=1)/diag
        va,vb=rows[0][1],rows[1][1];conf_diff=vb[[allnames.index(n) for n in PER]]-va[[allnames.index(n) for n in PER]]
        t0=va[[allnames.index(f't_{k}_m') for k in 'xyz']];t1=vb[[allnames.index(f't_{k}_m') for k in 'xyz']];td=np.linalg.norm(t1-t0)
        R0=va[[allnames.index(f'R_cf_{i}{j}') for i in range(3) for j in range(3)]].reshape(3,3);R1=vb[[allnames.index(f'R_cf_{i}{j}') for i in range(3) for j in range(3)]].reshape(3,3)
        angle=np.degrees(np.arccos(np.clip((np.trace(R0.T@R1)-1)/2,-1,1)))
        cross=[iou,*centers,*size,*kd,kd.mean(),kd.max(),*conf_diff,float(choice0!=choice1),td,td/np.linalg.norm(dims),angle,margin1-margin0]
    assert len(cross)==len(CROSS)
    return np.nan_to_num(np.r_[features,cross,ctx],nan=0,posinf=0,neginf=0).astype(np.float32)

def choices(z,preds,base,ck=None):
    result={}
    for a in C.ARMS:
        if base=='SYNTH_SCORER':
            scores=M.scores(ck,M.pack_inputs(z,a,ck['variant']));idx=M.selection(scores,C.HYP)
        else:
            scores=np.array([[h['score'] if h['score'] is not None else 0 for h in preds[a][str(i)]['geometry']['hypotheses']] if len(preds[a][str(i)]['geometry']['hypotheses'])==2 else [0,0] for i in z['ids']])
            idx=z[a+'_current']
        idx=np.where(z[a+'_valid'],idx,z[a+'_current'])
        result[a]=dict(selected=idx,margin=np.abs(scores[:,0]-scores[:,1]))
    return result

if __name__=='__main__':contract()

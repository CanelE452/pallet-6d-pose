"""V94 single-scalar inference. Source/root entry before current/checkpoint I/O."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse
import csv
import importlib.util
import time

import numpy as np

import current_common as c


def decode_rows(base,z,probability,operator_available):
    require=c.require;n=len(base)
    require(z.shape==probability.shape==operator_available.shape==(n,),'Output shape changed')
    require(z.dtype==probability.dtype==np.float32 and np.isfinite(z).all()and np.isfinite(probability).all(),'Finite FP32 outputs required')
    rows=[]
    for i,old in enumerate(base):
        prior=bool(old['valid']);op=bool(operator_available[i]);valid=prior and op;visible=bool(z[i]<0)
        geometry=old['geometry_self_hidden'];guarded=geometry if valid else None
        r=dict(old);r.update(prior_valid=prior,operator_query_available=op,additional_operator_U=prior and not op,
            valid=valid,status='OK'if valid else'UNKNOWN',
            reason=''if valid else(old['prior_reason']if not prior else'EXTRA_OPERATOR_INPUT_SUPPORT_U'),
            raw_query_aux_hidden_logit=float(z[i]),raw_p_visible_FP32=float(probability[i]),
            logit=-float(z[i])if valid else None,p_visible=float(probability[i])if valid else None,
            authoritative_decision=('D'if visible else'E')if valid else'UNKNOWN',
            decision_visible=visible if valid else None,present_at_frozen_threshold=visible if valid else None,
            geometry_self_hidden_if_valid=guarded,
            old_geometry_route_visible=(False if geometry is True else visible)if valid else None,
            self_occlusion_learned=False,external_vs_self_identified=False,
            semantic_U_detector=False,native_association_known=False)
        rows.append(r)
    return rows


def predict_all(model,features,torch,device,batch_size=64):
    """One identical scalar operator; no GT/region/mosaic readout selection."""
    n=len(features['p3_roi']);zs=[];available=[];prob=[];attention=[]
    with torch.inference_mode():
        for first in range(0,n,batch_size):
            ix=slice(first,min(first+batch_size,n))
            kwargs=dict(p3_roi=torch.from_numpy(np.ascontiguousarray(features['p3_roi'][ix])).to(device),
                        local_rgb96=torch.from_numpy(np.ascontiguousarray(features['rgb96'][ix])).to(device),
                        local_valid96=torch.from_numpy(np.ascontiguousarray(features['crop_valid96'][ix])).to(device))
            a=model(**kwargs);b=model(**kwargs)
            for name in ('query_logit','query_available','attention_diagnostic_only'):
                c.require(torch.equal(a[name],b[name]),'Same-batch repeat differs: '+name)
            z=a['query_logit'].float();valid=a['query_available']
            c.require(z.shape==(len(z),)and torch.isfinite(z).all(),'Scalar output changed')
            zs.append(z.cpu().numpy());available.append(valid.cpu().numpy())
            prob.append(torch.sigmoid(-z).cpu().numpy());attention.append(a['attention_diagnostic_only'].float().cpu().numpy())
    return np.concatenate(zs),np.concatenate(available),np.concatenate(prob),np.concatenate(attention)


def run(entry_path):
    entry,policy,b=c.validate_entry(entry_path,__file__,'INFER')
    out=Path(entry['output_directory']);out.mkdir(parents=True,exist_ok=False)
    c.write_new(out/'STARTED.json',dict(entry=c.bind(entry_path),source_freeze=b['source_model_freeze'],
                current_GT_read=False,new_backbone_initializer_optimizer_PnP=0))
    import torch
    c.require(str(torch.__version__)=='2.1.1+cu118'and torch.cuda.is_available(),'Frozen Torch/CUDA runtime required')
    for name in c.CURRENT_KEYS:c.check(b[name])
    # Pinned V91 reader has no import-time I/O. It checks original 223x8 order,
    # q/L/preparation/factual45U, and deserializes only the five allowed members.
    old=c.prior_reader();features,base,q,L=old.load_current(b)
    torch.set_num_threads(1);torch.manual_seed(194);torch.cuda.manual_seed_all(194)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=importlib.util.spec_from_file_location('v94_source_selected_scalar',b['model_code']['path'])
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    model=mod.QueryCenterScalar().to('cuda:0');c.require(sum(v.numel()for v in model.parameters())==26573,'Architecture changed')
    payload=torch.load(b['checkpoint']['path'],map_location='cuda:0',weights_only=True)
    freeze=c.read(b['source_model_freeze']['path'])
    c.require(payload['update']==freeze['selected_update']and payload['parameter_count']==26573,'Selected checkpoint changed')
    c.require(c.same(payload['model_code'],b['model_code'])and c.same(payload['protocol'],b['source_fit_protocol']),'Checkpoint metadata mismatch')
    model.load_state_dict(payload['state_dict'],strict=True);model.eval().requires_grad_(False)
    before=old.state_sha(model);start=time.monotonic()
    z,op,pv,attention=predict_all(model,features,torch,'cuda:0',64)
    c.require(old.state_sha(model)==before,'Frozen model state changed')
    rows=decode_rows(base,z,pv,op);valid=np.array([r['valid']for r in rows]);prior=np.array([r['valid']for r in base])
    c.require(len(rows)==1784 and prior.sum()==1739 and not np.any(valid&~prior),'Prior slots/U changed')
    with(out/'REAL_QUERY_SCORES.csv').open('x',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out/'RAW_QUERY_SCALAR_OUTPUTS.npz',query_logit=z,query_available=op,prior_valid=prior,valid=valid,
        p_visible_FP32=pv,attention_diagnostic_only=attention,query_xy=features['query_xy'],side_L=features['side_L'],
        original_query_xy=q,original_side_L=L,id=np.array([r['id']for r in base]),native_corner=np.array([r['native_corner']for r in base],np.int8))
    c.write_new(out/'INPUT_PREPARATION_AUDIT.json',dict(completed=True,frames=223,slots=1784,prior_valid=1739,prior_U=45,
        original_q_L_slot_candidate_input_order_exact=True,source_model_freeze=b['source_model_freeze'],
        prior_artifact_bindings={k:b[k]for k in c.CURRENT_KEYS},original_P3_dtype=str(features['p3_roi'].dtype),
        original_P3_shape=list(features['p3_roi'].shape),P3_input_requantization=False,original_native65_binary=True,
        original_L_min=float(L.min()),original_L_max=float(L.max()),original_L_percentiles={str(k):float(np.percentile(L,k))for k in(0,25,50,75,90,95,99,100)},
        operator_available=int(op.sum()),extra_operator_U=int(np.sum(prior&~op)),input_whitelist=['p3_roi','rgb96','crop_valid96'],
        metadata_only=['query_xy','side_L'],GT_reference_region_or_mosaic_read=False,new_RGB_decode_backbone_initializer_PnP=0))
    for name in ('checkpoint','model_code'):c.check(b[name])
    c.write_new(out/'INFERENCE_SEAL.json',dict(completed=True,frames=223,slots=1784,prior_valid=1739,prior_U=45,
        valid=int(valid.sum()),U=int((~valid).sum()),additional_U=int(np.sum(prior&~valid)),original45U_preserved=True,
        entry=c.bind(entry_path),protocol=entry['protocol'],artifact_bindings=entry['artifact_bindings'],
        source_model_freeze=b['source_model_freeze'],source_gate_results=b['source_gate_results'],source_postfit_audit=b['source_postfit_audit'],
        checkpoint=b['checkpoint'],code=c.bind(__file__),preparation=c.bind(out/'INPUT_PREPARATION_AUDIT.json'),
        outputs=[c.bind(out/'REAL_QUERY_SCORES.csv'),c.bind(out/'RAW_QUERY_SCALAR_OUTPUTS.npz')],raw_E_threshold_logit=0.,D_rule='z<0',
        probability_is_display_only=True,rounded_pvisible_half_vs_rawsign_mismatch_N=int(np.sum(valid&((pv>=.5)!=(z<0)))),
        runtime_torch=str(torch.__version__),source_runtime_matched=True,model_unchanged=True,model_state_SHA256=before,repeat_scalar_available_attention_bitexact=True,batch_size=64,
        current_used_for_selection=False,current_GT_read=False,new_backbone=0,new_initializer=0,new_RGB_decodes=0,optimizer_updates=0,PnP=0,
        self_occlusion_learned=False,external_vs_self_identified=False,geometry_route_separate=True,
        meaning='Public weak AUX D/hiddenE to human D/E transfer; native association UNKNOWN. Binary hidden on humanS is not learnedself.',seconds=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)

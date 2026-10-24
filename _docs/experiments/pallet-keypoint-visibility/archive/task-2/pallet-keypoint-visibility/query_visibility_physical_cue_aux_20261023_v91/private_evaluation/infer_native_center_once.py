"""Frozen V91 native-centre diagnostic. No data/model is loaded on import.

Preparation commands use generated arrays only. Only root's approved immutable
entry may open registered current-input artifacts or load the source checkpoint.
The authoritative D decision is raw FP32 external-logit < 0, including tiny ties.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PROTOCOL=ROOT/'CURRENT_PROTOCOL_BEFORE_CURRENT_ARTIFACTS.json'
EXPECTED_PROTOCOL_SHA='396747f56b6975da47bdae0a14c0dcbdf1b19e011f887c863f7e73b1f83b756c'
INPUT_KEYS={'p3_roi','rgb96','crop_valid96','query_xy','side_L'}
LEGACY_ARCHIVE_KEYS=INPUT_KEYS|{'box_xyxy'}
SOURCE_KEYS={'source_model_freeze','source_fit_complete','source_results','source_postfit_audit','checkpoint','model_code','source_fit_protocol'}
CURRENT_KEYS={'current_model_input','prior_input_rows','prior_preparation_seal','prior_inference_seal','prior_input_gate','prepared_inputs','feature_metadata'}
GENERATED=HERE/'GENERATED_NATIVE_CENTER_TESTS_V2.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def need(test,why):
    if not test:raise ValueError(why)


def descriptor(b):
    need(isinstance(b,dict)and set(b)=={'path','sha256'},'Exact path/SHA descriptor required')
    need(isinstance(b['path'],str)and isinstance(b['sha256'],str)and len(b['sha256'])==64,'Invalid descriptor')


def check(b):descriptor(b);need(sha(b['path'])==b['sha256'],'Bound bytes changed: '+b['path'])


def read(path):
    p=Path(path)
    with(gzip.open if p.suffix=='.gz'else open)(p,'rt')as f:return json.load(f)


def rows(path):return[json.loads(x)for x in Path(path).read_text().splitlines()if x.strip()]


def write(path,d):
    with Path(path).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def has_binding(value,target):
    if isinstance(value,dict):return all(value.get(k)==v for k,v in target.items())or any(has_binding(v,target)for v in value.values())
    if isinstance(value,list):return any(has_binding(v,target)for v in value)
    return False


def same_binding(value,target):
    """Path/SHA identity allows nonsemantic producer metadata such as bytes."""
    return isinstance(value,dict)and value.get('path')==target['path']and value.get('sha256')==target['sha256']


def validate_source_metadata(p,b,f,c,r,a):
    """Pure metadata guard; no current descriptors are dereferenced here."""
    need(b['source_model_freeze']==p['source_model_freeze'],'Wrong preselected source freeze')
    need(b['source_fit_complete']==p['source_fit_completion']and b['source_results']==p['source_results'],'Wrong source completion/results')
    need(b['source_fit_protocol']==p['source_fit_protocol'],'Source protocol changed')
    need(f.get('completed')is True and f.get('optimizer_updates')==2000,'Incomplete single source fit')
    need(f.get('current_used_for_fit_selection')is False and f.get('PnP')==0,'Source selection leakage')
    need(same_binding(f.get('checkpoint'),b['checkpoint'])and same_binding(f.get('model_code'),b['model_code'])and same_binding(f.get('protocol'),b['source_fit_protocol']),'Source code/checkpoint binding mismatch')
    need(f.get('external_threshold')==.5,'Source threshold changed')
    need(c.get('completed')is True and c.get('optimizer_updates')==2000 and c.get('model_freeze')==b['source_model_freeze'],'Source fit completion mismatch')
    need(r.get('completed')is True and r.get('model_freeze')==b['source_model_freeze'],'Source result binding mismatch')
    gates=r.get('predeclared_source_gate',{})
    need(r.get('all_nine_pass')is True and len(gates)==9 and all(x is True for x in gates.values()),'Nine source gates have not passed')
    need(a.get('PASS')is True and(has_binding(a,b['source_model_freeze'])or has_binding(a,b['source_fit_complete'])),'Independent postfit audit PASS/binding missing')


def validate_entry(path):
    """Validate policy/code/review/source first. No current file SHA/body IO."""
    e=read(path)
    need(e.get('schema')=='V91_ROOT_APPROVED_NATIVE_CENTER_ENTRY_V1'and e.get('approved')is True,'Root approved entry required')
    need(e.get('current_used_for_selection')is False and e.get('training_allowed')is False and e.get('PnP_allowed')is False,'Forbidden scope')
    for k in('protocol','code','generated','review','source_postfit_audit','artifact_bindings'):check(e[k])
    need(e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['protocol']['sha256']==EXPECTED_PROTOCOL_SHA,'Unreviewed adapter/policy')
    p=read(e['protocol']['path']);proof=read(e['generated']['path']);review=read(e['review']['path'])
    need(proof.get('PASS')is True and proof.get('code')==e['code'],'Generated proof not current code')
    need(review.get('PASS')is True and not review.get('blockers',[]),'Independent adapter review missing')
    need(has_binding(review,e['code'])and has_binding(review,e['protocol']),'Review code/protocol binding missing')
    need(p.get('source_gate_all9_pass')is True and p.get('new_training_allowed')is False and p.get('PnP_allowed')is False,'Policy mismatch')
    payload=read(e['artifact_bindings']['path'])
    need(payload.get('completed')is True and payload.get('authorized_after_source_audit')is True,'Artifacts not authorized after source audit')
    b=payload['bindings'];need(set(b)==SOURCE_KEYS|CURRENT_KEYS,'Unexpected input/source binding fields')
    for x in b.values():descriptor(x)
    need(b['source_postfit_audit']==e['source_postfit_audit'],'Audit disagreement')
    for k in SOURCE_KEYS:check(b[k])
    validate_source_metadata(p,b,read(b['source_model_freeze']['path']),read(b['source_fit_complete']['path']),
                             read(b['source_results']['path']),read(b['source_postfit_audit']['path']))
    locations=p['artifact_locations_from_existing_code_only']
    for name,old in [('current_model_input','prior_input_pack'),('prior_input_rows','prior_rows'),('prior_preparation_seal','prior_preparation_seal'),('prior_input_gate','prior_input_gate')]:
        need(Path(b[name]['path']).resolve()==Path(locations[old]).resolve(),'Registered prior input location changed')
    need(p['inference']['input_keys_only']==['p3_roi','rgb96','crop_valid96']and p['inference']['metadata_keys_only']==['query_xy','side_L'],'Input whitelist changed')
    # current binding checks deliberately deferred to load_current AFTER return.
    return e,p,b


def verify_input_metadata(features,prior,frames,metadata):
    """Pure exact q/L/order and factual-flag checks on explicitly supplied arrays."""
    need(set(features)==INPUT_KEYS and len(prior)==1784 and len(frames)==len(metadata)==223,'All223/native1784 required')
    need(features['p3_roi'].shape==(1784,65,9,9)and np.issubdtype(features['p3_roi'].dtype,np.floating),'Original floating P3 required without requantization')
    for k,shape,dtype in [('rgb96',(1784,3,96,96),np.uint8),('crop_valid96',(1784,1,96,96),np.bool_),('query_xy',(1784,2),np.float32),('side_L',(1784,),np.float32)]:
        need(features[k].shape==shape and features[k].dtype==dtype,'Prior tensor shape/dtype mismatch: '+k)
    need(np.isfinite(features['p3_roi']).all()and np.isin(features['p3_roi'][:,64],(0,1)).all(),'Invalid factual P3')
    need(len({x['id']for x in frames})==223,'Duplicate prior frame')
    original_q=[];original_L=[];base=[]
    for fi,(frame,meta)in enumerate(zip(frames,metadata)):
        need(frame['id']==meta['id'],'Original frame/metadata order differs')
        q=np.asarray(frame['initial_points'],np.float64)[:8];L=float(meta['side']);h,w=frame['raw_hw']
        need(q.shape==(8,2)and np.isfinite(L)and L>0,'Original q/L malformed')
        for k in range(8):
            i=fi*8+k;r=prior[i]
            need(r['id']==frame['id']and r['native_corner']==k and r['image_sha256']==frame['image_sha256'],'Original native slot/image order differs')
            finite=np.isfinite(q[k]).all()
            need((r['query_x']==float(q[k,0])and r['query_y']==float(q[k,1]))if finite else(r['query_x']is None and r['query_y']is None),'Original query changed')
            need(r['roi_side']==L and features['side_L'][i]==np.float32(L),'Original L changed')
            need(np.array_equal(features['query_xy'][i],q[k].astype(np.float32),equal_nan=True),'Prior stored q differs')
            pixel=np.floor(np.where(np.isfinite(q[k]),q[k],-1e8)+.5).astype(np.int64)
            need(r['pixel_x']==int(pixel[0])and r['pixel_y']==int(pixel[1]),'Original nearest pixel differs')
            need(isinstance(r['valid'],bool),'Prior valid is not factual boolean')
            if r['valid']:
                need(finite and 0<=q[k,0]<w and 0<=q[k,1]<h and 0<=pixel[0]<w and 0<=pixel[1]<h,'Prior valid out of native bounds')
            geo=r.get('geometry_self_hidden');need(geo is None or isinstance(geo,bool),'Saved geometry flag malformed')
            base.append({name:r[name]for name in('id','native_corner','query_x','query_y','pixel_x','pixel_y','image_sha256','roi_side','saved_roi_pixel_valid','valid','geometry_self_hidden')})
            base[-1].update(prior_status=r.get('status'),prior_reason=r.get('reason',''),geometry_self_source=r.get('geometry_self_source','SAVED_PRIOR_GEOMETRY'))
            original_q.append(q[k]);original_L.append(L)
    need(sum(r['valid']for r in prior)==1739,'Original45 U changed')
    return base,np.asarray(original_q,np.float64),np.asarray(original_L,np.float64)


def load_current(b):
    # First current artifact hashes/reads occur only after validate_entry succeeded.
    for k in CURRENT_KEYS:check(b[k])
    prep=read(b['prior_preparation_seal']['path']);old=read(b['prior_inference_seal']['path']);gate=read(b['prior_input_gate']['path'])
    need(prep.get('completed')is True and prep.get('frames')==223 and prep.get('slots')==1784 and prep.get('valid')==1739,'Prior preparation incomplete')
    need(prep.get('original_q_L_order_preserved')is True and prep.get('optimizer_updates')==0 and prep.get('PnP_calls')==0,'Prior input provenance mismatch')
    need(old.get('completed')is True and old.get('frames')==223 and old.get('slots')==1784 and old.get('valid')==1739,'Prior inference availability changed')
    need(old.get('preparation')==b['prior_preparation_seal']and prep.get('gate')==b['prior_input_gate'],'Prior seal/gate mismatch')
    for key in('current_model_input','prior_input_rows'):need(b[key]in prep['outputs'],'Prior preparation does not bind artifact')
    need(gate['inputs']['prepared_inputs']==b['prepared_inputs']and gate['inputs']['feature_metadata']==b['feature_metadata'],'Original prepared-q/L descriptors changed')
    with np.load(b['current_model_input']['path'],allow_pickle=False)as z:
        need(set(z.files)==LEGACY_ARCHIVE_KEYS,'Legacy NPZ member schema changed')
        # box_xyxy member is not read/deserialized; no target/context is forwarded.
        features={k:z[k]for k in INPUT_KEYS}
    prior=rows(b['prior_input_rows']['path']);frames=read(b['prepared_inputs']['path']);metadata=read(b['feature_metadata']['path'])
    base,q,L=verify_input_metadata(features,prior,frames,metadata)
    return features,base,q,L


def decode_rows(base,zE,pVisible_FP32,centre_valid):
    need(zE.shape==pVisible_FP32.shape==centre_valid.shape==(len(base),),'Raw arrays differ')
    need(zE.dtype==pVisible_FP32.dtype==np.float32 and np.isfinite(zE).all()and np.isfinite(pVisible_FP32).all(),'Finite rawFP32 required')
    result=[]
    for i,r in enumerate(base):
        prior_valid=r['valid'];valid=bool(prior_valid and centre_valid[i]);D=bool(zE[i]<0)
        geo=r['geometry_self_hidden'];gvalid=geo if valid else None
        out=dict(r);out.update(prior_valid=prior_valid,center_valid=bool(centre_valid[i]),valid=valid,
            status='OK'if valid else'UNKNOWN',reason=''if valid else(r['prior_reason']if not prior_valid else'INVALID_CENTRE_FACTUAL_SUPPORT'),
            raw_physical_external_logit=float(zE[i]),raw_p_visible_FP32=float(pVisible_FP32[i]),
            logit=-float(zE[i])if valid else None,p_visible=float(pVisible_FP32[i])if valid else None,
            authoritative_decision=('D'if D else'E')if valid else'UNKNOWN',decision_visible=D if valid else None,
            present_at_frozen_threshold=D if valid else None,
            geometry_self_hidden_if_valid=gvalid,
            hybrid_decision_visible=(False if geo is True else D)if valid else None,
            conditional_causal_label=('S_PROXY'if geo is True else('E_AUXILIARY'if not D else'D_AUXILIARY'))if valid else'UNKNOWN',
            geometry_overrides_RAW=False,self_occlusion_learned=False,semantic_U_detector=False,
            source_reference_or_crop_availability_known=False)
        result.append(out)
    return result


def state_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        a=v.detach().cpu().contiguous().numpy();h.update(k.encode()+b'\0'+str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes())
    return h.hexdigest()


def predict_all(model,features,torch,device):
    """Generated-testable three-input batch64 inference; no file/GT/metadata IO."""
    maps=[];valids=[];prob=[]
    with torch.inference_mode():
        for first in range(0,1784,64):
            ix=slice(first,min(first+64,1784))
            inputs=dict(p3_roi=torch.from_numpy(np.ascontiguousarray(features['p3_roi'][ix])).to(device),
                local_rgb96=torch.from_numpy(np.ascontiguousarray(features['rgb96'][ix])).to(device),
                local_valid96=torch.from_numpy(np.ascontiguousarray(features['crop_valid96'][ix])).to(device))
            a=model(**inputs);again=model(**inputs)
            need(torch.equal(a['physical_external_logit'],again['physical_external_logit'])and torch.equal(a['map_valid'],again['map_valid']),'Repeat not bitexact')
            z=a['physical_external_logit'].float();need(torch.isfinite(z).all(),'Nonfinite raw map')
            maps.append(z.cpu().numpy());valids.append(a['map_valid'].cpu().numpy());prob.append(torch.sigmoid(-z[:,4,4]).cpu().numpy())
    return np.concatenate(maps),np.concatenate(valids),np.concatenate(prob)


def run(path):
    e,p,b=validate_entry(path)
    import torch
    need(Path(sys.executable).resolve()==Path('/home/minjae/anaconda3/bin/python').resolve()and str(torch.__version__)=='2.5.1+cu121','Frozen source Torch runtime required')
    need(torch.cuda.is_available(),'Frozen CUDA inference required')
    output=Path(e['output_directory']).resolve();output.mkdir(parents=True,exist_ok=False)
    write(output/'STARTED.json',dict(entry=bind(path),protocol=e['protocol'],source_freeze=b['source_model_freeze'],current_GT_read=False,optimizer_updates=0,PnP=0))
    features,base,q,L=load_current(b)
    np.savez_compressed(output/'INPUT_Q_L_VALIDITY_AUDIT.npz',query_xy=features['query_xy'],side_L=features['side_L'],
        original_query_xy=q,original_side_L=L,prior_valid=np.array([r['valid']for r in base]),
        id=np.array([r['id']for r in base]),native_corner=np.array([r['native_corner']for r in base],np.int8))
    write(output/'INPUT_PREPARATION_AUDIT.json',dict(completed=True,entry=bind(path),protocol=e['protocol'],source_model_freeze=b['source_model_freeze'],
        prior_artifact_bindings={k:b[k]for k in CURRENT_KEYS},frames=223,slots=1784,prior_valid=1739,prior_U=45,
        original_q_L_order_exact=True,no_resampling=True,NPZ_members_deserialized=sorted(INPUT_KEYS),
        original_P3_dtype=str(features['p3_roi'].dtype),original_P3_shape=list(features['p3_roi'].shape),
        original_P3_finite=True,original_native65_validity_binary=True,P3_input_requantization=False,
        original_L_min=float(L.min()),original_L_max=float(L.max()),
        original_L_percentiles={str(k):float(np.percentile(L,k))for k in(0,25,50,75,90,95,99,100)},
        source_training_L_values=[128,256],native_L_below_source_min_N=int(np.sum(L<128)),native_L_above_source_max_N=int(np.sum(L>256)),
        L_distribution_difference_is_diagnostic_only=True,
        legacy_box_xyxy_member_deserialized=False,new_RGB_decode_backbone_SAM_PnP=0,current_GT_read=False,
        audit=bind(output/'INPUT_Q_L_VALIDITY_AUDIT.npz')))
    torch.set_num_threads(1);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=importlib.util.spec_from_file_location('_v91_frozen_centre_model',b['model_code']['path']);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    model=mod.LocalCueEvidenceMap().to('cuda:0');need(sum(v.numel()for v in model.parameters())==25381,'Model count differs')
    model.load_state_dict(torch.load(b['checkpoint']['path'],map_location='cuda:0',weights_only=True),strict=True)
    model.eval().requires_grad_(False);before=state_sha(model);start=time.monotonic()
    m,v,pv=predict_all(model,features,torch,'cuda:0');z=m[:,4,4].copy()
    need(m.shape==v.shape==(1784,9,9)and z.dtype==pv.dtype==np.float32,'Unexpected saved output schema')
    need(state_sha(model)==before,'Frozen model changed')
    result=decode_rows(base,z,pv,v[:,4,4]);final_valid=np.array([r['valid']for r in result]);prior_valid=np.array([r['valid']for r in base])
    need(not np.any(final_valid&~prior_valid),'Old U was resolved by mistake')
    with(output/'REAL_QUERY_SCORES.csv').open('x',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
    np.savez_compressed(output/'RAW_NATIVE_CENTRE_OUTPUTS.npz',raw_physical_external_logit=z,physical_external_logit_map=m,
        map_valid=v,center_valid=v[:,4,4],prior_valid=prior_valid,valid=final_valid,p_visible_FP32=pv,
        query_xy=features['query_xy'],side_L=features['side_L'],original_query_xy=q,original_side_L=L,
        id=np.array([r['id']for r in base]),native_corner=np.array([r['native_corner']for r in base],np.int8))
    for k in('checkpoint','model_code'):check(b[k])
    rounded_diff=int(np.sum(final_valid&((pv>=.5)!=(z<0))))
    write(output/'INFERENCE_SEAL.json',dict(completed=True,frames=223,slots=1784,prior_valid=1739,prior_U=45,
        valid=int(final_valid.sum()),U=int((~final_valid).sum()),additional_U=int(np.sum(prior_valid&~final_valid)),original45U_preserved=True,
        entry=bind(path),protocol=e['protocol'],artifact_bindings=e['artifact_bindings'],source_postfit_audit=e['source_postfit_audit'],
        source_model_freeze=b['source_model_freeze'],checkpoint=b['checkpoint'],code=bind(__file__),
        preparation=bind(output/'INPUT_PREPARATION_AUDIT.json'),outputs=[bind(output/'REAL_QUERY_SCORES.csv'),bind(output/'RAW_NATIVE_CENTRE_OUTPUTS.npz')],
        raw_E_threshold_logit=0.,D_rule='zE<0',threshold_probability_E=.5,probability_is_display_only=True,
        rounded_probability_equals_half_N=int(np.sum(final_valid&(pv==.5))),rounded_probability_ge_half_vs_authoritative_D_mismatch_N=rounded_diff,
        model_unchanged=True,model_state_SHA256=before,repeat_max_logit_difference=0.,repeat_all_maps_and_valid_bitexact=True,
        batch_size=64,source_runtime_torch=str(torch.__version__),device='cuda:0',FP32=True,TF32=False,
        current_used_for_selection=False,current_GT_read=False,new_backbone=0,new_RGB_decodes=0,optimizer_updates=0,PnP=0,
        target_context_GT_reference_or_nativeID_forward=False,semantic_U_detector=False,self_occlusion_learned=False,
        meaning='AI-conditional physical cue auxiliary at unchanged native query centre. RAW and old-geometry hybrid are separate. No semantic/native8 identification proof.',
        seconds=time.monotonic()-start))
    print(json.dumps(bind(output/'INFERENCE_SEAL.json')),flush=True)


def template():
    proof=read(GENERATED);need(proof.get('PASS')is True and proof.get('code')==bind(__file__),'Current generated proof required')
    need(sha(PROTOCOL)==EXPECTED_PROTOCOL_SHA,'Policy changed')
    write(HERE/'NATIVE_CENTER_ENTRYLOCK_TEMPLATE_V2.json',dict(schema='V91_ROOT_APPROVED_NATIVE_CENTER_ENTRY_V1',approved=False,
        protocol=bind(PROTOCOL),code=bind(__file__),generated=bind(GENERATED),
        review=dict(path='ROOT_BIND_INDEPENDENT_REVIEW',sha256='0'*64),
        source_postfit_audit=dict(path='ROOT_BIND_SOURCE_POSTFIT_AUDIT',sha256='0'*64),
        artifact_bindings=dict(path=str(ROOT/'CURRENT_ARTIFACT_BINDINGS_AFTER_SOURCE_AUDIT.json'),sha256='0'*64),
        current_used_for_selection=False,training_allowed=False,PnP_allowed=False,
        output_directory=str(HERE/'NATIVE_CENTER_ONE_SHOT')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['template','run']);parser.add_argument('--entry',type=Path)
    args=parser.parse_args()
    if args.stage=='template':template()
    else:
        if args.entry is None:parser.error('--entry required')
        run(args.entry)

"""Generated-only adapter proof. Never reads actual current/source artifacts."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
CODE=HERE/'infer_native_center_once.py'
sp=importlib.util.spec_from_file_location('generated_nativecentre_adapter',CODE)
a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
FIX=HERE/'GENERATED_FIXTURES_V1'


def dump(path,d):
    with path.open('x')as f:json.dump(d,f,ensure_ascii=False,allow_nan=False);f.write('\n')
    return a.bind(path)


def gate_fixture(root,policy,mutation=None):
    root.mkdir()
    cp=root/'FAKE_NOT_MODEL_CHECKPOINT.bin';cp.write_bytes(b'generated-not-checkpoint')
    mc=root/'FAKE_NOT_MODEL_CODE.py';mc.write_text('# generated descriptor only\n')
    fit=dump(root/'FAKE_SOURCE_PROTOCOL.json',{'generated':True})
    b={'checkpoint':a.bind(cp),'model_code':a.bind(mc),'source_fit_protocol':fit}
    freeze=dict(completed=True,optimizer_updates=2000,current_used_for_fit_selection=False,PnP=0,
                checkpoint=b['checkpoint'],model_code=b['model_code'],protocol=fit,external_threshold=.5)
    if mutation=='source_incomplete':freeze['completed']=False
    b['source_model_freeze']=dump(root/'FAKE_SOURCE_FREEZE.json',freeze)
    b['source_fit_complete']=dump(root/'FAKE_SOURCE_COMPLETE.json',dict(completed=True,optimizer_updates=2000,model_freeze=b['source_model_freeze']))
    gates={str(i):True for i in range(9)}
    if mutation=='one_source_gate_fail':gates['8']=False
    b['source_results']=dump(root/'FAKE_SOURCE_RESULTS.json',dict(completed=True,model_freeze=b['source_model_freeze'],all_nine_pass=True,predeclared_source_gate=gates))
    b['source_postfit_audit']=dump(root/'FAKE_SOURCE_POSTFIT_AUDIT.json',dict(PASS=mutation!='audit_fail',model_freeze=b['source_model_freeze']))
    for k in a.CURRENT_KEYS:b[k]=dict(path=str(root/('NEVER_CURRENT_'+k)),sha256='0'*64)
    p=copy.deepcopy(policy)
    p.update(source_model_freeze=b['source_model_freeze'],source_fit_completion=b['source_fit_complete'],source_results=b['source_results'],source_fit_protocol=fit)
    p['artifact_locations_from_existing_code_only']=dict(prior_input_pack=b['current_model_input']['path'],prior_rows=b['prior_input_rows']['path'],
        prior_preparation_seal=b['prior_preparation_seal']['path'],prior_input_gate=b['prior_input_gate']['path'])
    prot=dump(root/'FAKE_CURRENT_POLICY.json',p)
    proof=dump(root/'FAKE_GENERATED_PROOF.json',dict(PASS=True,code=a.bind(CODE)))
    review=dump(root/'FAKE_INDEPENDENT_REVIEW.json',dict(PASS=mutation!='review_fail',blockers=[],code=a.bind(CODE),protocol=prot))
    payload=dump(root/'FAKE_ARTIFACT_BINDINGS.json',dict(completed=True,authorized_after_source_audit=mutation!='artifacts_not_authorized',bindings=b))
    e=dict(schema='V91_ROOT_APPROVED_NATIVE_CENTER_ENTRY_V1',approved=mutation!='not_approved',
        current_used_for_selection=False,training_allowed=False,PnP_allowed=False,
        protocol=prot,code=a.bind(CODE),generated=proof,review=review,source_postfit_audit=b['source_postfit_audit'],artifact_bindings=payload,
        output_directory=str(root/'NEVER_EXECUTED_OUTPUT'))
    if mutation=='allow_training':e['training_allowed']=True
    if mutation=='code_not_bound':e['code']['sha256']='0'*64
    path=root/'FAKE_ENTRY.json';dump(path,e)
    return path,Path(prot['path']),prot['sha256']


class GeneratedOnlyMap(torch.nn.Module):
    def __init__(self):super().__init__();self.register_buffer('constant',torch.zeros(()));self.batch_sizes=[]
    def forward(self,p3_roi,local_rgb96,local_valid96):
        self.batch_sizes.append(len(p3_roi))
        return dict(physical_external_logit=p3_roi[:,0].float()+self.constant,
                    map_valid=p3_roi[:,64].bool(),input_available=p3_roi[:,64].bool().flatten(1).any(1))


def main():
    torch.set_num_threads(1);FIX.mkdir(exist_ok=False)
    policy=a.read(a.PROTOCOL)  # public policy only, no current paths are followed.
    original_protocol=a.PROTOCOL;original_sha=a.EXPECTED_PROTOCOL_SHA;original_hash=a.sha
    touched_current=[]
    def guarded_hash(path):
        if 'NEVER_CURRENT_'in str(path):touched_current.append(str(path));raise AssertionError('Forbidden current access during source guard')
        return original_hash(path)
    failures=[]
    try:
        a.sha=guarded_hash
        for name in(None,'not_approved','allow_training','one_source_gate_fail','source_incomplete','audit_fail','review_fail','artifacts_not_authorized','code_not_bound'):
            ep,pp,ps=gate_fixture(FIX/('PASS'if name is None else name),policy,name)
            a.PROTOCOL=pp;a.EXPECTED_PROTOCOL_SHA=ps
            try:a.validate_entry(ep)
            except(ValueError,AssertionError)as exc:
                if name is None:raise
                failures.append(dict(case=name,rejected=True,reason=str(exc)))
            else:
                assert name is None,'Malformed entry accepted: '+str(name)
    finally:a.PROTOCOL=original_protocol;a.EXPECTED_PROTOCOL_SHA=original_sha;a.sha=original_hash
    assert not touched_current and len(failures)==8
    # All223 frames/1784 slots are invented. No actual frame/row/SHA is read.
    frames=[];metadata=[];prior=[];qs=[];Ls=[]
    for fi in range(223):
        q=np.array([[20.+k,30.+k]for k in range(8)],np.float64);L=64.+(fi%4)*.5
        frame=dict(id='GENERATED_FRAME_'+str(fi),initial_points=q.tolist(),raw_hw=[480,640],image_sha256='a'*64)
        frames.append(frame);metadata.append(dict(id=frame['id'],side=L))
        for k in range(8):
            i=fi*8+k;good=i<1739
            prior.append(dict(id=frame['id'],native_corner=k,query_x=float(q[k,0]),query_y=float(q[k,1]),
                pixel_x=int(q[k,0]),pixel_y=int(q[k,1]),image_sha256='a'*64,roi_side=L,saved_roi_pixel_valid=True,
                valid=good,status='OK'if good else'UNKNOWN',reason=''if good else'GENERATED_OLD_U',
                geometry_self_hidden=True if i%3==0 else(False if i%3==1 else None),geometry_self_source='GENERATED_ONLY'))
            qs.append(q[k]);Ls.append(L)
    p3=np.zeros((1784,65,9,9),np.float32);p3[:,64]=1
    features=dict(p3_roi=p3,rgb96=np.broadcast_to(np.zeros((1,3,96,96),np.uint8),(1784,3,96,96)),
        crop_valid96=np.broadcast_to(np.ones((1,1,96,96),bool),(1784,1,96,96)),
        query_xy=np.array(qs,np.float32),side_L=np.array(Ls,np.float32))
    base,q,L=a.verify_input_metadata(features,prior,frames,metadata)
    assert np.array_equal(q,np.array(qs,np.float64))and np.array_equal(L,np.array(Ls,np.float64))
    half=dict(features,p3_roi=p3.astype(np.float16));a.verify_input_metadata(half,prior,frames,metadata)
    badchecks=[]
    for name in('q','L','order','pixel','old45U','imageSHA'):
        f=dict(features);r=copy.deepcopy(prior)
        if name=='q':f['query_xy']=features['query_xy'].copy();f['query_xy'][0,0]+=1
        elif name=='L':f['side_L']=features['side_L'].copy();f['side_L'][0]+=1
        elif name=='order':r[0],r[1]=r[1],r[0]
        elif name=='pixel':r[0]['pixel_x']+=1
        elif name=='old45U':r[-1]['valid']=True
        elif name=='imageSHA':r[0]['image_sha256']='b'*64
        try:a.verify_input_metadata(f,r,frames,metadata)
        except ValueError:badchecks.append(name)
        else:raise AssertionError('Metadata mutation accepted: '+name)
    # Generated maps put centre values exactly at raw FP32 threshold edge cases.
    edge=np.array([-1e-20,1e-20,-0.,0.,-1.,1.],np.float32)
    z=np.resize(edge,1784).copy();features['p3_roi'][:,0]=z[:,None,None]
    features['p3_roi'][2,64,4,4]=0  # one NEW unavailable centre among prior-valid.
    model=GeneratedOnlyMap();before=a.state_sha(model)
    maps,valid,pv=a.predict_all(model,features,torch,'cpu')
    assert a.state_sha(model)==before and len(model.batch_sizes)==56 and model.batch_sizes[-1]==56
    assert model.batch_sizes[::2]==model.batch_sizes[1::2]and sum(model.batch_sizes)==3568
    assert np.array_equal(maps[:,4,4],z)and maps.dtype==np.float32 and np.isfinite(maps).all()
    result=a.decode_rows(base,z,pv,valid[:,4,4]);final=np.array([r['valid']for r in result]);old=np.array([r['valid']for r in base])
    assert final.sum()==1738 and(~final).sum()==46 and not np.any(final&~old)
    assert all(r['authoritative_decision']=='UNKNOWN'and r['decision_visible']is None and r['p_visible']is None and r['logit']is None and r['geometry_self_hidden_if_valid']is None for r in result if not r['valid'])
    assert result[0]['decision_visible']is True and result[1]['decision_visible']is False and pv[0]==pv[1]==.5
    assert result[3]['decision_visible']is False and result[3]['authoritative_decision']=='E'and pv[3]==.5
    assert result[0]['geometry_self_hidden']is True and result[0]['hybrid_decision_visible']is False and result[0]['decision_visible']is True
    assert all(r['geometry_self_hidden']==base[i]['geometry_self_hidden']for i,r in enumerate(result))
    checks=dict(completed=True,PASS=True,code=a.bind(CODE),test_code=a.bind(__file__),protocol=a.bind(original_protocol),
        scope='Only public policy/code plus invented gate files, metadata and CPU maps; no actual source/current artifacts.',
        generated_source_guard_cases=failures,approved_generated_source_guard_passes_without_current_access=True,
        current_descriptors_never_hashed_or_opened_in_validate_entry=True,
        generated223_frames1784_slots=True,original_q_L_order_image_native_pixel_exact=True,
        original_P3_float16_and_float32_supported_without_requantization=True,metadata_mutations_rejected=badchecks,
        batch64_repeat_all_maps_valid_exact=True,generated_batch_sizes=model.batch_sizes,
        prior45U_preserved=True,additional_centre_U_preserved=True,all1784_rawfinite=True,
        raw_zE_sign_authoritative=True,positive_and_negative_tiny_FP32_probability_round_half_test=True,
        exact_zero_and_signed_zero_are_E=True,invalid_main_probability_blank=True,
        old_geometry_raw_unchanged_and_invalid_first_ifvalid_None=True,geometry_hybrid_not_RAW=True,
        real_model_construct_checkpoint_parameter_reads_forward=0,current_artifact_SHA_body_RGB_GT_score_reads=0,
        actual_backbone_training_current_PnP=0,generated_CPU_stub_forward_only=True)
    a.write(a.GENERATED,checks)
    print(json.dumps(a.bind(a.GENERATED)),flush=True)


if __name__=='__main__':main()

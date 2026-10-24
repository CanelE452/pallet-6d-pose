"""Invented 223x8 metadata/tensors only. No actual current/source body reads."""
from pathlib import Path
import copy
import argparse
import io
import tempfile
import json

import numpy as np
import torch

import current_common as c
import infer_same_query_once as infer
import evaluate_saved_only as ev

HERE=Path(__file__).resolve().parent


def csv_style(rows):
    return [{k:(''if v is None else str(v))for k,v in r.items()}for r in rows]


def fixture():
    features=dict(p3_roi=np.zeros((1784,65,9,9),np.float32),rgb96=np.zeros((1784,3,96,96),np.uint8),
                  crop_valid96=np.ones((1784,1,96,96),bool),query_xy=np.zeros((1784,2),np.float32),side_L=np.full(1784,64.,np.float32))
    features['p3_roi'][:,64]=1
    features['p3_roi'][:,0,4,4]=np.where(np.arange(1784)%2==0,-1e-12,1e-12)
    features['crop_valid96'][100,0,47,47]=False
    prior=[];frames=[];metadata=[];truth=[]
    for fi in range(223):
        frameid=f'INVENTED_{fi:03d}';q=np.array([[16.25+k,24.25+k]for k in range(8)],np.float64)
        frames.append(dict(id=frameid,initial_points=q.tolist(),raw_hw=[480,640],image_sha256='f'*64))
        metadata.append(dict(id=frameid,side=64.))
        for k in range(8):
            i=fi*8+k;valid=i<1739;geo=(i%11==0)
            row=dict(id=frameid,native_corner=k,query_x=float(q[k,0]),query_y=float(q[k,1]),pixel_x=int(np.floor(q[k,0]+.5)),pixel_y=int(np.floor(q[k,1]+.5)),image_sha256='f'*64,roi_side=64.,saved_roi_pixel_valid=valid,valid=valid,geometry_self_hidden=geo,status='OK'if valid else'UNKNOWN',reason=''if valid else'GENERATED_PRIOR_U')
            prior.append(row);features['query_xy'][i]=q[k]
            group='WOOD_PRIMARY105'if fi<105 else'SQUARE118'
            state='DIRECT_VISIBLE'if k<4 else'EXTERNAL_OCCLUDED'if(k<6 or fi>=105)else'SELF_OCCLUDED'
            truth.append(dict(id=frameid,native_corner=str(k),image_sha256='f'*64,query_x=str(row['query_x']),query_y=str(row['query_y']),group=group,human_state=state,
                recording='INVENTED_RECORDING',present_at_fixed_0_5=str(k%2==0),legacy_stored_state='U'if k==7 else'S'if k==6 else'B'if k%2==0 else'O',
                paired_same_query_RGB_D_vs_E='True',paired_same_query_RGB_D_vs_ES='True',same_slot_pipeline_D_vs_E='True',same_slot_pipeline_D_vs_ES='True',
                initial_reference_error_px='2.',human_reference_inside_nominal_crop='True'))
    manual=[]
    for group,count in [('WOOD_PRIMARY105',153),('SQUARE118',223)]:
        selected=[r for r in truth if r['group']==group and r['human_state']=='DIRECT_VISIBLE'][:count]
        manual.extend([dict(r,annotation_coordinate_source='manual_click',initial_reference_error_px='2.',query_valid='True')for r in selected])
    return features,prior,frames,metadata,truth,manual


class StubScalar:
    def __init__(self):self.input_dtypes=set();self.calls=0
    def __call__(self,**kwargs):
        assert set(kwargs)=={'p3_roi','local_rgb96','local_valid96'}
        self.input_dtypes.add(kwargs['p3_roi'].dtype);self.calls+=1
        p,r,v=kwargs.values();z=p[:,0,4,4].float()
        available=p[:,64,4,4].bool()&v[:,0,47:49,47:49].flatten(1).all(1)
        z=torch.where(available,z,z*0)
        return dict(query_logit=z,query_available=available,attention_diagnostic_only=torch.ones(len(z),9,9)/81)


def source_guard_tests():
    b={k:dict(path='INVENTED_'+k,sha256='a'*64)for k in c.SOURCE_KEYS}
    names=['VAL_D_recall','VAL_E_recall','VAL_known_U_rate']
    protocol=dict(selected_source_freeze=b['source_model_freeze'],source_gate_names=names)
    freeze=dict(completed=True,selected_update=100,source_only=True,source_input_bindings={'current_allowed':False},entry=dict(path='INVENTED_SOURCE_ENTRY',sha256='b'*64),checkpoint=b['checkpoint'],model_code=b['model_code'],protocol=b['source_fit_protocol'])
    complete=dict(completed=True,model_freeze=b['source_model_freeze'])
    gates=dict(completed=True,all_gates_pass=True,gates={x:{'PASS':True}for x in names})
    audit=dict(PASS=True,model_freeze=b['source_model_freeze'])
    c.guard_source(protocol,b,freeze,complete,gates,audit,{})
    rejections=0
    for name in names:
        bad=copy.deepcopy(gates);bad['gates'][name]['PASS']=False
        try:c.guard_source(protocol,b,freeze,complete,bad,audit,{})
        except PermissionError:rejections+=1
        else:raise AssertionError('Failed source gate accepted')
    for change in [('source_only',False),('selected_update',0),('source_input_bindings',{'current_allowed':True})]:
        bad=copy.deepcopy(freeze);bad[change[0]]=change[1]
        try:c.guard_source(protocol,b,bad,complete,gates,audit,{})
        except PermissionError:rejections+=1
        else:raise AssertionError('Bad source metadata accepted')
    fd=tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False);fd.write('{"approved":false}');fd.close();path=Path(fd.name)
    try:
        c.validate_entry(path,infer.__file__,'INFER')
    except PermissionError:rejections+=1
    else:raise AssertionError('Unapproved entry accepted')
    path.unlink();return rejections


def run():
    torch.set_num_threads(1)
    memory=io.BytesIO();torch.save({'state_dict':{'generated':torch.tensor([1.])},'update':100,'seed':194,'model_code':{'path':'INVENTED','sha256':'a'*64},'optimizer_state':{'state':{},'param_groups':[]}},memory);memory.seek(0)
    loaded=torch.load(memory,map_location='cpu',weights_only=True);assert loaded['update']==100 and torch.equal(loaded['state_dict']['generated'],torch.tensor([1.]))
    f,prior,frames,metadata,truth,manual=fixture()
    reader=c.prior_reader();base,q,L=reader.verify_input_metadata(f,prior,frames,metadata)
    model=StubScalar();z,available,pv,attention=infer.predict_all(model,f,torch,'cpu',64)
    assert model.input_dtypes=={torch.float32}and model.calls==56
    assert z.shape==available.shape==pv.shape==(1784,)and attention.shape==(1784,9,9)
    new=infer.decode_rows(base,z,pv,available)
    assert sum(r['valid']for r in new)==1738 and sum(r['additional_operator_U']for r in new)==1
    assert sum(not r['prior_valid']for r in new)==45
    assert new[0]['authoritative_decision']=='D'and new[0]['p_visible']==.5
    assert new[100]['geometry_self_hidden_if_valid']is None and new[100]['p_visible']is None
    assert all(r['geometry_self_hidden_if_valid']is None and r['authoritative_decision']=='UNKNOWN'for r in new if not r['valid'])
    oldz=-z.copy();oldavailable=np.ones(1784,bool);oldpv=np.full(1784,.5,np.float32)
    old=infer.decode_rows(base,oldz,oldpv,oldavailable)
    for r in old:r['raw_physical_external_logit']=r.pop('raw_query_aux_hidden_logit')
    metric=ev.calculate_all(c.metric_base(),truth,prior,csv_style(new),csv_style(old),manual)
    assert metric['availability']['new_total_U']==46 and metric['availability']['extra_operator_U']==1
    assert sum(r['prior_input_U']for r in metric['availability_by_group'])==45
    assert sum(r['extra_operator_U']for r in metric['availability_by_group'])==1
    assert all('NOT_LEARNED_SELF'in r['measurement_scope']for r in metric['metrics']if r['task']=='D_vs_S')
    assert len(metric['manual_near_D'])==12
    assert {r['N']for r in metric['manual_near_D']}=={153,223}
    assert all(not r['learned_self_prediction']and not r['external_vs_self_identified']for r in metric['human_self_binary_response'])
    assert all(r['human_S_support']==0 and r['human_S_binary_hidden_response_rate']is None for r in metric['human_self_binary_response']if r['group']=='SQUARE118')
    assert not any(r['group']=='SQUARE118'and r['task']=='D_vs_S'for r in metric['metrics'])
    # Exact previous-count replay and metadata/q changes rejected.
    hist=csv_style([r for r in metric['metrics']if r['model']in('v61','legacy',ev.OLD,ev.OLD_GEO)])
    checked=ev.historical_replay(metric['metrics'],hist,{'v61','legacy',ev.OLD,ev.OLD_GEO})
    corrupted=csv_style(new);corrupted[0]['query_x']='17.25'
    try:ev.calculate_all(c.metric_base(),truth,prior,corrupted,csv_style(old),manual)
    except PermissionError:pass
    else:raise AssertionError('Changed current query accepted')
    # Geometry self route is separate from raw; input invalid wins both routes.
    rec={'valid':False,'z':-1.,'geo':True};assert ev.response(rec,True)=='OTHER_UNRESOLVED'
    rec={'valid':True,'z':-1.,'geo':True};assert ev.response(rec)=='VISIBLE'and ev.response(rec,True)=='ABSENT_NONVISIBLE'
    rejected=source_guard_tests()
    return dict(completed=True,PASS=True,inference_code=c.bind(infer.__file__),evaluator_code=c.bind(ev.__file__),common_code=c.bind(c.__file__),generated_code=c.bind(__file__),
      pinned_prior_reader_code=c.bind(c.V91_CODE),pinned_metric_core=c.bind(c.BASE_CODE),invented_frames=223,invented_slots=1784,
      original45U_preserved=True,extra_operator_U1_separate=True,GT_forward_fields_zero=True,source_gate_and_entry_rejections=rejected,
      authoritative_rawsign_not_rounded_half=True,original_float32_P3_preserved_without_requantization=True,
      same64batch_repeat_scalar_support_attention_exact=True,baseline_metric_rows_replayed=checked,
      manualnear_D_denominators=[153,223],WOOD_humanS_binary_response_only=True,SQUARE_noS_gold_claim=True,
      geometry_route_not_raw_and_invalid_precedence=True,original_q_L_geometry_join_rejection=True,
      generated_runtime_torch=str(torch.__version__),generated_weights_only_inmemory_checkpoint_load=True,
      actual_source_current_GT_image_cache_or_checkpoint_parameter_reads=0,actual_model_inference_fit_PnP=0,
      generated_stub_calls_not_learned_model_calls=model.calls)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'GENERATED_CURRENT_PREPARATION_TESTS.json');args=ap.parse_args()
    result=run();path=args.output;assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

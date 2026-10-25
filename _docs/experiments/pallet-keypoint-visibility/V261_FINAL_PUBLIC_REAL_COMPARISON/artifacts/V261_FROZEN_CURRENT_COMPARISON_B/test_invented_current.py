"""Invented current-shaped arrays only; no case, baseline, checkpoint or RGB I/O."""
import json,sys,types
from pathlib import Path
import numpy as np
import torch
H=Path(__file__).absolute().parent

def load(p,name):
    m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m;exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m

def run():
    driver=load(H/'run_frozen_current.py','v261_current_invented_driver');core=load(H/'current_core.py','v261_current_invented_core')
    closure=driver.document(driver.DEPENDENCY,[driver.DEPENDENCY]);cb=closure['code'];white=list(cb.values())
    raw=driver.pure(cb['spatial_crop'],white,['crop']);prep=driver.pure(cb['preprocess'],white,['preprocess_views'],{'MEAN':(.485,.456,.406),'STD':(.229,.224,.225)});canonical=driver.pure(cb['canonical'],white,['node_pixel_batch']);crop=driver.module(cb['crop_module'],white,'crop_modules')
    R=H.parent;P=R.parent
    condition={'path':str(R/'V227_FROZEN_FEATURE_CODE_C/extract_context_features.py'),'sha256':'0279eca179150242b0542d371201cdeb6c32e06fa38df07d4333ae532ae0ed9f','size_bytes':23251}
    cond=driver.pure(condition,[condition],['condition32'])
    class Encoder:
        def __init__(self):self.calls=0
        def forward_features(self,x):
            self.calls+=1;cls=torch.ones(len(x),384);tokens=torch.ones(len(x),256,384)*2
            return{'x_norm_clstoken':cls,'x_norm_patchtokens':tokens}
    class Head:
        def __init__(self,a,b):self.a=a;self.b=b
        def __call__(self,x):return torch.tensor([[self.a,self.b]]*len(x),dtype=torch.float32)
    q8=[[5+k/10,5]for k in range(8)];av=[True]*8;av[2]=False
    queries=[{'id':'INVENTED','group_id':'CURRENT_FRAME_0000','original_global_index':k,'row_id':f'INVENTED:native{k}','native_slot':k,'query_xy':q8[k],'L':4.,'original_query_available':av[k]}for k in range(8)]
    frame={'raw_hw':[13,13],'frame_index':0,'queries':queries,'query_xy':q8,'original_query_available':av}
    rgb=np.arange(13*13*3,dtype=np.uint16).reshape(13,13,3).astype(np.uint8);encoder=Encoder()
    op=types.SimpleNamespace(crop=raw.crop,node_pixel_batch=canonical.node_pixel_batch,preprocess_views=prep.preprocess_views,crop_module=crop,condition32=cond.condition32)
    x,support,reasons,facts=core.canonical_frame_features(frame,rgb,encoder,op,'cpu',True)
    assert x.shape==(8,1186)and support.sum()==7 and not x[2].any()and reasons[2]=='ORIGINAL_INPUT_UNAVAILABLE'and encoder.calls==2
    assert facts['first_batch_repeat_byteexact']and facts['global_bbox_or_context_encoder_used']is False
    for k in np.flatnonzero(support):assert x[k,1154+k]==1 and x[k,1154:1162].sum()==1 and x[k,1162+k]==0 and x[k,1162+2]==0
    rows=core.prediction_rows(frame,x,support,reasons,{'C':Head(-1,100),'T':Head(1,-100)})
    assert rows['C'][0]['prediction']=='D'and rows['T'][0]['prediction']=='E'and rows['C'][2]['prediction']=='U'
    checks=['Canonical1154+literalnative32, original U retained; firstbatchrepeat; no bbox/context gate','Both heads MAIN0 signs, AUX1 opposite cannot control decision']
    metric_b={'path':str(P/'query_visibility_cue_occlusion_20261023_v72/contract/saved_evaluate_current_scores.py'),'sha256':'d079650b65aef777675af552da61335a28b73f978580413d74d143433019bef6','size_bytes':16095}
    strata_b={'path':str(P/'query_visibility_direct_owner_frozen_current_benchmark_20261024_v178/evaluation_preparation/evaluate_saved_five_models.py'),'sha256':'5ff180a8e0924d43a0e10c5c5c0c25b65408e2903f7005ab78ff571cf7b33c07','size_bytes':29720}
    metric=driver.pure(metric_b,[metric_b],['ratio','calculate'],{'D':core.D});error=driver.pure(strata_b,[strata_b],['error_bucket']);metric.error_bucket=error.error_bucket
    ledger=[];pred={'C':[],'T':[]};flags=[]
    for i in range(1784):
        fi,k=divmod(i,8);identity=f'INVENTED{fi:04d}';human=core.D if i<1147 else core.E if i<1237 else core.D if i<1248 else core.S if i<1384 else'UNKNOWN';available=i<1739
        state_c='D'if available else'U';state_t=('E'if 1147<=i<1237 or i==0 else'D')if available else'U'
        ledger.append({'global_index':str(i),'frame_index':str(fi),'id':identity,'row_id':identity+':native'+str(k),'native_corner':str(k),'query_x':'5.25','query_y':'5.5','roi_side':'96','original_query_valid':str(available),'human_state':human,'paired_same_query_RGB_D_vs_E':str(i<1237),'same_slot_pipeline_D_vs_E':str(i<1248),'group':core.GROUPS[0]if fi<105 else core.GROUPS[1],'initial_reference_error_px':'2'if i%2 else'4','v61_D_E_U':state_c,'legacy_D_E_U':state_c,'v61_original_response':'VISIBLE'if available else'OTHER_UNRESOLVED','legacy_original_response':'VISIBLE'if available else'OTHER_UNRESOLVED'})
        flags.append(i<1237 or 1248<=i<1269)
        for model,state in [('C',state_c),('T',state_t)]:
            pred[model].append({'id':identity,'row_id':identity+':native'+str(k),'original_global_index':i,'frame_index':fi,'native_slot':k,'query_xy':[5.25,5.5],'L':96,'original_query_available':available,'prediction':state,'logit':None if state=='U'else 1. if state=='E'else -1.,'U_reason':'ORIGINAL_INPUT_UNAVAILABLE'if state=='U'else None})
    result=core.evaluate_saved_predictions(ledger,pred,metric,flags)
    assert result['primary_N']==1237 and result['paired_SELF_flag_present']and len(result['ledger'])==1784
    transition=next(r for r in result['tables']['PAIRED_RECOVERY_AND_DAMAGE']if r['group']=='ALL'and r['baseline']=='C'and r['new']=='T')
    assert transition['recovered']==90 and transition['damaged']==1 and transition['net']==89
    self_paired=next(r for r in result['tables']['HUMAN_SELF_BINARY_RESPONSE']if r['group']=='ALL'and r['model']=='T'and r['cohort']=='PAIRED_SAME_QUERY')
    assert self_paired['N']==21
    checks.append('Invented1784 strict samequery join, fixed1237/1147/90 and SELF136/21; paired recovered90/damaged1')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='v261_current_invented_')as tmp:
        p=Path(tmp)/'false.json';p.write_text(json.dumps({'approved_actual_final_current_comparison':False}))
        try:driver.run(p)
        except ValueError:pass
        else:raise AssertionError('False entry accepted')
        target=Path(tmp)/'regular.txt';target.write_text('invented');alias=Path(tmp)/'alias.txt';alias.symlink_to(target)
        try:driver.regular(alias)
        except ValueError:pass
        else:raise AssertionError('Generic alias follow allowed')
    checks.append('Falseentry beforedata; generic symlink payload rejected; alias transport must be explicit Root canonical mapping')
    return{'completed':True,'PASS':True,'invented_checks':checks,'actual_current_or_baseline_or_GT_or_RGB_or_CP_reads':0,'actual_encoder_or_checkpoint_model_calls':0,'mock_encoder_calls':encoder.calls,'optimizer_updates':0}

if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,sort_keys=True,indent=2))

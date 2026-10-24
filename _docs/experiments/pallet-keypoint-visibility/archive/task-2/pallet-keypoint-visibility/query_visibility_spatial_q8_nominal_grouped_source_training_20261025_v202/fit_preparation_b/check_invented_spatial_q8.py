"""Authored CPU tensors only. No production arrays, checkpoints or scores."""
from pathlib import Path
import ast,hashlib,importlib.util,os,stat,sys
import numpy as np
import torch
H=Path(__file__).absolute().parent;P=H.parent.parent
V199=P/'query_visibility_native_conditioned_grouped_source_training_20261024_v199'

def safe(path):
    p=Path(path);assert p.is_absolute() and '..'not in p.parts and all('v112'not in x.casefold()for x in p.parts)
    current=Path(p.anchor)
    for part in p.parts[1:]:
        current/=part;assert not stat.S_ISLNK(os.lstat(current).st_mode)
    return p

def module(path,name):
    p=safe(path);s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def byteequal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.detach().contiguous().numpy().tobytes()==b.detach().contiguous().numpy().tobytes()
def rejects(call):
    try:call()
    except (AssertionError,ValueError):return True
    return False

def context(native):
    native=np.asarray(native,np.int64);n=len(native);mask=np.ones((n,8),np.float32);mask[np.arange(n),native]=0
    xy=np.broadcast_to(np.arange(n,dtype=np.float32)[:,None,None]/32,(n,8,2)).copy();xy[mask==0]=0
    return dict(native_slot=native,other_mask8=mask,other_dxdy16=xy.reshape(n,16),fit_nominal_eligible=np.ones(n,bool))

def run_checks():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    t=module(H/'train_spatial_q8_nominal_source.py','spatial_trainer_cpu_fixture')
    base=module(t.c.HEAD,'spatial_base_cpu_fixture');candidate=module(t.c.CANDIDATE,'spatial_candidate_cpu_fixture');h=module(H/'spatial_q8_conditioned_head.py','spatial_head_cpu_fixture');adapter=module(t.MASK,'spatial_masks_cpu_fixture');pure=t.pure_training(base)
    cls=h.make_spatial_q8_head_class(base,candidate);original=candidate.make_candidate_head_class(base)
    torch.manual_seed(1810);old=original().float()
    torch.manual_seed(1810);control=cls(spatial_q8=False).float()
    torch.manual_seed(1810);actual=cls(spatial_q8=True).float();checks={}
    checks['matched118359_original115222_plus3137']=sum(v.numel()for v in old.parameters())==115222 and sum(v.numel()for v in control.parameters())==sum(v.numel()for v in actual.parameters())==118359
    checks['all_original_parameter_bytes_and_both_fresh_states_equal']=all(byteequal(v,control.state_dict()[k])for k,v in old.state_dict().items()) and set(control.state_dict())==set(actual.state_dict()) and all(byteequal(v,actual.state_dict()[k])for k,v in control.state_dict().items())
    checks['residual_final_weight_bias_exactzero']=bool((actual.spatial_residual[-1].weight==0).all()and(actual.spatial_residual[-1].bias==0).all())
    rgb=torch.arange(4*96*96*3,dtype=torch.int64).remainder(256).to(torch.uint8).reshape(4,96,96,3);valid=torch.ones((4,96,96),dtype=torch.bool);x=base.assemble_local_input(rgb,valid,use_sam=False)
    ctx=context([0,1,6,7]);native,mask,xy=t.context_tensors(ctx,np.arange(4),'cpu')
    with torch.no_grad():a=old(x);b=control(x,native,mask,xy);d=actual(x,native,mask,xy)
    checks['zero_residual_all_three_logits_original_bytes_equal']=all(byteequal(a[k],b[k])and byteequal(a[k],d[k])for k in a)
    checks['encoder12x12_pool3x3_and_RAW_SAMzero']=tuple(control.encoder(x).shape)==(4,64,12,12)and tuple(control.pool(control.encoder(x)).shape)==(4,64,3,3)and bool((x[:,6:]==0).all())
    checks['native_dtype_shape_range_invalid_rejected']=all(rejects(lambda n=n:actual(x,n,mask,xy))for n in [native.to(torch.int8),native.float(),native.bool(),native[:,None],torch.tensor([-1,1,6,7]),torch.tensor([0,1,6,8])])
    badmask=mask.clone();badmask[0,0]=1
    badxy=xy.clone();badxy[0,0]=1
    checks['self_unavailable_nonbinary_nonfinite_context_rejected']=rejects(lambda:actual(x,native,badmask,xy))and rejects(lambda:actual(x,native,mask,badxy))and rejects(lambda:actual(x,native,mask*.5,xy))and rejects(lambda:actual(x,native,mask,xy+float('nan')))
    checks['arm_flag_bool_and_xy_shape_dtype_bound_rejected']=rejects(lambda:cls(spatial_q8=1))and rejects(lambda:actual(x,native,mask,xy.double()))and rejects(lambda:actual(x,native,mask,xy[:,:15]))and rejects(lambda:actual(x,native,mask,xy+9))
    # Activate authored branch to establish which condition columns can affect direct.
    with torch.no_grad():
        for model in (control,actual):
            for v in model.spatial_residual.parameters():v.zero_()
            model.spatial_residual[0].weight[0,0]=1;model.spatial_residual[0].weight[0,64]=1;model.spatial_residual[0].weight[0,73]=1;model.spatial_residual[0].weight[0,82]=1;model.spatial_residual[-1].weight[0,0]=1
    features=torch.zeros((4,64));features[:,0]=torch.arange(4,dtype=torch.float32)/4
    xy2=xy.clone();xy2[0,2]=1
    checks['control_zeros_only16XY_actual_can_interact']=byteequal(control.residual_from_features(features,native,mask,xy),control.residual_from_features(features,native,mask,xy2))and not byteequal(actual.residual_from_features(features,native,mask,xy),actual.residual_from_features(features,native,mask,xy2))
    # Both control and actual keep original category and same other-mask active.
    mask2=mask.clone();mask2[0,1]=0;xy3=xy.clone();xy3[0,2:4]=0
    checks['both_arms_keep_same_mask_and_native_conditions']=all(not byteequal(m.residual_from_features(features,native,mask,xy),m.residual_from_features(features,native,mask2,xy3))for m in [control,actual])
    with torch.no_grad():u=actual(x,native,mask,xy);v=actual(x,native,mask,xy2)
    checks['context_no_direct_injection_support_owner_shared_training_caveat']=all(byteequal(u[k],v[k])for k in ['support_logits','owner_logits'])
    for model in (control,actual):
        model.zero_grad(set_to_none=True);model.residual_from_features(features,native,mask,xy2).sum().backward()
    checks['control_XY_gradient_zero_actual_XY_nonzero']=bool((control.spatial_residual[0].weight.grad[:,80:]==0).all()and(actual.spatial_residual[0].weight.grad[:,80:]!=0).any())
    # Mixed original inputU with JSON None and stored NaN stays unknown and allzero context.
    n=4;xx=dict(row_id=np.array(['AUTHORED%d'%i for i in range(n)]),original_global_index=np.arange(n,dtype=np.int64),native_slot=native.numpy().astype(np.int8),query_xy=np.array([[1.,2.],[3.,4.],[5.,6.],[np.nan,np.nan]],np.float64),L=np.array([10.,20.,30.,np.nan],np.float64),original_query_available=np.array([True,True,True,False]))
    view=context(xx['native_slot']);view['other_mask8'][-1]=0;view['other_dxdy16'][-1]=0;view.update(other_mask8_bool=view['other_mask8'].astype(bool),native_onehot8=np.eye(8,dtype=np.float32)[view['native_slot']],row_id=xx['row_id'].copy(),context_origin_group_id=['AUTHORED_GROUP']*n,meta=[dict(row_id=str(xx['row_id'][i]),original_global_index=i,native_slot=int(xx['native_slot'][i]),query_xy=xx['query_xy'][i].tolist()if i<3 else None,L=float(xx['L'][i])if i<3 else None,original_query_available=bool(xx['original_query_available'][i]))for i in range(n)])
    checks['context_ordered_META_F64_U_NaN_validation']=t.validate_context_view(dict(x=xx),view)is view
    altered=dict(view,row_id=view['row_id'][::-1].copy());checks['context_same_count_wrong_order_rejected']=rejects(lambda:t.validate_context_view(dict(x=xx),altered))
    family_view=dict(view,meta=[dict(r,family='SELF_SCENE15',state_name='NOMINAL',state_index=0)for r in view['meta']]);assert t.validate_context_view(dict(x=xx),family_view)is family_view
    wrong_eligible=dict(family_view,fit_nominal_eligible=np.zeros(n,bool));checks['actual_family_state_guard_rejects_eligibility_drift']=rejects(lambda:t.validate_context_view(dict(x=xx),wrong_eligible))
    # Tiny complete pools retain original TRAIN versus HOLDOUT role separately.
    oldc={};masks={};contexts={};offsets={'SYNTH_AUG65536':0,'PUBLIC_TRAIN1480':32,'CF3816':64}
    for name,offset in offsets.items():
        rr=np.arange(16,dtype=np.int8);oldc[name]={'x':dict(rgb96=np.broadcast_to((rr.astype(np.uint8)+offset)[:,None,None,None],(16,96,96,3)).copy(),valid96=np.ones((16,96,96),bool),original_query_available=np.ones(16,bool),native_slot=rr%8)}
        masks[name]=np.arange(16)%2==0;contexts[name]=context(rr%8)
    direct={'x':dict(rgb96=np.broadcast_to((np.arange(16,dtype=np.uint8)+96)[:,None,None,None],(16,96,96,3)).copy(),valid96=np.ones((16,96,96),bool),native_slot=np.arange(16,dtype=np.int8)%8)}
    masks['DIRECT_CONTROL']=np.arange(16)%2==0;contexts['DIRECT_CONTROL']=context(direct['x']['native_slot']);contexts['SYNTH_AUG65536']['fit_nominal_eligible'][[0,4]]=False
    original_masks={k:v.copy()for k,v in masks.items()};fitm=t.nominal_fit_masks(masks,contexts)
    checks['TRAIN_excluded_jitter_not_promoted_HOLDOUT']=all(np.array_equal(masks[k],original_masks[k])for k in masks)and bool(masks['SYNTH_AUG65536'][0])and not fitm['SYNTH_AUG65536'][0]and not (~masks['SYNTH_AUG65536'])[0]
    badcontexts=dict(contexts);badcontexts['PUBLIC_TRAIN1480']=dict(contexts['PUBLIC_TRAIN1480'],fit_nominal_eligible=np.zeros(16,bool));checks['only_SYNTH_nominal_eligibility_can_filter']=rejects(lambda:t.nominal_fit_masks(masks,badcontexts))
    full=dict(support=[[np.arange(0,8),np.arange(8,16)]]*2,conditional=[np.ones(16)/16]*3,public=[np.arange(0,5),np.arange(5,10),np.arange(10,16)],CF=[np.arange(0,8),np.arange(8,16)])
    masked=adapter.masked_old_pools(oldc,fitm,lambda _:full);assert masked['feasible'];target=np.tile(np.array([0,1],np.int8),8);target[2]=1;dp=adapter.masked_direct_pools(np.ones(16,bool),target,fitm['DIRECT_CONTROL']);assert dp['feasible']
    draws=[]
    for arm in range(2):
        rng=np.random.default_rng(1810);drng=np.random.default_rng(1811);records=[]
        for step in range(3):
            draw=pure['draw_step'](rng,masked['pools']);di=np.concatenate([drng.choice(a,16,replace=True)for a in dp['pools']]);adapter.assert_draws_train(draw,di,fitm)
            gathered=t.gather_context(contexts,draw,di);gn=t.gather_native(oldc,direct,draw,di);g=t.gather(oldc,direct,draw,di,False,pure)
            assert np.array_equal(gathered['native_slot'],gn)and np.array_equal(gn,g['rgb96'][:,0,0,0].astype(np.int64)%8)and gathered['other_mask8'].shape==(224,8)and gathered['other_dxdy16'].shape==(224,16)
            assert np.count_nonzero(target[di]==0)==16 and np.count_nonzero(target[di]==1)==16
            records.append({**{k:v.tolist()for k,v in draw.items()},'direct':di.tolist()})
        draws.append(records)
    checks['all6losses_fit_masks_same224context_pixel_native_gather_and_balanced_paired_RNG']=draws[0]==draws[1]
    a=dict(support_logits=torch.linspace(-2,2,448).reshape(224,2).requires_grad_(),owner_logits=torch.linspace(1,-1,672).reshape(224,3).requires_grad_(),direct_end_external_logit=torch.linspace(-1,1,224).requires_grad_());target=torch.tensor([0]*16+[1]*16,dtype=torch.int8)
    loss,base_loss,components,direct_loss=t.objective(a,target,pure,candidate);loss.backward()
    checks['unchanged_five_equalmeans_plus_balanced_direct_weight1_gradient']=bool(torch.equal(loss,torch.stack(components).mean()+direct_loss))and all(v.grad is not None and torch.isfinite(v.grad).all()for v in a.values())
    sp,op,dd,proxy=t.decode(np.zeros((3,2),np.float32),np.zeros((3,3),np.float32),np.array([0.,-1e-8,10.],np.float32),np.array([True,True,False]))
    checks['direct_raw_ge0_E_negative_D_inputU_preserved']=dd.tolist()==[1,0,-1]and sp[2].tolist()==[-1,-1]and op[2]==proxy[2]==-1
    def funcs(path):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(safe(path).read_text()).body if isinstance(n,ast.FunctionDef)}
    before=funcs(V199/'fit_preparation_a/train_native_conditioned_grouped_source.py');after=funcs(H/'train_spatial_q8_nominal_source.py')
    unchanged=('pure_training','metadata_matches','load_old','load_direct','gather','native_indices','gather_native','assemble','decode','objective','join_direct_split','split_masks','cohort_counts','fresh_run')
    checks['14_original_recipe_join_mask_decoder_functions_AST_exact']=all(before[k]==after[k]for k in unchanged)
    tree=ast.parse(safe(H/'train_spatial_q8_nominal_source.py').read_text());fit=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='fit');text=ast.unparse(fit)
    checks['4000_final_both_freeze_before_any_source_predictions_sourceonly']=('range(1, 4001)'in text and text.index("barrier =")<text.index('full = save_outputs') and 'all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs=True'in text and 'protected_real_COMMON_VAL_access=0'in text)
    read=t.c.read;bound=t.c.read_bound
    try:
        t.c.read=lambda _:dict(approved_spatial_q8_nominal_source_fit=False,source_only=True,automatic_retry=False);t.c.read_bound=lambda _:(_ for _ in ()).throw(RuntimeError('Bound body gate crossed'))
        checks['false_actual_entry_rejected_before_bound_body']=rejects(lambda:t.fit(Path('/authored_false_entry.json')))
    finally:t.c.read=read;t.c.read_bound=bound
    assert all(checks.values()),checks
    return {k:bool(v)for k,v in checks.items()}

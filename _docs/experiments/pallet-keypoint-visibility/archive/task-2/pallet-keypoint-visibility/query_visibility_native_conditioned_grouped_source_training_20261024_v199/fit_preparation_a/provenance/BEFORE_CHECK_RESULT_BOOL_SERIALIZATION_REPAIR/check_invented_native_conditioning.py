"""CPU authored fixtures only. Never open cache/target/log/checkpoint/score assets."""
from pathlib import Path
import ast, hashlib, importlib.util, os, stat, sys
import numpy as np
import torch

H=Path(__file__).absolute().parent;P=H.parent.parent
V194=P/'query_visibility_source_group_split_direct_end_training_20261024_v194'

def safe(path):
    p=Path(path)
    assert p.is_absolute() and '..' not in p.parts and all('v112' not in x.casefold() for x in p.parts)
    cur=Path(p.anchor)
    for x in p.parts[1:]:
        cur/=x;assert not stat.S_ISLNK(os.lstat(cur).st_mode)
    return p

def module(path,name):
    p=safe(path);spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def byteequal(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.detach().contiguous().numpy().tobytes()==b.detach().contiguous().numpy().tobytes()

def rejects(call):
    try:call()
    except (ValueError,AssertionError):return True
    return False

def run_checks():
    torch.set_num_threads(1)
    t=module(H/'train_native_conditioned_grouped_source.py','native_trial_trainer_fixture')
    base=module(t.c.HEAD,'native_trial_original_head_fixture');candidate=module(t.c.CANDIDATE,'native_trial_candidate_fixture')
    h=module(H/'native_conditioned_head.py','native_trial_residual_fixture')
    adapter=module(t.MASK,'native_trial_mask_fixture');pure=t.pure_training(base)
    cls=h.make_native_conditioned_head_class(base,candidate);original=candidate.make_candidate_head_class(base)
    torch.manual_seed(1810);old=original().float()
    torch.manual_seed(1810);zero=cls(native_condition=False).float()
    torch.manual_seed(1810);one=cls(native_condition=True).float()
    checks={}
    checks['matched116407_parameters']=sum(p.numel()for p in zero.parameters())==sum(p.numel()for p in one.parameters())==116407
    checks['original115222_parameter_initial_bytes_unchanged']=sum(p.numel()for p in old.parameters())==115222 and all(byteequal(v,zero.state_dict()[k])for k,v in old.state_dict().items())
    checks['both_arms_initial_state_every_tensor_byteequal']=set(zero.state_dict())==set(one.state_dict()) and all(byteequal(v,one.state_dict()[k])for k,v in zero.state_dict().items())
    checks['residual_final_exactzero']=bool((zero.native_residual[2].weight==0).all()and(zero.native_residual[2].bias==0).all())
    rgb=torch.arange(4*96*96*3,dtype=torch.int64).remainder(256).to(torch.uint8).reshape(4,96,96,3);valid=torch.ones((4,96,96),dtype=torch.bool)
    x=base.assemble_local_input(rgb,valid,use_sam=False);native=torch.tensor([0,1,6,7],dtype=torch.int64)
    with torch.no_grad():a=old(x);b=zero(x,native);d=one(x,native)
    checks['initial_all3_logits_original_and_both_arms_byteequal']=all(byteequal(a[k],b[k])and byteequal(a[k],d[k])for k in a)
    checks['native_invalid_dtype_shape_range_rejected']=all(rejects(lambda n=n:one(x,n))for n in [native.to(torch.int8),native.float(),native.bool(),native[:,None],torch.tensor([-1,1,2,3]),torch.tensor([0,1,2,8])])
    checks['condition_flag_bool_only']=rejects(lambda:cls(native_condition=1))
    # Authored activated weights demonstrate image x native interaction, not a bias-only branch.
    with torch.no_grad():
        for model in (zero,one):
            for p in model.native_residual.parameters():p.zero_()
            model.native_residual[0].weight[0,0]=1.;model.native_residual[0].weight[0,64]=1.;model.native_residual[2].weight[0,0]=1.
    z=torch.zeros((2,64),dtype=torch.float32);z[1,0]=1.
    n0=torch.zeros(2,dtype=torch.int64);n1=torch.ones(2,dtype=torch.int64)
    delta=one.residual_from_features(z,n0)-one.residual_from_features(z,n1)
    checks['activated_path_has_nonadditive_image_native_interaction']=bool(delta[0]!=delta[1])
    checks['zero_control_ignores_native_values']=byteequal(zero.residual_from_features(z,n0),zero.residual_from_features(z,n1))
    with torch.no_grad():u=one(x,native);v=one(x,native.flip(0))
    checks['native_branch_never_changes_support_or_owner_at_same_state']=all(byteequal(u[k],v[k])for k in ('support_logits','owner_logits'))
    for model in (zero,one):
        model.zero_grad(set_to_none=True);model.residual_from_features(z,n0).sum().backward()
    checks['zero_condition_columns_gradient_zero_onehot_columns_trainable']=bool((zero.native_residual[0].weight.grad[:,64:]==0).all()and(one.native_residual[0].weight.grad[:,64:]!=0).any())
    # Tiny source-like containers: actual same draw helper and original TRAIN masks.
    oldc={};masks={};offsets={'SYNTH_AUG65536':0,'PUBLIC_TRAIN1480':32,'CF3816':64}
    for name,offset in offsets.items():
        rr=np.arange(16,dtype=np.int8)
        oldc[name]={'x':dict(rgb96=np.broadcast_to((rr.astype(np.uint8)+offset)[:,None,None,None],(16,96,96,3)).copy(),valid96=np.ones((16,96,96),bool),original_query_available=np.ones(16,bool),native_slot=rr%8)}
        masks[name]=np.arange(16)%2==0
    direct={'x':dict(rgb96=np.broadcast_to((np.arange(16,dtype=np.uint8)+96)[:,None,None,None],(16,96,96,3)).copy(),valid96=np.ones((16,96,96),bool),native_slot=np.arange(16,dtype=np.int8)%8)}
    masks['DIRECT_CONTROL']=np.arange(16)%2==0
    full=dict(support=[[np.arange(0,8),np.arange(8,16)]]*2,conditional=[np.ones(16)/16]*3,public=[np.arange(0,5),np.arange(5,10),np.arange(10,16)],CF=[np.arange(0,8),np.arange(8,16)])
    masked=adapter.masked_old_pools(oldc,masks,lambda _:full);assert masked['feasible']
    target=np.tile(np.array([0,1],np.int8),8);target[2]=1
    dp=adapter.masked_direct_pools(np.ones(16,bool),target,masks['DIRECT_CONTROL']);assert dp['feasible']
    draws=[]
    for arm in range(2):
        rng=np.random.default_rng(1810);drng=np.random.default_rng(1811);records=[]
        for step in range(3):
            draw=pure['draw_step'](rng,masked['pools']);di=np.concatenate([drng.choice(a,16,replace=True)for a in dp['pools']]);adapter.assert_draws_train(draw,di,masks)
            gn=t.gather_native(oldc,direct,draw,di);g=t.gather(oldc,direct,draw,di,False,pure)
            expected=np.concatenate([oldc[d]['x']['native_slot'][draw[k]].astype(np.int64)for k,d in zip(('visible','foreign','synthetic_owner','public_owner','CF_owner'),('SYNTH_AUG65536',)*3+('PUBLIC_TRAIN1480','CF3816'))]+[direct['x']['native_slot'][di].astype(np.int64)])
            assert gn.dtype==np.int64 and np.array_equal(gn,expected) and np.array_equal(gn,g['rgb96'][:,0,0,0].astype(np.int64)%8)
            records.append({**{k:v.tolist()for k,v in draw.items()},'direct':di.tolist()})
        draws.append(records)
    checks['same3step1810_1811_paired_draws_and_exact224_native_pixel_alignment']=draws[0]==draws[1]
    checks['all5plusdirect_original_TRAIN_mask_draws_and_negative']=rejects(lambda:adapter.assert_draws_train({'visible':np.array([1])},np.array([0]),masks)) and rejects(lambda:adapter.assert_draws_train({},np.array([1]),masks))
    checks['native_numeric_float_or_outofrange_not_coerced']=rejects(lambda:t.native_indices(np.array([1.],np.float64))) and rejects(lambda:t.native_indices(np.array([8],np.int8)))
    # TRAIN-only native/domain class prior census. HOLDOUT triplet is retained but excluded.
    ns=np.repeat(np.array([0,1,0,5,3],np.int8),3);ys=np.tile(np.array([0,1,0],np.int8),5);tr=np.arange(15)<12
    dc=dict(x={'native_slot':ns,'original_query_available':np.ones(15,bool)},y={'direct_end_external_target':ys},metadata=[dict(domain='SYNTH'if i<6 or i>=12 else'REAL')for i in range(15)])
    prior=t.train_native_class_priors(dc,tr);known=[r for r in prior['rows']if r['direct_D']+r['direct_E']]
    checks['TRAIN_only_domain_native_raw2to1_balanced_expected_half_not_holdout']=prior['TRAIN_known_class_totals']==dict(D=8,E=4) and all(r['raw_D_equals_2E']and r['balanced_draw_expected_E_given_domain_native']==.5 for r in known)and all(r['direct_D']==r['direct_E']==0 for r in prior['rows']if r['native_slot']==3)
    checks['native6_7_no_direct_known_not_imputed']=all(r['direct_D']==r['direct_E']==0 and r['balanced_draw_expected_E_given_domain_native']is None for r in prior['rows']if r['native_slot']in(6,7))
    same_rgb=np.zeros((7,96,96,3),np.uint8);different_valid=np.ones((7,96,96),bool);different_valid[2:4,0,0]=False
    collision_fixture=dict(x=dict(rgb96=same_rgb,valid96=different_valid,native_slot=np.array([0,1,2,2,6,7,0],np.int8),original_query_available=np.array([True,True,True,True,True,False,True]),row_id=np.array(['TOY%d'%i for i in range(7)])),y=dict(direct_end_external_target=np.array([0,1,0,1,-1,-1,1],np.int8)))
    census=t.train_raw_identity_collisions(collision_fixture,np.array([True]*6+[False]))
    checks['raw_vs_raw_native_conflicts_and_validbyte_identity_exact']=census['RAW_only']['unique_identities']==2 and census['RAW_only']['conflicting_identities']==2 and census['RAW_only']['rows_in_conflicting_identities']==4 and census['RAW_plus_native']['conflicting_identities']==1 and census['RAW_plus_native']['rows_in_conflicting_identities']==2
    checks['collision_census_TRAIN_known_only_U_UNKNOWN_retained']=census['TRAIN_rows']==6 and census['known_available_TRAIN_rows']==4 and census['excluded_TRAIN_inputU']==census['excluded_TRAIN_available_UNKNOWN']==1 and collision_fixture['y']['direct_end_external_target'].tolist()==[0,1,0,1,-1,-1,1]
    sp,op,dd,px=t.decode(np.zeros((2,2),np.float32),np.zeros((2,3),np.float32),np.array([0.,10.],np.float32),np.array([True,False]))
    checks['fixed_direct_ge0_and_original_inputU_first']=dd.tolist()==[1,-1]and sp[1].tolist()==[-1,-1]and op[1]==px[1]==-1
    # Confirm unchanged inherited functions as source AST, not imported original runner.
    def funcs(path):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(safe(path).read_text()).body if isinstance(n,ast.FunctionDef)}
    before=funcs(V194/'grouped_fit_preparation_b/train_grouped_direct_end.py');after=funcs(H/'train_native_conditioned_grouped_source.py')
    names=('pure_training','metadata_matches','load_old','load_direct','gather','assemble','decode','objective','join_direct_split','split_masks','cohort_counts','save_cohorts','fresh_run')
    checks['13_inherited_loader_mask_loss_decode_cohort_functions_AST_exact']=all(before[k]==after[k]for k in names)
    read=t.c.read;bound=t.c.read_bound
    try:
        t.c.read=lambda _:dict(approved_native_conditioned_grouped_source_fit=False,source_only=True,automatic_retry=False)
        t.c.read_bound=lambda _:(_ for _ in ()).throw(RuntimeError('Body gate crossed'))
        checks['false_actual_entry_rejected_before_any_bound_body']=rejects(lambda:t.fit(Path('/invented_false_entry.json')))
    finally:t.c.read=read;t.c.read_bound=bound
    assert all(checks.values()),checks
    return checks

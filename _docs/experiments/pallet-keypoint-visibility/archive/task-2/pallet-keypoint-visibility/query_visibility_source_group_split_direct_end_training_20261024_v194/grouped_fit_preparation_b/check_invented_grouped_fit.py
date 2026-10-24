"""Invented CPU inputs only; no production source arrays/checkpoints/fit."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,tempfile
import numpy as np
import torch
H=Path(__file__).absolute().parent

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def check():
    t=module(H/'train_grouped_direct_end.py','invented_grouped_trainer');m=module(t.MASK,'invented_mask_adapter');pure=t.pure_training(None);candidate=t.c.module(t.c.CANDIDATE,'invented_direct_loss_only')
    tests={}
    old=t.P/'query_visibility_sampled_domain_direct_end_20261024_v189/paired_direct_end_fit_preparation_c/train_paired_direct_end.py'
    def funcs(p):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
    aa,bb=funcs(old),funcs(H/'train_grouped_direct_end.py')
    tests['nine_shared_functions_AST_exact']=all(aa[k]==bb[k]for k in ['pure_training','metadata_matches','load_old','load_direct','gather','assemble','decode','objective','save_outputs'])
    def row(k):return dict(row_id='G/native'+str(k),original_global_index=k,native_slot=k,query_xy=[10.+k,20.],L=96.,original_query_available=True)
    packet=dict(path='/invented/packet.npz',sha256='a'*64)
    plan=[dict(row(k),cache_index=k,control_join_key=['SYNTH_FINITE_END','A','BEFORE',k],source_packet=packet,authority='CONSTRUCTIVE_WEAK',stored_weak_direct_target=0 if k==0 else-1,role='TRAIN'if k==0 else'HOLDOUT',component_id='invented')for k in range(2)]
    metadata=[dict(row(k),cache_index=i,domain='SYNTH_FINITE_END',attempt_id='A',variant='BEFORE',control_join_key=plan[k]['control_join_key'],source_packet=packet,authority='CONSTRUCTIVE_WEAK',root_admitted=True)for i,k in enumerate((1,0))]
    aligned=t.join_direct_split(metadata,plan,np.array([-1,0],np.int8));mask=m.fit_mask(metadata,aligned)
    tests['direct_reordered_key_join_preserves_original_source_index']=mask.tolist()==[False,True]and[ r['source_split_cache_index']for r in aligned]==[1,0]
    for name,modify in [('duplicate',lambda x:x.__setitem__(1,dict(x[0],cache_index=1))),('packet_drift',lambda x:x[0].__setitem__('source_packet',dict(packet,sha256='b'*64)))]:
        bad=copy.deepcopy(metadata);modify(bad)
        try:t.join_direct_split(bad,plan,np.array([-1,0],np.int8));ok=False
        except AssertionError:ok=True
        tests['direct_'+name+'_rejected']=ok
    bad=copy.deepcopy(metadata);bad[0]['L']=95.
    try:m.fit_mask(bad,t.join_direct_split(bad,plan,np.array([-1,0],np.int8)));ok=False
    except AssertionError:ok=True
    tests['direct_qL_drift_rejected']=ok
    # Small invented source population, retaining the original hard-owner guards.
    support=np.array([[i%2,(i//2)%2]for i in range(12)],np.int8)
    mass=np.array([[.1,.2,.3]if i%2 else [.2,.1,.1]for i in range(12)],np.float64)
    public_target=np.repeat(np.arange(3,dtype=np.int8),[53,52,148]);cf_target=np.repeat(np.arange(2,dtype=np.int8),157)
    oldc={'SYNTH_AUG65536':dict(x=dict(original_query_available=np.ones(12,bool)),y=dict(support_target=support,owner_soft=mass)),
        'PUBLIC_TRAIN1480':dict(x=dict(original_query_available=np.ones(253,bool)),y=dict(support_target=np.full((253,2),-1,np.int8),owner_target=public_target)),
        'CF3816':dict(x=dict(original_query_available=np.ones(314,bool)),y=dict(support_target=np.full((314,2),-1,np.int8),owner_target=cf_target))}
    masks={k:np.arange(len(v['x']['original_query_available']))%3!=0 for k,v in oldc.items()};masked=m.masked_old_pools(oldc,masks,pure['pools_for']);tests['every_original_loss_class_remains_and_holdout_masked']=masked['feasible']
    tests['raw_partial_mass_classconditional_preserved']=all(np.allclose(p,np.where(masks['SYNTH_AUG65536'],mass[:,j],0)/mass[masks['SYNTH_AUG65536'],j].sum(),rtol=0,atol=3e-16)for j,p in enumerate(masked['pools']['conditional']))
    av=np.array([True,True,True,True,True,False]);target=np.array([0,0,1,1,-1,-1],np.int8);dm=np.array([True,False,True,False,True,True]);direct=m.masked_direct_pools(av,target,dm);masks['DIRECT_CONTROL']=dm
    tests['direct_unknown_U_and_holdout_excluded']=[x.tolist()for x in direct['pools']]==[[0],[2]]
    sequences=[]
    for arm in range(2):
        rng=np.random.default_rng(1810);drng=np.random.default_rng(1811);chunks=[]
        for step in range(16):
            draw=pure['draw_step'](rng,masked['pools']);dd=np.concatenate([drng.choice(x,16,replace=True)for x in direct['pools']]);m.assert_draws_train(draw,dd,masks);chunks.append({k:v.tolist()for k,v in draw.items()}|{'direct':dd.tolist()})
        sequences.append(chunks)
    tests['two_arms_same192_plus32_seeded_streams']=sequences[0]==sequences[1]and all(len(x['direct'])==32 and sum(len(v)for k,v in x.items()if k!='direct')==192 for x in sequences[0])
    rng=np.random.default_rng(1810);reference=[{k:v.tolist()for k,v in pure['draw_step'](rng,masked['pools']).items()}for i in range(16)]
    tests['direct1811_rng_does_not_advance_old1810_stream']=all({k:v for k,v in r.items()if k!='direct'}==o for r,o in zip(sequences[0],reference))
    bad=copy.deepcopy(sequences[0][0]);bad['visible'][0]=0
    try:m.assert_draws_train({k:np.array(v)for k,v in bad.items()if k!='direct'},np.array(bad['direct']),masks);ok=False
    except AssertionError:ok=True
    tests['held_draw_injection_rejected']=ok
    empty=m.masked_direct_pools(av,target,np.array([True,True,False,False,True,True]));tests['empty_direct_class_fails_no_rebalance']=not empty['feasible']
    torch.manual_seed(7);a=dict(support_logits=torch.randn(224,2,requires_grad=True),owner_logits=torch.randn(224,3,requires_grad=True),direct_end_external_logit=torch.randn(224,requires_grad=True));y=torch.tensor([0]*16+[1]*16,dtype=torch.int8)
    total,base,parts,dl=t.objective(a,y,pure,candidate);oldbase,oldparts=pure['five_losses']({k:a[k][:192]for k in ('support_logits','owner_logits')})
    tests['oldfive_mean_byte_exact_plus_additive_weight1']=torch.equal(base,oldbase)and all(torch.equal(x,y)for x,y in zip(parts,oldparts))and torch.equal(total,oldbase+dl)
    total.backward();tests['direct_gradient_only_separate32']=bool((a['direct_end_external_logit'].grad[:192]==0).all()and(a['direct_end_external_logit'].grad[192:]!=0).all())
    # Three small AdamW steps: identical toy state/input/draws produce exact parity.
    torch.manual_seed(1810);prototype=torch.nn.Linear(2,6);states=[]
    inp=torch.arange(448,dtype=torch.float32).reshape(224,2)/448
    for i in range(2):
        model=copy.deepcopy(prototype);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        for step in range(3):
            opt.zero_grad(set_to_none=True);z=model(inp);loss,*_=t.objective(dict(support_logits=z[:,:2],owner_logits=z[:,2:5],direct_end_external_logit=z[:,5]),y,pure,candidate);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        states.append((pure['object_sha'](model.state_dict()),pure['object_sha'](opt.state_dict())))
    tests['invented_three_step_sameinit_state_optimizer_parity']=states[0]==states[1]
    av2=np.array([True,True,False]);s=np.array([[0.,-1.],[1.,0.],[np.nan,np.nan]],np.float32);o=np.array([[0.,0.,0.],[1.,2.,3.],[np.nan]*3],np.float32);z=np.array([-np.finfo(np.float32).tiny,0.,np.nan],np.float32);sp,op,dp,proxy=t.decode(s,o,z,av2)
    tests['direct_raw_ge_zero_originalU_and_owner_tie']=dp.tolist()==[0,1,-1]and op.tolist()==[0,2,-1]and sp.tolist()==[[0,0],[1,0],[-1,-1]]and proxy.tolist()==[-1,0,-1]
    # Exercise the actual cohort serializer on authored fixtures, not source data.
    with tempfile.TemporaryDirectory(prefix='invented_cohort_',dir=H)as directory:
        folder=Path(directory);full={k:np.zeros((3,2),np.float32)if k in ('raw_support_logits','support_prediction','query_xy')else np.zeros((3,3),np.float32)if k=='raw_owner_logits'else np.arange(3)for k in t.ROW_OUTPUTS}
        full.update(row_id=np.array(['r0','r1','r2']),original_query_available=np.array([True,False,True]),native_slot=np.array([0,1,2],np.int8),support_order=np.array(['v','f']),owner_order=np.array(['T','F','B']),direct_end_order=np.array(['D','E','U']))
        binding=t.c.save_arrays(folder/'FULL.npz',full);aligned=[dict(component_id='c'+str(i),authority='INVENTED')for i in range(3)];saved=t.save_cohorts(binding,np.array([True,False,True]),folder,'TOY',aligned)
        tr=t.c.arrays(saved['TRAIN']['outputs'],t.ROW_OUTPUTS+t.ORDERS);ho=t.c.arrays(saved['HOLDOUT']['outputs'],t.ROW_OUTPUTS+t.ORDERS)
        tests['cohort_serializer_preserves_orders_rows_and_U']=tr['row_id'].tolist()==['r0','r2']and ho['row_id'].tolist()==['r1']and ho['original_query_available'].tolist()==[False]and tr['owner_order'].tolist()==['T','F','B']and saved['HOLDOUT']['original_inputU']==1
    assert all(tests.values()),tests
    return dict(completed=True,PASS=True,invented_only=True,tests=tests,production_source_array_header_checkpoint_model_fit_or_eval_reads=0)

if __name__=='__main__':print(json.dumps(check()))

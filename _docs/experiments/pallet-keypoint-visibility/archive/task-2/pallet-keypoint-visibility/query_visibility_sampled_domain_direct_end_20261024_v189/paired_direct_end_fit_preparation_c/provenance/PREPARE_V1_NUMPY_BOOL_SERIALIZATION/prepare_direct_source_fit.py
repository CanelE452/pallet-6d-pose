"""Metadata/code/invented arrays only. Does not load source arrays or models."""
import argparse, copy, json
import numpy as np
import torch
import direct_source_common as c
import build_direct_control_cache as cache
import train_paired_direct_end as trainer
H,V=c.H,c.V
PROTOCOL=H/'PROTOCOL_BEFORE_DIRECT_CONTROL_CACHE_OR_FIT.json'

def invented():
    tests={};sam=c.sampled_helpers();runtime,_,_=c.sam_helpers()
    z=np.arange(35,dtype=np.float32).reshape(5,7)-17
    mask,av=sam['sample_sam'](z,[2.25,1.75],8.,True,[5,7]);grid=sam['grid96']([2.25,1.75],8.).reshape(96,96,2)
    expected=np.zeros((96,96),bool);ea=np.zeros_like(expected)
    for j in range(96):
        for i in range(96):
            x,y=grid[j,i];ix,iy=int(np.floor(x)),int(np.floor(y));dx,dy=x-ix,y-iy;value=0.
            for ax,ay,w in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
                if 0<=ix+ax<7 and 0<=iy+ay<5:value+=float(z[iy+ay,ix+ax])*w
            ea[j,i]=-.5<=x<6.5 and -.5<=y<4.5;expected[j,i]=value>0 and ea[j,i]
    tests['exact_inherited_SAM96_independent_scalar_fourtap_and_domain']=np.array_equal(mask,expected) and np.array_equal(av,ea)
    m,a=sam['sample_sam'](None,None,None,False,[5,7]);tests['inputU_precedes_SAM_body_and_stays_zero_channels']=not m.any() and not a.any()
    box,reason=runtime['prompt_box']([[-2.,1.],[8.,1.],[8.,9.],[-2.,9.],[0.,2.],[1.,3.],[2.,4.],[3.,5.]])
    tests['unclipped_unexpanded_original_q8_box_only']=reason is None and box.tolist()==[-2.,1.,8.,9.]
    tests['changed_RGB_recompute_key_differs_even_same_q8']=c.ash(np.zeros((3,4,3),np.uint8))!=c.ash(np.ones((3,4,3),np.uint8))
    # Exercise source population/join guards using invented administrative
    # objects; no real completion/result is opened by these fixtures.
    synth=[dict(attempt_id='s%d'%i,native_slot=i,outputs={'path':'S%d'%i,'sha256':'a'*64} if i==0 else None,direct_end_states=[0,1,0] if i==0 else [-1]*3,status='SUPPORTED' if i==0 else 'UNKNOWN') for i in range(2)]
    real=[dict(pair_id='r%d'%i,original_input={'native_slot':i},variants={v:({'path':'R%d_%s'%(i,v),'sha256':'b'*64} if i==0 or v=='BEFORE' else {'constructed':False}) for v in ['BEFORE','COVER','ADJACENT_SHAM']},PASS=i==0,direct_end_weak_states=dict(BEFORE='VISIBLE_WEAK',COVER='EXTERNALLY_BLOCKED_WEAK',SHAM='VISIBLE_WEAK') if i==0 else dict(BEFORE='UNKNOWN',COVER='UNKNOWN',SHAM='UNKNOWN'),status='SUPPORTED' if i==0 else 'FAILED') for i in range(2)]
    domain_old=cache.DOMAINS;read_old,lines_old=c.read_bound,c.lines
    try:
        cache.DOMAINS={k:2 for k in domain_old};data={};ad=dict(completed=True,PASS=True,source_only=True,training_admission_direct_end=True,sources=[],groups=[])
        for domain,rows in [('SYNTH_FINITE_END',synth),('REAL_SEMANTIC_END_WEAK',real)]:
            done={'completed':True,'PASS':True,('attempts' if domain=='SYNTH_FINITE_END' else 'manifest'):{'key':domain+'rows'}};data[domain]=done;data[domain+'review']={'completed':True,'PASS':True};data[domain+'rows']=rows
            ad['sources'].append(dict(domain=domain,complete={'key':domain},independent_review={'key':domain+'review'},census={'key':domain+'rows'},expected_attempts=2))
            for r in rows:
                for vi,var in enumerate(cache.VARIANTS):
                    packet=r['outputs'] if domain=='SYNTH_FINITE_END' else r['variants'][{'SHAM':'ADJACENT_SHAM'}.get(var,var)]
                    if packet is None or 'path' not in packet:continue
                    aid=r.get('attempt_id',r.get('pair_id'));gid=domain+'/'+aid+'/'+var;admitted=(aid.endswith('0'))
                    mm=[dict(row_id=gid+'/native%d'%k,original_occurrence_id='old%d'%k,original_global_index=k,native_slot=k,query_xy=[3.,4.],L=12.,original_query_available=k!=7,original_input_flags={'query_available':k!=7}) for k in range(8)]
                    ad['groups'].append(dict(group_id=gid,domain=domain,attempt_id=aid,variant=var,source_packet=packet,layout='SYNTH_TRIPLET' if domain=='SYNTH_FINITE_END' else 'REAL_SINGLE',variant_index=vi,selected_native_slot=0 if admitted else 1,source_status=r['status'],admitted=admitted,admitted_target=(1 if var=='COVER' else 0) if admitted else -1,authority='CONSTRUCTIVE_SOURCE_FIXED_NEAR_ANCHOR_END_TOY' if domain=='SYNTH_FINITE_END' else 'SOURCE_RGB_SEMANTIC_END_WEAK_TOY',metadata8=mm))
        c.read_bound=lambda b:data[b['key']];c.lines=lambda b:data[b['key']]
        groups,census,_=cache.source_population(ad);tests['all_census_including_missing_and_failed_BEFORE_only_no_fake_images']=len(groups)==7 and len(census)==12 and sum(r['source_packet'] is None for r in census)==5
        for name,change in [('missing_physical_group',lambda a:a['groups'].pop()),('admit_UNKNOWN',lambda a:a['groups'][-1].update(admitted=True,admitted_target=0)),('wrong_requested_slot',lambda a:a['groups'][0].update(selected_native_slot=1))]:
            bad=copy.deepcopy(ad);change(bad)
            try:cache.source_population(bad)
            except AssertionError:tests[name+'_rejected']=True
            else:raise AssertionError('Invented bad source closure must fail')
    finally:cache.DOMAINS=domain_old;c.read_bound=read_old;c.lines=lines_old
    candidate=c.module(c.CANDIDATE,'v189_invented_loss_only_candidate')
    pure=c.functions(c.OLD_TRAINER,('five_losses','draw_step'),dict(torch=torch,F=torch.nn.functional,np=np,require=c.need))
    a=dict(support_logits=torch.zeros((224,2),requires_grad=True),owner_logits=torch.zeros((224,3),requires_grad=True),direct_end_external_logit=torch.zeros(224,requires_grad=True))
    target=torch.tensor([0]*16+[1]*16,dtype=torch.int8)
    total,base,parts,direct=trainer.objective(a,target,pure,candidate);total.backward()
    tests['old_five_mean_plus_separate_balanced_direct32_no_owner_support_alias']=len(parts)==5 and torch.equal(base,torch.stack(parts).mean()) and torch.equal(total,base+direct) and bool((a['support_logits'].grad[192:]==0).all()) and bool((a['owner_logits'].grad[192:]==0).all()) and bool((a['direct_end_external_logit'].grad[:192]==0).all())
    tests['balanced_direct32_exact_analytic_gradient']=torch.equal(a['direct_end_external_logit'].grad[192:],torch.tensor([1/64]*16+[-1/64]*16))
    p=dict(support=[[np.arange(3),np.arange(3,6)],[np.arange(6,9),np.arange(9,12)]],conditional=[np.ones(12)/12]*3,public=[np.arange(3),np.arange(3,6),np.arange(6,9)],CF=[np.arange(3),np.arange(3,6)])
    r1=np.random.default_rng(1810);r2=np.random.default_rng(1810);drng=np.random.default_rng(1811)
    for _ in range(3):
        d1=pure['draw_step'](r1,p);drng.choice(7,32);d2=pure['draw_step'](r2,p);c.need(all(np.array_equal(d1[k],d2[k]) for k in d1),'Extra direct RNG cannot alter original192 draw stream')
    tests['separate_direct_rng_leaves_original192_draws_byteexact']=True
    av=np.array([True,True,True,False]);s=np.array([[1.,-1.],[-1.,1.],[0.,0.],[np.nan,np.nan]],np.float32);o=np.zeros((4,3),np.float32);o[3]=np.nan;d=np.array([0.,1.,-1.,np.nan],np.float32)
    sp,op,dp,proxy=trainer.decode(s,o,d,av)
    tests['support_strict_gt0_direct_ge0_owner_firsttie_U_first']=sp.tolist()==[[1,0],[0,1],[0,0],[-1,-1]] and op.tolist()==[0,0,0,-1] and dp.tolist()==[1,1,0,-1] and proxy.tolist()==[0,1,-1,-1]
    d[0]=np.nan;tests['direct_nonfinite_available_is_U_not_visible']=trainer.decode(s,o,d,av)[2][0]==-1
    c.need(all(tests.values()),'Invented cache/training guards')
    return dict(completed=True,PASS=True,invented_only=True,tests=tests,count=len(tests),new_head_model_instantiations=0,optimizer_steps_this_prepare=0,source_body_header_weight_feature_or_evaluation_reads=0,CUDA_initialized=torch.cuda.is_initialized())

def prepare():
    c.need(not PROTOCOL.exists(),'Prepare once; preserve prior attempts before explicit repair')
    c.need(c.sha(c.SCOPE)=='c5f17638e41289b20080c73e33ebe7acd1b19d73d37186957e131919688d03e5','Frozen parent source scope')
    add=c.read(c.SAM_ADDENDUM);c.need(add['root_authorized'] is True and add['prospective_direct_objective']['direct_draws_per_step']==32 and add['prospective_direct_objective']['additive_direct_loss_weight']==1,'Root prospective direct objective')
    prior=V/'direct_end_training_preparation_c';bindings=c.read(prior/'EXACT_TRAIN_CACHE_BINDINGS.json');headcheck=prior/'GENERATED_INVENTED_HEAD_ARITHMETIC_CHECKS.json';c.need(c.read(headcheck)['PASS'],'Prior root-authorized exact three-step lambda0 head arithmetic')
    _,sam_protocol,_=c.sam_helpers();checks=invented();c.write(H/'GENERATED_INVENTED_CACHE_TRAINING_CHECKS.json',checks)
    sources={k:c.bind(p) for k,p in [('scope',c.SCOPE),('V190_prospective_train_scope',c.TRAIN_SCOPE),('SAM_and_direct_design_addendum',c.SAM_ADDENDUM),('admission_schema',H/'ADMISSION_SCHEMA.json'),('common_code',H/'direct_source_common.py'),('cache_code',H/'build_direct_control_cache.py'),('trainer_code',H/'train_paired_direct_end.py'),('prepare_code',__file__),('existing_head',c.HEAD),('candidate',c.CANDIDATE),('original_trainer',c.OLD_TRAINER),('original_train_protocol',c.OLD_TRAIN_PROTOCOL),('original_pixel_helper',c.PIXEL_HELPER),('original_SAM_helper',c.SAM_HELPER),('original_SAM_protocol',c.SAM_PROTOCOL),('SAM_source_pin',sam_protocol['source_pin']['path']),('head_lambda0_arithmetic_checks',headcheck),('exact_TRAIN_cache_readiness',prior/'EXACT_TRAIN_CACHE_BINDINGS.json')]}
    p=dict(completed=True,prepared_only=True,cache_code=sources['cache_code'],trainer_code=sources['trainer_code'],source_bindings=sources,checks=c.bind(H/'GENERATED_INVENTED_CACHE_TRAINING_CHECKS.json'),unchanged_TRAIN_caches=bindings,checkpoint=sam_protocol['checkpoint'],training=dict(arms=[dict(name=a,use_sam=b) for a,b in trainer.ARMS],old_five_mean_unchanged=True,old_batch=192,direct_batch=[16,16],direct_weight=1,total_batch=224,direct_rng_seed=1811,old_rng_seed=1810,seed=1810,steps=8000,optimizer='AdamW',lr=3e-4,weight_decay=1e-4,clip=5,FP32=True,TF32=False,AMP=False,strict_deterministic=True,head_parameters=115222,source_only_design_basis=True),cache=dict(admission_manifest='Exact schema in ADMISSION_SCHEMA.json; complete full1024+66 census and all physical groups, no fake missing inputs; only root-admitted requested-native direct labels',input_keys=list(c.INPUTS),target_keys=list(c.META)+['direct_end_external_target','authority'],source_index_repeats_allowed=True,original_qL_native_flags_U_unchanged=True,SAM='Original fullRGB and all8 original q minmax box, no clipping/GT prompt; frozen original SAM1 vit_b state/API/runtime; new changed RGB must forward, exact fullRGB+box-only raw4 reuse needs hash equality then fresh metadata. Root entry before any body/checkpoint/model.',SAM96='Pinned V181 FP64 bilinear original V90 pixel-center grid and strict >0; no new crop sampling, original saved raw96/valid copied. Missing original prompt gives separate zero SAM channels, no change to original input availability.'),fit=dict(final_only=True,all2_final8000_freeze_before_any_saved_outputs=True,evaluation_or_COMMON_or_held_or_private_loader_exists=False,no_target_or_GT_or_reference_inputs=True,new_native_owner_quality_targets=False,original_public253_CF314_owner_only=True,source_authorities_kept_distinct=True),output='Fresh root-entry-supplied child under V189 or new V190; no preexisting output overwrite',training_admission_pending_separate_root_manifest=True,no_actual_source_body_header_or_asset_hash_reads=True,no_production_forward_or_fit=True,selected_model=None,adopted=False,fully_solved=False)
    c.write(PROTOCOL,p)
    cache_entry=dict(approved_direct_control_cache=False,source_only=True,code=sources['cache_code'],protocol=c.bind(PROTOCOL),root_scope=None,admission=None,checkpoint=sam_protocol['checkpoint'],output_directory=None)
    fit_entry=dict(approved_paired_direct_end_source_fit=False,source_only=True,code=sources['trainer_code'],protocol=c.bind(PROTOCOL),root_scope=None,augmentation_complete=bindings['augmentation_TRAIN_complete'],original_public_CF_complete=bindings['original_public_CF_cache_complete'],direct_cache_complete=None,output_directory=None)
    c.write(H/'ROOT_ACTUAL_CACHE_ENTRY_TEMPLATE_APPROVED_FALSE.json',cache_entry);c.write(H/'ROOT_ACTUAL_FIT_ENTRY_TEMPLATE_APPROVED_FALSE.json',fit_entry)
    c.write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,FINAL_READY=True,protocol=c.bind(PROTOCOL),cache_code=sources['cache_code'],trainer_code=sources['trainer_code'],checks=p['checks'],inherited_head_lambda0_checks=sources['head_lambda0_arithmetic_checks'],cache_entry_template=c.bind(H/'ROOT_ACTUAL_CACHE_ENTRY_TEMPLATE_APPROVED_FALSE.json'),fit_entry_template=c.bind(H/'ROOT_ACTUAL_FIT_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_source_assets_headers_checkpoint_or_model_reads=0,actual_fit_or_frozenSAM_forward=0,production_training_admission=False,selected_model=None,fully_solved=False))
    print(json.dumps(dict(completed=True,PASS=True,cache_code=sources['cache_code'],trainer_code=sources['trainer_code'],protocol=c.bind(PROTOCOL),preparation=c.bind(H/'PREPARATION_COMPLETE.json'))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true',required=True);p.parse_args();prepare()

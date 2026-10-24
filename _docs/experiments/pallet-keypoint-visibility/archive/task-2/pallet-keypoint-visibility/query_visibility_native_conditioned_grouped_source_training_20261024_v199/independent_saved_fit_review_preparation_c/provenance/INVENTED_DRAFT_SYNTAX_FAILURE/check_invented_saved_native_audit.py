"""Invented NumPy/scalar fixtures only. No torch/model/source asset access."""
from pathlib import Path
import ast,json,tempfile
import numpy as np
import audit_saved_native_source_fit as a

def reject(fn):
    try:fn()
    except (AssertionError,ValueError):return
    raise AssertionError('Invalid authored fixture accepted')

def save(path,packet):
    with path.open('xb')as f:np.savez_compressed(f,**packet)
    return a.bind(path)

def authored_direct():
    n=8
    x=dict(row_id=np.asarray(['r%d'%i for i in range(n)]),original_global_index=np.arange(n,dtype=np.int64),native_slot=np.asarray([0,0,1,0,2,2,6,7],np.int8),query_xy=np.zeros((n,2),np.float64),L=np.full(n,96.,np.float64),original_query_available=np.asarray([1,1,1,1,1,1,0,1],bool),rgb96=np.zeros((n,96,96,3),np.uint8),valid96=np.ones((n,96,96),bool))
    x['rgb96'][4,0,0,0]=1;x['rgb96'][5,0,0,0]=1
    target=np.asarray([0,1,0,0,0,1,-1,-1],np.int8)
    rows=[dict(domain='SYNTH_FINITE_END'if i<6 else'REAL_WEAK_END',attempt_id='a%d'%(i//3),variant=('BEFORE','COVER','SHAM')[i%3],native_slot=int(x['native_slot'][i]))for i in range(n)]
    return dict(x=x,y=dict(direct_end_external_target=target,authority=np.asarray(['SYNTHETIC_FINITE_END']*6+['SOURCE_RGB_SEMANTIC_END_WEAK']*2)),rows=rows),np.asarray([1,1,1,0,1,1,1,1],bool)

def checks():
    result=[]
    def test(name,fn):fn();result.append(dict(name=name,PASS=True))
    def losses():
        r=dict(update=1,old_five_losses=[.1,.2,.3,.4,.5],old_five_mean=float(np.float32(.3)),direct_loss=float(np.float32(.7)),total_loss=1.,grad_norm=6.)
        a.loss_record(r,1)
        for key,value in [('update',2),('grad_norm',float('nan')),('total_loss',1.1),('old_five_mean',.7)]:reject(lambda:a.loss_record(dict(r,**{key:value}),1))
    test('logged5mean_directweight1_finitegrad_and_badrecord_rejection',losses)
    def mass_draws():
        soft=np.asarray([[.1,.2,.3]if i%2 else[.3,.1,.1]for i in range(12)],np.float64);av=np.ones(12,bool);av[11]=False
        dom={'SYNTH_AUG65536':dict(x=dict(original_query_available=av),y=dict(support_target=np.asarray([[i%2,(i//2)%2]for i in range(12)],np.int8),owner_soft=soft)),'PUBLIC_TRAIN1480':dict(x=dict(original_query_available=np.ones(253,bool)),y=dict(support_target=np.full((253,2),-1,np.int8),owner_target=np.repeat(np.arange(3,dtype=np.int8),[53,52,148]))),'CF3816':dict(x=dict(original_query_available=np.ones(314,bool)),y=dict(support_target=np.full((314,2),-1,np.int8),owner_target=np.repeat(np.arange(2,dtype=np.int8),157))),'DIRECT_CONTROL':dict(x=dict(original_query_available=np.asarray([1,1,1,1,0],bool)),y=dict(direct_end_external_target=np.asarray([0,0,1,1,-1],np.int8)))}
        masks={k:np.arange(len(v['x']['original_query_available']))%3!=0 for k,v in dom.items()};masks['DIRECT_CONTROL']=np.asarray([1,0,1,0,1],bool);pool,dp=a.filtered_pools(dom,masks)
        assert [v.tolist()for v in dp]==[[0],[2]]
        for j,p in enumerate(pool['conditional']):
            expected=np.asarray([soft[i,j]if masks['SYNTH_AUG65536'][i]and av[i]else 0. for i in range(12)]);expected/=expected.sum();assert np.allclose(p,expected,rtol=0,atol=2e-16)
        streams=[]
        for _ in range(2):
            rng=np.random.default_rng(1810);drng=np.random.default_rng(1811);stream=[]
            for step in range(32):
                draw=a.old_draw(rng,pool);direct=np.concatenate([drng.choice(v,16,replace=True)for v in dp]);assert sum(map(len,draw.values()))==192 and len(direct)==32;stream.append({k:v.tolist()for k,v in draw.items()}|{'direct':direct.tolist()})
            streams.append(stream)
        assert streams[0]==streams[1]
        bad={k:v.copy()for k,v in masks.items()};bad['DIRECT_CONTROL'][:]=False;reject(lambda:a.filtered_pools(dom,bad))
    test('independent_partialmass_TRAIN_CDF_and_paired192_32_rng',mass_draws)
    def schema():
        shapes=a.parameter_shapes();assert len(shapes)==36 and sum(int(np.prod(s))for s in shapes.values())==116407
        assert shapes['native_residual.0.weight']==(16,72)and shapes['native_residual.2.weight']==(1,16)
    test('independent36tensor116407_shape_schema_no_model',schema)
    def envelope():
        proto={'path':'AUTHORED','sha256':'0'*64};split={'path':'AUTHORED_SPLIT','sha256':'1'*64}
        for j,arm in enumerate(a.ARMS):
            z=dict(state_dict={},optimizer_state={},update=8000,seed=1810,arm=arm,use_sam=False,native_condition=bool(j),protocol=proto,source_split=split);a.checkpoint_envelope(z,proto,arm,split)
            for key,value in [('native_condition',not bool(j)),('native_condition',j),('use_sam',True),('update',7999),('seed',0)]:reject(lambda:a.checkpoint_envelope(dict(z,**{key:value}),proto,arm,split))
    test('two_RAW_native_bool_flags_and_checkpoint_envelope_reject_wrongarms',envelope)
    def codes():
        native=np.arange(8,dtype=np.int64);assert np.array_equal(a.native_condition_codes(native,True),np.eye(8,dtype=np.float32));assert not a.native_condition_codes(native,False).any()
        reject(lambda:a.native_condition_codes(native.astype(np.float32),True));reject(lambda:a.native_condition_codes(np.asarray([8],np.int64),True));reject(lambda:a.native_condition_codes(native,1))
    test('zero8_vs_onehot8_originalslots_no_modulo_or_floatfallback',codes)
    def prior():
        cache,mask=authored_direct();p=a.independent_native_priors(cache,mask);assert p['TRAIN_known_class_totals']=={'D':3,'E':2}
        by={(r['domain'],r['native_slot']):r for r in p['rows']};assert by['ALL',0]['direct_D']==1 and by['ALL',0]['direct_E']==1
        assert by['ALL',0]['raw_known_E_fraction']==.5 and by['ALL',0]['balanced_draw_expected_E_given_domain_native']==(.5/(1/3+.5))
        assert by['ALL',7]['direct_UNKNOWN']==1 and by['ALL',6]['original_inputU']==1 and by['ALL',7]['raw_known_E_fraction']is None
        # HOLDOUT known row cannot affect any TRAIN prior.
        cache['y']['direct_end_external_target'][3]=1;assert a.independent_native_priors(cache,mask)==p
    test('independent_TRAIN_native_domain_raw_and_balanced_expectation_prior',prior)
    def collision():
        cache,mask=authored_direct();r=a.independent_raw_collisions(cache,mask);assert r['known_available_TRAIN_rows']==5 and r['excluded_TRAIN_inputU']==1 and r['excluded_TRAIN_available_UNKNOWN']==1
        assert r['RAW_only']['unique_identities']==2 and r['RAW_only']['conflicting_identities']==2 and r['RAW_plus_native']['unique_identities']==3
        assert r['RAW_only']['rows_in_conflicting_identities']==5
        old=a.canonical_census(r);cache['x']['rgb96'][3]=123;assert a.canonical_census(a.independent_raw_collisions(cache,mask))==old
        cache['x']['valid96'][1,0,0]=False;assert a.independent_raw_collisions(cache,mask)['RAW_only']['unique_identities']==3
    test('RAW_byteidentity_native_split_conflicts_U_UNKNOWN_and_no_HOLDOUT',collision)
    def canonical():
        assert a.canonical_census({'conflicts':[{'identity':'b','rows':[2]},{'identity':'a','rows':[1]}]})==a.canonical_census({'conflicts':[{'identity':'a','rows':[1]},{'identity':'b','rows':[2]}]})
        assert a.canonical_census([1,2])!=a.canonical_census([2,1])
    test('census_conflict_order_agnostic_but_row_and_value_exact',canonical)
    def triplets():
        n=8;x=dict(row_id=np.asarray(['r%d'%i for i in range(n)]),original_query_available=np.ones(n,bool));truth=np.asarray([0,1,0,0,1,0,-1,-1],np.int8)
        rr=[dict(domain='S',attempt_id='A'if i<3 else'B'if i<6 else'C',native_slot=0 if i<6 else1,variant=('BEFORE','COVER','SHAM')[i%3])for i in range(n)]
        cache=dict(x=x,y=dict(direct_end_external_target=truth,authority=np.asarray(['WEAK']*n)),rows=rr);mask=np.asarray([1,1,1,0,0,0,1,1],bool);pred=np.asarray([0,1,0,0,0,0,-1,-1],np.int8)
        r=a.triplet_readout(cache,pred,mask);assert r['ALL']['known_D_E_D_triplets']==2 and r['ALL']['all3_correct']==1 and r['ALL']['incomplete_variant_groups']==1 and r['TRAIN']['known_D_E_D_triplets']==1 and r['HOLDOUT']['all3_correct']==0
        mask[1]=False;reject(lambda:a.triplet_readout(cache,pred,mask))
    test('paired_DED_source_truth_and_allattemptnative_UNKNOWN_retention',triplets)
    with tempfile.TemporaryDirectory(prefix='invented_only_',dir=a.H)as folder:
        folder=Path(folder);n=3;av=np.asarray([1,0,1],bool);x=dict(row_id=np.asarray(['r0','r1','r2']),original_global_index=np.asarray([7,7,9],np.int64),native_slot=np.asarray([0,1,2],np.int8),query_xy=np.asarray([[1.,2.],[np.nan,np.nan],[3.,4.]],np.float64),L=np.asarray([96.,np.nan,384.],np.float64),original_query_available=av)
        full=dict(x,raw_support_logits=np.asarray([[0.,1.],[np.nan,np.nan],[1.,0.]],np.float32),raw_owner_logits=np.asarray([[0.,0.,0.],[np.nan]*3,[-1.,-1.,-2.]],np.float32),raw_direct_end_external_logit=np.asarray([0.,np.nan,-np.finfo(np.float32).tiny],np.float32),support_prediction=np.asarray([[0,1],[-1,-1],[1,0]],np.int8),owner_prediction=np.asarray([0,-1,0],np.int8),direct_end_prediction_D0_E1_Uminus1=np.asarray([1,-1,0],np.int8),regional_support_proxy_D0_E1_Uminus1=np.asarray([1,-1,0],np.int8),support_order=np.asarray(['visible_end_support','foreign_front_support']),owner_order=np.asarray(['TARGET','FOREIGN','BACKGROUND']),direct_end_order=np.asarray(['VISIBLE_END0','EXTERNAL_BLOCKED_END1','UNKNOWNminus1']))
        def decode():
            assert a.validate_output(full,x).tolist()==[1,-1,0];bad=dict(full,direct_end_prediction_D0_E1_Uminus1=np.asarray([0,-1,0],np.int8));reject(lambda:a.validate_output(bad,x))
        test('scalar_strict_support_ge0_direct_firsttie_owner_and_U',decode)
        def subsets():
            mask=np.asarray([1,0,1],bool);aligned=[dict(component_id='component%d'%i,authority='AUTHORED',control_join_key=['D','A','SHAM',i])for i in range(n)];record={'source_cohorts':{'TOY':{}}};allowed={}
            for role,take in [('TRAIN',mask),('HOLDOUT',~mask)]:
                indices=np.flatnonzero(take);packet={k:full[k][indices]for k in a.ROWKEYS};packet.update({k:full[k]for k in a.ORDERS});b=save(folder/(role+'.npz'),packet);allowed[b['path']]=b;path=folder/(role+'.jsonl');line=[dict(cache_index=int(i),row_id=str(x['row_id'][i]),component_id=aligned[i]['component_id'],role=role,original_global_index=int(x['original_global_index'][i]),native_slot=int(x['native_slot'][i]),original_query_available=bool(av[i]),direct_target_authority=aligned[i]['authority'],control_join_key=aligned[i]['control_join_key'])for i in indices];path.write_text(''.join(json.dumps(r)+'\n'for r in line));record['source_cohorts']['TOY'][role]=dict(outputs=b,metadata=a.bind(path),rows=len(indices),original_inputU=int((~av[indices]).sum()))
            assert a.audit_subsets(record,'TOY',full,{'TOY':mask},{'TOY':aligned},allowed)['HOLDOUT']==dict(rows=1,inputU=1)
            old=record['source_cohorts']['TOY']['TRAIN']['outputs'];packet=a.arrays(old,a.OUTKEYS,allowed,True);packet['raw_direct_end_external_logit'][0]=.5;b=save(folder/'BAD.npz',packet);allowed[b['path']]=b;record['source_cohorts']['TOY']['TRAIN']['outputs']=b;reject(lambda:a.audit_subsets(record,'TOY',full,{'TOY':mask},{'TOY':aligned},allowed))
        test('cohort_disjoint_union_META_scores_U_authority_and_bytedrift',subsets)
        def whitelist():reject(lambda:a.arrays({'path':str(folder/'NEVER_EXISTS.npz'),'sha256':'0'*64},[],{}))
        test('binary_whitelist_guard_precedes_header_hash_load',whitelist)
        def false_entry():
            p=folder/'FALSE.json';p.write_text(json.dumps(dict(approved_actual_independent_native_sourcefit_audit=False,both_models_completed=False,source_only=True,automatic_retry=False)));reject(lambda:a.run(p))
        test('false_entry_stops_before_source_or_output_access',false_entry)
    def lexical():
        for tail in ('forbidden_v112/never_stat','example_current223/never_stat','example_public_held/never_stat','example_common4096/never_stat'):
            reject(lambda:a.safe(a.P/tail))
    test('protected_lexical_names_rejected_before_stat_read_hash',lexical)
    def cm():
        r=a.metric([0,0,1,1,-1],[0,-1,1,0,1]);assert r['CM_D_E_by_D_E_U']==[[1,0,1],[1,1,0]]and r['known']==4 and r['truth_UNKNOWN']==1
    test('weak_truth_UNKNOWN_and_prediction_U_are_distinct',cm)
    def imports():
        tree=ast.parse(a.raw(Path(a.__file__)))
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):assert all(not n.name.startswith('torch')for n in node.names)
        assert 'torch'not in __import__('sys').modules
    test('preparation_does_not_import_torch_or_construct_model',imports)
    return dict(completed=True,PASS=True,invented_only=True,checks=result,count=len(result),production_source_cache_target_log_prediction_checkpoint_header_body_reads=0,production_model_instantiation_forward_fit_calls=0,torch_imports=0)

if __name__=='__main__':
    r=checks();a.write(a.H/'GENERATED_INVENTED_SAVED_NATIVE_CHECKS.json',r);print('INVENTED_SAVED_NATIVE_AUDIT_PASS',r['count'],flush=True)

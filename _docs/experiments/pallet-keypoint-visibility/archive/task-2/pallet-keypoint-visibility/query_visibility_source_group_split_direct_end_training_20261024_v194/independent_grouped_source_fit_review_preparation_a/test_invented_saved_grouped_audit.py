"""Authored CPU arrays/tensors only; no source cache/output/weight/model reads."""
from pathlib import Path
import json,tempfile
import numpy as np
import torch
import audit_saved_grouped_source_fit as a

def reject(fn):
    try:fn()
    except AssertionError:return
    raise AssertionError('Invalid authored fixture was accepted')

def save(path,packet):
    with path.open('xb')as f:np.savez_compressed(f,**packet)
    return a.bind(path)

def checks():
    result=[]
    def test(name,fn):fn();result.append(dict(name=name,PASS=True))
    def losses():
        r=dict(update=1,old_five_losses=[.1,.2,.3,.4,.5],old_five_mean=float(np.float32(.3)),direct_loss=float(np.float32(.7)),total_loss=1.,grad_norm=6.)
        a.loss_record(r,1)
        for key,value in [('update',2),('grad_norm',float('nan')),('total_loss',1.1),('old_five_mean',.7)]:reject(lambda:a.loss_record(dict(r,**{key:value}),1))
    test('logged5mean_directweight1_finitegrad_and_badrecord_rejection',losses)
    def mass_draws():
        soft=np.array([[.1,.2,.3]if i%2 else[.3,.1,.1]for i in range(12)],np.float64);av=np.ones(12,bool);av[11]=False
        dom={'SYNTH_AUG65536':dict(x=dict(original_query_available=av),y=dict(support_target=np.array([[i%2,(i//2)%2]for i in range(12)],np.int8),owner_soft=soft)),'PUBLIC_TRAIN1480':dict(x=dict(original_query_available=np.ones(253,bool)),y=dict(support_target=np.full((253,2),-1,np.int8),owner_target=np.repeat(np.arange(3,dtype=np.int8),[53,52,148]))),'CF3816':dict(x=dict(original_query_available=np.ones(314,bool)),y=dict(support_target=np.full((314,2),-1,np.int8),owner_target=np.repeat(np.arange(2,dtype=np.int8),157))),'DIRECT_CONTROL':dict(x=dict(original_query_available=np.array([True,True,True,True,False])),y=dict(direct_end_external_target=np.array([0,0,1,1,-1],np.int8)))}
        masks={k:np.arange(len(v['x']['original_query_available']))%3!=0 for k,v in dom.items()};masks['DIRECT_CONTROL']=np.array([True,False,True,False,True]);pool,dp=a.filtered_pools(dom,masks)
        assert [v.tolist()for v in dp]==[[0],[2]]
        for j,p in enumerate(pool['conditional']):
            expected=np.array([soft[i,j]if masks['SYNTH_AUG65536'][i]and av[i]else 0. for i in range(12)]);expected/=expected.sum();assert np.allclose(p,expected,rtol=0,atol=2e-16)
        streams=[]
        for _ in range(2):
            rng=np.random.default_rng(1810);drng=np.random.default_rng(1811);stream=[]
            for step in range(32):
                draw=a.old_draw(rng,pool);direct=np.concatenate([drng.choice(v,16,replace=True)for v in dp]);assert sum(len(v)for v in draw.values())==192 and len(direct)==32;stream.append({k:v.tolist()for k,v in draw.items()}|{'direct':direct.tolist()})
            streams.append(stream)
        assert streams[0]==streams[1]
        bad={k:v.copy()for k,v in masks.items()};bad['DIRECT_CONTROL'][:]=False;reject(lambda:a.filtered_pools(dom,bad))
    test('independent_partialmass_TRAIN_CDF_and_paired192_32_rng',mass_draws)
    with tempfile.TemporaryDirectory(prefix='invented_only_',dir=a.H)as folder:
        folder=Path(folder)
        def shape_and_checkpoint():
            shapes=a.parameter_shapes();assert len(shapes)==32 and sum(np.prod(s)for s in shapes.values())==115222
            sd={k:torch.zeros(s,dtype=torch.float32)for k,s in shapes.items()};opt=dict(param_groups=[dict(lr=3e-4,weight_decay=1e-4,params=list(range(32)))],state={i:dict(step=torch.tensor(8000.),exp_avg=torch.zeros(s),exp_avg_sq=torch.zeros(s))for i,s in enumerate(shapes.values())})
            proto=dict(path='AUTHORED_PROTOCOL',sha256='0'*64);split=dict(path='AUTHORED_SPLIT',sha256='1'*64);z=dict(state_dict=sd,optimizer_state=opt,update=8000,seed=1810,arm=a.ARMS[0],use_sam=False,protocol=proto,source_split=split)
            path=folder/'AUTHORED_FINAL.pt';torch.save(z,path);b=a.bind(path);a.checkpoint(b,a.state_digest(sd),proto,a.ARMS[0],{b['path']:b},split)
            opt['state'][0]['step']=torch.tensor(7999.);bad=folder/'AUTHORED_BAD.pt';torch.save(z,bad);bb=a.bind(bad);reject(lambda:a.checkpoint(bb,a.state_digest(sd),proto,a.ARMS[0],{bb['path']:bb},split))
        test('authored32tensor115222_finalcheckpoint_AdamW8000_and_badstep',shape_and_checkpoint)
        n=3;av=np.array([True,False,True]);x=dict(row_id=np.array(['r0','r1','r2']),original_global_index=np.array([7,7,9],np.int64),native_slot=np.array([0,1,2],np.int8),query_xy=np.array([[1.,2.],[np.nan,np.nan],[3.,4.]],np.float64),L=np.array([96.,np.nan,384.],np.float64),original_query_available=av)
        s=np.array([[0.,1.],[np.nan,np.nan],[1.,0.]],np.float32);o=np.array([[0.,0.,0.],[np.nan]*3,[-1.,-1.,-2.]],np.float32);d=np.array([0.,np.nan,-np.finfo(np.float32).tiny],np.float32)
        full=dict(x,raw_support_logits=s,raw_owner_logits=o,raw_direct_end_external_logit=d,support_prediction=np.array([[0,1],[-1,-1],[1,0]],np.int8),owner_prediction=np.array([0,-1,0],np.int8),direct_end_prediction_D0_E1_Uminus1=np.array([1,-1,0],np.int8),regional_support_proxy_D0_E1_Uminus1=np.array([1,-1,0],np.int8),support_order=np.array(['visible_end_support','foreign_front_support']),owner_order=np.array(['TARGET','FOREIGN','BACKGROUND']),direct_end_order=np.array(['VISIBLE_END0','EXTERNAL_BLOCKED_END1','UNKNOWNminus1']))
        def decode():
            assert a.validate_output(full,x).tolist()==[1,-1,0]
            bad=dict(full,direct_end_prediction_D0_E1_Uminus1=np.array([0,-1,0],np.int8));reject(lambda:a.validate_output(bad,x))
        test('scalar_strict_support_ge0_direct_firsttie_owner_and_U',decode)
        def subsets():
            mask=np.array([True,False,True]);aligned=[dict(component_id='component%d'%i,authority='AUTHORED',control_join_key=['D','A','SHAM',i])for i in range(n)];record={'source_cohorts':{'TOY':{}}};allowed={}
            for role,take in [('TRAIN',mask),('HOLDOUT',~mask)]:
                indices=np.flatnonzero(take);packet={k:full[k][indices]for k in a.ROWKEYS};packet.update({k:full[k]for k in a.ORDERS});b=save(folder/(role+'.npz'),packet);allowed[b['path']]=b;path=folder/(role+'.jsonl')
                line=[dict(cache_index=int(i),row_id=str(x['row_id'][i]),component_id=aligned[i]['component_id'],role=role,original_global_index=int(x['original_global_index'][i]),native_slot=int(x['native_slot'][i]),original_query_available=bool(av[i]),direct_target_authority=aligned[i]['authority'],control_join_key=aligned[i]['control_join_key'])for i in indices]
                path.write_text(''.join(json.dumps(r)+'\n'for r in line));record['source_cohorts']['TOY'][role]=dict(outputs=b,metadata=a.bind(path),rows=len(indices),original_inputU=int((~av[indices]).sum()))
            counts=a.audit_subsets(record,'TOY',full,{'TOY':mask},{'TOY':aligned},allowed);assert counts['HOLDOUT']==dict(rows=1,inputU=1)
            # One output-byte drift is rejected even when rows/U/META remain.
            role='TRAIN';old=record['source_cohorts']['TOY'][role]['outputs'];packet=a.arrays(old,a.OUTKEYS,allowed,True);packet['raw_direct_end_external_logit'][0]=.5;b=save(folder/'BAD_SUBSET.npz',packet);allowed[b['path']]=b;record['source_cohorts']['TOY'][role]['outputs']=b;reject(lambda:a.audit_subsets(record,'TOY',full,{'TOY':mask},{'TOY':aligned},allowed))
        test('full_cohort_union_disjoint_META_scores_U_authority_and_bytedrift',subsets)
        def false_entry():
            p=folder/'FALSE_ENTRY.json';p.write_text(json.dumps(dict(approved_actual_independent_grouped_sourcefit_audit=False,both_models_completed=False,source_only=True,automatic_retry=False)));reject(lambda:a.run(p))
        test('false_entry_stops_before_any_source_or_output_access',false_entry)
    def lexical():reject(lambda:a.safe(a.P/'forbidden_v112'/'never_stat'))
    test('forbidden_lexical_name_before_stat_read_hash',lexical)
    def cm():
        r=a.metric([0,0,1,1,-1],[0,-1,1,0,1]);assert r['CM_D_E_by_D_E_U']==[[1,0,1],[1,1,0]]and r['known']==4 and r['truth_UNKNOWN']==1
    test('weak_truth_unknown_denominator_not_native_gold',cm)
    return dict(completed=True,PASS=True,invented_only=True,checks=result,count=len(result),production_source_cache_target_log_prediction_checkpoint_header_body_reads=0,production_model_instantiation_forward_fit_calls=0)

if __name__=='__main__':
    result=checks();a.write(a.H/'GENERATED_INVENTED_SAVED_GROUPED_FIT_CHECKS.json',result);print('INVENTED_SAVED_GROUPED_AUDIT_PASS',len(result['checks']),flush=True)

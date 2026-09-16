"""Validate all actual training updates and weights before performance is opened."""
import csv,hashlib
import numpy as np,torch
import env as E
from a_data import records,order_indices
from a_train import verify_binding

def main():
    torch.set_num_threads(4)
    complete=E.read(E.DOC/'A/MAIN_TRAINING_COMPLETE.json');assert complete['complete']
    verify_binding(complete['runs']);verify_binding(E.read(E.DOC/'SOURCE_BINDING.json')['files'])
    lock=E.read(E.DOC/'A/JOINT_TRAINING_LOCK.json');verify_binding(lock['bindings'])
    summaries=[];traces=[];orders={};states={}
    for binding in complete['runs']:
        r=E.read(E.ROOT/binding['path']);name=f"{r['cohort']}_{r['arm']}_seed{r['seed']}"
        assert r['status']=='COMPLETE' and r['optimizer_updates']==2000 and r['BN_unchanged']
        assert len(r['trace'])==2000 and r['physical_batch']==16 and not r['AMP'] and not r['EMA']
        assert r['initial_R0_SHA']==E.R0_SHA
        verify_binding([r['last_weights'],r['resume_checkpoint']])
        key=r['sampler_lock']
        if key not in orders:
            rows=records(r['cohort'],'train');indices=order_indices(len(rows),r['seed'])
            assert hashlib.sha256(indices.tobytes()).hexdigest()==lock['sampler'][key]['index_sequence_sha256']
            orders[key]=[[rows[j]['id'] for j in indices[i:i+16]] for i in range(0,len(indices),16)]
        for i,t in enumerate(r['trace']):
            assert t['step']==i+1 and t['batch_ids']==orders[key][i] and t['lr']==[1e-4]
            assert all(np.isfinite(t[x]) for x in ['loss','gradient_pre','gradient_post','parameter_delta_L2'])
            assert 0<=t['gradient_post']<=10.0001 and t['parameter_delta_L2']>0
            traces.append(dict(run=name,**{k:t[k] for k in ['step','loss','gradient_pre','gradient_post','parameter_delta_L2','selector']},
              batch_ids_SHA=hashlib.sha256('\n'.join(t['batch_ids']).encode()).hexdigest(),nonidentity_GT_rows=sum(v!=0 for v in t['branches'])))
        final=torch.load(E.ROOT/r['last_weights']['path'],map_location='cpu',weights_only=False)
        resume=torch.load(E.ROOT/r['resume_checkpoint']['path'],map_location='cpu',weights_only=False)
        sd=final['model'].state_dict()
        assert final['updates']==resume['step']==2000
        assert final['recipe_SHA']==resume['recipe_SHA']==E.sha(E.DOC/'A/JOINT_TRAINING_LOCK.json')
        assert all(torch.equal(v,resume['state_dict'][k]) and torch.isfinite(v).all() for k,v in sd.items())
        states[name]={k:v.clone() for k,v in sd.items()}
        summaries.append({k:v for k,v in r.items() if k not in ['trace','GPU']})
        summaries[-1]['trace_binding']=binding
        summaries[-1]['global_gradient_pre_max']=max(t['gradient_pre'] for t in r['trace'])
        summaries[-1]['nonidentity_fraction']=sum(int(v) for k,v in r['branch_histogram'].items() if k!='0')/sum(r['branch_histogram'].values())
        print('TRAIN AUDIT',name,2000,'nonidentity',summaries[-1]['nonidentity_fraction'],flush=True)
    pairs={}
    for cohort in ['RECT','SQUARE']:
        for seed in [1,2,3]:
            a=states[f'{cohort}_INDEXED_seed{seed}'];b=states[f'{cohort}_EQUIV_seed{seed}']
            pairs[f'{cohort}_{seed}']=dict(state_bit_exact=all(torch.equal(v,b[k]) for k,v in a.items()),
              state_L2=float(sum((v.double()-b[k].double()).square().sum() for k,v in a.items()).sqrt()))
    path=E.DOC/'A/training_step_trace.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(traces[0]),lineterminator='\n');w.writeheader();w.writerows(traces)
    old=E.DOC/'A/training_audit.json'
    E.freeze(E.DOC/'A/history/TRAINING_AUDIT_BEFORE_MAIN.json',E.read(old))
    E.write(old,dict(status='ALL_12_FITS_COMPLETE_VERIFIED',main_fits=12,main_updates=24000,smoke_optimizer_updates=12,
      training_forward_images=384000,probe_forward_images=768,paired_sample_order_exact=True,
      same_trainable_parameters=len({r['trainable_names_SHA'] for r in summaries})==1,
      final_weights_equal_resume_state=True,BN_unchanged=True,source_and_locked_code_hashes_unchanged=True,
      joint_objective_user_approved=True,full_step_trace=E.bound(path),runs=summaries,pair_final_weight_differences=pairs,
      training_paper_GT_inputs=0,FINAL_access=False,performance_not_yet_opened=True))
    assert len(summaries)==12 and len(traces)==24000
    print('A TRAINING AUDIT PASS',pairs,flush=True)
if __name__=='__main__':main()

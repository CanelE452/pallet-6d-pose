"""Exactly four conditional fits; direct unchanged historical training loop."""
import sys
from contracts import *
from prepare import simulation,reserved_guard


def audit_training():
    lock=read(DOC/'STUDENT_IMPLEMENTATION_LOCK.json');audits={}
    s=simulation();expected=s.tensor_sha(s.load_model().state_dict())
    for name in lock['fits']:
        folder=RAW/'runs'/name;a=read(folder/'TRAINING_AUDIT.json');trace=read(folder/'EXPOSURE.json')
        seed=a['seed'];old=read(OLD_RAW/f'runs/diversity_seed{seed}/EXPOSURE.json')
        assert len(trace)==300 and all(t['synthetic']==o['synthetic'] for t,o in zip(trace,old))
        assert a['init_state_sha256']==expected and a['last_only'] and a['optimizer_updates']==300
        assert a['checkpoint_sha256']==sha(folder/'last.pt')
        ck=s.torch.load(folder/'last.pt',map_location='cpu');model=ck['model']
        assert ck['train_args']==s.HYP and model.args['task']=='pose'
        assert all(s.torch.isfinite(v).all() for v in model.state_dict().values())
        a['same_seed_synthetic_300_step_hash_parity']=True;audits[name]=a
    write(DOC/'TRAINING_AUDIT.json',dict(status='PASS',fits=4,optimizer_updates=1200,
        init_parity=True,actual_synthetic_augmented_input_parity=True,
        optimizer='Unchanged SGD groups/warmup/epoch schedule',loss='Unchanged stock E2ELoss',
        BN_training='Unchanged normal BN updates',last_step_only=True,evaluation_GT_read=False,audits=audits))
    return audits


def main():
    verify_lock();reserved_guard();lock=read(DOC/'STUDENT_IMPLEMENTATION_LOCK.json')
    assert read(DOC/'TASK_RISK_VERDICT.json')['verdict']=='TASK_RISK_MECHANISM_PASS'
    for key in ('sources','implementation'):
        for p,h in lock[key].items():assert sha(ROOT/p)==h
    s=simulation();original_digest=s.batch_digest;original_optimizer=s.optimizer
    for name in lock['fits']:
        method,seed=name.rsplit('_seed',1);seed=int(seed);folder=RAW/'runs'/name
        if (folder/'TRAINING_AUDIT.json').exists():continue
        assert not (folder/'TRAINING_STARTED.json').exists(),'Never silently repeat an interrupted fit'
        resource=gpu();write(folder/'TRAINING_STARTED.json',dict(name=name,resource=resource))
        oldtrace=read(OLD_RAW/f'runs/diversity_seed{seed}/EXPOSURE.json');calls=[0];steps=[0];resources=[]
        def digest(batch):
            value=original_digest(batch);idx=calls[0]//2
            if calls[0]%2==0:assert value==oldtrace[idx]['synthetic'],('Synthetic parity failed before optimizer update',name,idx)
            calls[0]+=1;return value
        def optimizer(model):
            opt=original_optimizer(model)
            def post_step(opt,args,kwargs):
                steps[0]+=1
                if steps[0]%30==0:
                    r=gpu();resources.append(r);print('GPU',name,steps[0],r['gpu'],flush=True)
            opt.register_step_post_hook(post_step)
            return opt
        s.batch_digest=digest;s.optimizer=optimizer
        print('START',name,'real_labels',30 if method=='proposed' else 174,flush=True)
        s.train(method,seed)
        assert calls[0]==600 and steps[0]==300
        write(folder/'SAFETY_AND_PARITY.json',dict(status='PASS',synthetic_before_update_checks=300,
            optimizer_post_step_checks=10,resources=resources))
        print('FINISHED',name,flush=True)
    audit_training();print('ALL4_FITS_COMPLETE; reserved145 scoring now allowed',flush=True)


if __name__=='__main__':main()

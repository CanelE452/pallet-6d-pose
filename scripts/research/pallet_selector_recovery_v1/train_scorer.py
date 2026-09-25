"""Only synthetic files opened; real outcomes are not imported or read."""
from . import common as C
READS=C.synth_guard()
import numpy as np
import torch
from . import models as M

def main():
    C.setup();gpu=C.gpu();lock=C.read(C.sdoc(2)/'SYNTH_PREDICTION_LOCK.json');labels=C.read(C.sdoc(2)/'EXACT_LABEL_AUDIT.json')
    C.verify(lock['features']);C.verify(labels['labels']);z=dict(np.load(C.ROOT/lock['features']['path']));l=dict(np.load(C.ROOT/labels['labels']['path']));assert np.array_equal(z['ids'],l['ids'])
    y=np.tile(l['parity'],2);parts=np.tile(l['split'],2);valid=np.concatenate([z[a+'_valid'] for a in C.ARMS]);train=(parts=='TRAIN')&valid;val=(parts=='VAL')&valid
    variants=['GEO_LINEAR']+(['GEO_IMG_MLP'] if lock['feature_mode']=='GEO_IMG' else [])
    vals={};paths={}
    for variant in variants:
        x=np.concatenate([M.pack_inputs(z,a,variant) for a in C.ARMS]);x,mean,std=M.normalize(x,train)
        state,result=M.fit(x,y,train,val,linear=variant=='GEO_LINEAR');vals[variant]=result
        path=C.sraw(2)/(variant+'.pt');assert not path.exists();torch.save(dict(state=state,mean=mean,std=std,d=x.shape[-1],variant=variant),path);paths[variant]=C.bind(path)
    winner=min(variants,key=lambda a:(-vals[a]['best_val_accuracy'],a!='GEO_LINEAR'))
    C.freeze(C.sdoc(2)/'SCORER_VAL_RESULTS.json',dict(variants=vals,valid_train=int(train.sum()),valid_val=int(val.sum()),invalid_excluded=int((~valid).sum())))
    C.freeze(C.sdoc(2)/'SCORER_SELECTION_LOCK.json',dict(created_at=C.now(),winner=winner,checkpoint=paths[winner],variants=paths,
        criterion='Synthetic VAL parity accuracy only; tie GEO_LINEAR; earliest best epoch',real_used=False,feature_contract=C.bind(C.DOC/'SELECTOR_FEATURE_CONTRACT.json')))
    C.freeze(C.sdoc(2)/'SCORER_TRAINING_AUDIT.json',dict(real_GT_reads=0,guard_active=True,read_paths=sorted(set(READS)),gpu=gpu,base_optimizer_steps=0,seed=42,
        allowed_variants=variants,TEST_used_in_fit=False,checkpoint_hashes=paths))
    test(z,l)

def test(z,l):
    assert not (C.sdoc(2)/'SCORER_SYNTH_TEST.json').exists()
    lock=C.read(C.sdoc(2)/'SCORER_SELECTION_LOCK.json');C.verify(lock['checkpoint']);ck=torch.load(C.ROOT/lock['checkpoint']['path'],map_location='cpu',weights_only=False)
    C.freeze(C.sdoc(2)/'SYNTH_TEST_ONCE_LOCK.json',dict(created_at=C.now(),selection_lock=C.bind(C.sdoc(2)/'SCORER_SELECTION_LOCK.json'),test_reads=1))
    output={};raw={}
    for arm in C.ARMS:
        score=M.scores(ck,M.pack_inputs(z,arm,lock['winner']));sel=M.selection(score,C.HYP);swapped=M.selection(score[:,::-1],C.HYP[::-1]);assert np.array_equal(sel,1-swapped)
        usable=z[arm+'_valid'];selected=np.where(usable,sel,z[arm+'_current']);prob=1/(1+np.exp(np.clip(score[:,1]-score[:,0],-80,80)));raw[arm]=dict(selected=selected,prob=prob)
        subsets={'TEST':l['split']=='TEST'}
        for label,mask in [('LOW',l['elevation']<15),('MID',(l['elevation']>=15)&(l['elevation']<45)),('HIGH',l['elevation']>=45),('SMALL',l['size']<.1),('LARGE',l['size']>=.1)]:subsets['TEST_'+label]=(l['split']=='TEST')&mask
        output[arm]={}
        for name,mask in subsets.items():
            n=int(mask.sum());target=l['parity'][mask];p=prob[mask]
            output[arm][name]=dict(n=n,current_accuracy=float(np.mean(z[arm+'_current'][mask]==target)) if n else None,learned_accuracy=float(np.mean(selected[mask]==target)) if n else None,
                learned_correct=int(np.sum(selected[mask]==target)),current_correct=int(np.sum(z[arm+'_current'][mask]==target)),valid_pairs=int(usable[mask].sum()),
                brier=float(np.mean((p-target)**2)) if n else None,mean_probability=float(p.mean()) if n else None,
                abs_score_margin_quantiles=np.quantile(np.abs(score[mask,0]-score[mask,1]),[.1,.5,.9]) if n else [])
    aggregate=dict(n=sum(output[a]['TEST']['n'] for a in C.ARMS),current_correct=sum(output[a]['TEST']['current_correct'] for a in C.ARMS),learned_correct=sum(output[a]['TEST']['learned_correct'] for a in C.ARMS))
    aggregate.update(current_accuracy=aggregate['current_correct']/aggregate['n'],learned_accuracy=aggregate['learned_correct']/aggregate['n'])
    C.freeze(C.sdoc(2)/'SCORER_SYNTH_TEST.json',dict(winner=lock['winner'],aggregate=aggregate,by_expert=output,candidate_order_swap_invariance=True,TEST_once=True,
        caveat='Exact renderer parity test, not real transfer; inherited base R0 trained broader synthetic source. No TEST tuning.'))
    print('SYNTH_TEST',lock['winner'],aggregate,flush=True)

if __name__=='__main__':main()

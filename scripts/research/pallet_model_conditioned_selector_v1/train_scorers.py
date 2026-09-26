"""Two exact legacy linear fits. Runtime guard forbids opening real references."""
from . import common as C
READS=C.guard('fit')
import numpy as np
import torch
from . import common as C
from scripts.research.pallet_selector_recovery_v1 import models as M

def main():
    C.U.setup();gpu=C.U.gpu()
    assert not (C.stage(2)/'SCORER_LOCK.json').exists(),'Two fits already complete; no retraining'
    fl=C.read(C.stage(2)/'SYNTH_FEATURES_LOCK.json')
    for b in fl['files']:C.verify(b)
    C.verify(fl['prediction_lock']);z=dict(np.load(C.ROOT/fl['files'][0]['path']))
    # Bind existing exact parity only after both models' prediction/features locks.
    start=C.now();la=C.read(C.OLD/'stage2_synth_scorer/EXACT_LABEL_AUDIT.json');C.verify(la['labels']);labels=dict(np.load(C.ROOT/la['labels']['path']))
    assert fl['created_at']<start and np.array_equal(labels['ids'],z['ids'])
    C.save(C.stage(2)/'EXACT_LABEL_REUSE.json',dict(created_at=start,labels=la['labels'],source_audit=C.bind(C.OLD/'stage2_synth_scorer/EXACT_LABEL_AUDIT.json'),prediction_lock=fl['prediction_lock'],exact_reuse=True))
    mask=labels['split']!='TEST';part=labels['split'][mask];y=labels['parity'][mask];paths={};audits={}
    for model in C.MODELS:
        path=C.RAW/(C.NEW[model]+'.pt');assert not path.exists(),'Partial fit present; do not silently train again'
        valid=z[model+'_valid'][mask];train=(part=='TRAIN')&valid;val=(part=='VAL')&valid
        raw=z[model+'_geo'][mask];x,mean,std=M.normalize(raw,train)
        assert x.shape[1:]==(2,94) and train.sum()>0 and val.sum()>0
        state,result=M.fit(x,y,train,val,linear=True)
        assert set(state)=={'net.weight','net.bias'} and tuple(state['net.weight'].shape)==(1,94)
        torch.save(dict(state=state,mean=mean,std=std,d=94,variant='GEO_LINEAR',model_condition=model),path);paths[model]=C.bind(path)
        C.save(C.stage(2)/('S1_SCORER_VAL.json' if model=='S1' else 'HMAN_SCORER_VAL.json'),result)
        audits[model]=dict(valid_train=int(train.sum()),valid_val=int(val.sum()),train_ids=[str(i) for i in z['ids'][mask][train]],
            val_ids=[str(i) for i in z['ids'][mask][val]],invalid_train=int(((part=='TRAIN')&~valid).sum()),invalid_val=int(((part=='VAL')&~valid).sum()),
            best_epoch=result['best_epoch'],epochs=result['epochs'],checkpoint=paths[model],model_inputs_only=model,normalization_train_only=True,TEST_tensors_in_fit=False)
    C.save(C.stage(2)/'SCORER_LOCK.json',dict(created_at=C.now(),checkpoints=paths,features=C.bind(C.stage(2)/'SYNTH_FEATURES_LOCK.json'),labels=la['labels'],
        protocol=C.bind(C.DOC/'PROTOCOL_LOCK.json'),model_code=C.bind(C.ROOT/'scripts/research/pallet_selector_recovery_v1/models.py'),selection='own synthetic VAL parity; earliest best; no TEST or real selection'))
    C.save(C.stage(2)/'SCORER_TRAINING_AUDIT.json',dict(created_at=C.now(),models=audits,real_GT_reads=0,guard_active=True,read_paths=sorted(set(READS)),
        trainings=2,seed=42,lr=.001,weight_decay=.0001,batch=256,max_epoch=30,patience=5,optimizer='AdamW',architecture='shared nn.Linear(94,1)',
        keypoint_optimizer_steps=0,TEST_used_in_fit=False,TRAIN_only_normalization=True,VAL_only_earlystop=True,gpu_start=gpu,gpu_end=C.U.gpu(),
        caveat='Existing NPZ container includes historically known TEST labels, but TEST rows are removed before all fit/normalization/VAL tensors. No TEST scoring before lock.'))
    print('TWO_SCORERS_FROZEN',paths,flush=True)

if __name__=='__main__':main()

"""2x2 stored-parameter / BatchNorm-running-buffer swap; no new learning."""
import copy
import torch
from scripts.self_training_yolo.v3 import true_ignore_trainer  # checkpoint import
from . import recovery_common as R
C=R.C
PHASE='bn_probe'
ARMS={'STUDENT_R0BN':('student','r0'),'R0_STUDENTBN':('r0','student')}


def is_bn_buffer(name):
    return name.endswith(('.running_mean','.running_var','.num_batches_tracked'))


def main():
    paths={'r0':C.N.E.R0,'student':R.BASE_RAW/'runs/PLASTIC/weights/last.pt'}
    C.verify(C.read(R.BASE_DOC/'FIT_PLASTIC.json')['checkpoint'])
    protocol=dict(question='Does changing stored BN statistics account for current student degradation?',
        arms=ARMS,baseline_arms='Unmodified R0 and unmodified PLASTIC student are cached controls',
        swap='Only running_mean, running_var, num_batches_tracked; convolution and affine BN parameters untouched',
        new_training=False,GT_used_for_transform=False,threshold_sweep=False,
        interpretation='Component intervention at inference, not proof that BN-frozen training improves or pseudo labels are correct',
        sources=[C.bound(p) for p in paths.values()]+[C.bound(__file__),C.bound(R.__file__)])
    with R.scope(PHASE):C.freeze(C.DOC/'PROTOCOL.json',protocol)
    R.evaluation_protocol(PHASE,list(ARMS),protocol['sources'])
    payloads={k:torch.load(p,map_location='cpu',weights_only=False) for k,p in paths.items()}
    for arm,(w,b) in ARMS.items():
        with R.scope(PHASE):
            ck=copy.deepcopy(payloads[w]);base=ck['model'].state_dict();donor=payloads[b]['model'].state_dict()
            names=[n for n in base if is_bn_buffer(n)];assert len(names)==378
            before={n:v.clone() for n,v in base.items()}
            for n in names:base[n].copy_(donor[n])
            ck['model'].load_state_dict(base)
            assert all(torch.equal(before[n],base[n]) for n in base if n not in names)
            assert all(torch.equal(base[n],donor[n]) for n in names)
            ck['ema']=None;ck['recovery_probe']=dict(arm=arm,weights=w,buffers=b)
            dest=C.RAW/'checkpoints'/f'{arm}.pt';dest.parent.mkdir(parents=True,exist_ok=True)
            fit=C.DOC/f'FIT_{arm}.json'
            if fit.exists():C.verify(C.read(fit)['checkpoint']);continue
            assert not dest.exists(),'Preserve unbound probe checkpoint'
            with dest.open('xb') as handle:torch.save(ck,handle)
            C.freeze(fit,dict(complete=True,arm=arm,checkpoint=C.bound(dest),protocol=C.bound(C.DOC/'PROTOCOL.json'),
                new_optimizer_steps=0,changed_buffers=sum(not torch.equal(before[n],base[n]) for n in names),
                buffer_count=len(names),learned_parameters_unchanged=True))
            print('PROBE_EXPORTED',arm,flush=True)


if __name__=='__main__':main()

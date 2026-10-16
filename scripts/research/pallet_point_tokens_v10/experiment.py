"""Bounded four-arm point-coordinate refinement experiment."""
from pathlib import Path
import hashlib
import time
import json
import numpy as np
import torch
from data import RUN,ROOT,KIT,PARENT,EXPORT,observations,line_inputs,read,write,sha
from refiner import Refiner,direction_matrix,loss
from cli_train import state_sha

ARMS=('local','isotropic','directional','donor')
STEPS=2000
BATCH=16


def target(split,device):
    rows=read(RUN/f'targets_{split}.json')['rows']
    return torch.tensor([r['points'] for r in rows],dtype=torch.float32,device=device),torch.tensor([r['valid'] for r in rows],device=device)


def preflight(O,lines,mass,gt,valid):
    obs={k:v[:8].cpu() for k,v in O.items()};l=lines[:8].cpu();m=mass[:8].cpu();g=gt[:8].cpu();v=valid[:8].cpu()
    model=Refiner(1);opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    for arm in ARMS:
        out=model(obs,l,m,arm);assert torch.equal(out['points'],obs['points'])
    horizontal=torch.tensor([[[0.,1.]]]);w=torch.ones(1,1,1);A=direction_matrix(w,horizontal)
    result=(A@torch.tensor([[[[7.],[2.]]]]))[...,0];assert torch.equal(result,torch.tensor([[[0.,2.]]]))
    norms=[]
    for step in range(8):
        out=model(obs,l,m);obj=loss(out,obs['points'],g,v);opt.zero_grad();obj.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)));opt.step()
    with torch.no_grad():
        out=model(obs,l,m);flipped=model(obs,-l,m);perm=torch.arange(35,-1,-1)
        shuffled=model(obs,l[:,perm],m[:,perm]);off=model(obs,l,m,projection='off')
        assert torch.allclose(out['points'],flipped['points'],atol=1e-4,rtol=0)
        assert torch.allclose(out['points'],shuffled['points'],atol=1e-4,rtol=0)
        assert torch.equal(off['points'],obs['points'])
        eigen=torch.linalg.eigvalsh(out['A']);assert eigen.min()>-1e-6 and eigen.max()<1.00001
        assert out['delta'].abs().max()>0
    checks=dict(status='PASS',exact_initial_yolo_parity=True,horizontal_line_no_horizontal_motion=True,
        line_sign_invariance=True,line_order_invariance=True,line_off_exact_yolo=True,PSD_direction_matrices=True,
        finite_gradient_norms=norms,smoke_steps=8,parameters=sum(p.numel() for p in model.parameters()),
        scope='Smoke uses first8 synthetic training images; state discarded. Learned attention is not a calibrated probability.')
    write(RUN/'PREFLIGHT.json',checks);print('PREFLIGHT',checks,flush=True)


def main():
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert read(RUN/'DATA_AUDIT.json')['status']=='PASS'
    data,O=observations('train','cuda:0');gt,valid=target('train','cuda:0')
    l,m=line_inputs('train',1,O);preflight(O,l,m,gt,valid)
    protocol=dict(arms=ARMS,seeds=[1,2,3],steps=STEPS,batch=BATCH,lr=3e-4,
        source_sha256={p.name:sha(p) for p in [Path(__file__),KIT/'refiner.py',KIT/'data.py']},
        target_sha256={s:sha(RUN/f'targets_{s}.json') for s in ('train','calibration','synth_val')},
        observation_manifests={s:sha(EXPORT/f'{s}.json') for s in ('train','calibration','synth_val')},
        point_sha256={s:sha(PARENT/f'points_{s}.json') for s in ('train','calibration','synth_val')},
        line_cache_sha256={f'{split}_seed{seed}':sha(RUN/f'lines_{split}_seed{seed}.npz') for split in ('train','calibration','synth_val') for seed in (1,2,3)},
        architecture='Separate token for each YOLO point: local image patches at 1/5% box diagonal, exact floating coordinates, '
          'camera-order point identity. Associate each token with 36 independently predicted DHT lines and null option. '
          'Context contains full-line sampled image features, distance, normal orientation, mass. '
          'Predict residual v; directional delta=sum(w*n*n^T)*v. No point-connecting lines.',
        controls='local: image-only residual A=I, line context zero. isotropic: same line context but A=trace(A)/2*I. '
          'donor: same directional network with wrong-image lines affinely mapped to current ROI. '
          'All same initialized full module and update/sample budget; local leaves association parameters inactive.',
        prediction_bound='v=.02*observed_box_diagonal*tanh(head); zero-initialized head gives exact YOLO output. '
          'A is PSD, eigenvalues<=1, null contributes0. No hard runtime filter.',
        loss='Original camera-order supervised point SmoothL1 beta1px + .25*relu(new_error-base_error); '
          'all valid labelled corners. No GT matching or oracle assignment during inference or primary scoring.',
        frozen='Original YOLO and v9 DHT checkpoint. Train feature projection, point tokens, line association and residual head only.',
        acceptance='Every seed on synth_val: directional conditional frame-mean point error improves >=3% over stock; '
          'median/P90 nonworse; mean no worse than local and isotropic; beats trained donor. '
          'Zero new corner catastrophes base<=5px to refined>20px. Full512 normalized failure-inclusive mean lower. '
          'No validation-based checkpoint selection or extra updates.',
        evaluation='Original camera-facing IDs, v>0. Full requested512/256 with missing matched detections7/3 '
          'retained as normalized failure1; conditional pixel metrics also reported. Reused synthetic sets, no fresh holdout/real claim.',
        optimizer_steps=STEPS*len(ARMS)*3,preflight_sha256=sha(RUN/'PREFLIGHT.json'))
    lock=RUN/'PROTOCOL.json'
    if lock.exists():raise SystemExit('protocol exists; refusing new training')
    write(lock,protocol);started=time.monotonic();binding={}
    for seed in (1,2,3):
        lines,mass=line_inputs('train',seed,O);dl,dm=line_inputs('train',seed,O,donor=True)
        rng=np.random.default_rng(seed);plan=[]
        while len(plan)<STEPS*BATCH:plan.extend(rng.permutation(len(data.records)).tolist())
        plan=np.array(plan[:STEPS*BATCH],dtype=np.int64).reshape(STEPS,BATCH);psha=hashlib.sha256(plan.tobytes()).hexdigest()
        initial=None
        for arm in ARMS:
            cell=RUN/'models'/f'{arm}_seed{seed}';cell.mkdir(parents=True,exist_ok=False)
            model=Refiner(seed).to('cuda:0');state=state_sha(model.state_dict())
            if initial is None:initial=state
            assert state==initial
            info=dict(initial_state_sha256=state,plan_sha256=psha,parameters=sum(p.numel() for p in model.parameters()),
                      arm=arm,seed=seed,steps=STEPS,batch=BATCH,protocol_sha256=sha(lock))
            write(cell/'START.json',info);binding[f'{arm}_seed{seed}']=info
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            ll,mm=(dl,dm) if arm=='donor' else (lines,mass)
            with (cell/'TRACE.jsonl').open('w') as f:
                for step in range(STEPS):
                    ii=torch.tensor(plan[step],device='cuda:0');obs={k:v[ii] for k,v in O.items()}
                    out=model(obs,ll[ii],mm[ii],arm);obj=loss(out,obs['points'],gt[ii],valid[ii])
                    opt.zero_grad(set_to_none=True);obj.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True));opt.step()
                    f.write(json.dumps(dict(step=step+1,loss=float(obj.detach()),grad_norm=gn))+'\n')
                    if (step+1)%500==0:print(arm,seed,step+1,'loss',round(float(obj),4),flush=True);f.flush()
            state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save(dict(state_dict=state,seed=seed,arm=arm,steps=STEPS),cell/'final.pt')
            write(cell/'COMPLETE.json',dict(complete=True,steps=STEPS,elapsed_seconds=time.monotonic()-tick,checkpoint_sha256=sha(cell/'final.pt')))
            del model,opt
    for seed in (1,2,3):
        for key in ('initial_state_sha256','plan_sha256','parameters'):assert len({binding[f'{a}_seed{seed}'][key] for a in ARMS})==1
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,steps=24000,bindings=binding,elapsed_seconds=time.monotonic()-started))
    print('ALL TRAINING COMPLETE',flush=True)


if __name__=='__main__':main()

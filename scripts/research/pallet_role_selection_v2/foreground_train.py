"""Final bounded control: supervise object corners only on GT-matched proposals."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from common import *
from model import *
from evaluate import iou
import numpy as np,time

FOLDER=OUT/'foreground_matched'
def register_fg():
    if (FOLDER/'PROTOCOL.json').exists():return read(FOLDER/'PROTOCOL.json')
    rows=read(PREVIOUS/'train_records.json');idx=[i for i,r in enumerate(rows) if iou(r['pred_box'],r['box'],r['raw_hw'])>=.5]
    assert len(idx)==1400
    o=dict(reason='136/1536 top proposals have GT IoU<.5. Supervising their corner IDs as if they were aligned object proposals is a confound for the fixed-proposal selector.',
        supervision='Use GT IoU>=.5 for training foreground assignment only. No confidence threshold, no selftraining, no pseudolabels. All test detections/misses still scored.',
        train_indices=idx,train_records_sha256=sha(PREVIOUS/'train_records.json'),
        preceding_summary_sha256=sha(OUT/'mean_risk_followup'/'SUMMARY.json'),
        arms=['direct','selected'],seeds=[1,2,3],steps=1200,batch=32,lr=.0003,wd=.0001,
        selector='Same MSE normalized candidate error; fixed2px switching cost.',
        residual='Same16px residual and supervision. Both branches initialized from same seed, trained on the matched subset; no loaded proposal-specific checkpoint.',
        fresh_scene_seed_base=1051000,observations=576,
        primary='all-frame normalized error on unseen meshes; compare selected against direct and stock',
        acceptance='Every seed>=2pct vs stock AND direct; paired scene CI upper<0; no stock<=10px to>20px cases; each seen/unseen condition<=1.02stock',
        bounded_stop='No further model/loss/threshold sweep in this experiment; report supported and unsupported claims explicitly.')
    write(FOLDER/'PROTOCOL.json',o);return o

def main():
    cfg=register_fg();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    idx=cfg['train_indices'];z={k:v[idx].cuda() for k,v in torch.load(PREVIOUS/'train.pt',weights_only=True).items()}
    rows=read(PREVIOUS/'train_records.json');gt=torch.tensor(np.array([rows[i]['gt'] for i in idx]),dtype=torch.float32,device='cuda')
    cells={};start=time.time()
    for seed in [1,2,3]:
        rng=np.random.default_rng(seed);plan=[]
        while len(plan)<1200*32:plan.extend(rng.permutation(len(idx)).tolist())
        plan=np.array(plan[:1200*32]).reshape(1200,32)
        for arm in ['direct','selected']:
            m=SelectorResidual('A_direct' if arm=='direct' else 'C_select_residual',seed).cuda()
            assert torch.equal(m({k:v[:4] for k,v in z.items()})['points'],z['stock'][:4])
            opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.0001);history=[]
            for step,ii in enumerate(plan):
                opt.zero_grad(set_to_none=True);o=m({k:v[ii] for k,v in z.items()})
                error=(o['candidates'][:,:,:8]-gt[ii,None]).norm(dim=-1).mean(-1)
                sel=F.mse_loss(o['risk'],error/o['scale'][:,None])
                best=error.argmin(-1) if arm=='selected' else torch.zeros(len(ii),dtype=torch.long,device='cuda')
                xy=o['allpoints'][torch.arange(len(ii),device='cuda'),best,:8]
                res=F.smooth_l1_loss(xy,gt[ii],beta=4)/16
                loss=res+(sel if arm=='selected' else sel*0);assert torch.isfinite(loss);loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),10,error_if_nonfinite=True);opt.step()
                if (step+1)%400==0:
                    history.append(dict(step=step+1,loss=float(loss.detach()),selector=float(sel.detach()),residual=float(res.detach())))
                    print('FG',seed,arm,history[-1],flush=True)
            path=FOLDER/'models'/f'{arm}_s{seed}.pt';path.parent.mkdir(exist_ok=True)
            torch.save({k:v.cpu() for k,v in m.state_dict().items()},path)
            cells[f'{arm}_s{seed}']=dict(history=history,plan_sha256=hashlib.sha256(plan.tobytes()).hexdigest(),checkpoint_sha256=sha(path))
    write(FOLDER/'TRAIN_COMPLETE.json',dict(cells=cells,optimizer_steps=7200,elapsed_seconds=time.time()-start,source_sha256=sha(KIT/'foreground_train.py')))
if __name__=='__main__':main()

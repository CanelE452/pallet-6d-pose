"""Synthetic-only D rule selection with exact P objective and tie rule."""
import math
import numpy as np
import torch
from env import *
from direct_residual_control import DirectResidualControl,decode_direct,REF
@torch.no_grad()
def cache():
    verify();assert read(DOC/'D_TRAINING_AUDIT.json')['complete'];assert not gpu()['foreign_compute']
    data=dataset();bindings={}
    for s in (1,2,3):
        path=RAW/f'validation_D{s}.npz';ckp=RAW/f'runs/D{s}/last.pt'
        ck=torch.load(ckp,map_location='cpu',weights_only=False);assert ck['complete'] and ck['step']==6000
        if not path.exists():
            model=DirectResidualControl(**ck['config']).cuda().eval();model.load_state_dict(ck['model_state_dict']);delta=[];support=[]
            for start in range(0,len(data.validation_rows),16):
                out=fwd(model,data.batch(data.validation_rows[start:start+16]));assert torch.isfinite(out['delta_normalized']).all()
                delta.append(out['delta_normalized'].cpu().numpy());support.append(out['point_support'].cpu().numpy())
            with path.open('xb') as f:np.savez(f,rows=data.validation_rows,delta=np.concatenate(delta),support=np.concatenate(support),checkpoint_sha256=sha(ckp))
            del model;torch.cuda.empty_cache()
        arr=np.load(path);assert str(arr['checkpoint_sha256'])==sha(ckp)
        bindings[str(s)]=bound(path);print('D_VALIDATION_CACHED',s,flush=True)
    write(DOC/'D_VALIDATION_CACHE.json',dict(complete=True,bindings=bindings))
class Inputs:
    def __init__(self):
        self.data=dataset();self.arrays=self.data.arrays
        self.logs={s:np.load(RAW/f'validation_D{s}.npz') for s in (1,2,3)}
        self.rowmap={int(r):i for i,r in enumerate(self.data.validation_rows)}
        for s,a in self.logs.items():assert np.array_equal(a['rows'],self.data.validation_rows) and str(a['checkpoint_sha256'])==sha(RAW/f'runs/D{s}/last.pt')
    def evaluate(self,s,partition,rules):
        rows=np.flatnonzero(self.data.partitions==partition);result=[[] for _ in rules];legacy=old('select_synthetic')
        for start in range(0,len(rows),64):
            selected=rows[start:start+64];a={k:np.array(v[selected],copy=True) for k,v in self.arrays.items() if k not in ('p3','p4')}
            t={k:torch.from_numpy(a[k]) for k in ('points','boxes','point_valid','gt_points','gt_valid')};ix=[self.rowmap[int(r)] for r in selected]
            boxes=t['boxes'];valid=torch.isfinite(boxes).all(-1)&(boxes[:,2:]>boxes[:,:2]).all(-1)
            safe=torch.where(valid[:,None],boxes,boxes.new_tensor([0,0,1,1]));diag=(safe[:,2:]-safe[:,:2]).norm(dim=-1).clamp_min(1)
            o=dict(points_raw=t['points'],box_diagonal=diag,delta_normalized=torch.from_numpy(self.logs[s]['delta'][ix]),point_support=torch.from_numpy(self.logs[s]['support'][ix]))
            records=[self.data.source['records'][int(self.data.indices[r])] for r in selected];gain=a['gain'].reshape(-1)
            for i,rule in enumerate(rules):
                fraction=rule['max_move_image_diagonal_fraction'];cap=None if fraction is None else fraction*np.array([math.hypot(*r['raw_shape_hw']) for r in records])*gain
                q=decode_direct(o,rule['lam'],cap).numpy()
                result[i].extend(legacy.frame_metrics(q,a['point_valid'],a['gt_points'],a['gt_valid'],gain,records,a['matched'],a['detected'],a['matched_gt_index']))
        return result
def select():
    verify();inp=Inputs();protocol=read(DOC/'D_PROTOCOL_LOCK.json');rules=protocol['selection_rules'];legacy=old('select_synthetic');dst=DOC/'D_SELECTION.json'
    if not dst.exists():
        byseed={s:inp.evaluate(s,'selection',rules) for s in (1,2,3)};candidates=[]
        for i,r in enumerate(rules):
            values=np.array([[v['score'] for v in byseed[s][i]] for s in (1,2,3)])
            candidates.append(dict(**r,score=float(values.mean(0).mean()),per_seed={str(s):legacy.summarize(byseed[s][i]) for s in (1,2,3)}))
        freeze(dst,dict(complete=True,no_real_selection=True,selection_population='synth_val',selected_rule=legacy.choose_rule(candidates),candidates=candidates,
            checkpoints={str(s):sha(RAW/f'runs/D{s}/last.pt') for s in (1,2,3)},temperature=None,temperature_tuning=False,rule_count=len(rules),
            exact_P_lambda_cap_grid=True,same_objective_and_tie=True,P_selection_unchanged=True,heldout_used_for_selection=False,source_protocol_sha256=sha(DOC/'D_PROTOCOL_LOCK.json')))
    selected=read(dst);heldout={}
    for s in (1,2,3):
        rows=inp.evaluate(s,'heldout',[dict(lam=0,max_move_image_diagonal_fraction=None),selected['selected_rule']])
        heldout[f'D{s}']=legacy.summarize(rows[1]);heldout['R0']=legacy.summarize(rows[0])
    write(DOC/'D_SYNTH_HELDOUT.json',dict(complete=True,methods=heldout,selection_sha256=sha(dst)))
    print('D_SELECTION_FROZEN',selected['selected_rule'],flush=True)
if __name__=='__main__':cache() if 'cache' in sys.argv else select()

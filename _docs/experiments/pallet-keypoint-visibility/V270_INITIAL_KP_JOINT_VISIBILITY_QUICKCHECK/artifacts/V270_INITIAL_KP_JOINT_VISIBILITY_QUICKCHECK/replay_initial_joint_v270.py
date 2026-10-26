"""Portable CPU actual-head replay on five full pyramids, all1784 semantic readouts."""
import sys,json,types,time,argparse
from pathlib import Path
import numpy as np
import torch
W=Path(__file__).parent;p=W/'model_initial_joint_v270.py'
for a in [p,*p.parents]:assert not a.is_symlink()
b=p.read_bytes();m=types.ModuleType('v270_model');m.__file__=str(p);sys.modules[m.__name__]=m;exec(compile(b,str(p),'exec'),m.__dict__)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='cpu_replay_v1');args=parser.parse_args();out=m.safe(W/args.out);assert not out.exists();out.mkdir()
    start=time.time();torch.set_num_threads(2)
    seal=json.loads(m.safe(W/'CURRENT_PREDICTIONS_FROZEN.json').read_text());config=json.loads(m.safe(W/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json').read_text())
    # Public copies retain original source binding paths; locate owned byte-identical files.
    def localbind(binding):
        original=Path(binding['path']);parts=original.parts
        at=parts.index(W.name) if W.name in parts else None
        local=W.joinpath(*parts[at+1:]) if at is not None else W/'weights/BASE_R0.pt'
        assert m.binding(local)['sha256']==binding['sha256'];return local
    pred=np.load(localbind(seal['predictions_npz']),allow_pickle=False);fixture=np.load(localbind(seal['forward_replay_inputs']),allow_pickle=False)
    m.R0=W/'weights/BASE_R0.pt';assert m.binding(m.R0)['sha256']==m.R0_SHA
    results={};total=0
    for arm in m.ARMS:
        model=m.InitialKPJoint(arm,device='cpu');cp=torch.load(localbind(config['checkpoints'][arm]),map_location='cpu',weights_only=False);model.load_head_state(cp['head_state']);model.eval()
        with torch.no_grad():
            packet=model.forward_pyramid([torch.from_numpy(np.array(fixture[f'pyramid_P{k}'])) for k in (3,4,5)],with_det=True)
            anchor=torch.from_numpy(fixture['anchor_index']);selected=m.selected(packet,anchor)
            raw=packet['raw9'][torch.arange(5),anchor].numpy().reshape(5,27);E=selected['E_logit'].numpy()
        rawerr=float(np.max(np.abs(raw-fixture[arm+'_raw27'])));eerr=float(np.max(np.abs(E-fixture[arm+'_E_logit'])));assert rawerr<1e-3 and eerr<1e-3,(arm,rawerr,eerr)
        scores=packet['det']['scores'].sigmoid().amax(1);assert np.array_equal(scores.argmax(1).numpy(),fixture['anchor_index'])
        record=dict(full_pyramid_frames=5,max_raw27_absolute_error=rawerr,max_E_logit_absolute_error=eerr,candidate_anchor_exact=True)
        if arm!='XY_ONLY':
            logits=pred[arm+'_E_logit'][:,:8];prob=pred[arm+'_E_probability'][:,:8];replayed=1/(1+np.exp(-logits.astype(np.float64)))
            assert np.allclose(replayed,prob,atol=8e-8,rtol=0);assert np.array_equal(logits>=0,prob>=.5)
            record.update(all1784_threshold_labels_exact=True,readouts=1784,max_sigmoid_error=float(np.max(np.abs(replayed-prob))));total+=1784
        results[arm]=record
    m.dump(out/'REPLAY_RESULT.json',dict(PASS=True,completed=True,results=results,total_semantic_readouts=total,CPU_actual_head_frames_each=5,optimizer_steps=0,RGB_reads=0,human_GT_reads=0,elapsed_seconds=time.time()-start,code=m.binding(Path(__file__))))
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()

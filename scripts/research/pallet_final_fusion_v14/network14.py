"""Complete inference graph from a frozen YOLO observation, with no target inputs."""
import argparse
from context14 import *
from model14 import FusionEvaluator

class FinalPointLineNetwork(torch.nn.Module):
    def __init__(self,seed=1,device='cuda:0'):
        super().__init__();self.seed=seed
        self.local=local(seed,device).requires_grad_(False)
        self.base=C13.parent_model(seed,'learned_gate',device)
        self.amodal=C13.parent_model(seed,'gated_amodal',device)
        state=torch.load(V11/'models'/f'lines_joint_seed{seed}'/'final.pt',map_location='cpu',weights_only=False)
        self.dht=P.C.dht(seed,device);self.dht.load_state_dict(state['dht_state_dict']);self.dht.requires_grad_(False)
        self.line_refiner=OptionalRefiner(seed).to(device).requires_grad_(False);self.line_refiner.load_state_dict(state['state_dict'])
        self.evaluator=FusionEvaluator(seed).to(device)
        self.evaluator.load_state_dict(torch.load(RUN/'models'/f'heldout_fusion_seed{seed}'/'final.pt',map_location='cpu',weights_only=False)['state_dict'])
        self.eval()

    def forward(self,O):
        capture={};hook=self.amodal.feed.register_forward_hook(lambda m,args,out:capture.update(context=args[0]+out))
        try:
            local_points=base_points(self.local,O);b=self.base(O,local_points,'learned_gate');a=self.amodal(O,local_points,'gated_amodal')
        finally:hook.remove()
        lines,mass=predict_lines(self.dht,O);r=self.line_refiner(O,b['points'],lines,mass,'lines_joint')
        n=lines[...,:2]/lines[...,:2].norm(dim=-1,keepdim=True).clamp_min(1e-8);c=lines[...,2]/lines[...,:2].norm(dim=-1).clamp_min(1e-8)
        w=r['weights'][...,:-1];diag=(O['box'][:,2:]-O['box'][:,:2]).norm(dim=-1).clamp_min(1)
        residual=((b['points'][:,:,None]*n[:,None]).sum(-1)+c[:,None])/diag[:,None,None]
        vec=(w[...,None]*residual[...,None]*n[:,None]).sum(2);A=r['A']
        desc=torch.cat((A[...,0,0,None],A[...,1,1,None],A[...,0,1,None],vec,w.sum(-1,keepdim=True)),-1)
        rep=dict(base=b['points'],amodal=a['points'],context=capture['context'],probabilities=a['probabilities'],line_delta=r['delta'],line_descriptor=desc)
        out=self.evaluator(O,rep);out.update(lines=lines,line_weights=r['weights'],base=b['points'])
        return out

def verify_and_export():
    setup();assert read(RUN/'INFERENCE_COMPLETE.json')['complete'];rows=[];maximum=0.
    directory=RUN/'export';directory.mkdir(exist_ok=True)
    for seed in SEEDS:
        model=FinalPointLineNetwork(seed);nparams=sum(p.numel() for p in model.parameters())
        with torch.inference_mode():
            for condition in ('clean','corner20','crop20','crop40'):
                m,O=load('real',condition,'cuda:0');obs=subset(O,slice(0,8));out=model(obs)
                saved=np.load(RUN/'predictions/real'/condition/f'heldout_fusion_seed{seed}.npz');expected=saved['points'][saved['detected']][:8]
                error=float(np.max(np.abs(out['points'].cpu().numpy()-expected)));maximum=max(maximum,error)
                assert error<.001 and np.array_equal(out['choice'].cpu().numpy(),saved['choice'][saved['detected']][:8])
                rows.append(dict(seed=seed,condition=condition,frames=8,max_error_px=error))
            single=subset(O,slice(0,1))
            for _ in range(10):model(single)
            torch.cuda.synchronize();tick=time.perf_counter()
            for _ in range(30):model(single)
            torch.cuda.synchronize();ms=(time.perf_counter()-tick)*1000/30
        path=directory/f'final_seed{seed}.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=seed,architecture='FinalPointLineNetwork',
            yolo_checkpoint=str(PREP.R0.relative_to(ROOT)),yolo_sha256=PREP.R0_SHA,scope='Experimental final candidate, not production approved; input observation from canonical frozen YOLO extractor.'),path)
        rows.append(dict(seed=seed,total_post_yolo_parameters=nparams,post_yolo_batch1_mean_ms=ms,latency_scope='30 GPU-synchronized sequential forwards after10warmup; excludes YOLO, image loading and host transfer',export_sha256=sha(path)))
        del model
    write(RUN/'INTEGRATION_CHECK.json',dict(status='PASS',rows=rows,max_parity_error_px=maximum,no_gt_in_forward=True,shared_observation_interface=True))
    print('INTEGRATION PASS',maximum,flush=True)

if __name__=='__main__':verify_and_export()

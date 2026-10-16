"""Integrated risk-aware inference without GT or the unused V14 score head."""
from context16 import *
from context15 import FinalPointLineNetwork
from model16 import EvidenceEvaluator

class RepresentationOnly(torch.nn.Module):
    def forward(self,O,rep):return dict(rep=rep)

class EvidencePointLineNetwork(torch.nn.Module):
    def __init__(self,seed=1,arm='role'):
        super().__init__();self.arm=arm;self.seed=seed
        self.parent=FinalPointLineNetwork(seed).eval().requires_grad_(False)
        self.parent.evaluator=RepresentationOnly()
        self.evaluator=EvidenceEvaluator(seed).cuda()
        self.evaluator.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',map_location='cpu',weights_only=False)['state_dict'])
        self.eval()

    def forward(self,O,evidence_off=False):
        p=self.parent(O);rep=dict(p['rep'],lines=p['lines'],line_weights=p['line_weights'])
        out=self.evaluator(O,rep,self.arm,evidence_off=evidence_off);out.update(lines=p['lines'],line_weights=p['line_weights'],base=rep['base'])
        return out

def main():
    setup();assert read(RUN/'INFERENCE_COMPLETE.json')['complete'];rows=[];maximum=0.;d=RUN/'export';d.mkdir(exist_ok=True)
    for seed in SEEDS:
        for arm in ARMS:
            model=EvidencePointLineNetwork(seed,arm)
            with torch.inference_mode():
                for condition in ('clean','corner20','crop20','crop40','crop10','crop30'):
                    m,O=load('real',condition,'cuda:0');o=subset(O,slice(0,4));out=model(o)
                    saved=np.load(RUN/'predictions/real'/condition/f'{arm}_seed{seed}.npz');det=saved['detected'];err=float(np.abs(out['points'].cpu().numpy()-saved['points'][det][:4]).max());maximum=max(maximum,err)
                    assert err<.001 and np.array_equal(out['choice'].cpu().numpy(),saved['choice'][det][:4])
                    rows.append(dict(seed=seed,arm=arm,condition=condition,frames=4,max_error_px=err))
            path=d/f'{arm}_seed{seed}.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,
                architecture='EvidencePointLineNetwork',yolo_checkpoint=str(PREP.R0.relative_to(ROOT)),yolo_sha256=PREP.R0_SHA,scope='Experimental architecture; no real-driven checkpoint selection.'),path)
            rows.append(dict(arm=arm,seed=seed,parameters=sum(p.numel() for p in model.parameters()),sha256=sha(path)));del model
    write(RUN/'INTEGRATION_CHECK.json',dict(status='PASS',rows=rows,max_error_px=maximum,no_gt_forward=True,unused_v14_evaluator_removed=True))
    print('INTEGRATION PASS',maximum,flush=True)

if __name__=='__main__':main()

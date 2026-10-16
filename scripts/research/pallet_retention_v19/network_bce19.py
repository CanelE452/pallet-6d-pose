"""Integrated frozen parent/vector plus learned retention, without GT."""
from context19 import *
ARMS=("local_bce","area_bce","line_bce")
from network18 import VectorPointLineNetwork
from model19 import RetentionGate

class RetentionNetwork(torch.nn.Module):
    def __init__(self,seed=1,arm='line_gate'):
        super().__init__();self.arm=arm;self.seed=seed;self.parent=VectorPointLineNetwork(seed,'line_vector').eval().requires_grad_(False);self.gate=RetentionGate(seed).cuda();self.gate.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True)['state_dict']);self.eval()
    def forward(self,O,evidence_off=False):
        p=self.parent.parent(O);rep=dict(p['rep'],lines=p['lines'],line_weights=p['line_weights']);proposal=self.parent.evaluator(O,rep,'line_vector');rep['proposal_delta']=proposal['points']-rep['base'];out=self.gate(O,rep,self.arm.replace("_bce","_gate"),evidence_off=evidence_off);return out

def main():
    setup();rows=[];maximum=0.;alpha_max=0.;folder=RUN/'export';folder.mkdir(exist_ok=True)
    for seed in SEEDS:
        for arm in ARMS:
            model=RetentionNetwork(seed,arm)
            with torch.inference_mode():
                for condition in ('clean','corner20','crop20','crop40','crop10','crop30'):
                    m,O=load('real',condition,'cuda:0');out=model(subset(O,slice(0,4)));p=np.load(RUN/'predictions/real'/condition/f'{arm}_seed{seed}.npz');det=p['detected'];err=float(np.abs(out['points'].cpu().numpy()-p['points'][det][:4]).max());ae=float(np.abs(out['alpha'].cpu().numpy()-p['alpha'][det][:4]).max());maximum=max(maximum,err);alpha_max=max(alpha_max,ae);assert err<.001 and ae<1e-4;rows.append(dict(seed=seed,arm=arm,condition=condition,frames=4,max_error_px=err,max_alpha_error=ae))
            path=folder/f'{arm}_seed{seed}.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,architecture='RetentionNetwork',yolo_checkpoint=str(PREP.R0.relative_to(ROOT)),yolo_sha256=PREP.R0_SHA,scope='Frozen upstream, experimental scalar-retention architecture. No real-driven checkpoint selection.'),path);rows.append(dict(arm=arm,seed=seed,parameters=sum(p.numel() for p in model.parameters()),sha256=sha(path)));del model
    write(RUN/'INTEGRATION_CHECK_BCE.json',dict(status='PASS',rows=rows,max_error_px=maximum,max_alpha_error=alpha_max,no_gt_forward=True));print('INTEGRATION PASS',maximum,alpha_max,flush=True)

if __name__=='__main__':main()

"""Training-only surrogate-gradient mismatch and feature intervention diagnostics."""
from context22 import *
from model20 import CauseGate,oracle_alpha

def main():
    setup();RUN.mkdir(parents=True,exist_ok=True);assert not (RUN/'PROBE_PROTOCOL.json').exists()
    write(RUN/'PROBE_PROTOCOL.json',dict(source_sha256=sha(Path(__file__)),scope='Training-only before fitting V22. Earlier real oracle results motivate hypotheses but no new real intervention for model selection.',hypotheses=['Weighted alpha BCE has the same per-sample optimum as distance but different relative gradient weights under perpendicular residuals.','Frozen64D parent context may dominate correction judgment; test removal, not assume it is causal.']))
    rows=[]
    for seed in SEEDS:
        O,r,gt,valid,ids,kind=proposal_training(seed);d=r['proposal_delta'];b=r['base'];w=d.norm(dim=-1);target=oracle_alpha(b,d,gt);mid=b+.5*d;res=mid-gt;point_grad=.25*(res*d).sum(-1)/res.norm(dim=-1).clamp_min(1e-9);bce_grad=w*(.5-target)
        model=CauseGate(seed).cuda().eval();model.load_state_dict(torch.load(OLD.RUN/'models'/f'unit_weighted_seed{seed}'/'final.pt',weights_only=True)['state_dict']);outs={k:[] for k in ('actual','no_context','no_extra_lines')}
        with torch.no_grad():
            for first in range(0,len(ids),32):
                sl=slice(first,first+32);o=subset(O,sl);rep=subset(r,sl)
                for k in outs:
                    rr=dict(rep,context=torch.zeros_like(rep['context'])) if k=='no_context' else rep
                    outs[k].append(model(o,rr,'unit_weighted',evidence_off=k=='no_extra_lines')['points'])
        outs={k:torch.cat(v) for k,v in outs.items()};stats={}
        for typ,mask in [('all',valid),('hidden',valid&(kind==1)),('outside',valid&(kind==2))]:
            x=bce_grad[mask].abs();y=point_grad[mask].abs();top=x>=torch.quantile(x,.8);stats[typ]=dict(points=int(mask.sum()),bce_top20_actual_gradient_mass=float(y[top].sum()/y.sum()),bce_top20_bce_gradient_mass=float(x[top].sum()/x.sum()),point_error={k:float((v-gt).norm(dim=-1)[mask].mean()) for k,v in outs.items()},intervention_mean_displacement={k:float((v-outs['actual']).norm(dim=-1)[mask].mean()) for k,v in outs.items() if k!='actual'},opposite_gradient_sign_fraction=float(((point_grad[mask]*bce_grad[mask])< -1e-7).float().mean()))
        # Independent analytic example: identical optimal scale but different cost of moving.
        example=[]
        for perpendicular in (0.,20.):
            base=np.array([0.,0.]);delta=np.array([10.,0.]);truth=np.array([0.,perpendicular]);e0=np.linalg.norm(base-truth);e1=np.linalg.norm(base+delta-truth);example.append(dict(perpendicular_px=perpendicular,alpha_star=0.,same_BCE_weight=10.,full_move_harm_px=float(e1-e0)))
        rows.append(dict(seed=seed,statistics=stats,analytic_example=example));del model
    write(RUN/'PROBE.json',dict(status='PASS',rows=rows,scope='Context removal on an already-trained model is OOD sensitivity, not a causal performance claim. Same-sign gradient does not imply identical population optimum.'));print('PROBE PASS',flush=True)

if __name__=='__main__':main()

"""Training-only cause probes before choosing a corrective comparison."""
from pathlib import Path
import sys
KIT=Path(__file__).resolve().parent
sys.path.insert(0,str(KIT.parent/'pallet_retention_v19'))
from context19 import *
import context19 as T
from model19 import RetentionGate,oracle_alpha
RUN=ROOT/'data/pallet/results/pallet_cause_v20'
BLOCKS={'patch':(0,144),'parent_context':(144,208),'point_id':(208,216),'observability':(216,219),'ROI_xy':(219,221),'image_xy':(221,223),'amodal_delta':(223,225),'proposal_delta':(225,227),'proposal_length':(227,228)}

def main():
    setup();RUN.mkdir(parents=True,exist_ok=True);write(RUN/'PROBE_PROTOCOL.json',dict(scope='Training-only source feature scales and target/movement mass. No final real evaluation for design selection.',hypotheses=['Movement weighting favors large helpful corrections.','Feature block scale obscures geometric/proposal evidence.','Point-wise representation lacks trainable cross-point context.'],source_sha256=sha(Path(__file__))));rows={};scales={}
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=T.proposal_training(seed);target=oracle_alpha(rep['base'],rep['proposal_delta'],gt);movement=rep['proposal_delta'].norm(dim=-1);conditions=np.array([x.rsplit('::',1)[1] for x in ids]);rs={}
        for name,mask in [('all',valid),('outside',valid&(kind==2)),('hidden',valid&(kind==1))]+[(c+'_outside',valid&(kind==2)&torch.tensor(conditions==c,device=valid.device)[:,None]) for c in ['crop20','crop40']]:
            if not mask.any():continue
            z=target[mask];w=movement[mask];m=[]
            for tag,selected in [('stop',z<=.05),('partial',(z>.05)&(z<.95)),('full',z>=.95)]:m.append(dict(target=tag,points=int(selected.sum()),point_fraction=float(selected.float().mean()),movement_mass_fraction=float(w[selected].sum()/w.sum())))
            rs[name]=dict(points=len(z),mean_oracle_alpha=float(z.mean()),movement_weighted_oracle_alpha=float((z*w).sum()/w.sum()),groups=m)
        rows[str(seed)]=rs
        if seed==1:
            model=RetentionGate(seed).cuda().eval();model.load_state_dict(torch.load(T.RUN/'models/line_bce_seed1/final.pt',weights_only=True)['state_dict']);inputs=[];hook=model.point_encode.register_forward_pre_hook(lambda module,args:inputs.append(args[0].detach().cpu()))
            with torch.no_grad():
                for i in range(0,len(gt),32):model(subset(O,slice(i,i+32)),subset(rep,slice(i,i+32)),'line_gate')
            hook.remove();x=torch.cat(inputs)[valid.cpu()];weight=model.point_encode[0].weight.detach().cpu()
            for name,(lo,hi) in BLOCKS.items():
                z=x[:,lo:hi];contribution=z@weight[:,lo:hi].T;scales[name]=dict(rms=float(z.square().mean().sqrt()),max_abs=float(z.abs().max()),linear_contribution_rms=float(contribution.square().mean().sqrt()))
    write(RUN/'PROBE.json',dict(status='PASS',training_targets=rows,input_scales=scales));print(json.dumps(dict(targets=rows['1'],scales=scales),indent=2),flush=True)

if __name__=='__main__':main()

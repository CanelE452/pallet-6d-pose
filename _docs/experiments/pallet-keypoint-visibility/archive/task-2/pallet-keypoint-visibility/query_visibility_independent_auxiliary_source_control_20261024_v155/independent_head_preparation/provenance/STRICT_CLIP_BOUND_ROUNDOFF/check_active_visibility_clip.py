"""Appendix: invented CPU visibility gradients deliberately exceed clip5."""
from pathlib import Path
import json,hashlib
import torch
from independent_query_head import IndependentAuxiliaryQueryHead,OriginalHead,per_output_loss
from check_invented_independence import same,visstate,optimizer,digest
H=Path(__file__).resolve().parent
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.manual_seed(1552);ref=OriginalHead().float()
    models=[]
    for _ in range(2):torch.manual_seed(1552);models.append(IndependentAuxiliaryQueryHead().float())
    for m in models:same(ref.state_dict(),visstate(m))
    ropt=optimizer(ref.parameters());opts=[(optimizer(m.visibility_parameters()),optimizer(m.auxiliary_parameters()))for m in models]
    gen=torch.Generator().manual_seed(1553);G=torch.zeros(16,42);av=torch.ones(16,dtype=torch.bool)
    labels={key:torch.tensor([0.,1.]*8)for key in ('external_occlusion','target_entity','coarse_point_quality')}
    records=[]
    for step in range(1,3):
        x=torch.randn(16,1928,generator=gen);ropt.zero_grad(set_to_none=True)
        z=ref(x,G);loss=10*per_output_loss({'external_occlusion':z},{'external_occlusion':labels['external_occlusion']},av,av)['external_occlusion'];loss.backward()
        rawgrad=[p.grad.clone()for p in ref.parameters()];rn=torch.nn.utils.clip_grad_norm_(list(ref.parameters()),5)
        assert float(rn)>5;post=float(torch.linalg.vector_norm(torch.cat([p.grad.reshape(-1)for p in ref.parameters()])))
        assert post<=5;clipped=[p.grad.clone()for p in ref.parameters()];ropt.step()
        outrecords=[]
        for model,(vo,ao),scale in zip(models,opts,(1.,1.e6)):
            vo.zero_grad(set_to_none=True);ao.zero_grad(set_to_none=True);out=model(x,G);same(out['external_occlusion'],z)
            terms=per_output_loss(out,labels,av,av)
            (10*terms['external_occlusion']+scale*(terms['target_entity']+terms['coarse_point_quality'])).backward()
            for a,p in zip(rawgrad,model.visibility_parameters()):same(a,p.grad)
            vn=torch.nn.utils.clip_grad_norm_(model.visibility_parameters(),5);an=torch.nn.utils.clip_grad_norm_(model.auxiliary_parameters(),5)
            same(vn,rn);vpost=float(torch.linalg.vector_norm(torch.cat([p.grad.reshape(-1)for p in model.visibility_parameters()])))
            assert vpost<=5 and vpost==post
            for a,p in zip(clipped,model.visibility_parameters()):same(a,p.grad)
            vo.step();ao.step();same(ref.state_dict(),visstate(model));same(ropt.state_dict(),vo.state_dict())
            outrecords.append(dict(auxiliary_loss_scale=scale,visibility_preclip=float(vn),visibility_postclip=vpost,auxiliary_preclip=float(an),visibility_gradient_params_and_optimizer_state_bitexact=True))
        records.append(dict(update=step,reference_preclip=float(rn),reference_postclip=post,arms=outrecords))
    result=dict(completed=True,PASS=True,appendix_only=True,original_checks=bind(H/'GENERATED_INDEPENDENCE_CHECKS.json'),head=bind(H/'independent_query_head.py'),code=bind(__file__),invented_CPU_only=True,visibility_loss_scale=10,visibility_clip_threshold=5,visibility_clip_active_each_update=True,visibility_postclip_norm_at_most5=True,updates=2,separate_optimizers_and_separate_clipping=True,millionfold_aux_loss_cannot_change_visibility_clipped_gradients_parameters_or_AdamW_state=True,records=records,final_visibility_state_sha256=digest(visstate(models[0])),source_feature_weight_private_V112_reads_or_calls=0)
    target=H/'ACTIVE_VISIBILITY_CLIP_APPENDIX.json'
    with target.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(PASS=True,appendix=bind(target),records=records)))
if __name__=='__main__':main()

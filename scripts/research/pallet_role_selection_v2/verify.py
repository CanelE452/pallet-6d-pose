from common import *
from model import *
import numpy as np

def main():
    register();torch.set_num_threads(2)
    z={k:v[:12] for k,v in torch.load(PREVIOUS/'train.pt',weights_only=True).items()}
    checks={};edges={frozenset(e) for e in EDGES}
    checks['candidate_bijections_and_center']=all(sorted(p)==list(range(9)) and p[8]==8 for p in PERMS)
    checks['candidate_topology_preserved']=all({frozenset((p[a],p[b])) for a,b in EDGES}==edges for p in PERMS)
    checks['all_arms_initial_identity']=all(torch.equal(SelectorResidual(a,1)(z)['points'],z['stock']) for a in ARMS)
    m=SelectorResidual('C_select_residual',1);o=m(z)
    target=o['candidates'][:,1,:8].detach().clone()
    checks['large_relabel_exactly_representable']=torch.equal(o['allpoints'][:,1,:8],target)
    loss,sel,res=objective(o,target,'C_select_residual')
    risk_params=[p for n,p in m.named_parameters() if n.startswith('risk')]
    residual_params=[p for n,p in m.named_parameters() if not n.startswith('risk')]
    leak1=torch.autograd.grad(sel,residual_params,allow_unused=True,retain_graph=True)
    leak2=torch.autograd.grad(res,risk_params,allow_unused=True,retain_graph=True)
    checks['no_cross_branch_trainable_gradient']=all(g is None for g in leak1+leak2)
    loss.backward();checks['finite_gradient']=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in m.parameters())
    assert all(checks.values()),checks
    write(OUT/'PREFLIGHT.json',dict(PASS=True,checks=checks,source_sha256=sha(KIT/'verify.py')))
    print(checks)
if __name__=='__main__':main()

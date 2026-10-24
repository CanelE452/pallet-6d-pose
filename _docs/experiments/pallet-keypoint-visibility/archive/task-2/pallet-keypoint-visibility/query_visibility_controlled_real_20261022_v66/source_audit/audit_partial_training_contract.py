"""CPU-only independent algebra/normalization/guard tests; no model or cache load."""
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import ast,hashlib,json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
TRAINER=ROOT/'model/fit_partial_head.py'
PROTOCOL=ROOT/'FIT_PROTOCOL_V2.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    text=TRAINER.read_text();tree=ast.parse(text)
    functions=[x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name in ['partial_nll','domain_loss','fit']]
    cache_reads=[]
    class ReachedApprovedRead(Exception):pass
    def no_cache_read(path):cache_reads.append(str(path));raise ReachedApprovedRead()
    env=dict(torch=torch,np=np,Path=Path,json=json,sha=sha,__file__=str(TRAINER),load_cache=no_cache_read)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(TRAINER),'exec'),env)
    partial,domain=env['partial_nll'],env['domain_loss'];torch.manual_seed(20261022)
    z=torch.randn(63,6,dtype=torch.float64,requires_grad=True);a=torch.arange(1,64)
    got=partial(z,a);p=torch.softmax(z,1)
    expected=torch.stack([-torch.log(p[i,[j for j in range(6)if int(m)&(1<<j)]].sum())for i,m in enumerate(a)])
    assert torch.allclose(got,expected,atol=1e-14,rtol=1e-14)
    shifted=partial(z+1.e6,a);assert torch.allclose(got,shifted,atol=3e-10,rtol=1e-10)
    unknown=partial(z,torch.full((63,),63));unknown.sum().backward()
    assert torch.equal(unknown,torch.zeros_like(unknown))and torch.equal(z.grad,torch.zeros_like(z))
    for cls in range(6):
        mask=torch.full((63,),1<<cls);truth=torch.full((63,),cls)
        assert torch.allclose(partial(z,mask),torch.nn.functional.cross_entropy(z,truth,reduction='none'),atol=1e-14,rtol=1e-14)
    dense=torch.randn(5,6,9,9,dtype=torch.float64,requires_grad=True)
    zero=partial(dense,torch.full((5,9,9),63));zero.sum().backward()
    assert torch.equal(zero,torch.zeros_like(zero))and torch.equal(dense.grad,torch.zeros_like(dense))
    loss=torch.tensor([[1.,3.,12345.],[8.,88888.,77777.],[4.,66666.,55555.]],requires_grad=True)
    masks=torch.tensor([[1,1,63],[1,63,63],[1,63,63]]);valid=torch.ones_like(masks,dtype=torch.bool);codes=torch.tensor([0,0,1])
    assert domain(loss,masks,valid,codes)==4 # sourceA informative-cell mean4; B4.
    assert domain(loss,masks,valid,codes,per_target=True)==4.5 # A(target2,target8)mean5;B4.
    domain(loss,masks,valid,codes,per_target=True).backward()
    assert torch.equal(loss.grad[masks==63],torch.zeros_like(loss.grad[masks==63]))
    empty=domain(loss,torch.full_like(masks,63),valid,codes,per_target=True)
    assert empty==0 and torch.isfinite(empty)
    protocol=json.loads(PROTOCOL.read_text())
    for b in protocol['bindings']:assert sha(b['path'])==b['sha256'],b['path']
    assert sha(ROOT/'model/query_global_amodal_head.py')==sha(ROOT.parent/'query_visibility_amodal_global_20261022_v65/model/query_global_amodal_head.py')
    checked=[]
    with TemporaryDirectory(prefix='partial_guard_mock_',dir=ROOT/'source_audit')as tmp:
        d=Path(tmp)
        for name in ['train','val','denylist']:(d/name).write_text('MOCK_METADATA_ONLY_NO_ARRAY')
        (d/'protocol').write_text(json.dumps(dict(updates=2000,validation_every=100,seed=20261022)))
        args=SimpleNamespace(out=str(d/'out'),input_lock=str(d/'lock'),candidate='MOCK',
          train=str(d/'train'),val=str(d/'val'),denylist=str(d/'denylist'),protocol=str(d/'protocol'),
          updates=2000,val_every=100,seed=20261022)
        approved=dict(bindings=[],trainer_sha256=sha(TRAINER),model_sha256=sha(TRAINER.with_name('query_global_amodal_head.py')),
          runs={'MOCK':{**{k:dict(path=getattr(args,k),sha256=sha(getattr(args,k)))for k in ['train','val','protocol','denylist']},'out':args.out}})
        def run(case,changed_args=None,changed_lock=None,pass_guard=False):
            candidate=SimpleNamespace(**vars(args));candidate.__dict__.update(changed_args or {})
            lock=json.loads(json.dumps(approved));lock.update(changed_lock or {});Path(args.input_lock).write_text(json.dumps(lock))
            n=len(cache_reads)
            try:env['fit'](candidate)
            except ReachedApprovedRead:
                assert pass_guard and len(cache_reads)==n+1,case
            except (ValueError,AssertionError,KeyError):
                assert not pass_guard and len(cache_reads)==n,case
            else:raise AssertionError('Guard test did not stop '+case)
            checked.append(case)
        run('approved-entry-only',pass_guard=True)
        for key in ['train','val','protocol','denylist']:run('reject-unapproved-'+key+'-path',{key:str(d/'WRONG')})
        run('reject-trainer-code-SHA',changed_lock={'trainer_sha256':'0'*64})
        run('reject-model-code-SHA',changed_lock={'model_sha256':'0'*64})
        run('reject-seed-change',{'seed':17});run('reject-budget-change',{'updates':2001});run('reject-validation-period-change',{'val_every':50})
        run('reject-output-path-change',{'out':str(d/'other')});run('reject-unregistered-candidate',{'candidate':'OTHER'})
        run('reject-source-file-SHA-change',changed_lock={'bindings':[dict(path=str(d/'train'),sha256='0'*64)]})
    report=dict(schema='V66_INDEPENDENT_PARTIAL_TRAINER_CONTRACT_AUDIT',PASS=True,
      trainer_sha256=sha(TRAINER),protocol_sha256=sha(PROTOCOL),model_sha256=sha(TRAINER.with_name('query_global_amodal_head.py')),
      algebra=dict(all63_nonempty_masks_probability_sum_parity=True,all63_unknown_exact_zero_loss_and_gradient_center_and_dense=True,
        onehot_cross_entropy_parity_all6=True,logit_shift_invariance=True,source_informative_only_mean=True,
        global_unique_target_then_source_mean=True,no_information_zero_loss_and_no_divide_by_zero=True),
      pre_cache_guard_tests=checked,actual_source_or_private_NPZ_RGB_labels_scores_opened=0,
      actual_model_instantiations_optimizer_updates_GPU_PnP_calls=0,
      code_review=dict(common4domain_informative_center_partialNLL=True,first_strict_minimum_checkpoint=True,
        protocol_updates_seed_val_every_checked=True,input_code_cache_path_SHA_guard_before_load=True,
        source_train_val_identity_SHA_overlap_rejection=True,current_data_read_paths_found=False),
      qualifications=['Partial likelihood scores only permitted-state consistency; it does not establish calibrated external/self confidence.',
       'Source teachers and controlled cutouts require separate provenance/coverage reporting.',
       'Candidate training has3vs4equally weighted source domains; source mixture weight changes with adding PALLOX.',
       'Candidate selection across two completed runs is external to this trainer and must follow frozen common4domain criterion.',
       'Approved input lock must be created/bound to all real caches before fitting; this audit uses synthetic guard fixtures only.'])
    out=ROOT/'source_audit/INDEPENDENT_PARTIAL_TRAINER_AUDIT.json';assert not out.exists()
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()

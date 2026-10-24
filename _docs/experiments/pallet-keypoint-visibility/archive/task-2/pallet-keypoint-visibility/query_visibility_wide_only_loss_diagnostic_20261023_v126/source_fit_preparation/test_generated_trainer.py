"""C18 bounded derivative/gradient checks, invented tensors only; no model/source load."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json
import torch
H=Path(__file__).resolve().parent;CODE=H/'train_source_once.py'
s=importlib.util.spec_from_file_location('C18_generated',CODE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
p=importlib.util.spec_from_file_location('C17_generated_parent',m.C17);parent=importlib.util.module_from_spec(p);p.loader.exec_module(parent)

def functions(path):return {n.name:n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)}
def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)

def main(out):
 checks={};a=functions(CODE);b=functions(m.C17)
 unchanged=('join_wide','wide_pools','draw_wide64','selector18','earliest_history','wide_report','clone_cpu','capture_rng')
 checks['eight_data_sampling_selector_report_helpers_AST_identical']=all(same(a[k],b[k])for k in unchanged)
 r=a['run'];pr=b['run'];tryA=next(i for i,n in enumerate(r.body)if isinstance(n,ast.Try));tryB=next(i for i,n in enumerate(pr.body)if isinstance(n,ast.Try))
 checks['all_model_data_optimizer_sampler_seed_setup_AST_identical']=same(ast.Module(body=r.body[:tryA],type_ignores=[]),ast.Module(body=pr.body[:tryB],type_ignores=[]))
 insideA={n.name:n for n in r.body[tryA].body if isinstance(n,ast.FunctionDef)};insideB={n.name:n for n in pr.body[tryB].body if isinstance(n,ast.FunctionDef)}
 checks['predict3matrices_checkpoint_RNG_eval18_AST_identical']=all(same(insideA[k],insideB[k])for k in('checkpoint','predict','evaluate'))
 y=torch.tensor(([1.,0.]*64)+([1.]*32+[0.]*32)+([1.]*32+[0.]*32));available=torch.ones(256,dtype=torch.bool);pairs=torch.arange(48).reshape(24,2)
 z=torch.linspace(-3,4,256,requires_grad=True);total,ob,hinge,fb,wb=m.loss256(z,y,available,pairs);po=parent.loss256(z,y,available,pairs)
 checks['old_BCE_hinge_flat_and_wide_logs_exact_C17']=all(torch.equal(x,q)for x,q in zip((ob,hinge,fb,wb),po[1:]))
 expected=torch.nn.functional.binary_cross_entropy_with_logits(z[192:],y[192:]);checks['objective_exact_wide64_BCE']=torch.equal(total,expected)and torch.equal(total,wb)
 total.backward();ref=z.detach().clone().requires_grad_();torch.nn.functional.binary_cross_entropy_with_logits(ref[192:],y[192:]).backward()
 checks['old192_output_gradient_exact0']=bool(torch.count_nonzero(z.grad[:192])==0)
 checks['all256_gradient_exact_independent_wide_only']=torch.equal(z.grad,ref.grad)
 changed=z.detach().clone();changed[:192]+=19;checks['objective_invariant_to_old192_outputs']=torch.equal(m.loss256(changed,y,available,pairs)[0],total.detach())and not torch.equal(m.loss256(changed,y,available,pairs)[1],ob.detach())
 z2=torch.linspace(-3,4,256,requires_grad=True);y2=y.clone();a2=available.clone();a2[254]=False;y2[255]=-1;loss,*_=m.loss256(z2,y2,a2,pairs);loss.backward()
 checks['wide_teacherU_unavailable_grad0']=bool(z2.grad[254]==0 and z2.grad[255]==0 and torch.count_nonzero(z2.grad[:192])==0)
 # A shared parameter gradient matches an independent wide-only tensor computation.
 x=(torch.arange(256*3).reshape(256,3)%7).float();w=torch.zeros(3,requires_grad=True);loss,*_=m.loss256(x@w,y,available,pairs);loss.backward();w2=torch.zeros(3,requires_grad=True);torch.nn.functional.binary_cross_entropy_with_logits(x[192:]@w2,y[192:]).backward()
 checks['shared_parameter_gradient_wide_only_exact']=torch.equal(w.grad,w2.grad)
 body=ast.unparse(r.body[tryA]);final=body.index("save_stage(RUN, 'FINAL_', 1000, lastz");load=body.index("model.load_state_dict(best_state['state_dict'])");secondary=body.index("save_stage(RUN, '', selected")
 checks['final1000_saved_before_selected_load']=final<load<secondary and 'lastz = evaluate(step)'in body
 checks['runtime_old192_gradient_zero_guard_logged']="oldgrad_nonzero == 0"in body and 'old192_output_gradient_nonzero=oldgrad_nonzero'in body
 checks['final1000_and_selected_output_names_distinct']="FINAL_ALL55900_SOURCE_SCORES.npz"in ast.unparse(a['save_stage'])and"SELECTED_ALL55900_SOURCE_SCORES.npz"in ast.unparse(a['save_stage'])
 oldsave=next(n for n in ast.walk(b['run'])if isinstance(n,ast.Call)and ast.unparse(n.func)=='np.savez_compressed')
 newsave=next(n for n in ast.walk(a['save_scores'])if isinstance(n,ast.Call)and ast.unparse(n.func)=='np.savez_compressed')
 checks['all_source_score_schema_fields_preserved']=[k.arg for k in oldsave.keywords]==[k.arg for k in newsave.keywords]
 assert all(checks.values()),checks
 result=dict(completed=True,PASS=True,code=dict(path=str(CODE),sha256=hashlib.sha256(CODE.read_bytes()).hexdigest()),parent_C17=dict(path=str(m.C17),sha256=m.C17_SHA),runtime=dict(torch=torch.__version__,device='CPU invented tensors only'),checks={k:bool(v)for k,v in checks.items()},actual_source_parameters_features_model_forward_fit_current_V112=0)
 with Path(out).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);main(ap.parse_args().output)

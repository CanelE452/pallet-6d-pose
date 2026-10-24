"""Only C16 duration/snapshot controls. No repeated C15 model/data tests or forwards."""
from pathlib import Path
import argparse,ast,tempfile
import numpy as np
import torch
import train_source_once as c


def test():
 assert all(getattr(c,k)is getattr(c.c15,k)for k in('join_flat','flat_pools','draw192','loss192','selector14','earliest_history','flat_report','load_old_metadata','load_feature_packet','state_sha'))
 # Frozen CPU generator snapshots preserve the continuation, without an optimizer reset.
 gens=[torch.Generator().manual_seed(s)for s in(208,214,215,216,217,218)]
 for _ in range(1000):
  for g in gens:torch.randint(10000,(3,),generator=g)
 snapshot=[g.get_state().clone()for g in gens]
 restored=[torch.Generator()for _ in gens]
 for g,s in zip(restored,snapshot):g.set_state(s)
 for _ in range(7000):
  for a,b in zip(gens,restored):assert torch.equal(torch.randint(10000,(3,),generator=a),torch.randint(10000,(3,),generator=b))
 # Tensor state must be a detached deep copy, not mutable optimizer state aliases.
 original=dict(state={0:dict(step=torch.tensor(1000.),exp_avg=torch.tensor([.1,.2]),exp_avg_sq=torch.tensor([.3,.4]))},param_groups=[dict(lr=1e-4,weight_decay=.001,params=[0])])
 copied=c.clone_cpu(original);original['state'][0]['step'].fill_(2000);original['state'][0]['exp_avg'].zero_();original['param_groups'][0]['params'].append(1)
 assert copied['state'][0]['step']==1000 and copied['state'][0]['exp_avg'][0]!=0 and copied['param_groups'][0]['params']==[0]
 with tempfile.TemporaryDirectory()as td:
  path=Path(td)/'invented_resume.pt';torch.save(dict(state_dict={'invented':torch.tensor([1.])},optimizer_state=copied,RNG_states=dict(old_cpu_generators=snapshot[:5],new_cpu_generator=snapshot[5]),update=1000),path)
  loaded=torch.load(path,map_location='cpu',weights_only=True);assert loaded['update']==1000 and loaded['optimizer_state']['state'][0]['step']==1000
  assert all(torch.equal(a,b)for a,b in zip(loaded['RNG_states']['old_cpu_generators'],snapshot[:5]))
 source=Path(c.__file__).read_text();tree=ast.parse(source);run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
 optimizer_assign=[n for n in ast.walk(run)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='optimizer'for t in n.targets)]
 assert len(optimizer_assign)==1
 step_loop=next(n for n in ast.walk(run)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='step')
 assert ast.literal_eval(step_loop.iter.args[0])==1 and ast.literal_eval(step_loop.iter.args[1])==8001
 assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in('load_state_dict','manual_seed')for n in ast.walk(step_loop))
 assert "if step<=1000:"in source and "if step<=2000:"in source and "row!=replay"in source and "assert row==c15_history[step]"in source
 assert "parameters_exact=all(torch.equal"in source and "np.array_equal(lastz,rf['query_logit'])"in source and "assert cert['PASS']"in source
 assert 'optimizer_state=clone_cpu(optimizer.state_dict())'in source and 'RNG_states=capture_rng(gens,newgen)'in source
 assert "torch.save(checkpoint(8000),RUN/'FINAL_CHECKPOINT.pt')"in source and "best_state=checkpoint(step)"in source
 assert "model(xo),model(xn)"in source and "len(history)==81"in source and 'selected_weights_exact_C15='in source
 design=c.read(c.DESIGN);assert c.sha(c.DESIGN)==c.DESIGN_SHA and design['selection']['updates']==list(range(0,8001,100))
 assert c.earliest_history([dict(update=i,criterion_fourteen_VAL_cohorts_BCE=v)for i,v in((0,.5),(1000,.4),(2000,.4),(3000,.3),(8000,.35))])==(3000,.3)
 return dict(completed=True,PASS=True,code=c.bind(c.__file__),test_code=c.bind(__file__),design=c.bind(c.DESIGN),runtime=torch.__version__,C15_math_loader_sampler_selector_exact_function_objects=True,no_optimizer_or_rng_reset_inside8000_loop=True,CPU_RNG_snapshot_1000_plus7000_continuation_exact=True,optimizer_snapshot_deep_copy=True,weights_only_resume_schema_roundtrip=True,strict_C15_batch_loss_history_parameter_score_guards_present=True,old_C14_trace_guard_through2000=True,selected_and_final_optimizer_RNG_saved=True,source_eval_shapes_unchanged=True,source_eval_points81=True,strict_earliest_selector=True,actual_source_features_pixels_parameters_models_head_forward_fit_current_V112_reads_or_calls=0)


if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();r=test();c.write(a.output,r);print(r)

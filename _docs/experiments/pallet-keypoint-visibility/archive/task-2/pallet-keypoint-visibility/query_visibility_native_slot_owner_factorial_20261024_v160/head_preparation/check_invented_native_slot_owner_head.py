"""Invented tensors only: parity, arm isolation, gradient routes and clipping."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import torch
from torch.nn import functional as F

H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v160_head_under_check',H/'native_slot_owner_head.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
Head=m.NativeSlotOwnerQueryHead;Base=m.IndependentAuxiliaryQueryHead
torch.set_num_threads(1)
checks={}
def record(name, ok, **details):
    assert bool(ok),name
    checks[name]={'PASS':True,**details}
def seed_head(cls, **kwargs):
    torch.manual_seed(1550);return cls(**kwargs)
def copy_state(model):return {k:v.detach().clone() for k,v in model.state_dict().items()}
def nonzero_grad_names(model):
    return [n for n,p in model.named_parameters() if p.grad is not None and bool((p.grad!=0).any())]
def no_grad(model,prefixes):
    return all(p.grad is None for n,p in model.named_parameters() if n.startswith(prefixes))
def zero_grad(model,prefixes):
    return all(p.grad is not None and not bool((p.grad!=0).any()) for n,p in model.named_parameters() if n.startswith(prefixes))

gen=torch.Generator().manual_seed(1601)
features=torch.randn(24,1928,generator=gen,dtype=torch.float32)
geometry=torch.zeros(24,42,dtype=torch.float32)
slots=torch.arange(24,dtype=torch.int64)%8
binary=(torch.arange(24)%2).float();owner_target=torch.arange(24)%3
base=seed_head(Base);original=copy_state(base)
arms={}
rng=[]
for native,owner in [(False,False),(True,False),(False,True),(True,True)]:
    arm=seed_head(Head,native_slot_hint=native,owner_auxiliary=owner)
    arms[(native,owner)]=arm;rng.append(torch.get_rng_state().clone())
record('four_arms_same_original_initialization',all(all(torch.equal(arm.state_dict()[k],v) for k,v in original.items()) for arm in arms.values()),original_keys=len(original),seed=1550)
record('four_arms_same_RNG_consumption',all(torch.equal(rng[0],x) for x in rng[1:]))
record('original_state_keys_preserved',all(set(original)<=set(arm.state_dict()) for arm in arms.values()))
expected=base(features,geometry)
for flags,arm in arms.items():
    out=arm(features,geometry,slots)
    record(f'initial_exact_output_parity_native{int(flags[0])}_owner{int(flags[1])}',all(torch.equal(expected[k],out[k]) for k in expected),original_outputs=list(expected))
    record(f'zero_branches_native{int(flags[0])}_owner{int(flags[1])}',all(not bool(p.detach().count_nonzero()) for p in arm.new_parameters()) and not bool(out['query_visible_owner'].detach().count_nonzero()),owner_logits_shape=list(out['query_visible_owner'].shape))
record('exact_new_keys',set(arms[(False,False)].state_dict())-set(original)=={'native_hint.weight','owner_head.weight','owner_head.bias'})

arm=arms[(True,True)]
groups={'visibility':arm.visibility_parameters(),'entity':arm.entity_parameters(),'quality':arm.quality_parameters(),'new':arm.new_parameters()}
group_ids={k:{id(p) for p in ps} for k,ps in groups.items()}
record('optimizer_partitions_disjoint_and_complete',all(not group_ids[a]&group_ids[b] for a in groups for b in groups if a<b) and set.union(*group_ids.values())=={id(p) for p in arm.parameters()})
counts={k:{'tensors':len(v),'scalars':sum(p.numel() for p in v)} for k,v in groups.items()}
record('original_five_tensor_visibility_partition',counts['visibility']=={'tensors':5,'scalars':224513} and group_ids['visibility']=={id(p) for n,p in arm.named_parameters() if n.startswith(('head.','geometry.'))},partitions=counts,total_scalars=sum(p.numel() for p in arm.parameters()))
record('independent_auxiliary_accessor_unchanged',[id(p) for p in arm.auxiliary_parameters()]==[id(p) for p in groups['entity']+groups['quality']])
record('new_partition_exact_707',counts['new']=={'tensors':3,'scalars':707})
record('flags_constructor_only',arm.native_slot_hint is True and arm.owner_auxiliary is True)
try:
    arm.native_slot_hint=False
except AttributeError:
    record('public_native_flag_readonly',True)
else:
    raise AssertionError('native_slot_hint unexpectedly mutable')

# Loading original states must never absorb/overwrite any new branch tensor.
load=seed_head(Head,native_slot_hint=True,owner_auxiliary=True)
with torch.no_grad():
    base.geometry.weight.copy_(torch.randn(base.geometry.weight.shape,generator=gen)*.02)
modified=copy_state(base);new_before={k:v.clone() for k,v in load.state_dict().items() if k.startswith(('native_hint.','owner_head.'))}
load.load_original_state(modified)
record('load_full_original_state_and_nonzero_geometry_parity',all(torch.equal(load.state_dict()[k],v) for k,v in modified.items()) and all(torch.equal(base(features,geometry)[k],load(features,geometry,slots)[k]) for k in expected) and all(torch.equal(load.state_dict()[k],v) for k,v in new_before.items()))
aux_before={k:v.clone() for k,v in load.state_dict().items() if k.startswith(('entity_head.','quality_head.'))}
vis_state={k:v.clone() for k,v in original.items() if k.startswith(('head.','geometry.'))}
load.load_visibility_state(vis_state)
record('load_original_five_key_visibility_state_only',all(torch.equal(load.state_dict()[k],v) for k,v in vis_state.items()) and all(torch.equal(load.state_dict()[k],v) for k,v in aux_before.items()) and all(torch.equal(load.state_dict()[k],v) for k,v in new_before.items()))
try:load.load_visibility_state(original)
except ValueError:record('visibility_loader_rejects_nonvisibility_keys',True)
else:raise AssertionError('visibility loader accepted full auxiliary state')

# Invented nonzero weights make invariance meaningful rather than all-zero.
probe=seed_head(Head,native_slot_hint=True,owner_auxiliary=True)
with torch.no_grad():
    probe.head[0].weight.copy_(torch.randn(probe.head[0].weight.shape,generator=gen)*.0005)
    probe.head[0].bias.fill_(4.)
    probe.head[2].weight.copy_(torch.linspace(.1,1.,64).reshape(1,64));probe.head[2].bias.zero_()
    probe.native_hint.weight.copy_(torch.arange(1,9).reshape(1,8).expand(64,8)*.02)
    probe.owner_head.weight.copy_(torch.linspace(-.3,.4,192).reshape(3,64));probe.owner_head.bias.copy_(torch.tensor([.1,-.2,.3]))
    for h in [probe.entity_head,probe.quality_head]:
        h[2].weight.copy_(torch.linspace(-.2,.3,64).reshape(1,64));h[2].bias.fill_(.1)
invented=copy_state(probe)
permuted=(slots+3)%8
for native,owner in arms:
    arm=arms[(native,owner)];arm.load_state_dict(invented)
    out=arm(features,geometry,slots);per=arm(features,geometry,permuted)
    unchanged=['target_entity','coarse_point_quality','query_visible_owner']
    record(f'slot_permutation_aux_owner_invariant_native{int(native)}_owner{int(owner)}',all(torch.equal(out[k],per[k]) for k in unchanged))
    if native:
        predicted=((probe.native_hint.weight[:,permuted]-probe.native_hint.weight[:,slots]).T*probe.head[2].weight).sum(1)
        delta=per['external_occlusion']-out['external_occlusion']
        record(f'known_slot_effect_enabled_owner{int(owner)}',bool((delta.abs()>0).all()) and torch.allclose(delta,predicted,atol=3e-5,rtol=1e-5),max_delta_error=float((delta-predicted).abs().max()))
    else:
        record(f'nonzero_hint_has_no_slot_effect_when_disabled_owner{int(owner)}',torch.equal(out['external_occlusion'],per['external_occlusion']))
record('owner_aux_flag_does_not_change_diagnostic_math',all(all(torch.equal(arms[(native,False)](features,geometry,slots)[k],arms[(native,True)](features,geometry,slots)[k]) for k in ['external_occlusion','target_entity','coarse_point_quality','query_visible_owner']) for native in [False,True]))
record('owner_invariance_is_nontrivial',not torch.equal(probe(features,geometry,slots)['query_visible_owner'],probe(features+.25,geometry,slots)['query_visible_owner']))

# At zero owner initialization, first CE trains owner weights; shared gradient
# is exactly zero until owner weights become nonzero (invented below).
z=seed_head(Head,native_slot_hint=True,owner_auxiliary=True)
F.cross_entropy(z(features,geometry,slots)['query_visible_owner'],owner_target).backward()
record('zero_owner_first_gradient_behavior',zero_grad(z,('head.0.','geometry.')) and no_grad(z,('head.2.','native_hint.','entity_head.','quality_head.')) and bool((z.owner_head.weight.grad!=0).any()),nonzero_parameters=nonzero_grad_names(z))

for native in [False,True]:
    g=seed_head(Head,native_slot_hint=native,owner_auxiliary=True);g.load_state_dict(invented)
    F.cross_entropy(g(features,geometry,slots)['query_visible_owner'],owner_target).backward()
    record(f'owner_loss_routes_only_to_owner_and_shared_visual_native{int(native)}',bool((g.head[0].weight.grad!=0).any()) and no_grad(g,('head.2.','native_hint.','entity_head.','quality_head.')) and zero_grad(g,('geometry.',)),nonzero_parameters=nonzero_grad_names(g))
for native in [False,True]:
    g=seed_head(Head,native_slot_hint=native,owner_auxiliary=True);g.load_state_dict(invented)
    # Stable BCE on this invented large positive visibility still exercises all paths.
    F.binary_cross_entropy_with_logits(g(features,geometry,slots)['external_occlusion'],binary).backward()
    native_ok=bool((g.native_hint.weight.grad!=0).any()) if native else g.native_hint.weight.grad is None
    record(f'primary_loss_native_path_enabled{int(native)}',native_ok and bool((g.head[0].weight.grad!=0).any()) and bool((g.head[2].weight.grad!=0).any()) and no_grad(g,('owner_head.','entity_head.','quality_head.')),nonzero_parameters=nonzero_grad_names(g))
for output,prefix in [('target_entity','entity_head.'),('coarse_point_quality','quality_head.')]:
    g=seed_head(Head,native_slot_hint=True,owner_auxiliary=True);g.load_state_dict(invented)
    F.binary_cross_entropy_with_logits(g(features,geometry,slots)[output],binary).backward()
    active=nonzero_grad_names(g)
    record(f'{output}_gradient_independent',bool(active) and all(n.startswith(prefix) for n in active) and all(p.grad is None for n,p in g.named_parameters() if not n.startswith(prefix)),nonzero_parameters=active)

# Huge new-branch gradients cannot affect old clipping under separate lists.
base=seed_head(Base);g=seed_head(Head,native_slot_hint=True,owner_auxiliary=True)
for p in base.visibility_parameters():p.grad=torch.ones_like(p)
for p in g.visibility_parameters():p.grad=torch.ones_like(p)
for p in g.new_parameters():p.grad=torch.full_like(p,1e6)
norm_base=torch.nn.utils.clip_grad_norm_(base.visibility_parameters(),5.)
norm_new=torch.nn.utils.clip_grad_norm_(g.visibility_parameters(),5.)
torch.nn.utils.clip_grad_norm_(g.new_parameters(),5.)
record('separate_clipping_preserves_original_visibility_gradients',torch.equal(norm_base,norm_new) and all(torch.equal(a.grad,b.grad) for a,b in zip(base.visibility_parameters(),g.visibility_parameters())),preclip_visibility_norm=float(norm_new))

bad=[slots.float(),slots.reshape(-1,1),torch.full_like(slots,8),torch.full_like(slots,-1)]
for i,b in enumerate(bad):
    try:g(features,geometry,b)
    except ValueError:record(f'input_slot_validation_{i}',True)
    else:raise AssertionError('bad input native_slot accepted')
record('all_values_finite',all(torch.isfinite(x).all() for x in probe(features,geometry,slots).values()))

bind=lambda p:{'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
result={'schema':'V160_INVENTED_NATIVE_SLOT_OWNER_MEANINGFUL_CHECKS','completed':True,'PASS':True,
        'UTC':datetime.now(timezone.utc).isoformat(),'head':bind(H/'native_slot_owner_head.py'),
        'check_code':bind(Path(__file__)),'base_source':bind(m.BASE_SOURCE),
        'torch_version':torch.__version__,'device':'cpu','invented_batch':[24,1928],
        'checks':checks,'check_count':len(checks),'partition_counts':counts,
        'source_feature_or_checkpoint_bodies_read':0,'source_or_private_inputs_forwarded':0,
        'new_training_or_optimizer_steps':0,'PnP':0,'V112_or_private_access':0,
        'limits':'Architecture/gradient preparation checks use invented tensors only; no source performance or training admission claim.'}
(H/'GENERATED_MEANINGFUL_CHECKS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'PASS':True,'checks':len(checks),'partition_counts':counts,'receipt':bind(H/'GENERATED_MEANINGFUL_CHECKS.json')},indent=2))

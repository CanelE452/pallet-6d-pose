"""Code-only deterministic generation; never reads source features or weights."""
from pathlib import Path
import hashlib

P = Path(__file__).resolve().parents[2]
BASE = P/'query_visibility_independent_auxiliary_source_control_20261024_v155/source_fit_preparation/train_independent_warm_fresh_control.py'
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == '86e534653cb7ff27fe412bed5b5a58875fbbd1f925591f85804ccebebadca6a2'
original = BASE.read_text()
fit = original[original.index('def fit(entry):'):original.index("if __name__=='__main__':")]

def replace(old, new):
    global fit
    assert fit.count(old) == 1, old[:100]
    fit = fit.replace(old, new)

replace("    e=read(entry);assert e['approved_actual_source_fit'] and e['source_only']", "    e=read(entry);assert e['approved_actual_source_fit'] is True and e['source_only'] is True")
replace("    p=read(PROTOCOL);assert p['design']==bind(DESIGN) and sha(DESIGN)==DESIGN_SHA", "    p=read(PROTOCOL);assert p['design']==bind(DESIGN) and sha(DESIGN)==DESIGN_SHA\n    assert p['pinned_V155_trainer']==bind(BASE_TRAINER) and p['pinned_V155_protocol']==bind(BASE_PROTOCOL)\n    assert p['arms']==list(ARMS) and p['updates_each']==8000 and p['seed_each']==1550\n    assert e['arms']==list(ARMS) and e['updates_each']==8000 and e['seed_each']==1550")
replace("    h=load_module('v155_saved_v148_helpers',HELPER,HELPER_SHA)\n    head=load_module('v155_separated_head',HEAD,HEAD_SHA)", "    h=load_module('v160_saved_v148_helpers',HELPER,HELPER_SHA)\n    head=load_module('v160_slot_owner_head',HEAD,p['head']['sha256'])")
replace("    frozen=[p[k] for k in ('code','design','root_policy','helper','head','parent_protocol','anchor_checkpoint','anchor_freeze',", "    frozen=[p[k] for k in ('code','design','helper','head','parent_protocol',")
replace("        'Pallox_features','Pallox_inputfacts','quality','original_head_generated_checks','original_native35')]+p['quality_labels']", "        'Pallox_features','Pallox_inputfacts','quality','original_head_generated_checks','original_native35',\n        'pinned_V155_trainer','pinned_V155_protocol','input_metadata_contract','expected_source_metadata',\n        'owner_bridge_complete','owner_labels','head_generated_checks','trainer_generated_checks','interpretation_note')]+p['quality_labels']")
replace("    allpacks=packs+[public,mr]", "    allpacks=packs+[public,mr]\n    expected_rows=lines(check(p['expected_source_metadata']))\n    metadata_by_family={name:[r for r in expected_rows if r['family']==name] for name in FAMILIES}\n    for i,(name,pack) in enumerate(zip(FAMILIES,allpacks)):\n        check_packet_native_metadata(name,pack,metadata_by_family[name],metas[i]['split'] if i<2 else None)\n    assert sum((~a['query_available']).sum() for a in allpacks)==535\n    owner_rows=lines(check(p['owner_labels']));owner=owner_metadata(owner_rows,facts)\n    owner_known=np.flatnonzero(owner>=0)\n    assert len(owner_known)==253 and np.bincount(owner[owner_known],minlength=3).tolist()==[53,52,148]\n    assert np.isin(owner_known,vi).all() and len(vi)==390\n    owner_positions=np.flatnonzero(owner[vi]>=0)\n    assert np.array_equal(vi[owner_positions],owner_known)")
replace("    xs=[torch.as_tensor(a['features'],device='cuda') for a in allpacks]", "    xs=[torch.as_tensor(a['features'],device='cuda') for a in allpacks]\n    native_slots=[torch.as_tensor(a['native_slot'].astype(np.int64),device='cuda') for a in allpacks]\n    owner_y=torch.as_tensor(owner[owner_known],dtype=torch.int64,device='cuda')")
replace("                raw=np.full((len(x),3),np.nan,np.float32)", "                raw=np.full((len(x),6),np.nan,np.float32)")
replace("            for a,x in zip(allpacks,xs):", "            for a,x,ns in zip(allpacks,xs,native_slots):")
replace("                    ii=ix[lo:lo+512];o=model(x[ii],z42[:len(ii)])\n                    raw[ii]=torch.stack([o[k] for k in ('external_occlusion','target_entity','coarse_point_quality')],1).cpu().numpy()", "                    ii=ix[lo:lo+512];o=model(x[ii],z42[:len(ii)],ns[ii])\n                    raw[ii]=torch.cat((torch.stack([o[k] for k in ('external_occlusion','target_entity','coarse_point_quality')],1),o['query_visible_owner']),1).cpu().numpy()")
replace("            np.savez_compressed(directory/f'{prefix_name}_{name}_OUTPUTS.npz',raw_three_logits=raw,", "            owner_logits=raw[:,3:].copy();owner_probs=owner_softmax(owner_logits)\n            np.savez_compressed(directory/f'{prefix_name}_{name}_OUTPUTS.npz',raw_three_logits=raw[:,:3].copy(),\n                raw_owner_three_logits=owner_logits,owner_three_probabilities=owner_probs,raw_six_logits=raw,\n                owner_class_order=np.asarray(['TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED']),")
replace("    RUN.mkdir(parents=True);started=time.monotonic();pending={};common_aux_initial=None;common_aux_final=None;common_aux_outputs=None;common_draw=None", "    RUN.mkdir(parents=True);started=time.monotonic();pending={};common_aux_initial=None;common_aux_final=None;common_aux_outputs=None;common_draw=None;common_original_initial=None;common_full_initial=None")
replace("    write(RUN/'STARTED.json',dict(completed=False,entry=bind(entry),protocol=bind(PROTOCOL),actual_updates=0,automatic_retry=False))", "    write(RUN/'STARTED.json',dict(completed=False,pid=os.getpid(),entry=bind(entry),protocol=bind(PROTOCOL),actual_updates=0,arms=list(ARMS),automatic_retry=False))")
replace("        for arm_index,arm in enumerate(ARMS):", "        for arm_index,arm in enumerate(ARMS):\n            arm_config=ARM_CONFIGS[arm_index];native_enabled=arm_config['native_slot_hint'];owner_weight=arm_config['owner_auxiliary_weight']")
replace("            model=head.IndependentAuxiliaryQueryHead().cuda().float()\n            factory_fresh_hash=h.state_sha(model)\n            if arm_index==0:\n                payload=torch.load(p['anchor_checkpoint']['path'],map_location='cpu',weights_only=True)\n                assert payload['update']==8000;model.load_visibility_state(payload['state_dict']);del payload\n            else:assert h.state_sha(model)==factory_fresh_hash", "            model=head.NativeSlotOwnerQueryHead(native_slot_hint=native_enabled,owner_auxiliary=bool(owner_weight)).cuda().float()\n            factory_fresh_hash=h.state_sha(model)\n            original_initial=original_state_sha(model)\n            if common_original_initial is None:common_original_initial=original_initial;common_full_initial=factory_fresh_hash\n            else:assert common_original_initial==original_initial and common_full_initial==factory_fresh_hash\n            assert not model.native_hint.weight.any() and not model.owner_head.weight.any() and not model.owner_head.bias.any()")
replace("            assert sum(t.numel() for t in model.parameters())==471043 and not model.geometry.weight.any()", "            assert sum(t.numel() for t in model.parameters())==471750 and not model.geometry.weight.any()")
replace("            if arm_index==0:\n                for a,b in zip(initial[:2],baselines):assert h.eq(a[:,0].copy(),b['raw_head_logit_before_U'])\n", "")
replace("                ix=a.get('feature_constructed',np.ones(len(raw),bool));assert not raw[ix,1:].any()", "                ix=a.get('feature_constructed',np.ones(len(raw),bool));assert not raw[ix,1:].any()")
replace("            write(directory/'INITIAL_PARITY.json',dict(completed=True,PASS=True,C32_visibility_old_new_raw_bitexact=(arm_index==0),fresh_visibility_factory_state_unchanged_before_fit=(arm_index==1),auxiliary_state_sha=aux_initial,\n                all4_datasets_repeated_three_outputs_bitexact=True,auxiliary_logits_initial_zero=True,initial_state=initial_hash))", "            write(directory/'INITIAL_PARITY.json',dict(completed=True,PASS=True,fresh_visibility_factory_state_unchanged_before_fit=True,all4_original_initial_state_sha=original_initial,all4_full_initial_state_sha=factory_fresh_hash,auxiliary_state_sha=aux_initial,\n                all4_datasets_repeated_six_outputs_bitexact=True,auxiliary_and_owner_logits_initial_zero=True,initial_state=initial_hash,native_slot_hint=native_enabled,owner_auxiliary_weight=owner_weight))")
replace("            vp=list(model.visibility_parameters());ap=list(model.auxiliary_parameters())\n            assert not (set(map(id,vp)) & set(map(id,ap)))\n            assert set(map(id,vp+ap))==set(map(id,model.parameters()))", "            vp=list(model.visibility_parameters());ap=list(model.auxiliary_parameters());np_=list(model.new_parameters())\n            assert len(vp)==5 and sum(t.numel() for t in vp)==224513\n            assert sum(t.numel() for t in ap)==246530 and sum(t.numel() for t in np_)==707\n            ids=[set(map(id,group)) for group in (vp,ap,np_)]\n            assert all(not(ids[i]&ids[j]) for i in range(3) for j in range(i))\n            assert set.union(*ids)==set(map(id,model.parameters()))")
replace("            optimizer_aux=torch.optim.AdamW(ap,lr=1e-4,weight_decay=.001)", "            optimizer_aux=torch.optim.AdamW(ap,lr=1e-4,weight_decay=.001)\n            optimizer_new=torch.optim.AdamW(np_,lr=1e-4,weight_decay=.001)")
replace("                    model.train();optimizer_vis.zero_grad(set_to_none=True);optimizer_aux.zero_grad(set_to_none=True)", "                    model.train();optimizer_vis.zero_grad(set_to_none=True);optimizer_aux.zero_grad(set_to_none=True);optimizer_new.zero_grad(set_to_none=True)")
replace("                    o=[model(xs[i][ix],z42[:len(ix)]) for i,ix in enumerate((oi,ni))]\n                    po=model(xs[2][pi],z42[:len(pi)]);mo=model(xs[3][vi],z42[:len(vi)])\n                    eo=model(xs[3][ei],z42[:len(ei)])['target_entity'];eo.retain_grad()\n                    qo=[model(xs[i][ix],z42[:len(ix)])['coarse_point_quality'] for i,ix in enumerate(qi)]", "                    o=[model(xs[i][ix],z42[:len(ix)],native_slots[i][ix]) for i,ix in enumerate((oi,ni))]\n                    po=model(xs[2][pi],z42[:len(pi)],native_slots[2][pi]);mo=model(xs[3][vi],z42[:len(vi)],native_slots[3][vi])\n                    eo=model(xs[3][ei],z42[:len(ei)],native_slots[3][ei])['target_entity'];eo.retain_grad()\n                    qo=[model(xs[i][ix],z42[:len(ix)],native_slots[i][ix])['coarse_point_quality'] for i,ix in enumerate(qi)]")
replace("                    loss=sum(nl)+pl+ml+el+ql", "                    original_loss=sum(nl)+pl+ml+el+ql\n                    owner_logits=mo['query_visible_owner'][owner_positions]\n                    ol=balanced_owner_ce(owner_logits if owner_weight else owner_logits.detach(),owner_y)\n                    loss=original_loss+ol if owner_weight else original_loss")
replace("                    grad_vis=torch.nn.utils.clip_grad_norm_(vp,5);grad_aux=torch.nn.utils.clip_grad_norm_(ap,5)\n                    assert torch.isfinite(grad_vis) and torch.isfinite(grad_aux)\n                    optimizer_vis.step();optimizer_aux.step();assert not model.geometry.weight.any()", "                    grad_vis=torch.nn.utils.clip_grad_norm_(vp,5);grad_aux=torch.nn.utils.clip_grad_norm_(ap,5);grad_new=torch.nn.utils.clip_grad_norm_(np_,5)\n                    assert torch.isfinite(grad_vis) and torch.isfinite(grad_aux) and torch.isfinite(grad_new)\n                    optimizer_vis.step();optimizer_aux.step();optimizer_new.step();assert not model.geometry.weight.any()")
replace("                        auxiliary_weight=1,total_loss=float(loss.detach()),visibility_grad_norm=float(grad_vis),auxiliary_grad_norm=float(grad_aux)),allow_nan=False)+'\\n')\n                    if step%250==0:print(json.dumps(dict(arm=arm,update=step,seconds=time.monotonic()-armstart)),flush=True)", "                        auxiliary_weight=1,native_slot_hint=native_enabled,owner_auxiliary_weight=owner_weight,owner_known_indices=owner_known.tolist(),owner_class_counts=[53,52,148],owner_balanced_CE=float(ol.detach()),total_loss=float(loss.detach()),visibility_grad_norm=float(grad_vis),auxiliary_grad_norm=float(grad_aux),new_branch_grad_norm=float(grad_new)),allow_nan=False)+'\\n')\n                    if step%250==0:\n                        progress=dict(pid=os.getpid(),arm=arm,arm_index=arm_index,update=step,updates_each=8000,elapsed_seconds=time.monotonic()-started,arm_elapsed_seconds=time.monotonic()-armstart,validation_outcomes_included=False)\n                        progress_write(RUN/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)")
replace("                common_aux_final=aux_final;common_aux_outputs=[a[:,1:].copy() for a in final]", "                common_aux_final=aux_final;common_aux_outputs=[a[:,1:3].copy() for a in final]")
replace("                assert all(h.eq(a[:,1:].copy(),b) for a,b in zip(final,common_aux_outputs))", "                assert all(h.eq(a[:,1:3].copy(),b) for a,b in zip(final,common_aux_outputs))")
replace("            torch.save(dict(state_dict=h.q.clone_cpu(model.state_dict()),visibility_optimizer_state=h.q.clone_cpu(optimizer_vis.state_dict()),auxiliary_optimizer_state=h.q.clone_cpu(optimizer_aux.state_dict()),\n                update=8000,parent_C32_update=8000,seed=1550,arm=arm,auxiliary_weight=1,visibility_initialization=(\"C32\" if arm_index==0 else \"FRESH\"),architecture=bind(HEAD),", "            torch.save(dict(state_dict=h.q.clone_cpu(model.state_dict()),visibility_optimizer_state=h.q.clone_cpu(optimizer_vis.state_dict()),auxiliary_optimizer_state=h.q.clone_cpu(optimizer_aux.state_dict()),new_branch_optimizer_state=h.q.clone_cpu(optimizer_new.state_dict()),\n                update=8000,seed=1550,arm=arm,auxiliary_weight=1,visibility_initialization='FRESH',native_slot_hint=native_enabled,owner_auxiliary_weight=owner_weight,architecture=bind(HEAD),")
replace("            del model,optimizer_vis,optimizer_aux;torch.cuda.empty_cache()\n        # All held validation reports are emitted only after both fixed fits finish.", "            write(directory/'FROZEN_BEFORE_ANY_SOURCE_COMPARISON.json',dict(completed=True,checkpoint=bind(checkpoint),update=8000,arm=arm,source_comparison_done=False,selected_model=None))\n            del model,optimizer_vis,optimizer_aux,optimizer_new;torch.cuda.empty_cache()\n        # All source validation reports are emitted only after all four fixed fits freeze.\n        write(RUN/'ALL4_FROZEN_BEFORE_SOURCE_COMPARISON.json',dict(completed=True,arms=list(ARMS),checkpoints={arm:item['checkpoint'] for arm,item in pending.items()},common_original_initial_state=common_original_initial,common_full_initial_state=common_full_initial,common_draw_sha=common_draw,source_validation_reports_emitted=False))")
replace("            report=dict(completed=True,arm=arm,update=8000,auxiliary_weight=1,visibility_initialization=(\"C32\" if ARMS.index(arm)==0 else \"FRESH\"),native=nr,Pallox=pr,MR=mrrep,", "            owner_report=owner_diagnostic(raw[3][:,3:],owner,mr['query_available'])\n            report=dict(completed=True,arm=arm,update=8000,auxiliary_weight=1,visibility_initialization='FRESH',native_slot_hint=ARM_CONFIGS[ARMS.index(arm)]['native_slot_hint'],owner_auxiliary_weight=ARM_CONFIGS[ARMS.index(arm)]['owner_auxiliary_weight'],owner_diagnostic=owner_report,native=nr,Pallox=pr,MR=mrrep,")
replace("                model_freeze=bind(directory/'MODEL_FREEZE.json'),draw_sha=item['draw_sha']))", "                model_freeze=bind(directory/'MODEL_FREEZE.json'),draw_sha=item['draw_sha']))")
replace("            write(directory/'MODEL_FREEZE.json',dict(completed=True,checkpoint=item['checkpoint'],parameters=471043,update=8000,", "            write(directory/'MODEL_FREEZE.json',dict(completed=True,checkpoint=item['checkpoint'],parameters=471750,update=8000,")
replace("            equal_auxiliary_initial_and_final_states_and_outputs_and_common_draws=True,visibility_initialization_is_only_between_arm_intervention=True,draw_sha=common_draw,entity_frame_support=entity_frame_support,", "            equal_original_initial_states_and_auxiliary_initial_final_outputs_and_common_draws=True,factorial_native_slot_and_owner_auxiliary_are_only_arm_interventions=True,all4_frozen_before_source_comparison=bind(RUN/'ALL4_FROZEN_BEFORE_SOURCE_COMPARISON.json'),draw_sha=common_draw,entity_frame_support=entity_frame_support,")

prefix = r'''"""V160 fixed source-only four-arm preparation and root-entry runner.

Pinned V155 C38 losses, draws, masks and reports are retained. Preparation only
reads code/JSON/JSONL and runs invented CPU checks. Source NPZ bodies are loaded
only inside fit after a root-approved entry; no checkpoint is loaded for init.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, hashlib, importlib.util, json, time, traceback
import numpy as np
import torch
from torch.nn import functional as F

H=Path(__file__).resolve().parent
V,P=H.parent,H.parent.parent
BASE_TRAINER=P/'query_visibility_independent_auxiliary_source_control_20261024_v155/source_fit_preparation/train_independent_warm_fresh_control.py'
BASE_TRAINER_SHA='86e534653cb7ff27fe412bed5b5a58875fbbd1f925591f85804ccebebadca6a2'
BASE_PROTOCOL=BASE_TRAINER.parent/'PROTOCOL_BEFORE_SOURCE_FEATURE_OR_CHECKPOINT_BODIES.json'
BASE_PROTOCOL_SHA='a042ca1c7daa02959f08ccc45ee65920870d40090518b8ec61c685b7e50910d8'
DESIGN=V/'ROOT_DESIGN_BEFORE_NEW_SOURCE_FEATURE_OR_MODEL_BODIES.json'
DESIGN_SHA='5b49cc780ca99fae895fe7cc785fbee74ba34f26301ee8fa5e41c1f1cabbfd2b'
HEAD=V/'head_preparation/native_slot_owner_head.py'
HEAD_CHECKS=V/'head_preparation/GENERATED_MEANINGFUL_CHECKS.json'
CONTRACT=V/'input_contract_audit/INPUT_BINDINGS_AND_NATIVE_SLOT_PROVENANCE.json'
OWNER_COMPLETE=P/'query_visibility_native_part_support_feasibility_20261024_v159/owner_supervision_bridge/COMPLETE.json'
OWNER_LABELS=OWNER_COMPLETE.parent/'ALL1840_SEPARATE_OWNER_AUXILIARY_SUPERVISION.jsonl'
INTERPRETATION=V/'ROOT_PROSPECTIVE_COMPONENT_AND_FULL_GATE_INTERPRETATION.json'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_FEATURE_OR_MODEL_BODIES.json'
RUN=V/'runs/C39_C40_C41_C42_NATIVE_SLOT_OWNER_FIXED8000'
FAMILIES=('OLD6144','NEW20480','PALLOX512','MR1840')
ARM_CONFIGS=[dict(name='C39_BASE_NATIVE0_OWNER0',native_slot_hint=False,owner_auxiliary_weight=0),dict(name='C40_NATIVE_SLOT8_OWNER0',native_slot_hint=True,owner_auxiliary_weight=0),dict(name='C41_NATIVE0_OWNER3_AUX',native_slot_hint=False,owner_auxiliary_weight=1),dict(name='C42_NATIVE_SLOT8_OWNER3_AUX',native_slot_hint=True,owner_auxiliary_weight=1)]
ARMS=tuple(a['name'] for a in ARM_CONFIGS)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def check(b):
    assert '_v112' not in b['path'].lower()
    assert sha(b['path'])==b['sha256'],b['path']
    return Path(b['path'])
def write(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def progress_write(p,obj):
    temporary=Path(str(p)+'.tmp');temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,p)
def load_module(name,path,digest):
    assert sha(path)==digest
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

base=load_module('v160_exact_pinned_v155_trainer',BASE_TRAINER,BASE_TRAINER_SHA)
HELPER,HELPER_SHA=base.HELPER,base.HELPER_SHA
pools,draw_balanced,labels_and_masks=base.pools,base.draw_balanced,base.labels_and_masks
binary_metric,mr_report=base.binary_metric,base.mr_report

def owner_metadata(rows,facts):
    assert len(rows)==len(facts)==1840
    for i,(r,f) in enumerate(zip(rows,facts)):
        for key in ('global_index','frame_index','frame_id','source_id','query_id','native_slot','split','query_xy','L'):assert r[key]==f[key],(i,key)
        assert r['owner_state'] in OWNER_MAP and r['owner_aux_target']==OWNER_MAP[r['owner_state']]
        assert not r['training_admitted'] and not r['native_visibility_target_defined'] and not r['point_quality_target_defined']
        assert not r['human_gold'] and not r['physical_native_gold'] and not r['pixel_coordinate_gold']
        if r['owner_aux_target']>=0:assert r['split']=='SOURCE_DEVELOPMENT' and r['prior_owner_observation_present'] and r['center4_valid'] is True
        if r['split']!='SOURCE_DEVELOPMENT':assert r['owner_aux_target']==-1 and not r['prior_owner_observation_present']
    target=np.asarray([r['owner_aux_target'] for r in rows],np.int64)
    assert np.bincount(target[target>=0],minlength=3).tolist()==[53,52,148]
    assert int((target<0).sum())==1587 and target[1043]==-1 and rows[1043]['center4_valid'] is False
    return target

OWNER_MAP={'TARGET_VISIBLE_SUPPORTED':0,'FOREIGN_VISIBLE_SUPPORTED':1,'BACKGROUND_SUPPORTED':2,'UNKNOWN':-1}
def balanced_owner_ce(logits,target):
    assert logits.ndim==2 and logits.shape==(len(target),3) and target.dtype==torch.int64
    assert bool(((target>=0)&(target<3)).all())
    assert all(bool((target==k).any()) for k in range(3))
    return sum(F.cross_entropy(logits[target==k],target[target==k]) for k in range(3))/3

def check_packet_native_metadata(family,pack,expected,synthetic_split=None):
    n=len(expected);assert n==len(pack['features'])
    slot=pack['native_slot'];assert slot.shape==(n,) and np.issubdtype(slot.dtype,np.integer)
    assert ((slot>=0)&(slot<8)).all()
    assert np.array_equal(slot,np.asarray([r['native_slot'] for r in expected]))
    assert np.array_equal(pack['global_index'],np.asarray([r['global_index'] for r in expected]))
    identity=pack['occurrence_id'] if family in FAMILIES[:2] else pack['query_id']
    assert identity.astype(str).tolist()==[r['query_id'] for r in expected]
    q=np.asarray([r['query_xy'] if r['query_xy'] is not None else [np.nan,np.nan] for r in expected],np.float64)
    L=np.asarray([r['L'] if r['L'] is not None else np.nan for r in expected],np.float64)
    assert np.array_equal(pack['query_xy'],q,equal_nan=True) and np.array_equal(pack['L'],L,equal_nan=True)
    assert np.array_equal(pack['query_available'],np.asarray([r['query_available'] for r in expected],bool))
    constructed=pack['feature_constructed'] if family!='PALLOX512' else pack.get('feature_constructed',np.ones(n,bool))
    assert np.array_equal(constructed,np.asarray([r['feature_constructed'] for r in expected],bool))
    split=synthetic_split if synthetic_split is not None else pack['split']
    assert np.asarray(split).astype(str).tolist()==[r['split'] for r in expected]
    if family!='PALLOX512':assert np.array_equal(pack['no_detection'],np.asarray([r['no_detection'] for r in expected],bool))
    assert not (pack['query_available'] & ~constructed).any()
    return slot

def original_state_sha(model):
    digest=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        if name.startswith(('native_hint.','owner_head.')):continue
        a=t.detach().cpu().contiguous().numpy();digest.update(name.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return digest.hexdigest()

def owner_softmax(logits):
    result=np.full_like(logits,np.nan);valid=np.isfinite(logits).all(1)
    z=logits[valid]-logits[valid].max(1,keepdims=True);e=np.exp(z);result[valid]=e/e.sum(1,keepdims=True)
    return result

def owner_diagnostic(logits,target,available):
    known=(target>=0)&available;prediction=np.argmax(logits,axis=1)
    cm=[[int((known&(target==k)&(prediction==j)).sum()) for j in range(3)] for k in range(3)]
    return dict(weak_TRAIN_only=True,known253=int(known.sum()),CM_rows_truth_T_F_BG_columns_prediction_T_F_BG=cm,held_owner_accuracy=None,held360_owner_truth_all_UNKNOWN=True,diagnostic_only_not_native_visibility_or_quality_gold=True)

def generated_adapter_checks():
    old=base.generated_adapter_checks()
    logits=torch.tensor([[.3,-.7,.2],[-.4,.6,.1],[.8,-.2,.4],[.2,.3,-.9],[.7,.1,.2],[-.8,.6,.4]],requires_grad=True)
    target=torch.tensor([0,1,1,2,2,2],dtype=torch.int64)
    state=torch.get_rng_state().clone();loss=balanced_owner_ce(logits,target);loss.backward()
    desired=F.softmax(logits.detach(),1)-F.one_hot(target,3).float()
    desired=desired*torch.tensor([1/3,1/6,1/6,1/9,1/9,1/9])[:,None]
    assert torch.allclose(logits.grad,desired,atol=1e-8,rtol=1e-6) and torch.equal(state,torch.get_rng_state())
    # Shuffled/repeated native input rules out row-modulo hint reconstruction.
    slot=np.array([6,2,6,0],np.int8);expected=[dict(global_index=i,query_id=f'invented:{i}',native_slot=int(s),query_xy=[i+.5,i+.25],L=20.,query_available=True,feature_constructed=True,split='TRAIN',no_detection=False) for i,s in enumerate(slot)]
    pack=dict(features=np.zeros((4,1928),np.float32),native_slot=slot,global_index=np.arange(4),occurrence_id=np.asarray([r['query_id'] for r in expected]),query_xy=np.asarray([r['query_xy'] for r in expected]),L=np.full(4,20.),query_available=np.ones(4,bool),feature_constructed=np.ones(4,bool),no_detection=np.zeros(4,bool))
    assert np.array_equal(check_packet_native_metadata('OLD6144',pack,expected,np.asarray(['TRAIN']*4)),slot)
    rejected=0
    for bad in (np.arange(4),slot.astype(float),np.array([6,2,8,0],np.int8)):
        badpack=dict(pack,native_slot=bad)
        try:check_packet_native_metadata('OLD6144',badpack,expected,np.asarray(['TRAIN']*4))
        except AssertionError:rejected+=1
    assert rejected==3
    probs=owner_softmax(np.array([[1.,2.,3.],[np.nan]*3],np.float32));assert np.isclose(probs[0].sum(),1) and np.isnan(probs[1]).all()
    return dict(PASS=True,invented_CPU_arrays_only=True,base_V155_adapter_checks=old,balanced_owner_CE_unequal_class_gradient_checked=True,owner_loss_no_RNG_draws=True,direct_shuffled_repeated_cached_native_metadata_checked=True,slot_modulo_float_or_out_of_range_rejected=True,owner_probability_no_detection_NaN_preserved=True,source_feature_NPZ_or_weight_bodies_read=0,actual_fit_performed=False)

def prepare():
    assert sha(DESIGN)==DESIGN_SHA and sha(BASE_PROTOCOL)==BASE_PROTOCOL_SHA
    design=read(DESIGN);assert design['arms']==ARM_CONFIGS and not design['current_allowed'] and not design['automatic_retry']
    head_checks=read(HEAD_CHECKS);assert head_checks['PASS'] and head_checks['head']==bind(HEAD)
    contract=read(CONTRACT);assert contract['PASS'] and contract['source_rows_total']==28976 and contract['owner_known_train_rows']==253
    prior=read(BASE_PROTOCOL);assert prior['code']==bind(BASE_TRAINER)
    owner_complete=read(OWNER_COMPLETE);assert owner_complete['PASS'] and not owner_complete['training_admitted']
    assert owner_complete['outputs'][0]==bind(OWNER_LABELS)
    facts=lines(check(prior['MR_inputfacts']));owner_metadata(lines(OWNER_LABELS),facts)
    expected=lines(check(contract['expected_metadata']));assert len(expected)==28976
    generated=generated_adapter_checks();write(H/'GENERATED_TRAINER_CHECKS.json',dict(code=bind(__file__),**generated))
    p=dict(prior)
    # Historical checkpoint is lineage only and is never checked or loaded by fit.
    lineage_checkpoint=p.pop('anchor_checkpoint');lineage_freeze=p.pop('anchor_freeze')
    p.pop('root_policy',None)
    p.update(completed=True,prepared_only=True,code=bind(__file__),design=bind(DESIGN),head=bind(HEAD),arms=list(ARMS),arm_configs=ARM_CONFIGS,updates_each=8000,seed_each=1550,
        pinned_V155_trainer=bind(BASE_TRAINER),pinned_V155_protocol=bind(BASE_PROTOCOL),
        historical_anchor_checkpoint_NOT_LOADED=lineage_checkpoint,historical_anchor_freeze_NOT_LOADED=lineage_freeze,
        input_metadata_contract=bind(CONTRACT),expected_source_metadata=contract['expected_metadata'],
        owner_bridge_complete=bind(OWNER_COMPLETE),owner_labels=bind(OWNER_LABELS),owner_class_counts=[53,52,148],owner_unknown_rows=1587,
        owner_loss='mean of per-class mean CE over all253 known SOURCE_DEVELOPMENT owner rows selected from existing MR vi390 forward;weight0/1;no extra RNG or forward',
        owner_primary_role_quality_independence='No relabel or gate; UNKNOWN only excluded from owner loss; original all28976 rows retained',
        owner_reviewer_provenance={'SINGLE_REVIEW_WEAK':350,'CROSS40_THREE_AGREE_WEAK':30,'CROSS40_CONSERVATIVE_UNKNOWN':10},
        head_generated_checks=bind(HEAD_CHECKS),trainer_generated_checks=bind(H/'GENERATED_TRAINER_CHECKS.json'),interpretation_note=bind(INTERPRETATION),
        all4_fresh_same_original_state=True,all4_fixed_before_source_comparison=True,
        original_visibility_AdamW_clip5_unchanged=True,original_independent_entity_quality_optimizer_clip5_unchanged=True,new707_parameters_separate_AdamW_clip5=True,
        primary_decision='original factual inputU first;otherwise visibility logit>=0;no owner/entity/quality gate',
        source_features_checkpoint_or_saved_score_bodies_read_during_preparation=False,training_admitted_by_preparation=False,actual_fit_performed=False,selected_model=None,current_allowed=False,automatic_retry=False)
    write(PROTOCOL,p)
    template=dict(approved_actual_source_fit=False,source_only=True,private_allowed=False,new_encoder_R0_allowed=False,PnP_allowed=False,q_L_change_allowed=False,automatic_retry=False,
        code=bind(__file__),protocol=bind(PROTOCOL),design=bind(DESIGN),head_generated_checks=bind(HEAD_CHECKS),input_metadata_contract=bind(CONTRACT),interpretation_note=bind(INTERPRETATION),arms=list(ARMS),updates_each=8000,seed_each=1550)
    write(H/'ROOT_ACTUAL_FIT_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
    next_command=f"{p['runtime']['executable']} {Path(__file__).resolve()} --entry {H/'ROOT_APPROVED_ACTUAL_FIT_ENTRY.json'}"
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,protocol=bind(PROTOCOL),code=bind(__file__),head=bind(HEAD),head_checks=bind(HEAD_CHECKS),trainer_checks=bind(H/'GENERATED_TRAINER_CHECKS.json'),input_contract=bind(CONTRACT),entry_template=bind(H/'ROOT_ACTUAL_FIT_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_fit_performed=False,source_feature_NPZ_or_weights_read=False,root_must_approve_entry_before_feature_body_read=True,next_command_after_root_approval=next_command,automatic_retry=False))
    print(json.dumps(dict(completed=True,prepared_only=True,protocol=bind(PROTOCOL),preparation=bind(H/'PREPARATION_COMPLETE.json'),next_command_after_root_approval=next_command)),flush=True)

'''

suffix = r'''if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--entry',type=Path)
    args=parser.parse_args()
    if args.prepare and not args.entry:prepare()
    elif args.entry and not args.prepare:fit(args.entry)
    else:raise SystemExit('Use --prepare or separately root-approved --entry; no automatic retry')
'''
target=Path(__file__).with_name('train_slot_owner_factorial.py')
assert not target.exists()
target.write_text(prefix+fit+suffix)
print(target)

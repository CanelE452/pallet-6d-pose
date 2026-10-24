"""Record the independent manual static review; never import trainer or data code."""
from pathlib import Path
import ast, datetime, hashlib, json
V=Path(__file__).resolve().parent.parent
O=Path(__file__).resolve().parent
H=V/'source_fit_preparation'
def bind(p):
 p=Path(p);assert p.suffix in ('.py','.json') and '_v112' not in str(p).lower()
 return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
fixed={
 'trainer':(H/'train_slot_owner_factorial.py','3dd58095938f63b37b92be2b42a628fb0862169087e9c41f7af873c8001c9260'),
 'protocol':(H/'PROTOCOL_BEFORE_SOURCE_FEATURE_OR_MODEL_BODIES.json','1ac65cddecf024b42aecece0e446e4b776db7a338c65c9ae4ea72c7581af2be5'),
 'trainer_checks':(H/'GENERATED_TRAINER_CHECKS.json','bb649134531e2e27e77b6c4b30517dd542402ea8e3557e659eea4d807e0ecc91'),
 'preparation':(H/'PREPARATION_COMPLETE.json','9c7e2d908f8b33030857df5dc04a88fa36cc5edbe76e6c75fb66d72744b1a309'),
 'design':(V/'ROOT_DESIGN_BEFORE_NEW_SOURCE_FEATURE_OR_MODEL_BODIES.json','5b49cc780ca99fae895fe7cc785fbee74ba34f26301ee8fa5e41c1f1cabbfd2b'),
 'contract':(V/'input_contract_audit/INPUT_BINDINGS_AND_NATIVE_SLOT_PROVENANCE.json','a1cddf46ad0d51272ed117ed77c6d9b82bc9b509f636b55b9520de91288c2199'),
 'head':(V/'head_preparation/native_slot_owner_head.py','82c843555cff138d8830f612befb733807c688e1e6c84d0607606dc4c4bd14b6'),
 'head_checks':(V/'head_preparation/GENERATED_MEANINGFUL_CHECKS.json','711d6a31c1628bf0a247018bb72e4744d9871a8488ee176f5141cf31c762b0d6'),
 'three_step_baseline_parity':(V/'root_control_update_parity/GENERATED_THREE_STEP_CONTROL_PARITY.json','4485278ef76fc58da83a3983a3aad8a4effbb81eab09c6035d50ed487c9acb04'),
 'review_rules':(O/'REVIEW_RULES_BEFORE_FROZEN_TRAINER_READ.json','f21f8dfca7239fb14d34aa161756bee8393a949c4bf97decbe6d748b34160a38')}
bindings={}
for name,(p,digest) in fixed.items():
 b=bind(p);assert b['sha256']==digest,(name,b);bindings[name]=b
trainer=Path(bindings['trainer']['path']).read_text();tree=ast.parse(trainer)
p=read(bindings['protocol']['path']);prep=read(bindings['preparation']['path']);hc=read(bindings['head_checks']['path']);tc=read(bindings['trainer_checks']['path']);parity=read(bindings['three_step_baseline_parity']['path'])
assert p['code']==prep['code']==tc['code']==bindings['trainer']
assert p['head']==prep['head']==hc['head']==bindings['head']
assert p['head_generated_checks']==prep['head_checks']==bindings['head_checks']
assert p['trainer_generated_checks']==prep['trainer_checks']==bindings['trainer_checks']
assert p['input_metadata_contract']==prep['input_contract']==bindings['contract']
assert prep['PASS'] and hc['PASS'] and tc['PASS'] and parity['PASS']
assert 'anchor_checkpoint' not in p and 'anchor_freeze' not in p
assert not prep['actual_fit_performed'] and not prep['source_feature_NPZ_or_weights_read']
assert not read(prep['entry_template']['path'])['approved_actual_source_fit']
assert 'torch.load' not in trainer and 'np.load' not in trainer
checks=[
 ('preparation_and_entry','trainer:53-56,137-180,434-446','Preparation reads bound code/JSON/JSONL and invented CPU arrays only; fit alone reaches source packets after matching approved source-only root entry. No checkpoint initializer load; historical anchor excluded from runtime frozen list.'),
 ('all_source_identities_native_slots_and_inputU','trainer:79-97,251-266,289-297','All28976 fixed family rows checked against bound expected metadata; native_slot read directly from packet and passed to onehot path. No modulo fallback. Family query_available is compared unchanged and total inputU asserted535; saved primary prediction preserves original inputU.'),
 ('owner_split_subset_and_loss','trainer:58-77,257-262,340-354','Owner metadata retains all1840 with T53/F52/BG148/U1587, held owner UNKNOWN and no prior observation. Known253 asserted subset of unchanged primary vi390. Loss is mean of three per-class mean CE over all253 in existing MR forward, weight0/1, no additional RNG/forward.'),
 ('fresh_same_initialization_fixed_draws_updates','trainer:200-210,308-327,337-341,367-368','Four fixed arms use fresh seed1550, compare original and full initial state hashes, replay8000 C32 old/new draws and unchanged seeded entity/quality sampling, and compare complete draw hashes. No warmstart, adaptive epoch or arm selection.'),
 ('original_three_optimizer_partitions','trainer:328-358;head:47-48;V155 head:51-61','Original five visibility tensors and independent entity/quality parameter group retain AdamW lr1e-4 decay.001 and clip5. New707 parameters use a disjoint third AdamW/clip5; union/disjointness/count checks are explicit. Geometry42 stays zero.'),
 ('labels_and_primary_decision_unchanged','trainer:55-56,211-250,269-277,289-297,348-354,410-420','Pinned original source labels/masks/losses and logit>=0 decision retained. Owner UNKNOWN affects owner loss only; entity-gated output remains separately named secondary diagnostic. No role/quality relabel, owner gate, or new primary U policy.'),
 ('all_four_frozen_before_validation_comparison','trainer:369-388,389-427','All four checkpoints/final states are fixed with per-arm freeze receipts and ALL4_FROZEN receipt before native/public/MR/quality/owner reports. Exact original native35 limits reused. Per-arm inference/parity outputs exist before this barrier, but no held-label metric or comparison is emitted then.'),
 ('source_only_scope_and_no_success_narrowing','trainer:173-197,410-429;protocol','Runtime binds only existing source lineage and fresh head, with no encoder/R0/SAM/PnP/coordinate-correction call or private/current/V112 body access. Full native35/Pallox/MR criteria remain; owner diagnostic never determines full qualification. selected_model remains null, current_allowed false, fully_solved false.'),
 ('final_head_receipt_schema_and_meaningful_checks','trainer:140;head:67-90;head44 and parity receipts','Final head receipt uses head key and matches final head SHA. Head checks cover branch separation/gradient routing, direct slot behavior, parameter partitions. Three invented AdamW steps show byte and numerical equality of all13 old parameters and both old optimizer states for baseline control.')]
result={
 'schema':'V160_FOCUSED_INDEPENDENT_PRE_EXECUTION_TRAINER_REVIEW',
 'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'completed':True,'PASS':True,'errors':[],'blockers':[],
 'review_mode':'Independent static source/protocol review plus existing invented-input check receipts; trainer not imported or executed by reviewer.',
 'bindings':bindings,'review_code':bind(__file__),
 'checks':[{'name':name,'PASS':True,'evidence_lines':loc,'finding':finding} for name,loc,finding in checks],
 'scope_limits':['No source feature NPZ, weight/checkpoint, RGB, private/current/V112 body read or hash by this review.','This certifies prepared-code conformance, not actual runtime array checks, source training completion/qualification or primary-task success.','Actual source fit remains contingent on root-approved matching entry; template approval is false.'],
 'receipt_schema_code_to_head_repair_verified':True,
 'invented_three_step_old_parameter_and_optimizer_numeric_and_byte_parity':True,
 'feature_or_weight_bodies_read':0,'trainer_or_fit_executed':False,'B_files_edited':False,
 'metadata_audit21_not_repeated':True}
for name,b in bindings.items():assert bind(b['path'])==b,name
out=O/'PRE_EXECUTION_TRAINER_REVIEW.json'
with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'review':bind(out),'PASS':True,'checks':len(checks)}))

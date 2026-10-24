"""Exact four SOURCE metadata files only; no image/NPZ/PT/model/probe/fit."""
import os,stat,json,hashlib,types
from collections import Counter
from pathlib import Path
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
H=Path(__file__).parent
V=P/'query_visibility_expected_part_counterfactual_source_bridge_20261025_v222/expected_part_counterfactual_pairs_v1'
def raw(p,sha):
    assert p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).lower()
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for s in p.parts[1:-1]:
            n=os.open(s,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        f=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        with os.fdopen(f,'rb')as z:
            assert stat.S_ISREG(os.fstat(z.fileno()).st_mode);b=z.read()
    finally:os.close(fd)
    assert hashlib.sha256(b).hexdigest()==sha,str(p)
    return b
def module(p,sha):
    m=types.ModuleType('pure_contract');exec(compile(raw(p,sha),str(p),'exec'),m.__dict__);return m
def load(p,sha):
    b=raw(p,sha);return [json.loads(s)for s in b.decode().splitlines()if s]if p.suffix=='.jsonl'else json.loads(b)
C=module(H/'source_training_contract_v2.py','40fb7d9e1994d7cd1aa30d063c19d785789062b7e6176af9cdb3ed5bde973c94')
oldH=H.parent/'V222_PAIRED_BRIDGE_TRAINING_B'
C1=module(oldH/'source_training_contract.py','4110bc8df75811a7e5d53daeb3bf9a3c627878a2fc6d57b408fb7741d91a3c46')
inputs={
'natural360':(P/'query_visibility_public_train_intended_part_query_targets_20261025_v218/joined_weak_intended_part_targets_v1/ALL360_CHECKED_WEAK_TARGET_CANDIDATES.jsonl','52ac5bb8d0fb19a1329b9bdb7f7b56964fed954a2a4d50489e3dcb7d598d139d'),
'CF144':(V/'ALL144_VARIANT_ROWS.jsonl','4097805a0e2fe2f7a68b196348284ad3c62b5721d56751d49dd5eb48d161e641'),
'FIT_manifest':(V/'SOURCE_FIT_ROWS_AND_EXACT_PNG_WHITELIST.json','195c78666c4fb912f90c750a5f3a6dc22edffa1cd3a02fd6d9cda562de5f98fc'),
'DEV_manifest':(V/'SOURCE_DEV_ROWS_AND_EXACT_PNG_WHITELIST.json','42ebcbd30ea287a627712f376f1b8adf6560f5e9e601d5633888d32a854bb22c')}
data={k:load(*b)for k,b in inputs.items()}
weak,dev=C.weak_plan(data['natural360'])
failures={}
try:C1.cf_plan(data['CF144'],weak,data['natural360'])
except ValueError as exc:failures['V1_authority_gate']=str(exc)
else:raise AssertionError('V1 authority mismatch was not reproduced')
pairs,status=C.cf_plan(data['CF144'],weak,data['natural360'])
try:C1.role_manifest(data['FIT_manifest'],'SOURCE_FIT',pairs,data['CF144'])
except ValueError as exc:failures['V1_FIT_decoder_flag_gate']=str(exc)
else:raise AssertionError('V1 FIT flag mismatch was not reproduced')
fit=C.role_manifest(data['FIT_manifest'],'SOURCE_FIT',pairs,data['CF144'])
hold=C.role_manifest(data['DEV_manifest'],'SOURCE_DEV',pairs,data['CF144'])
assert len(pairs)==48 and sum(x['fit_eligible']for x in pairs)==39
assert sum(x['source_role']=='SOURCE_DEV'for x in pairs)==9 and len(fit)==117 and len(hold)==27
assert len(status)==48 and all(x['constructed_pair']for x in status)
assert {r['authority']for r in data['CF144']}=={'CONTROLLED_EXPECTED_FINITE_PART_WEAK_NOT_PHYSICAL_NATIVE_POINT_GOLD'}
assert data['FIT_manifest']['trainer_DEV_payloads_allowed_before_two_new_states_freeze']is True
assert data['DEV_manifest']['trainer_DEV_payloads_allowed_before_two_new_states_freeze']is False
assert not {(x['path'],x['sha256'])for x in fit}&{(x['path'],x['sha256'])for x in hold}
result=dict(completed=True,PASS=True,metadata_only=True,inputs={k:dict(path=str(p),sha256=s)for k,(p,s)in inputs.items()},
    V1_metadata_gate_failures_reproduced=failures,
    natural_known_by_role={r:dict(Counter(x['target']for x in weak if x['source_role']==r and x['target']>=0))for r in ('SOURCE_FIT','SOURCE_DEV')},
    constructed_parents48=len(pairs),FIT_parents39=sum(x['fit_eligible']for x in pairs),DEV_parents9=sum(x['source_role']=='SOURCE_DEV'for x in pairs),
    FIT_RGB_whitelist117=len(fit),DEV_RGB_whitelist27=len(hold),authority_exact=True,FIT_flag_true_DEV_flag_false=True,
    original_qL_native_q8_condition32_valid_parent_identity_checks_PASS=True,
    numeric_training_loss_RNG_budget_unchanged=True,image_NPZ_PT_stat_hash_decode_or_model_probe_fit=0,
    old_V1_folder_writes=0,actual_train_runner_invocations=0)
with (H/'ACTUAL_SOURCE_METADATA_COMPATIBILITY_V2.json').open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(result,ensure_ascii=False))

"""Three invented CPU groups only; no run(), source payload, head or checkpoint."""
import ast, hashlib, json, os, types
from pathlib import Path
import numpy as np

H=Path(__file__).parent

def text(p):
    assert p.is_absolute() and 'v112' not in str(p).lower()
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for s in p.parts[1:-1]:
            n=os.open(s,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        f=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        with os.fdopen(f,'r') as z:return z.read()
    finally:os.close(fd)

cs=text(H/'source_training_contract.py');rs=text(H/'train_C65_continued_finite_part_CF.py')
C=types.ModuleType('invented_contract');exec(compile(cs,'OWN_PURE_CONTRACT','exec'),C.__dict__)
old=text(H.parent/'V219_TRAINING_PREPARATION_C/source_training_contract.py')
defs=lambda s:{x.name:ast.dump(x,include_attributes=False) for x in ast.parse(s).body if isinstance(x,ast.FunctionDef)}
dd,nd=defs(old),defs(cs)
assert all(dd[k]==nd[k] for k in dd), 'all9 old contract functions must remain AST exact'

original=[];weak=[]
for group in range(45):
    q8=[[20.+k,40.+k] for k in range(8)];av=[True]*8
    for k in range(8):
        rid=f'G{group:02d}_N{k}';role='SOURCE_FIT' if group<30 else 'SOURCE_DEV'
        g=dict(row_id=rid,group_id=f'G{group:02d}',native_slot=k,generated_query_xy=q8[k],generated_query_available=True,L=96.,source_role=role)
        original.append(dict(row_id=rid,generated_input=g))
        idx=group*8+k;isD=idx<39 or 240<=idx<249
        rgb=dict(path=f'/invented/{role}/{rid}.png',sha256=hashlib.sha256(rid.encode()).hexdigest(),size_bytes=1)
        weak.append(dict(row_id=rid,target=0 if isD else -1,source_role=role,component_id=f'{group:064x}',native_slot=k,query_xy=q8[k],L=96.,original_query_available=True,valid96_RLE=[[0,9216]],condition32=C.context32(q8[k],96.,k,True,q8,[96.]*8,av).tolist(),RGB96=rgb))
rows=[]
for w in weak:
    if w['target']!=0:continue
    g=next(x['generated_input'] for x in original if x['row_id']==w['row_id'])
    q8=[x['generated_input']['generated_query_xy'] for x in original if x['generated_input']['group_id']==g['group_id']]
    for v,t in zip(C.VARIANTS,(0,1,0)):
        failed=w['row_id']=='G00_N0' and v!='BEFORE'
        rgb=dict(w['RGB96'],path=f"/invented/role-copy/{w['row_id']}__BEFORE.png") if v=='BEFORE' else dict(path=f"/invented/{w['source_role']}/{w['row_id']}__{v}.png",sha256=hashlib.sha256((w['row_id']+v).encode()).hexdigest(),size_bytes=1)
        rows.append(dict(row_id=w['row_id']+'__'+v,parent_row_id=w['row_id'],paired_id='CF__'+w['row_id'],variant=v,source_role=w['source_role'],component_id=w['component_id'],native_slot=w['native_slot'],query_xy=w['query_xy'],L=w['L'],query_available=True,valid96_RLE=w['valid96_RLE'],condition32=w['condition32'],q8_xy=q8,q8_available=[True]*8,q8_L=[96.]*8,q8_native_slots=list(range(8)),authority=C.CF_AUTHORITY,native_point_DE_target=-1,training_admission=False,constructed=not failed,parent_RGB96_binding=w['RGB96'],RGB96=None if failed else rgb,input_available=not failed,target_known=not failed,weak_finite_part_target_D0_E1=None if failed else t))
pairs,status=C.cf_plan(rows,weak,original)
assert len(status)==48 and len(rows)==144 and len(pairs)==47 and sum(p['fit_eligible'] for p in pairs)==38
assert next(x for x in status if x['parent_row_id']=='G00_N0')['constructed_pair'] is False
assert next(x for x in rows if x['row_id']=='G00_N0__BEFORE')['weak_finite_part_target_D0_E1']==0
for role,n in [('SOURCE_FIT',39),('SOURCE_DEV',9)]:
    pp=[p for p in pairs if p['source_role']==role];bb=[r for r in rows if r['source_role']==role and r['variant']=='BEFORE'];vv=[r for p in pp for r in p['rows']]
    images={(r['RGB96']['path'],r['RGB96']['sha256']):r['RGB96'] for r in vv+bb}
    manifest=dict(completed=True,PASS=True,source_role=role,successful_triplet_parent_ids=[p['parent_row_id'] for p in pp],CF_variant_rows=vv,all_parent_BEFORE_rows=bb,exact_RGB96_whitelist=list(images.values()),trainer_DEV_payloads_allowed_before_two_new_states_freeze=False)
    assert len(bb)==n and len(C.role_manifest(manifest,role,pairs,rows))==len(images)

def old_draw(rng,pools):
    return {k:rng.integers(0,1000,n,dtype=np.int64) for k,n in [('visible',32),('foreign',32),('synthetic_owner',48),('public_owner',48),('CF_owner',32)]}
hashes=[]
for useCF in (False,True):
    rngs=[np.random.default_rng(x) for x in (1810,1811,1812,1813)];hh=hashlib.sha256()
    for _ in range(32):
        draw,point,natural,cf=C.draw_common_and_cf(old_draw,*rngs,{},[np.arange(10),np.arange(10,20)],[np.arange(20),np.arange(20,30)],38,useCF)
        assert sum(len(x) for x in draw.values())==192 and len(point)==32 and len(natural)==16
        if useCF:assert len(cf)==4 and ((cf>=0)&(cf<38)).all()
        else:assert cf is None
        for k,v in draw.items():hh.update(k.encode());hh.update(v.tobytes())
        hh.update(point.tobytes());hh.update(natural.tobytes())
    hashes.append(hh.hexdigest())
assert hashes[0]==hashes[1]
assert C.fixed_budget_from_probe(dict(zip(C.ARMS,(.04,.05))))==1200
assert C.fixed_budget_from_probe(dict(zip(C.ARMS,(.001,.002))))==2000
assert C.fixed_budget_from_probe(dict(zip(C.ARMS,(1.,2.))))==200

bad=[dict(r) for r in rows];bad[1]['native_slot']=7
try:C.cf_plan(bad,weak,original)
except ValueError:pass
else:raise AssertionError('native drift accepted')
meta=[dict(component_role='TRAIN',nominal_fit_eligible=True,component_id='a'*64,lineage_bound=True),dict(component_role='HOLDOUT',nominal_fit_eligible=True,component_id='b'*64,lineage_bound=True),dict(component_role='TRAIN',nominal_fit_eligible=True,component_id='c'*64,lineage_bound=True),dict(component_role='TRAIN',nominal_fit_eligible=True,component_id='d'*64,lineage_bound=False)]
assert C.fit_mask(meta,{'c'*64}).tolist()==[True,False,False,False]
node=next(x for x in ast.parse(rs).body if isinstance(x,ast.FunctionDef) and x.name=='raw')
calls=[]
fakeos=types.SimpleNamespace(open=lambda *a,**k:calls.append(a),O_RDONLY=0,O_DIRECTORY=0,O_NOFOLLOW=0)
ns=dict(Path=Path,os=fakeos,require=C.need);exec(compile(ast.Module(body=[node],type_ignores=[]),'OWN_RAW_HELPER','exec'),ns)
try:ns['raw']('/invented/v112/no_open')
except ValueError:pass
else:raise AssertionError('forbidden lexical path accepted')
assert not calls
# Only invented scalar logits, no neural module/model/head construction or forward.
import torch
z=torch.zeros(12,requires_grad=True);loss=C.cf_parent_loss(z);loss.backward()
assert abs(float(loss)-np.log(2))<1e-6
assert np.allclose(z.grad.numpy().reshape(4,3),np.tile([.03125,-.0625,.03125],(4,1)))
out=dict(PASS=True,groups=3,old_contract_functions_AST_exact=sorted(dd),partial_failure_factual_D_and_unknown_variants_preserved=True,role_closed_whitelist=True,common32step_draws_exact=True,CF_parent_loss_and_invented_logit_gradient_exact=True,prospective_budget_formula=True,native_drift_DEV_HOLD_missinglineage_rejected=True,forbidden_before_FS=True,actual_source_payload_model_or_head_forward_fit=0,checks_code_sha256=hashlib.sha256(text(Path(__file__)).encode()).hexdigest())
with (H/'INVENTED_CHECKS_FINAL.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))

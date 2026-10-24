"""Bounded independent code/metadata + invented-logit preflight, no source assets."""
import ast
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

V=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_dense_point_weak_bridge_source_training_20261025_v212')
H=V/'independent_preflight_c';A=V/'fit_preparation_a'
def need(x,m):
    if not bool(x):raise AssertionError(m)
def safe(p):
    p=Path(p);need('v112'not in str(p).lower()and'..'not in p.parts and p.is_absolute(),'Lexical guard')
    need(p.suffix in('.py','.json','.jsonl','.md'),'Text/code only')
    need(not any(x.is_symlink()for x in(p,*p.parents)),'No symlinks');return p
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(safe(p).read_bytes()).hexdigest())
def read(b):
    need(bind(b['path'])==b,'Exact binding');z=safe(b['path']).read_text()
    return [json.loads(x)for x in z.splitlines()]if Path(b['path']).suffix=='.jsonl'else json.loads(z)
def functions(b,names,ns):
    need(bind(b['path'])==b,'Exact code');tree=ast.parse(safe(b['path']).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
    need(len(nodes)==len(names),'Only requested pure functions');exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen_functions_only','exec'),ns);return ns

def main():
    r=read(bind(A/'FINAL_READY.json'));need(bind(A/'FINAL_READY.json')['sha256']=='1ef23fb991e042b26c1670154f1764a8dd34c7a6eec8e943bd68f6574163a073','Exact final READY')
    p=read(r['protocol']);need(p['trainer_code']==r['code']and p['target_contract']==r['target_contract'],'Final protocol code bindings')
    for b in p['source_bindings'].values():need(bind(b['path'])==b,'Frozen inherited source code')
    checks=read(r['checks']);need(checks['PASS']and checks['trainer_code']==r['code']and checks['target_contract']==r['target_contract']and checks['checks']==21,'A authored21 CPU checks exact')
    pa=read(p['point_admission']);wa=read(p['weak_admission']);weak=read(r['weak_metadata']);future=read(r['payload_whitelist']);false=read(r['false_entry'])
    need(pa['approved_prospective_SOURCE_point_teaching']and wa['approved_SOURCE_weak44_bridge'],'Separate source admissions')
    need(pa['eligibility_amendment']==p['eligibility_amendment']and wa['weak_metadata']==r['weak_metadata'],'Exact prospective target authority')
    need(false['approved_dense_point_weak_bridge_source_fit']is False and false['root_scope']['path']is None,'No automatic actual fit')
    need(len(future['files'])==58 and future['oldDIRECT_NPZ']==0,'No old DIRECT binary')
    need(Counter(Path(b['path']).suffix for b in future['files'])=={'.npz':6,'.png':52},'Only6 old input/target +52 weak PNG declarations')
    need(all('v112'not in b['path'].lower()for b in future['files']),'Future literals checked, never followed')
    ns=dict(np=np,math=math,need=need,META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available'))
    functions(r['target_contract'],{'finite_pair','point_loss_target','join_point_rows','balanced_pools','balanced_loss','objective'},ns)
    mapped=read(pa['exact_teacher_mapping']);need(len(mapped)==7040,'All admitted mapping metadata retained')
    results=[ns['point_loss_target'](m)for m in mapped]
    need(Counter(t for t,_ in results)=={0:5853,1:243,-1:944},'Independent final target-function count')
    need(all(t==-1 for m,(t,_)in zip(mapped,results)if m['point0_combined_state']not in('D','E')or not m['original_query_available']),'S/U/UNKNOWN never E')
    need(wa['counts']==dict(rows=44,D=26,E=18)and wa['weak_loss_weight']==.25,'Separate low weight weak branch')
    need(len(weak['rows'])==44 and len(weak['frames'])==3 and len({m['row_id']for m in weak['rows']})==44,'44 weak occurrences /3 source frames')
    need(Counter(m['variant']for m in weak['rows'])=={'FACTUAL':8,'SHAM':18,'COVER':18},'Factual once including failed cases; only18 actual pairs')
    need(Counter(m['proposed_weak_target']for m in weak['rows'])=={0:26,1:18},'Weak D/E only')
    need(len({(m['rgb96_sha256_expected'],m['valid96_sha256_expected'])for m in weak['rows']})==44,'Declared44 effective input hashes distinct, payload not reopened')
    for m in weak['rows']:
        group=weak['frames'][m['frame_q8_key']]['metadata8'];need([z['native_slot']for z in group]==list(range(8)),'Full original q8')
        active=group[m['native_slot']]
        need(m['component_role']=='TRAIN'and m['original_query_available']is True,'Only weak TRAIN/available')
        for k in('native_slot','query_xy','L','original_query_available'):need(m[k]==active[k],'Original weak active q8 join')
        need(m['source_row_id']==active['row_id']and m['source_original_global_index']==active['original_global_index'],'Source original identity')
        need(m['authority']=='ASSISTANT_SOURCE_LOCAL_VISIBLE_FEATURE_WEAK_NOT_NATIVE_GOLD','No native gold alias')
    # Independent invented logits exercise actual objective/gradient slices.
    oldns=dict(torch=torch,np=np,F=F,require=need)
    functions(p['source_bindings']['old_sampler_loss'],{'five_losses'},oldns)
    tests=[]
    def ok(name,value):need(value,name);tests.append(name)
    def bad(name,f):
        try:f()
        except (AssertionError,KeyError,ValueError,TypeError):tests.append(name);return
        raise AssertionError(name)
    sup=torch.zeros((240,2),requires_grad=True);owner=torch.zeros((240,3),requires_grad=True);direct=torch.zeros(240,requires_grad=True)
    pt=torch.tensor([0]*16+[1]*16,dtype=torch.int8);wt=torch.tensor([0]*8+[1]*8,dtype=torch.int8)
    loss,aux,_,pl,wl=ns['objective'](dict(support_logits=sup,owner_logits=owner,direct_end_external_logit=direct),pt,wt,oldns,True)
    ok('new_objective_exact_five_mean_point_quarter_weak',torch.equal(loss,aux+pl+.25*wl))
    loss.backward();expected=torch.zeros(240);expected[192:208]=1/64;expected[208:224]=-1/64;expected[224:232]=1/128;expected[232:240]=-1/128
    ok('known_class_gradient_scaling_and_slice_isolation',torch.equal(direct.grad,expected)and torch.isfinite(sup.grad).all()and torch.isfinite(owner.grad).all())
    d=torch.zeros(224,requires_grad=True);out=dict(support_logits=torch.zeros((224,2)),owner_logits=torch.zeros((224,3)),direct_end_external_logit=d)
    total,b,_,point,w=ns['objective'](out,pt,None,oldns,False);total.backward()
    ok('C62_no_weak_branch_and_common_point_gradients',w.item()==0 and torch.equal(d.grad,direct.grad[:224]))
    rng_hashes=[]
    for bridge in(False,True):
        old=np.random.default_rng(1810);point=np.random.default_rng(1811);wrng=np.random.default_rng(1812);h=hashlib.sha256()
        for _ in range(12):
            a=old.integers(0,40,192,dtype=np.int64);b=point.integers(0,20,32,dtype=np.int64)
            if bridge:wrng.integers(0,10,16,dtype=np.int64)
            h.update(a.tobytes());h.update(b.tobytes())
        rng_hashes.append(h.hexdigest())
    ok('weak1812_cannot_advance_common_streams',rng_hashes[0]==rng_hashes[1])
    valid=next(copy.deepcopy(m)for m,(t,_)in zip(mapped,results)if t==1)
    for field,value in [('point0_target_alone_state','S'),('point0_combined_state','S'),('point0_combined_state','U'),('point0_combined_state','UNKNOWN'),('original_query_available',False),('component_role','HOLDOUT')]:
        m=copy.deepcopy(valid);m[field]=value;ok('excluded_not_E_'+field+'_'+str(value),ns['point_loss_target'](m)[0]==-1)
    m=copy.deepcopy(valid);m['reference_locality_for_actual_input']['inside_original_frame_canvas_extent']=False
    ok('frame_outside_not_E',ns['point_loss_target'](m)[0]==-1)
    m=copy.deepcopy(valid);m['source_teacher_key']=[999999,m['native_slot']]
    cache=[{k:valid[k]for k in ns['META']}];cache[0].update(source_image_original_global_index=valid['source_image_original_global_index'],anchor_original_global_index=valid['anchor_original_global_index'])
    bad('image_teacher_key_cannot_alias_anchor_or_cache',lambda:ns['join_point_rows'](cache,[m],np.ones(1,bool),np.ones(1,bool)))
    # Actual root gate is tested before its first metadata/payload access.
    gate=dict(json=json,need=need,rb=lambda _:json.dumps(dict(approved_dense_point_weak_bridge_source_fit=False)).encode())
    functions(r['code'],{'fit'},gate);bad('false_root_gate_stops_before_any_body',lambda:gate['fit']('/unopened_entry'))
    # Verify input/architecture leaves and order of BOTH-final barrier statically.
    source=safe(r['code']['path']).read_text();tree=ast.parse(source);fit=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='fit')
    calls=[(n.lineno,ast.unparse(n.func))for n in ast.walk(fit)if isinstance(n,ast.Call)]
    barrier_line=next(n.lineno for n in ast.walk(fit)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='barrier'for t in n.targets))
    first_save=min(line for line,name in calls if name=='base.save_outputs')
    ok('both_final_barrier_precedes_source_predictions',barrier_line<first_save and 'model.eval().requires_grad_(False)'in source)
    ok('same_C61_factory_and_fresh_no_SAM_recipe','cls(spatial_q8=True)'in source and 'torch.manual_seed(1810)'in source and 'range(1,4001)'in source and "np.random.default_rng(1812)"in source and "module(p['inherited_trainer']"in source)
    review=dict(completed=True,PASS=True,code=r['code'],protocol=r['protocol'],target_contract=r['target_contract'],preparation=bind(A/'FINAL_READY.json'),independent_checker=bind(__file__),
        admissions=dict(point=p['point_admission'],weak=p['weak_admission']),metadata=dict(point7040=pa['exact_teacher_mapping'],weak44=r['weak_metadata'],future_payloads=r['payload_whitelist']),
        independently_checked_point_counts={'D':5853,'E':243,'UNKNOWN_loss':944,'all_rows':7040},weak_counts={'D':26,'E':18,'rows':44,'frames':3,'PNG_payloads_declared':52},
        essential_invented_checks=tests,A_authored_CPU21_checks_reused=r['checks'],
        conclusions=['Same exact C61 RAW8/native/mask/q8 spatial factory118359; GT reference/alpha/role/targets never passed to model.',
            'Point main target replaces old DIRECT pool. Target-aloneD/combinedDE/available/frame/actual crop predicate and image identity retained; S/U/outside never external.',
            'Old5 original mean/pools use immutable TRAIN & nominal fit masks; saved role cohorts use original masks. Weak44 separate weight0.25, balanced8/8 and RNG1812; old192/point16+16 shared1810/1811.',
            'Both fresh seeds1810/final4000 states precede any source outputs; saved direct decoder inherited logit>=0 with original U. Root actual entry still required.'],
        actual_SOURCE_NPZ_PNG_header_stat_hash_or_decode=0,original_GT_reopened=0,model_instantiations_or_fit_or_forward_or_optimizer_calls=0,
        limits=['Independent new objective uses invented CPU logits/gradients only. Same-factory model initialization/forward is supported by exact inherited code and A21 CPU checks, not re-run by C.',
            'Metadata hash uniqueness is declared provenance; actual weak PNG decoding/hashes and full source target joins remain runtime guards.',
            'Public weak feature authority is distinct from saved procedural native point and human/native physical gold. No protected metric/model selection.'])
    with (H/'REVIEW.json').open('x')as f:json.dump(review,f,ensure_ascii=False,indent=2);f.write('\n')
    with (H/'KO.md').open('x')as f:f.write('# V212 독립 코드·metadata preflight\n\nPASS. main point7040 중 D5853/E243/UNKNOWN loss944를 최종 pure predicate로 확인했다. image-scene teacher와 q-anchor를 분리하고 S/U·frame/crop 밖을 E로 바꾸지 않는다. weak44(D26/E18, 원3frame q8)는 별도 assistant local-feature 권위이며 0.25 추가항이다. old5 mean·immutable TRAIN/nominal fitmask, 같은 C61 RAW/native/q8 factory, fresh1810·paired old/point RNG와 독립 weak1812, 양 final4000 후 source outputs 순서를 확인했다.\n\n독립 가상 objective·gradient slice·mask·false-body-gate 검사 PASS. 원 source NPZ/PNG/header/GT/weights/보호 점수 접근0, 실제 모델/forward/fit/optimizer0이다. A의 정확21 CPU architecture/init 검사와 V211 닫힌 GT QA는 재사용했고 새로 반복했다고 주장하지 않는다. actual PNG·source array 검사는 별도 root 실제 entry 후 runtime guard가 수행한다. 새 실사 개선·gold·모델 선택을 주장하지 않는다.\n')
    print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),KO=bind(H/'KO.md'),checks=len(tests))))

if __name__=='__main__':main()

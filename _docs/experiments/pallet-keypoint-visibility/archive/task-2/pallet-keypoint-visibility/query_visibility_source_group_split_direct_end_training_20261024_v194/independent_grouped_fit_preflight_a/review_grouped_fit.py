"""Independent static/CPU-authored-logit preflight; production arrays/models=0."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,json,math,os,stat
import numpy as np
import torch
from torch.nn import functional as F
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
B=V/'grouped_fit_preparation_b';OLD=P/'query_visibility_sampled_domain_direct_end_20261024_v189/paired_direct_end_fit_preparation_c/train_paired_direct_end.py'
ORIGINAL=P/'query_visibility_query_jitter_paired_source_training_20261024_v183/source_fit_preparation/train_jitter_paired_source.py'
CANDIDATE=P/'query_visibility_sampled_domain_direct_end_20261024_v189/direct_end_training_preparation_c/synthetic_direct_end_candidate.py'

def need(ok,msg):
    if not bool(ok):raise AssertionError(msg)

def text(path):
    p=Path(path);need(p.is_absolute()and p.is_relative_to(P)and'..'not in p.parts and all('v112'not in s.casefold()for s in p.parts),'Lexical source whitelist before access')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for name in p.parts[1:-1]:
            nxt=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        end=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(end).st_mode),'Regular code/text only')
    with os.fdopen(end,'rb')as f:return f.read()

def binding(p):return dict(path=str(p),sha256=hashlib.sha256(text(p)).hexdigest())
def read(p):return json.loads(text(p))
def nodes(p):return {n.name:n for n in ast.parse(text(p)).body if isinstance(n,ast.FunctionDef)}
def extract(p,names,ns):
    nn=nodes(p);exec(compile(ast.Module(body=[nn[k]for k in names],type_ignores=[]),str(p)+'::PURE_FUNCTIONS_ONLY','exec'),ns);return ns
def fail(fn):
    try:fn()
    except AssertionError:return True
    return False

def main():
    ready=read(B/'FINAL_READY.json');code=Path(ready['code']['path']);protocol=read(Path(ready['protocol']['path']));adapter_path=Path(ready['mask_adapter']['path'])
    for key in ('code','protocol','mask_adapter','checks','test_code'):
        need(binding(Path(ready[key]['path']))==ready[key],'Final binding '+key)
    scope=read(Path(protocol['prospective_scope']['path']));checks=read(Path(ready['checks']['path']))
    need(protocol['training']==scope['training']and protocol['split_complete']==scope['source_split'],'Prospective recipe/source split exact')
    old,new=nodes(OLD),nodes(code);shared=['pure_training','metadata_matches','load_old','load_direct','gather','assemble','decode','objective','save_outputs']
    evidence={'nine_inherited_functions_AST_identical':all(ast.dump(old[n],include_attributes=False)==ast.dump(new[n],include_attributes=False)for n in shared),'producer17_invented_tests_PASS':checks['PASS']is True and len(checks['tests'])==17 and all(checks['tests'].values())}
    c=SimpleNamespace(need=need,META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available'))
    ns=dict(np=np,torch=torch,F=F,require=need,DOMAIN_ROWS=('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816'))
    pure=extract(ORIGINAL,['pools_for','draw_step','five_losses'],ns)
    masks=extract(adapter_path,['require','fit_mask','masked_old_pools','masked_direct_pools','assert_draws_train'],dict(np=np))
    cand=extract(CANDIDATE,['require','balanced_direct_end_loss'],dict(np=np))
    current=extract(code,['join_direct_split','objective','decode'],dict(np=np,torch=torch,c=c))
    # Independent scalar expected class masses include partial known rows and
    # original unavailable rows, without row normalization or hard argmax.
    mass=np.array([[.1,.2,.3]if i%2 else[.4,.1,.1]for i in range(12)],np.float64)
    available=np.ones(12,bool);available[-1]=False
    support=np.array([[i%2,(i//2)%2]for i in range(12)],np.int8);support[-1]=-1
    caches={'SYNTH_AUG65536':dict(x=dict(original_query_available=available),y=dict(support_target=support,owner_soft=mass)),'PUBLIC_TRAIN1480':dict(x=dict(original_query_available=np.ones(253,bool)),y=dict(support_target=np.full((253,2),-1,np.int8),owner_target=np.repeat(np.arange(3,dtype=np.int8),[53,52,148]))),'CF3816':dict(x=dict(original_query_available=np.ones(314,bool)),y=dict(support_target=np.full((314,2),-1,np.int8),owner_target=np.repeat(np.arange(2,dtype=np.int8),157)))}
    tm={k:np.arange(len(v['x']['original_query_available']))%3!=0 for k,v in caches.items()}
    p=masks['masked_old_pools'](caches,tm,pure['pools_for']);need(p['feasible'],'Invented full five-loss classes feasible')
    for j,prob in enumerate(p['pools']['conditional']):
        denom=math.fsum(float(mass[i,j])for i in range(12)if available[i]and tm['SYNTH_AUG65536'][i])
        expect=[float(mass[i,j])/denom if available[i]and tm['SYNTH_AUG65536'][i]else 0. for i in range(12)]
        need(np.allclose(prob,expect,rtol=0,atol=2e-16),'Independent raw partial-known classmass conditional')
    evidence['all_five_masks_and_partial_mass_scalar_CDF']=True
    av=np.array([True,True,True,True,False,True]);target=np.array([0,0,1,1,-1,-1],np.int8);tm['DIRECT_CONTROL']=np.array([True,False,True,False,True,True]);d=masks['masked_direct_pools'](av,target,tm['DIRECT_CONTROL'])
    need([v.tolist()for v in d['pools']]==[[0],[2]],'Only TRAIN+available+known direct rows')
    sequences=[]
    for _ in range(2):
        rng=np.random.default_rng(1810);drng=np.random.default_rng(1811);seq=[]
        for step in range(64):
            draw=pure['draw_step'](rng,p['pools']);di=np.concatenate([drng.choice(v,16,replace=True)for v in d['pools']]);masks['assert_draws_train'](draw,di,tm);seq.append({k:v.tolist()for k,v in draw.items()}|{'direct':di.tolist()})
        sequences.append(seq)
    need(sequences[0]==sequences[1]and all(sum(len(v)for k,v in r.items()if k!='direct')==192 and len(r['direct'])==32 for r in sequences[0]),'Identical paired192+32 streams')
    injection={k:np.asarray(v)for k,v in sequences[0][0].items()if k!='direct'};injection['visible'][0]=0
    need(fail(lambda:masks['assert_draws_train'](injection,np.asarray(sequences[0][0]['direct']),tm)),'HOLDOUT draw rejected')
    evidence['all_six_draw_routes_TRAIN_only_paired_seed_and_U']=True
    def r(k):return dict(row_id='g/native%d'%k,original_global_index=7,native_slot=k,query_xy=[10.+k,20.],L=384.,original_query_available=True)
    packet=dict(path='INVENTED_NO_FILE',sha256='0'*64)
    plan=[dict(r(k),cache_index=k,control_join_key=['REAL_SEMANTIC_END_WEAK','A','SHAM',k],source_packet=packet,authority='SOURCE_RGB_SEMANTIC_END_WEAK',stored_weak_direct_target=0 if k==0 else-1,role='TRAIN'if k==0 else'HOLDOUT')for k in range(2)]
    meta=[dict(r(k),cache_index=i,domain='REAL_SEMANTIC_END_WEAK',attempt_id='A',variant='SHAM',control_join_key=plan[k]['control_join_key'],source_packet=packet,authority=plan[k]['authority'],root_admitted=True)for i,k in enumerate((1,0))]
    aligned=current['join_direct_split'](meta,plan,np.array([-1,0],np.int8));need(masks['fit_mask'](meta,aligned).tolist()==[False,True],'Order-independent direct key and exact META')
    meta[0]['L']=383.;need(fail(lambda:masks['fit_mask'](meta,aligned)),'F64 q/L drift refused')
    evidence['reordered_direct_key_join_and_exact_META6']=True
    support_logits=torch.linspace(-1,1,448,dtype=torch.float32).reshape(224,2).requires_grad_();owner_logits=torch.linspace(-.8,.9,672,dtype=torch.float32).reshape(224,3).requires_grad_();direct_logits=torch.linspace(-.2,.3,224,dtype=torch.float32).requires_grad_();y=torch.tensor([0]*16+[1]*16,dtype=torch.int8)
    total,base,parts,dl=current['objective'](dict(support_logits=support_logits,owner_logits=owner_logits,direct_end_external_logit=direct_logits),y,pure,SimpleNamespace(balanced_direct_end_loss=cand['balanced_direct_end_loss']))
    binary=torch.tensor([1.]*16+[0.]*16);labels3=torch.arange(3).repeat_interleave(16);labels2=torch.arange(2).repeat_interleave(16)
    independent=[F.binary_cross_entropy_with_logits(support_logits[:32,0],binary),F.binary_cross_entropy_with_logits(support_logits[32:64,1],binary),F.cross_entropy(owner_logits[64:112],labels3),F.cross_entropy(owner_logits[112:160],labels3),F.cross_entropy(owner_logits[160:192],labels2)]
    direct_expected=.5*(F.binary_cross_entropy_with_logits(direct_logits[192:208],torch.zeros(16))+F.binary_cross_entropy_with_logits(direct_logits[208:],torch.ones(16)))
    need(torch.equal(base,torch.stack(independent).mean())and torch.equal(dl,direct_expected)and torch.equal(total,base+direct_expected),'Independent five-mean+balanced-direct weight1 arithmetic')
    total.backward();need(bool((support_logits.grad[192:]==0).all()and(owner_logits.grad[192:]==0).all()and(direct_logits.grad[:192]==0).all()and(direct_logits.grad[192:]!=0).all()),'No cross-task direct label leakage into old outputs')
    evidence['independent_sixloss_CPU_arithmetic_and_gradient_routes']=True
    fit=new['fit'];lines=text(code).decode().splitlines();barrier=next(i for i,s in enumerate(lines,1)if "barrier=out/"in s);emission=next(i for i,s in enumerate(lines,1)if 'full=save_outputs('in s)
    need(barrier<emission and not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='torch'and n.func.attr=='load'for n in ast.walk(ast.parse(text(code)))),'Both8000 barrier before source predictions; no checkpoint load')
    evidence['static_both8000_freeze_before_any_source_prediction']=True
    need(all(evidence.values()),'All independent preflight checks')
    result=dict(completed=True,PASS=True,code=ready['code'],protocol=ready['protocol'],mask_adapter=ready['mask_adapter'],prospective_scope=protocol['prospective_scope'],producer_checks=ready['checks'],review_code=binding(Path(__file__)),checks=evidence,blocking_issues=[],scope='Static code/text and authored CPU-logit/mask fixtures only; no production cache/header/array/weight/head-model/fit/evaluation or source-image reads.',limits=['Full source target containers are validated by runtime before masking; HOLDOUT values cannot enter any optimization pool/draw/loss or outcome selection.','Actual target-class feasibility/source-cache bytes and all16000 live updates are not audited by this preflight.','Grouped holdout is source-only and historically known; direct heldout labels are synthetic, not real end/native gold or transfer proof.'],production_array_header_weight_model_fit_or_eval_calls=0)
    with(H/'REVIEW.json').open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,review=binding(H/'REVIEW.json'),checks=len(evidence)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()

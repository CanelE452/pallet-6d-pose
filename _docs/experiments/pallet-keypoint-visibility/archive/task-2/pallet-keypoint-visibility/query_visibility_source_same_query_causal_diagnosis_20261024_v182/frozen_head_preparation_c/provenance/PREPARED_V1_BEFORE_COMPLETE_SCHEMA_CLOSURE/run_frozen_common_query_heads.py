"""V182 source-TRAIN common-COVER-query diagnostic; frozen heads only.

--prepare reads code/text metadata and invented CPU tensors only. Actual entry
loads separately bound common pixels/SAM predictions/V181 fixed checkpoints.
Targets, source GT buffers, labels, optimizer and fitting are never used.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,math,os,sys,time,traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;P=ROOT.parent
V181=P/'query_visibility_local_support_raw_head_20261024_v181'
SCOPE=ROOT/'ROOT_SCOPE_BEFORE_NEW_COMMON_QUERY_OR_PUBLIC_PIXELS.json'
SCOPE_SHA='142ba2fd62dfa16c79bda690bf90814d0a9cdd330898722b8f896edf76b396a1'
HEAD=V181/'head_preparation/local_support_head.py'
HEAD_SHA='14a40c4dd14d416c1648e62b408fd7acdf74f06d4113c69785bd737b45aa848b'
SAMPLER=V181/'raw_cache_preparation/build_source_raw96_cache.py'
SAMPLER_SHA='01b6be15ad7941384ef5a209b397ef1a0b240dec8b403564e5e54c40a702225f'
TRAINER=V181/'source_fit_preparation/train_local_support_ablation.py'
FIT=V181/'runs/C50_C51_LOCAL_SUPPORT_FIXED8000_v2/COMPLETE.json'
OLD_CACHE=V181/'source_raw96_cache_v1/COMPLETE.json'
SLOTS=ROOT/'common_query_preparation_a/COMMON4096_SLOTS_INPUT_ONLY.jsonl'
SLOTS_SHA='17fdddce08d0da37609e850963ab2ec55cc3a37a1a836c7d852cd59e08aeb75a'
GROUPS=ROOT/'common_query_preparation_a/COMMON512_SAM_GROUPS_INPUT_ONLY.jsonl'
GROUPS_SHA='b6a256150e33713448fed45818bab080b2b68af9cc1ceb98e7f90f062ccf93f9'
PROTOCOL=HERE/'PROTOCOL_BEFORE_COMMON_INPUT_SAM_OR_FROZEN_HEAD_WEIGHT_BODIES.json'
PYTHON='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
ARMS=[('C50_RAW96_DUAL_SUPPORT_OWNER_AUX',False),('C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX',True)]
META=['row_id','original_global_index','native_slot','query_xy','L','original_query_available']
INPUT_KEYS=set(META)|{'rgb96','valid96'}
SAM_KEYS={'image_logits','lowres_logits','predicted_quality','mask_logits_gt0','box_xyxy','original_query_xy','original_L','native_slot','original_global_index','original_occurrence_id','original_query_available','prediction_available'}
USED={}

def require(ok,msg):
    if not bool(ok):raise AssertionError(msg)
def permitted(p):
    p=Path(p).resolve();require(p.is_relative_to(P),'Outside fixed source project')
    require(not any(s in str(p).lower() for s in ('_v112','_v169','_v177','_v178','current223','/private_','synth_val','validation2048','source_validation')),'Forbidden non-TRAIN or private path')
    return p
def sha(p):
    h=hashlib.sha256()
    with permitted(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def binding(p):return {'path':str(permitted(p)),'sha256':sha(p)}
def checked(b):
    p=permitted(b['path']);require(sha(p)==b['sha256'],'SHA '+str(p));USED[str(p)]=b['sha256'];return p
def readj(p):return json.loads(permitted(p).read_text())
def rows(p):return [json.loads(x) for x in permitted(p).read_text().splitlines() if x]
def writej(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def writerows(p,rr):
    with Path(p).open('x') as f:
        for r in rr:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def qarray(q):return np.asarray([np.nan,np.nan] if q is None else q,np.float64)
def side(L):return np.nan if L is None else float(L)
def same(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and (np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b))
def load_npz(b,keys,exact=None):
    with np.load(checked(b),allow_pickle=False) as z:
        if exact is not None:require(set(z.files)==set(exact),'Exact NPZ whitelist')
        return {k:z[k].copy() for k in keys}

def pure_helpers():
    # AST-only functions. No source producers, inventory loaders or label code.
    tree=ast.parse(SAMPLER.read_text());names={'require','same','qarray','validq','grid96','sample_sam'}
    ns={'np':np}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    require(len(nodes)==len(names),'Pinned pure sampler functions')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SAMPLER),'exec'),ns)
    tree=ast.parse(TRAINER.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='object_sha']
    require(len(nodes)==1,'Pinned state hashing function')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(TRAINER),'exec'),dict(torch=torch,hashlib=hashlib),ns)
    return ns

def head_module():
    spec=importlib.util.spec_from_file_location('v182_pinned_v181_head',HEAD)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def validate_metadata(rr,gg):
    require(len(rr)==4096 and len(gg)==512,'All4096/512 groups')
    require([r['diagnostic_row_index'] for r in rr]==list(range(4096)) and [r['original_global_index'] for r in rr]==list(range(4096)),'New diagnostic ordering')
    require(len({r['row_id'] for r in rr})==4096 and len({g['group_id'] for g in gg})==512,'Distinct diagnostic IDs')
    gm={g['group_id']:g for g in gg};perbase={}
    for r in rr:
        require(r['split']=='TRAIN' and 0<=r['base_index']<128 and 0<=r['native_slot']<8,'TRAIN/native fixed')
        require(r['query_xy']==r['anchor_original_query_xy'] and r['L']==r['anchor_original_L'],'Only exact COVER anchor q/L')
        require(r['original_query_available']==r['anchor_original_flags']['query_available'],'Only anchor availability')
        require(r['group_id'] in gm,'Input group mapping')
        k=(r['base_index'],r['native_slot']);identity=(r['query_xy'],r['L'],r['original_query_available'],r['anchor_original_global_index'],r['anchor_original_query_id'])
        if k in perbase:require(perbase[k]==identity,'One fixed q/L/anchor per4variants')
        else:perbase[k]=identity
    for g in gg:
        group=[r for r in rr if r['group_id']==g['group_id']]
        require(len(group)==8 and [r['native_slot'] for r in group]==list(range(8)),'All8 native each group')
        require(g['split']=='TRAIN' and g['common_query_xy8']==[r['query_xy'] for r in group] and g['common_L8']==[r['L'] for r in group],'Group q8/L8 exact')
    require(sum(not r['original_query_available'] for r in rr)==96,'Fixed anchor inputU96')
    return gm

def validate_inputs(x,rr):
    n=len(rr);require(set(x)==INPUT_KEYS,'Only RGB/valid/identity; no GT/SAM in A input packet')
    dtype={'rgb96':np.uint8,'valid96':np.bool_,'original_global_index':np.int64,'native_slot':np.int8,'query_xy':np.float64,'L':np.float64,'original_query_available':np.bool_}
    for k,dt in dtype.items():require(x[k].dtype==dt,'Input dtype '+k)
    require(x['row_id'].dtype.kind=='U','Input row_id Unicode')
    shapes={'rgb96':(n,96,96,3),'valid96':(n,96,96),'query_xy':(n,2)}
    for k in x:require(x[k].shape==shapes.get(k,(n,)),'Input shape '+k)
    for i,r in enumerate(rr):
        for k in ('row_id','original_global_index','native_slot','original_query_available'):require(x[k][i]==r[k],'Input metadata '+k)
        require(same(x['query_xy'][i],qarray(r['query_xy'])) and same(np.asarray(x['L'][i]),np.asarray(side(r['L']))),'Input original common q/L')
    av=x['original_query_available'];require(np.isfinite(x['query_xy'][av]).all() and np.isfinite(x['L'][av]).all() and (x['L'][av]>0).all(),'Available anchor q/L finite')

def assemble_sam(x,rr,groups,sam_complete,helper):
    require(sam_complete['completed'] and sam_complete['PASS'],'Frozen SAM completion')
    actual={g['group_id']:g for g in sam_complete['groups']};require(set(actual)==set(groups),'All512 SAM group IDs')
    mask=np.zeros((4096,96,96),bool);available=np.zeros_like(mask);proof=[]
    for group_id,g in groups.items():
        inds=[i for i,r in enumerate(rr) if r['group_id']==group_id];a=actual[group_id]
        require(a['common_query_xy8']==g['common_query_xy8'] and a['common_L8']==g['common_L8'],'SAM common prompt qL identity')
        require(a['common_original_query_available8']==[bool(x['original_query_available'][i]) for i in inds],'SAM original query availability')
        logits=None;prediction_available=np.zeros(8,bool)
        if a['outputs'] is not None:
            z=load_npz(a['outputs'],SAM_KEYS-{'lowres_logits','predicted_quality','mask_logits_gt0'},exact=SAM_KEYS)
            require(a['prompt_available'] and a['runtime_available'],'SAM successful packet availability')
            require(z['image_logits'].dtype==np.float32 and z['image_logits'].shape==tuple(g['raw_hw']) and np.isfinite(z['image_logits']).all(),'Full original image logits')
            for k,want in [('original_query_xy',x['query_xy'][inds]),('original_L',x['L'][inds]),('native_slot',x['native_slot'][inds]),('original_global_index',x['original_global_index'][inds]),('original_occurrence_id',x['row_id'][inds]),('original_query_available',x['original_query_available'][inds])]:
                require(same(z[k],want),'SAM NEW diagnostic metadata '+k)
            require(z['prediction_available'].dtype==bool and same(z['prediction_available'],x['original_query_available'][inds]),'SAM predicted availability copies original anchor')
            require(same(z['box_xyxy'],np.asarray(g['common_q8_context_box_xyxy'],np.float64)),'Exact q8 box; no GT/clip/expand')
            logits=z['image_logits'];prediction_available=z['prediction_available']
        else:require(not a['prompt_available'] or not a['runtime_available'],'Missing SAM reason preserved')
        for j,i in enumerate(inds):
            mask[i],available[i]=helper['sample_sam'](logits,rr[i]['query_xy'],rr[i]['L'],bool(prediction_available[j]),g['raw_hw'])
            proof.append(dict(row_id=rr[i]['row_id'],group_id=group_id,original_global_index=i,sam_mask96_sha256=ash(mask[i]),sam_available96_sha256=ash(available[i]),SAM_output=a['outputs']))
    return mask,available,proof

def cover_input_parity(x,mask,sam_av,rr,old):
    oldmap={int(v):i for i,v in enumerate(old['original_global_index'])};require(len(oldmap)==4096,'Original source IDs unique')
    pairs=[]
    for i,r in enumerate(rr):
        if r['variant']!='COVER':continue
        j=oldmap[r['anchor_original_global_index']]
        require(r['source_image_original_global_index']==r['anchor_original_global_index'] and str(old['row_id'][j])==r['anchor_original_query_id'],'Original COVER identities')
        for k in ('rgb96','valid96','native_slot','query_xy','L','original_query_available'):require(same(x[k][i],old[k][j]),'COVER input parity '+k)
        require(same(mask[i],old['sam_mask96'][j]) and same(sam_av[i],old['sam_available96'][j]),'COVER same-grid SAM input parity')
        pairs.append((i,j))
    require(len(pairs)==1024,'All original COVER1024 retained inclU')
    return pairs

def decode(support,owner,available):
    n=len(available);require(support.dtype==owner.dtype==np.float32 and support.shape==(n,2) and owner.shape==(n,3),'FP32 two tasks')
    require(np.isfinite(support[available]).all() and np.isfinite(owner[available]).all() and np.isnan(support[~available]).all() and np.isnan(owner[~available]).all(),'U rawNaN first')
    sp=np.full((n,2),-1,np.int8);op=np.full(n,-1,np.int8);proxy=np.full(n,-1,np.int8)
    sp[available]=(support[available]>0).astype(np.int8);op[available]=owner[available].argmax(1).astype(np.int8)
    proxy[available]=np.where(sp[available,1]==1,1,np.where(sp[available,0]==1,0,-1)).astype(np.int8)
    return sp,op,proxy

def infer(model,x,mask,sam_av,use_sam,head,state_sha):
    n=len(x['row_id']);support=np.full((n,2),np.nan,np.float32);owner=np.full((n,3),np.nan,np.float32)
    ids=np.flatnonzero(x['original_query_available']);repeat=None
    with torch.inference_mode():
        for start in range(0,len(ids),96):
            ii=ids[start:start+96];rgb=torch.as_tensor(x['rgb96'][ii],device='cuda');valid=torch.as_tensor(x['valid96'][ii],device='cuda')
            extra=dict(sam_mask96=torch.as_tensor(mask[ii],device='cuda'),sam_available=torch.as_tensor(sam_av[ii],device='cuda')) if use_sam else {}
            inp=head.assemble_local_input(rgb,valid,use_sam=use_sam,**extra);first=model(inp)
            if start==0:
                second=model(inp);require(all(torch.equal(first[k],second[k]) for k in first),'Fixed first batch repeat exact')
                repeat=dict(rows=ii.tolist(),repeat2_exact=True,input_tensor_sha256=ash(inp.cpu().numpy()))
            support[ii]=first['support_logits'].cpu().numpy();owner[ii]=first['owner_logits'].cpu().numpy()
    require(repeat is not None and state_sha(model.state_dict())==state_sha.expected,'Frozen state unchanged after inference')
    sp,op,proxy=decode(support,owner,x['original_query_available'])
    return dict(raw_support_logits=support,raw_owner_logits=owner,support_prediction=sp,owner_prediction=op,regional_support_proxy_D0_E1_Uminus1=proxy,
        support_order=np.asarray(['visible_end_support','foreign_front_support']),owner_order=np.asarray(['TARGET','FOREIGN','BACKGROUND']),**{k:x[k] for k in META}),repeat

def score_delta(new,old,pairs):
    ii=np.array([p[0] for p in pairs]);jj=np.array([p[1] for p in pairs]);av=new['original_query_available'][ii]
    require(same(new['original_query_available'][ii],old['original_query_available'][jj]),'COVER score original availability')
    result={'rows':len(ii),'available':int(av.sum()),'input_U':int((~av).sum()),'measurement_only_no_tolerance_or_selection':True}
    for k in ('raw_support_logits','raw_owner_logits'):
        a=new[k][ii][av];b=old[k][jj][av];require(a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all(),'Old/new frozen score schema')
        delta=a.astype(np.float64)-b.astype(np.float64)
        result[k]={'exact_array_bytes_equal':a.tobytes()==b.tobytes(),'max_abs_delta':float(np.abs(delta).max()),'mean_abs_delta':float(np.abs(delta).mean()),'nonzero_elements':int(np.count_nonzero(delta))}
    for k in ('support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1'):result[k+'_changed_elements']=int(np.count_nonzero(new[k][ii]!=old[k][jj]))
    return result

def prepare():
    require(sha(SCOPE)==SCOPE_SHA and sha(HEAD)==HEAD_SHA and sha(SAMPLER)==SAMPLER_SHA,'Pinned scope/head/sampler')
    require(sha(SLOTS)==SLOTS_SHA and sha(GROUPS)==GROUPS_SHA,'Frozen metadata manifests')
    rr=rows(SLOTS);gg=rows(GROUPS);validate_metadata(rr,gg)
    helpers=pure_helpers();head=head_module();tests=[]
    def passed(name,ok):require(ok,name);tests.append(dict(name=name,PASS=True))
    passed('all4096diagIDs/512TRAINgroups/native8/constantCOVERqL/anchorU96',True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1820);torch.set_num_threads(1)
        rgb=torch.randint(0,256,(2,96,96,3),dtype=torch.uint8);valid=torch.ones((2,96,96),dtype=torch.bool);mask=torch.eye(96,dtype=torch.bool)[None].expand(2,-1,-1).clone();av=valid.clone();av[:,0]=False
        raw=head.assemble_local_input(rgb,valid,use_sam=False);sam=head.assemble_local_input(rgb,valid,use_sam=True,sam_mask96=mask,sam_available=av)
        passed('sameV1818channelsRGBvalidconstantXY_and_RAW_SAMzeros',raw.shape==(2,8,96,96) and torch.equal(raw[:,:6],sam[:,:6]) and torch.count_nonzero(raw[:,6:])==0 and torch.equal(sam[:,6],(mask&av).float()) and torch.equal(sam[:,7],av.float()))
        model=head.Raw96SupportHead().float().eval().requires_grad_(False);before=helpers['object_sha'](model.state_dict())
        with torch.inference_mode():a=model(raw);b=model(raw)
        passed('inventedCPU115157architecture_eval_repeat_stateunchanged',sum(p.numel() for p in model.parameters())==115157 and all(torch.equal(a[k],b[k]) for k in a) and before==helpers['object_sha'](model.state_dict()) and all(not p.requires_grad for p in model.parameters()))
    grid=helpers['grid96']([0.,0.],96.)
    left=(np.float64(47.5)/96-.5)*96;right=(np.float64(48.5)/96-.5)*96
    passed('exact96pixelcenter_FP64expression_not_linspace',np.array_equal(grid[47*96+47],[left,left]) and np.array_equal(grid[48*96+48],[right,right]) and grid[0,0]!=-48.)
    mask,av=helpers['sample_sam'](np.ones((2,2),np.float32),[-.5,-.5],.125,True,[2,2])
    upper,upper_av=helpers['sample_sam'](np.ones((2,2),np.float32),[1.5,1.5],.125,True,[2,2])
    passed('sameFP64fourtap_paddedSAM_domain_halfpixelbounds',mask[48,48] and av[48,48] and not av[47,47] and upper[47,47] and upper_av[47,47] and not upper_av[48,48])
    m,v=helpers['sample_sam'](None,None,None,False,[512,512]);passed('missingSAM_and_anchorU_do_not_create_inputavailability',not m.any() and not v.any())
    sub=np.nextafter(np.float32(0),np.float32(1));s=np.array([[0,-0.],[1,1],[sub,-sub],[np.nan,np.nan]],np.float32);o=np.array([[0,0,0],[0,1,1],[1,0,0],[np.nan]*3],np.float32)
    sp,op,px=decode(s,o,np.array([1,1,1,0],bool));passed('strict0_subnormal_firsttie_foreignpriority_UrawNaN',sp.tolist()==[[0,0],[1,1],[1,0],[-1,-1]] and op.tolist()==[0,1,0,-1] and px.tolist()==[-1,1,0,-1])
    passed('input_whitelist_excludes_targets_GT_and_nativehints',INPUT_KEYS=={'rgb96','valid96','row_id','original_global_index','native_slot','query_xy','L','original_query_available'})
    # Score deltas are descriptive, never equality/tolerance gates.
    old=dict(raw_support_logits=np.array([[0,1]],np.float32),raw_owner_logits=np.array([[0,1,2]],np.float32),support_prediction=np.array([[0,1]],np.int8),owner_prediction=np.array([2],np.int8),regional_support_proxy_D0_E1_Uminus1=np.array([1],np.int8),original_query_available=np.array([True]))
    new={k:v.copy() for k,v in old.items()};new['raw_support_logits'][0,0]=.25
    delta=score_delta(new,old,[(0,0)]);passed('nonzeroCOVER_score_delta_is_measured_not_admission_gate',delta['raw_support_logits']['max_abs_delta']==.25 and delta['measurement_only_no_tolerance_or_selection'])
    checks=HERE/'GENERATED_INVENTED_CHECKS.json';writej(checks,dict(completed=True,PASS=True,tests=tests,actual_checkpoint_commonpixel_SAM_or_target_bodies_read=0))
    fit=readj(FIT);old=readj(OLD_CACHE)
    require(fit['PASS'] and fit['head']==binding(HEAD) and fit['total_updates']==16000,'Frozen V181 metadata')
    p=dict(completed=True,prepared_only=True,scope=binding(SCOPE),code=binding(__file__),head=binding(HEAD),pure_sampler_source=binding(SAMPLER),pure_statehash_source=binding(TRAINER),checks=binding(checks),slot_metadata=binding(SLOTS),group_metadata=binding(GROUPS),frozen_V181_fit=binding(FIT),original_source_cache_metadata=binding(OLD_CACHE),
        common_input_complete='PENDING separately root-bound A completed COMMON4096_INPUTS exact8; targets never opened',SAM_complete='PENDING separately root-bound B completed512groups/exact12key packets',
        frozen_models={a:dict(checkpoint=fit['arms'][a]['checkpoint'],final_state_sha=fit['arms'][a]['final_state_sha'],updates=8000,use_sam=u,old_COVER_reference_scores=fit['arms'][a]['outputs']['SYNTH_TRAIN4096']) for a,u in ARMS},
        old_COVER_input_binding_inherited_not_opened=old['domains']['SYNTH_TRAIN4096']['inputs'],
        runtime=dict(python=PYTHON,torch='2.1.1+cu118',numpy='1.26.4',device='cuda',FP32=True,AMP=False,TF32=False,strict_deterministic=True,eval_no_grad_only=True),
        batch='Ascending available diagnosticindices, fixed batches96 (last residual); repeat fixed first availablebatch twice exact for eachhead. Same V181 inputassembly/inference batching; no target reads.',
        input='RGB3/255,valid,constantrelativeXY2,SAMmask,SAMavailability; C50SAMchannelszero. Source q/L only sampling metadata, never GT/ref/nativehint inputs.',
        SAM='Exact V181 grid96/paddedbilinearFP64/>0/domain; original anchor availability preserved independently; missingSAM represented0mask/0SAMavailability, never new inputU',
        COVER='All1024 COVER metadata/RGB96/valid96/SAM96/availability byteparity against V181 sourceinput beforeforward; all2newpredictionssealed beforeoldscorecomparison. Oldscore delta measured only, never tolerance/model/threshold selection.',
        output='COMPLETE.arms[arm].outputs single13keyV181style4096NPZ, plus separate whitelisted originlineage and SAMinput-onlycache. original_global_index in NEW output means diagnostic0..4095, not production-replay identity.',
        decoder='Eachsupportlogit>0; ownerargmaxfirsttie; unavailable rawNaN/decoded-1; regionalforeignYES E elsevisibleYES D elseU',
        target_file_reads=0,no_optimizer_or_fit=True,no_source_RGB_GT_buffers_or_private_held_VAL_reads=True,selected_model=None,fully_solved=False)
    writej(PROTOCOL,p)
    template=HERE/'ROOT_ACTUAL_COMMON_HEAD_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    writej(template,dict(approved_frozen_common_query_head_forward=False,source_only=True,scope=p['scope'],code=p['code'],protocol=binding(PROTOCOL),common_input_complete=None,SAM_complete=None,frozen_V181_fit=p['frozen_V181_fit'],original_source_cache_metadata=p['original_source_cache_metadata'],output_dir=str(ROOT/'frozen_common_query_heads_v1')))
    writej(HERE/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],head=p['head'],protocol=binding(PROTOCOL),checks=binding(checks),entry_template=binding(template),actual_learned_weight_pixel_SAM_target_body_reads_or_frozenforward=0))
    print(json.dumps(dict(completed=True,PASS=True,prepared_only=True,code=binding(__file__),protocol=binding(PROTOCOL))))

def run(entry_path):
    e=readj(entry_path);require(e.get('approved_frozen_common_query_head_forward') is True and e.get('source_only') is True,'Separate root actual head entry required')
    for k,path in [('scope',SCOPE),('code',Path(__file__)),('protocol',PROTOCOL)]:require(e[k]==binding(path),'Entry '+k)
    require(sha(SCOPE)==SCOPE_SHA and sha(HEAD)==HEAD_SHA and sha(SAMPLER)==SAMPLER_SHA,'Pinned scope/code')
    p=readj(PROTOCOL)
    for k in ('checks','head','pure_sampler_source','pure_statehash_source','slot_metadata','group_metadata'):checked(p[k])
    require(e['frozen_V181_fit']==p['frozen_V181_fit'] and e['original_source_cache_metadata']==p['original_source_cache_metadata'],'Exact frozen models/source reference metadata')
    require(Path(sys.executable).resolve()==Path(PYTHON).resolve() and torch.__version__=='2.1.1+cu118' and np.__version__=='1.26.4' and torch.cuda.is_available(),'Pinned CUDA runtime')
    complete=readj(checked(e['common_input_complete']));require(complete['completed'] and complete['PASS'] and complete['rows']==4096 and complete['groups']==512 and complete['SAM_arrays_present'] is False,'A completed input-only common4096')
    rr=rows(SLOTS);gg=rows(GROUPS);gm=validate_metadata(rr,gg)
    out=Path(e['output_dir']);require(out.parent.resolve()==ROOT.resolve() and not out.exists(),'Fresh root-bound output; no overwrite/retry')
    out.mkdir();writej(out/'STARTED.json',dict(entry=binding(entry_path),code=binding(__file__),protocol=binding(PROTOCOL),new_target_files_read=0))
    try:
        x=load_npz(complete['inputs'],INPUT_KEYS,exact=INPUT_KEYS);validate_inputs(x,rr)
        sam=readj(checked(e['SAM_complete']));helper=pure_helpers();head=head_module()
        mask,sam_av,samproof=assemble_sam(x,rr,gm,sam,helper)
        oldmeta=readj(checked(e['original_source_cache_metadata']));oldb=oldmeta['domains']['SYNTH_TRAIN4096']['inputs']
        require(oldb==p['old_COVER_input_binding_inherited_not_opened'],'Exact original source input cache')
        old=load_npz(oldb,INPUT_KEYS|{'sam_mask96','sam_available96'})
        pairs=cover_input_parity(x,mask,sam_av,rr,old);del old
        writej(out/'COVER1024_INPUT_PARITY.json',dict(completed=True,PASS=True,rows1024=True,original_input=oldb,common_input=complete['inputs'],rawRGB_valid_SAMmask_SAMavailability_qL_native_originalav_byteexact=True,new_target_files_read=0))
        sam_path=out/'COMMON4096_SAM96_INPUT_ONLY.npz';np.savez_compressed(sam_path,sam_mask96=mask,sam_available96=sam_av,**{k:x[k] for k in META})
        writerows(out/'ALL4096_SAM_INPUT_FACTS.jsonl',samproof)
        lineagekeys=['row_id','diagnostic_row_index','original_global_index','group_id','base_index','scene_index','variant','native_slot','query_xy','L','original_query_available','source_image_original_global_index','anchor_original_global_index','source_image_original_query_id','anchor_original_query_id','image_original_query_xy','image_original_L','image_original_flags','anchor_original_query_xy','anchor_original_L','anchor_original_flags']
        writerows(out/'COMMON4096_PREDICTION_INPUT_LINEAGE.jsonl',[{k:r[k] for k in lineagekeys} for r in rr])
        fit=readj(checked(e['frozen_V181_fit']));require(fit['PASS'] and fit['all2_frozen_before_outputs_or_VAL_access'],'Already frozen V181 fits')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        models={};arms={}
        for arm,use_sam in ARMS:
            ref=p['frozen_models'][arm];require(fit['arms'][arm]['checkpoint']==ref['checkpoint'] and fit['arms'][arm]['updates']==8000,'Fixed original final8000 checkpoint')
            packet=torch.load(checked(ref['checkpoint']),map_location='cpu',weights_only=True)
            require(packet['update']==8000 and packet['arm']==arm and packet['use_sam']==use_sam and packet['seed']==1810,'Checkpoint metadata')
            model=head.Raw96SupportHead().float();model.load_state_dict(packet['state_dict'],strict=True);del packet
            require(sum(t.numel() for t in model.parameters())==115157,'Exact architecture')
            model=model.cuda().eval().requires_grad_(False);state=helper['object_sha'](model.state_dict());require(state==ref['final_state_sha'],'Exact V181 final state')
            models[arm]=model
        raw_outputs={}
        for arm,use_sam in ARMS:
            statehash=helper['object_sha'];statehash.expected=p['frozen_models'][arm]['final_state_sha']
            data,repeat=infer(models[arm],x,mask,sam_av,use_sam,head,statehash)
            path=out/(arm+'_FINAL_COMMON4096_OUTPUTS.npz');np.savez_compressed(path,**data)
            raw_outputs[arm]=data;arms[arm]=dict(outputs=binding(path),source_checkpoint=p['frozen_models'][arm]['checkpoint'],final_state_sha=statehash.expected,state_after_inference_unchanged=True,first_batch_repeat=repeat,rows=4096,original_inputU=int((~x['original_query_available']).sum()),newfit=0)
            print(json.dumps(dict(arm=arm,rows=4096,inputU=arms[arm]['original_inputU'],frozen_forward_complete=True)),flush=True)
        del models
        seal=out/'ALL2_PREDICTIONS_SEALED_BEFORE_TARGET_READ.json';writej(seal,dict(completed=True,PASS=True,arms=arms,common_input=complete['inputs'],SAM_complete=e['SAM_complete'],all2_final8000_frozen=True,new_target_files_read=0,no_target_or_label_or_GT_model_inputs=True))
        comparison={}
        for arm,use_sam in ARMS:
            b=p['frozen_models'][arm]['old_COVER_reference_scores']
            old=load_npz(b,{'raw_support_logits','raw_owner_logits','support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1','original_query_available'})
            comparison[arm]=score_delta(raw_outputs[arm],old,pairs)
        writej(out/'COVER1024_OLD_NEW_SAVED_SCORE_DELTA.json',dict(completed=True,measurement_only=True,predictions_seal=binding(seal),comparison=comparison,old_inference_different_batch_companions_possible=True,no_equality_tolerance_gate_or_model_threshold_choice=True))
        for path,h in USED.items():require(sha(path)==h,'Consumed asset changed during frozen forward')
        writej(out/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,scope=binding(SCOPE),entry=binding(entry_path),code=binding(__file__),protocol=binding(PROTOCOL),head=binding(HEAD),common_input_complete=e['common_input_complete'],SAM_complete=e['SAM_complete'],frozen_V181_fit=e['frozen_V181_fit'],arms=arms,rows_each_arm=4096,original_inputU_each_arm=96,all2_predictions_seal=binding(seal),predictions_sealed_before_target_read=True,new_target_files_read=0,input_lineage=binding(out/'COMMON4096_PREDICTION_INPUT_LINEAGE.jsonl'),SAM96_input_only=binding(sam_path),COVER_input_parity=binding(out/'COVER1024_INPUT_PARITY.json'),COVER_score_delta=binding(out/'COVER1024_OLD_NEW_SAVED_SCORE_DELTA.json'),production_query_replay=False,original_prefixes_mean_new_diagnosticIDs=True,newfit_initializer_encoder_or_PnP=0,GT_model_inputs=0,private_held_VAL_access=0,selected_model=None,adopted=False,fully_solved=False))
        print(json.dumps(dict(completed=True,PASS=True,complete=binding(out/'COMPLETE.json'))))
    except BaseException:
        writej(out/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False,new_target_files_read=0));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--root-entry',type=Path);args=ap.parse_args();prepare() if args.prepare else run(args.root_entry)

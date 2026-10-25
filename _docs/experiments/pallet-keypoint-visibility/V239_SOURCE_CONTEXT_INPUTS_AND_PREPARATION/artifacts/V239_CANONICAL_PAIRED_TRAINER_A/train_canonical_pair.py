"""V239 canonical frozen-vector paired heads; Root true entry only.
No DINO, RGB, old prefix or old CONTROL checkpoint loader is present.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types
import numpy as np
import torch
from torch.nn import functional as F

R=Path(__file__).absolute().parent.parent
CONTROL='CONTROL_CANONICAL_LOCAL'
TREAT='TREAT_CANONICAL_CONTEXT'
MODULE_PIN=(str(R/'V237_RELATIONAL_QUERY_DESIGN_B/relational_query_head.py'),'dfd26ab7ccfa7037f0c3dca548d59c8574aa8b78c50699f70f01e03bf48935a9')
ROOT_PROTOCOL_PIN=(str(R/'V239_ROOT_CANONICAL_CONTEXT/PROTOCOL_BEFORE_CANONICAL_FEATURES_OR_PAIRED_FIT.json'),'4cbbd1ede0420e1cc6dfe83446b2204141bd99ac29781836b9663c49c77fa021')
INITIAL_BASE_SHA='d1b8da9721a5492dcfb166cf78943eb5db18c5b14b8e8fed328ccab31c12648c'
DRAW_SHA='20a1fc74ea15a04a16fe29f8e5b9a114b4629677710ed43c1628b2a1ac00b734'
GEOMETRY_KEYS=('node_available','node_fraction','context_fraction','q8','L8','original_available8','box_xyxy')
FEATURE_LAYOUT={
 'ORIGINAL_ALL_FEATURES.npy':(np.float32,(7617,1186)),
 'NODE_CLS.npy':(np.float32,(1767,8,384)),
 'CONTEXT_CLS.npy':(np.float32,(1767,384)),
 'SAMPLE_GROUP_INDEX.npy':(np.int64,(7617,)),
 'SAMPLE_NODE_INDEX.npy':(np.int64,(7617,)),
 'RELATIVE_XY8.npy':(np.float32,(7617,8,2))}
SAMPLE_KEYS=('full_view','central_view','fractions','central4','condition','target','family','fit','available','row_id')


def safe(value):
    text = str(value)
    if any(x in text.casefold() for x in ('v112', '__pycache__', '.pyc', 'current223', 'publicheld', 'oldval', 'v177', 'v178', 'source_holdout', 'v182', 'private_evaluation')):
        raise ValueError('Forbidden path before filesystem access')
    if '..' in Path(text).parts:
        raise ValueError('Parent path component forbidden')
    path = Path(os.path.abspath(text))
    for part in reversed((path, *path.parents)):
        if part.is_symlink():
            raise ValueError('Symlink rejected before leaf access')
    return path

def read(path):
    with os.fdopen(os.open(safe(path), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        return stream.read()

def identity(b):
    return str(safe(b['path'])), b['sha256']

def verify(b, whitelist, pin=None):
    if identity(b) not in {identity(x) for x in whitelist} or (pin is not None and identity(b) != pin):
        raise ValueError('Outside explicit path/SHA whitelist or pinned code')
    digest = hashlib.sha256(); size = 0
    with os.fdopen(os.open(safe(b['path']), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block); size += len(block)
    if digest.hexdigest() != b['sha256'] or ('size_bytes' in b and b['size_bytes'] != size):
        raise ValueError('Bound bytes changed')
    return safe(b['path'])

def binding(path):
    raw = read(path)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}

def write_json(path, obj):
    with os.fdopen(os.open(safe(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'w') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    return binding(path)

def load_code(b, whitelist, name, pin):
    path = verify(b, whitelist, pin)
    module = types.ModuleType(name); module.__file__ = str(path); sys.modules[name] = module
    exec(compile(read(path), str(path), 'exec'), module.__dict__)
    return module

def validate_geometry(g, count):
    if any(k not in g for k in GEOMETRY_KEYS):
        raise ValueError('Missing saved input geometry')
    for key in ('node_available', 'original_available8'):
        if g[key].dtype != bool or g[key].shape != (count, 8):
            raise ValueError('Exact original/support bool8')
    for key, shape in [('node_fraction', (count, 8)), ('context_fraction', (count,))]:
        if g[key].dtype != np.float32 or g[key].shape != shape or not np.isfinite(g[key]).all() or not ((g[key] >= 0) & (g[key] <= 1)).all():
            raise ValueError('FP32 fixed validity fractions')
    for key, shape in [('q8', (count, 8, 2)), ('L8', (count, 8)), ('box_xyxy', (count, 4))]:
        if g[key].dtype != np.float64 or g[key].shape != shape:
            raise ValueError('Original F64 geometry')
    if np.any(g['node_available'] & ~g['original_available8']):
        raise ValueError('Support cannot promote original input U')
    return g

def relative_xy(geometry, group, slot):
    q = geometry['q8'][group]; L = geometry['L8'][group, slot]
    active = q[np.arange(len(group)), slot]
    avail = geometry['original_available8'][group]
    if not np.isfinite(active).all() or not np.isfinite(L).all() or not (L > 0).all() or not np.isfinite(q[avail]).all():
        raise ValueError('Available original q and active L finite; no substitute')
    out = np.zeros((len(group), 8, 2), np.float64)
    np.divide(q-active[:, None, :], L[:, None, None], out=out, where=avail[:, :, None])
    out = np.clip(out, -8, 8).astype(np.float32)
    out[~avail] = 0; out[np.arange(len(group)), slot] = 0
    return out

def paired_changes(control_rows, treatment_rows):
    result = {}
    for family in (0, 1):
        for role in ('FIT', 'DEV'):
            take = [(a, b) for a, b in zip(control_rows, treatment_rows) if a['family'] == family and a['role'] == role]
            recovered = damaged = 0
            for a, b in take:
                truth = 'D' if a['target'] == 0 else 'E'
                recovered += a['prediction'] != truth and b['prediction'] == truth
                damaged += a['prediction'] == truth and b['prediction'] != truth
            result[f'family{family}_{role}'] = {'rows': len(take), 'recovered': recovered, 'damaged': damaged, 'net': recovered-damaged}
    return result


def state_sha(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(a.dtype).encode())
        digest.update(json.dumps(list(a.shape)).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu_copy(v) for v in value)
    return value


def validate_samples(data):
    if set(data)!=set(SAMPLE_KEYS):raise ValueError('Original sample metadata keys')
    n=len(data['target'])
    for key in ('full_view','central_view','central4','target','family','fit','available','row_id'):
        if data[key].shape!=(n,):raise ValueError('Original sample vector shape')
    if data['target'].dtype!=np.int8 or not np.isin(data['target'],[0,1]).all() or data['family'].dtype!=np.int8 or not np.isin(data['family'],[0,1]).all():raise ValueError('Separate known SOURCE target/family')
    if any(data[k].dtype!=bool for k in ('central4','fit','available')):raise ValueError('Original bool flags')
    if data['row_id'].dtype.kind!='U' or len(set(data['row_id'].tolist()))!=n:raise ValueError('Unique original row IDs')
    if any(data[k].dtype!=np.int64 for k in ('full_view','central_view')) or not np.array_equal(data['full_view'],np.arange(n)*2) or not np.array_equal(data['central_view'],np.arange(n)*2+1):raise ValueError('Original ordered view IDs; never read old prefix')
    if data['condition'].dtype!=np.float32 or data['condition'].shape!=(n,32) or not np.isfinite(data['condition']).all():raise ValueError('Original FP32condition32')
    if data['fractions'].dtype!=np.float32 or data['fractions'].shape!=(n,2) or not np.isfinite(data['fractions']).all() or not ((data['fractions']>=0)&(data['fractions']<=1)).all():raise ValueError('Original FP32validfractions')
    if np.count_nonzero(data['condition'][data['family']==1]):raise ValueError('YCB originalcondition32 must stay0')
    return data


def fixed_draws(data):
    eligible=data['fit']&data['available']&(data['target']>=0)
    pools=[np.flatnonzero(eligible&(data['family']==f)&(data['target']==y)) for f in (0,1) for y in (0,1)]
    if any(len(p)==0 for p in pools):raise ValueError('Empty required SOURCEFIT class pool; no fallback')
    rng=np.random.Generator(np.random.PCG64(17))
    draws=np.stack([np.concatenate([rng.choice(p,8,replace=True) for p in pools]) for _ in range(1000)]).astype(np.int64)
    return draws,eligible


def family_loss(logits,target,family):
    if logits.shape!=(32,2) or target.shape!=(32,) or family.shape!=(32,):raise ValueError('Fixed32/family2 objective')
    parts=[]
    for k in (0,1):
        take=family==k
        if int(take.sum())!=16 or int(((target==0)&take).sum())!=8 or int(((target==1)&take).sum())!=8:raise ValueError('Each family exactly8D8E')
        parts.append(F.binary_cross_entropy_with_logits(logits[take,k],target[take].to(logits.dtype)))
    return .5*(parts[0]+parts[1]),parts


def reduce_rows(rows):
    out={}
    for family in (0,1):
        for role in ('FIT','DEV'):
            rr=[r for r in rows if r['family']==family and r['role']==role]
            cm=[[sum(r['target']==y and r['prediction']==p for r in rr) for p in ('D','E','U')] for y in (0,1)]
            item={'rows':len(rr),'known_DE':len(rr),'CM_D_E_by_D_E_U':cm,'target_UNKNOWN':0,'original_input_U':sum(not r['available'] for r in rr)}
            for name,k in (('D',0),('E',1)):
                truth=sum(cm[k]);pred=cm[0][k]+cm[1][k];tp=cm[k][k]
                item[name]={'truth':truth,'predicted':pred,'precision':tp/pred if pred else None,'recall':tp/truth if truth else None}
            out[f'family{family}_{role}']=item
    return out


def source_gate(control,treat):
    gains=[];drops=[]
    for f in (0,1):
        a,b=control[f'family{f}_DEV'],treat[f'family{f}_DEV']
        old=[a[k]['recall'] for k in ('D','E')];new=[b[k]['recall'] for k in ('D','E')]
        if any(v is None for v in old+new):return {'PASS':False,'reason':'Missing fixed SOURCE class denominator','current_admission':False}
        gains.append((sum(new)-sum(old))/2);drops.extend([x-y for x,y in zip(old,new)])
    gain=100*sum(gains)/2;drop=100*max(drops)
    ycb=treat['family1_DEV']['E']['recall'];syn=[treat['family0_DEV'][k]['recall'] for k in ('D','E')]
    return {'PASS':gain>=3 and drop<=2 and ycb>=.5 and min(syn)>=.9,
            'mean_family_DEV_balanced_recall_gain_pp':gain,'max_family_class_recall_drop_pp':drop,
            'YCB_DEV_E_recall':ycb,'SYN_DEV_D_recall':syn[0],'SYN_DEV_E_recall':syn[1],
            'fixed_min_gain_pp':3,'fixed_max_class_drop_pp':2,'fixed_YCB_DEV_E_floor':.5,'fixed_SYN_DEV_D_E_floor':.9,
            'current_admission':False,'automatic_retry':False,'new_current_inference':0}


def byte_equal(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()


def global_feature_QA(features,data,geometry):
    x=features['ORIGINAL_ALL_FEATURES.npy'];node=features['NODE_CLS.npy'];context=features['CONTEXT_CLS.npy']
    group=features['SAMPLE_GROUP_INDEX.npy'];slot=features['SAMPLE_NODE_INDEX.npy'];relative=features['RELATIVE_XY8.npy']
    if not ((group>=0)&(group<1767)).all() or not ((slot>=0)&(slot<8)).all():raise ValueError('Original group/slot index bounds')
    if not np.array_equal(geometry['original_available8'][group,slot]&geometry['node_available'][group,slot],data['available']):raise ValueError('Original active availability equality')
    if set(group[data['fit']].tolist())&set(group[~data['fit']].tolist()):raise ValueError('No group FIT/DEV overlap')
    digest_active=hashlib.sha256();digest_node=hashlib.sha256()
    for start in range(0,len(x),64):
        indices=np.arange(start,min(start+64,len(x)))
        active=np.array(x[indices,:384],copy=True);same=np.array(node[group[indices],slot[indices]],copy=True)
        if not byte_equal(active,same):raise ValueError('Canonical active NODE_CLS must be byteexact; no overwrite/tolerance')
        if not byte_equal(np.array(x[indices,-32:],copy=True),data['condition'][indices]) or not byte_equal(np.array(x[indices,1152:1154],copy=True),data['fractions'][indices]):raise ValueError('Original condition32/fractions byteexact')
        if not np.isfinite(x[indices]).all():raise ValueError('Finite canonical active features')
        digest_active.update(active.tobytes());digest_node.update(same.tobytes())
    if not np.isfinite(node).all() or not np.isfinite(context).all() or not np.isfinite(relative).all():raise ValueError('Finite frozen context features')
    if np.count_nonzero(node[~geometry['node_available']]) or np.count_nonzero(context[geometry['context_fraction']==0]):raise ValueError('Unsupported node/context descriptors remainzero')
    if not byte_equal(relative,relative_xy(geometry,group,slot)):raise ValueError('Original F64relative→FP32 exact')
    syn=data['family']==0
    if not byte_equal(relative[syn],data['condition'][syn,16:].reshape(-1,8,2)) or not np.array_equal(slot[syn],data['condition'][syn,:8].argmax(1)):raise ValueError('Original SYN native/condition address unchanged')
    return {'completed':True,'PASS':True,'checked':len(x),'active_first384_byte_exact':True,'original_condition32_byte_exact':True,'original_fractions_byte_exact':True,'original_group_slot_availability_exact':True,'source_group_FIT_DEV_overlap':0,'active_bytes_sha256':digest_active.hexdigest(),'node_bytes_sha256':digest_node.hexdigest(),'tolerance_used':0,'features_overwritten':0,'head_DEV_forward_before_QA':0}


def context_args(indices,features,geometry,device):
    group=features['SAMPLE_GROUP_INDEX.npy'][indices];slot=features['SAMPLE_NODE_INDEX.npy'][indices]
    return (torch.as_tensor(np.array(features['NODE_CLS.npy'][group],copy=True),device=device),
            torch.as_tensor(np.array(features['CONTEXT_CLS.npy'][group],copy=True),device=device),
            torch.as_tensor(np.array(slot,copy=True),device=device),
            torch.as_tensor(geometry['original_available8'][group],device=device),
            torch.as_tensor(geometry['node_available'][group],device=device),
            torch.as_tensor(geometry['context_fraction'][group]>0,device=device),
            torch.as_tensor(np.array(features['RELATIVE_XY8.npy'][indices],copy=True),device=device))


def configure_runtime():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.manual_seed(17)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(17)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def train_arm(arm,head,data,features,geometry,draws,eligible,fit_features,fit_map,device,out,entry_binding,feature_complete,draw_sha,initial_base_sha):
    initial=cpu_copy(head.state_dict());initial_sha=state_sha(initial)
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    log=out/(arm+'_TRAIN1000.jsonl');head.train()
    with log.open('x') as stream:
        for step,indices in enumerate(draws):
            if not eligible[indices].all() or (fit_map[indices]<0).any():raise ValueError('Only original available known FIT optimizerinputs')
            x=fit_features[torch.as_tensor(fit_map[indices],device=device)]
            optimizer.zero_grad(set_to_none=True)
            logits=head(x) if arm==CONTROL else head(x,*context_args(indices,features,geometry,device))[0]
            loss,parts=family_loss(logits,torch.as_tensor(data['target'][indices],device=device),torch.as_tensor(data['family'][indices],device=device))
            if not torch.isfinite(loss):raise ValueError('Finite fixed SOURCE objective')
            loss.backward();optimizer.step()
            stream.write(json.dumps({'update':step+1,'sample_indices':indices.tolist(),'draws_sha256':draw_sha,'family0_BCE':float(parts[0].detach()),'family1_BCE':float(parts[1].detach()),'total_loss':float(loss.detach()),'family_counts':[16,16],'all_fit_available_known':True},allow_nan=False)+'\n')
            if (step+1)%100==0:print(json.dumps({'stage':arm,'updates':step+1,'fixed_final':1000}),flush=True)
    expected_states=4 if arm==CONTROL else 10
    if len(optimizer.state)!=expected_states or any(float(s['step'])!=1000 or any(not torch.isfinite(v).all() for v in s.values() if isinstance(v,torch.Tensor)) for s in optimizer.state.values()):raise ValueError('All head optimizer states finite1000')
    if any(not torch.isfinite(p).all() for p in head.parameters()):raise ValueError('Finite frozenfinalhead')
    path=out/(arm+'_FINAL1000.pt')
    torch.save({'arm':arm,'updates':1000,'head_parameters':76098 if arm==CONTROL else 86054,'adapter_parameters':0,'use_lora':False,'head_state_dict':cpu_copy(head.state_dict()),'optimizer_state_dict':cpu_copy(optimizer.state_dict()),'initial_head_state_dict':initial,'initial_head_sha256':initial_sha,'same_initial_base_sha256':initial_base_sha,'draws_sha256':draw_sha,'entry':entry_binding,'feature_complete':feature_complete},path)
    return binding(path),binding(log),state_sha(head.state_dict()),initial_sha


def run(entry_path):
    e=json.loads(read(entry_path))
    if e.get('approved_canonical_source_pair_training') is not True or e.get('source_only') is not True or e.get('automatic_retry') is not False:raise ValueError('Explicit Root SOURCE canonical pair entry required')
    if e['steps']!=1000 or e['threshold_logit']!=0 or e.get('old_CONTROL_or_prefix_payload_used') is not False or e.get('use_V235_counterfactuals') is not False:raise ValueError('Fixed1000/zero/nooldCONTROLprefix/noCF')
    codes,meta,payload=e['code_whitelist'],e['metadata_whitelist'],e['payload_whitelist']
    verify(e['code'],codes,(str(safe(__file__)),binding(__file__)['sha256']))
    verify(e['implementation_protocol'],codes);verify(e['root_protocol'],meta,ROOT_PROTOCOL_PIN);verify(e['root_scope'],meta)
    packet=json.loads(read(verify(e['feature_complete'],meta)))
    if packet.get('completed') is not True or packet.get('PASS') is not True:raise ValueError('Canonical producer must complete first')
    if identity(packet['root_protocol'])!=identity(e['root_protocol']):raise ValueError('Common prospective feature/trainer protocol')
    counts=packet['counts'];sharing=packet['canonical_sharing_QA']
    if counts['known']!=7617 or counts['FIT']!=5876 or counts['DEV']!=1741 or counts['groups']!=1767 or counts['supported_nodes']!=13801 or counts['all_original_rows']!=15040 or counts['masked_target_minus1']!=7423:raise ValueError('Original source denominators; no forcing')
    if sharing.get('completed') is not True or sharing.get('PASS') is not True or sharing['known_rows']!=7617 or any(sharing[k] is not True for k in ('active_first384_byte_exact','original_condition32_byte_exact','known_pixel_fractions_byte_exact')):raise ValueError('Producer canonical identity proof')
    for key in ('samples_npz','geometry_npz'):
        if identity(e[key])!=identity(packet[key]):raise ValueError('Original SOURCE metadata closure')
    if e['samples_npz']['sha256']!='837bbe7ff406e27c6ed2981493a83c2adab830cd5b2de498e4a48de456456e99':raise ValueError('Root original source sample pin')
    features={}
    for filename,(dtype,shape) in FEATURE_LAYOUT.items():
        if identity(e['features'][filename])!=identity(packet['features'][filename]):raise ValueError('Exact canonical feature identity')
        features[filename]=np.load(verify(e['features'][filename],payload),mmap_mode='r',allow_pickle=False)
        if features[filename].dtype!=dtype or features[filename].shape!=shape:raise ValueError('Canonical feature header')
    with np.load(verify(e['samples_npz'],payload),allow_pickle=False) as z:
        if set(z.files)!=set(SAMPLE_KEYS):raise ValueError('Original samples schema beforememberdecode')
        data=validate_samples({k:z[k] for k in SAMPLE_KEYS})
    with np.load(verify(e['geometry_npz'],payload),allow_pickle=False) as z:geometry=validate_geometry({k:z[k] for k in GEOMETRY_KEYS},1767)
    draws,eligible=fixed_draws(data);fit=np.flatnonzero(eligible);dev=np.flatnonzero(~data['fit']&data['available'])
    if len(data['target'])!=7617 or not data['available'].all() or len(fit)!=5876 or len(dev)!=1741:raise ValueError('Fixed known7617/FIT5876/DEV1741')
    draw_sha=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    if draw_sha!=DRAW_SHA:raise ValueError('Root exact original1000 draw hash')
    out=safe(e['output_directory'])
    if not out.is_relative_to(R) or 'v239' not in str(out).casefold() or out.exists():raise ValueError('Fresh owned V239 output')
    out.mkdir();started=time.monotonic()
    write_json(out/'STARTED.json',{'entry':binding(entry_path),'automatic_retry':False,'source_only':True})
    try:
        global_QA=write_json(out/'ALL7617_CANONICAL_FEATURE_QA_BEFORE_OPTIMIZERS.json',global_feature_QA(features,data,geometry))
        module=load_code(e['head_module'],codes,'v239_canonical_head',MODULE_PIN)
        configure_runtime();torch.manual_seed(17);control=module.CropFamilyHead().float().to(e['device'])
        torch.manual_seed(17);treat=module.RelationalQueryHead().float().to(e['device'])
        initial_base_sha=state_sha(control.state_dict())
        if initial_base_sha!=INITIAL_BASE_SHA or initial_base_sha!=state_sha(treat.base.state_dict()) or sum(p.numel() for p in control.parameters())!=76098 or sum(p.numel() for p in treat.parameters())!=86054:raise ValueError('Both originalseed17 baseinitialexact/params')
        if torch.count_nonzero(treat.relational.readout[-1].weight) or torch.count_nonzero(treat.relational.readout[-1].bias):raise ValueError('Initial residualoutputzero')
        initial_QA=write_json(out/'COMMON_INITIAL_BASE_AND_DRAW_QA.json',{'completed':True,'PASS':True,'seed':17,'same_initial_base_sha256':initial_base_sha,'draws_sha256':draw_sha,'updates_per_arm':1000,'new_CONTROL_training':True,'old_CONTROL_reuse':False})
        ledger=out/'COMMON_DRAW_LEDGER1000.jsonl'
        with ledger.open('x') as stream:
            for step,indices in enumerate(draws,1):stream.write(json.dumps({'update':step,'sample_indices':indices.tolist(),'draws_sha256':draw_sha,'family_counts':[16,16],'all_fit_available_known':True})+'\n')
        fit_features=torch.as_tensor(np.array(features['ORIGINAL_ALL_FEATURES.npy'][fit],copy=True),device=e['device'])
        fit_map=np.full(len(eligible),-1,np.int64);fit_map[fit]=np.arange(len(fit))
        checkpoints={};logs={};final_sha={};initial_sha={}
        for arm,head in ((CONTROL,control),(TREAT,treat)):
            checkpoints[arm],logs[arm],final_sha[arm],initial_sha[arm]=train_arm(arm,head,data,features,geometry,draws,eligible,fit_features,fit_map,e['device'],out,binding(entry_path),e['feature_complete'],draw_sha,initial_base_sha)
        freeze=write_json(out/'ALL2_FINAL1000_FROZEN_BEFORE_ANY_DEV_HEAD_OUTPUT.json',{'completed':True,'PASS':True,'checkpoints':checkpoints,'logs':logs,'all_final_updates':{CONTROL:1000,TREAT:1000},'canonical_feature_QA':global_QA,'common_initial_draw_QA':initial_QA,'same_initial_base_sha256':initial_base_sha,'draws_sha256':draw_sha,'new_DEV_head_output_before_bothfinalfreeze':0,'old_CONTROL_or_prefix_used':False})
        heads={CONTROL:control,TREAT:treat};scores={arm:{} for arm in heads}
        for head in heads.values():head.eval().requires_grad_(False)
        available=np.flatnonzero(data['available'])
        with torch.no_grad():
            for first in range(0,len(available),64):
                indices=available[first:first+64];x=torch.as_tensor(np.array(features['ORIGINAL_ALL_FEATURES.npy'][indices],copy=True),device=e['device'])
                args=context_args(indices,features,geometry,e['device'])
                for arm,head in heads.items():
                    z=(head(x) if arm==CONTROL else head(x,*args)[0]).cpu().numpy()
                    if not np.isfinite(z).all():raise ValueError('Finite frozen originalavailable logits')
                    scores[arm].update({int(i):v for i,v in zip(indices,z)})
        outputs={};rowsets={}
        for arm in heads:
            rows=[]
            for i in range(len(data['target'])):
                family=int(data['family'][i]);z=scores[arm].get(i);v=None if z is None else float(z[family])
                rows.append({'sample_index':i,'row_id':str(data['row_id'][i]),'family':family,'role':'FIT' if data['fit'][i] else 'DEV','available':bool(data['available'][i]),'target':int(data['target'][i]),'logit0':None if z is None else float(z[0]),'logit1':None if z is None else float(z[1]),'selected_family_logit':v,'prediction':'U' if v is None else 'E' if v>=0 else 'D'})
            csv_path=out/(arm+'_ALL_SOURCE_ROWS.csv')
            with csv_path.open('x',newline='') as stream:writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            outputs[arm]=binding(csv_path);rowsets[arm]=rows
            if state_sha(heads[arm].state_dict())!=final_sha[arm]:raise ValueError('Finalhead mutated afterfreeze')
        reductions={arm:reduce_rows(rows) for arm,rows in rowsets.items()}
        summary=write_json(out/'SOURCE_RESULTS.json',{'completed':True,'PASS':True,'arms':reductions,'paired_changes':paired_changes(rowsets[CONTROL],rowsets[TREAT]),'SOURCE_gate':source_gate(reductions[CONTROL],reductions[TREAT]),'two_SOURCE_referents_separate':True,'current_admission':False,'old_V233_CONTROL_reuse':False,'all_original_rows':15040,'known_output_rows':7617,'masked_target_minus1_retained':7423})
        complete={'completed':True,'PASS':True,'entry':binding(entry_path),'code':e['code'],'implementation_protocol':e['implementation_protocol'],'root_protocol':e['root_protocol'],'root_scope':e['root_scope'],'feature_complete':e['feature_complete'],'features':e['features'],'samples_npz':e['samples_npz'],'geometry_npz':e['geometry_npz'],'head_module':e['head_module'],'checkpoints':checkpoints,'logs':logs,'outputs':outputs,'summary':summary,'freeze':freeze,'canonical_feature_QA':global_QA,'common_initial_draw_QA':initial_QA,'draw_ledger':binding(ledger),'draws_sha256':draw_sha,'same_initial_base_sha256':initial_base_sha,'initial_head_sha256':initial_sha,'final_head_sha256':final_sha,'new_optimizer_updates':{CONTROL:1000,TREAT:1000},'head_parameters':{CONTROL:76098,TREAT:86054},'optimizer_state_tensors':{CONTROL:4,TREAT:10},'new_DEV_head_output_before_bothfinalfreeze':0,'old_CONTROL_prefix_DINO_RGB_GT_or_protected_payload_reads':0,'new_V235_CF':0,'automatic_retry':False,'current_admission':False,'actual_seconds':time.monotonic()-started}
        write_json(out/'COMPLETE.json',complete);print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(out/'COMPLETE.json')}),flush=True)
    except BaseException:
        write_json(out/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entry',required=True);run(parser.parse_args().entry)

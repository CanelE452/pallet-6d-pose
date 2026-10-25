"""V248 matched SOURCE continuation with separate publicpixel AUX.
CODE-only preparation: actual CP/features/arrays require a distinct Root true entry.
No encoder, RGB, teacher certificate or protected-input reader is present.
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
    if any(x in text.casefold() for x in ('v112', 'cache', '.pyc', 'current223', 'publicheld', 'public_held', 'oldval', 'oldhold', '/hold/', 'v177', 'v178', 'source_holdout', 'source_val', 'sourceval', 'v182', 'private_evaluation')):
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

# New V248 interface. SOURCE helper bodies are reused; safe() adds lexical cache/held guards.
CONTROL='CONTROL_NO_PUBLIC_AUX'
TREAT='TREAT_PUBLIC_PIXEL_AUX'
ARMS=(CONTROL,TREAT)
WEIGHTS={CONTROL:0.,TREAT:.25}
AUX_ROOT_PIN=(str(R/'V248_ROOT_PUBLIC_PIXEL_AUX/PROTOCOL_BEFORE_NEW_AUX_ARRAYS_OR_FEATURES_OR_FIT.json'),'f66d9e29f85141527aa2761311a29e512d611c6ba14b287d21dbb08c589e4fda')
PAIR_USAGE_PIN=(str(R/'V248_ROOT_PUBLIC_PIXEL_AUX/AUX_PAIR_USAGE_FROZEN_BEFORE_ARRAYS.json'),'6e8324f107a4b14903a531b591daac0bf9d4d10e2093b1d219f559a41eb45497')
ORIGINAL_CODE_PIN=(str(R/'V239_CANONICAL_PAIRED_TRAINER_A/train_canonical_pair.py'),'ce3380b5938132717d349b7be27545e5347110da8663d2b698ec44c96dbfe419')
WARM_COMPLETE_PIN=(str(R/'V239_ROOT_CANONICAL_CONTEXT/canonical_pair_source_fit_v1/COMPLETE.json'),'1fd6880528d1bd440376c68c7079b9cbee59ce736f006fdaef4244e3310a4df9')
WARM_CP_SHA='e645b50a2b9a41f6a06b502040c6683a7a5c3955093c07128b1095992409e65a'
OLD_ARM='TREAT_CANONICAL_CONTEXT'
AUX_KEYS=('target','fit','available','group_index','row_id','query_xy','L','pair_id','view','source_role','original_source_split')
RECIPE={'steps_each':1000,'prior_core_steps':1000,'cumulative_core_steps_each':2000,
 'seed':17,'main_batch':32,'main_family_D_E':[8,8],'aux_batch':32,'aux_V_E':[16,16],
 'aux_weight':WEIGHTS,'lr':.001,'weight_decay':.0001,'threshold_logit':0.,
 'aux_L':64.,'encoder_updates':0,'use_lora':False,'new_CF':False,'automatic_retry':False}
MAIN_COLS=('sample_index','row_id','family','role','available','target','logit0','logit1','selected_family_logit','prediction')
AUX_COLS=('aux_index','row_id','pair_id','view','original_source_split','role','available','target','logit','prediction')

class PublicPixelAuxModel(torch.nn.Module):
    """Unchanged relational core plus shared-stem, independent scalar AUX readout."""
    def __init__(self,core):
        super().__init__();self.core=core
        # Reset only the declared initialization RNG. Default Linear is nonzero.
        torch.manual_seed(17)
        self.public_pixel=torch.nn.Linear(64,1)
    def forward_main(self,x,*args):
        return self.core(x,*args)[0]
    def forward_aux(self,pixel1154):
        if pixel1154.ndim!=2 or pixel1154.shape[1]!=1154 or pixel1154.dtype!=torch.float32 or not torch.isfinite(pixel1154).all():
            raise ValueError('Finite FP32 AUX pixel1154 only; no native/q8')
        zero32=pixel1154.new_zeros((len(pixel1154),32))
        x=torch.cat((pixel1154,zero32),1)
        hidden=self.core.base.head[1](self.core.base.head[0](x))
        return self.public_pixel(hidden).squeeze(1)

def require_complete(obj,label):
    if obj.get('completed') is not True or obj.get('PASS') is not True:raise ValueError(label)

def exact_binding(a,b):
    if identity(a)!=identity(b):raise ValueError('Exact artifact path/SHA identity required')

def read_metadata(b,white,pin=None):
    return json.loads(read(verify(b,white,pin)))

def pair_roles(plan):
    if plan.get('schema')!='V248_AUX_INTERNAL_USAGE_WITHIN_ORIGINAL_PUBLIC_TRAIN22_V1' or plan.get('original_TRAIN22_or_native352_splits_changed') is not False:
        raise ValueError('Original public TRAIN22 usage role plan only')
    rows=plan['roles'];mapping={r['pair_id']:r['new_AUX_usage'] for r in rows}
    if len(rows)!=22 or len(mapping)!=22 or sum(v=='AUX_FIT' for v in mapping.values())!=17 or sum(v=='AUX_DEV' for v in mapping.values())!=5:
        raise ValueError('Exact fixed17 FIT/5 DEV pair groups')
    order=sorted(mapping,key=lambda p:hashlib.sha256(('V248:AUX_PAIR_USAGE:17:'+p).encode()).hexdigest())
    dev=set(order[:5])
    for r in rows:
        digest=hashlib.sha256(('V248:AUX_PAIR_USAGE:17:'+r['pair_id']).encode()).hexdigest()
        if r['original_source_split']!='TRAIN_LAB' or r['role_hash_sha256']!=digest or r['new_AUX_usage']!=('AUX_DEV' if r['pair_id'] in dev else 'AUX_FIT'):
            raise ValueError('Pair role hash/original TRAIN identity')
    return mapping

def validate_aux(d,features,roles):
    if set(d)!=set(AUX_KEYS):raise ValueError('Exact AUX nonpixel sample schema')
    n=len(d['target'])
    if n<1 or features.dtype!=np.float32 or features.shape!=(n,1154):raise ValueError('AUX features exact N×1154 FP32')
    for k in AUX_KEYS:
        if d[k].shape!=((n,2) if k=='query_xy' else (n,)):raise ValueError('AUX metadata shape')
    if d['target'].dtype!=np.int8 or not np.isin(d['target'],[0,1]).all():raise ValueError('Conditional V0/E1 only; no U/SELF as E')
    if any(d[k].dtype!=bool for k in ('fit','available')):raise ValueError('AUX fit/support bools')
    if d['group_index'].dtype!=np.int64 or not ((d['group_index']>=0)&(d['group_index']<44)).all():raise ValueError('AUX group index original22pair×2view')
    if d['query_xy'].dtype!=np.float64 or not np.isfinite(d['query_xy']).all() or d['L'].dtype!=np.float64 or not (d['L']==64.).all():raise ValueError('Prospective AUX query F64/L64, no GT recrop of native queries')
    for k in ('row_id','pair_id','view','source_role','original_source_split'):
        if d[k].dtype.kind!='U':raise ValueError('AUX string metadata only')
    if len(set(d['row_id'].tolist()))!=n or not np.isin(d['view'],['A','B']).all() or not (d['original_source_split']=='TRAIN_LAB').all():raise ValueError('Unique new AUX IDs and original TRAIN22 views')
    groups={};reverse={}
    for i in range(n):
        pair=str(d['pair_id'][i]);view=str(d['view'][i]);gi=int(d['group_index'][i])
        if pair not in roles or d['source_role'][i]!=roles[pair] or bool(d['fit'][i])!=(roles[pair]=='AUX_FIT'):
            raise ValueError('Every sibling inherits prospective pair usage; no split by pixel')
        key=(pair,view)
        if gi in groups and groups[gi]!=key or key in reverse and reverse[key]!=gi:raise ValueError('AUX group one-to-one pair/view')
        groups[gi]=key;reverse[key]=gi
    if set(d['pair_id'][d['fit']].tolist())&set(d['pair_id'][~d['fit']].tolist()):raise ValueError('No AUX pair overlap FIT/DEV')
    for start in range(0,n,256):
        take=np.arange(start,min(start+256,n));x=np.asarray(features[take])
        if not np.isfinite(x).all() or np.count_nonzero(x[~d['available'][take]]):raise ValueError('Unsupported AUX preserved/zero feature, no forced negative')
    return {'completed':True,'PASS':True,'records':n,'supported':int(d['available'].sum()),'unsupported_retained':int((~d['available']).sum()),'pair_usage_exact':True,'native_slot_assigned':False,'q8_or_native_forward':False,'fitted_normalizer':False}

def aux_draws(d):
    eligible=d['fit']&d['available'];pools=[np.flatnonzero(eligible&(d['target']==y)) for y in (0,1)]
    if any(len(x)==0 for x in pools):raise ValueError('Empty supported AUX_FIT class; no refill/split change')
    rng=np.random.Generator(np.random.PCG64(17))
    draws=np.stack([np.concatenate([rng.choice(p,16,replace=True) for p in pools]) for _ in range(1000)]).astype(np.int64)
    return draws,eligible

def aux_loss(logits,targets):
    if logits.shape!=(32,) or targets.shape!=(32,) or int((targets==0).sum())!=16 or int((targets==1).sum())!=16:raise ValueError('Fixed32 balanced AUX V16/E16')
    return F.binary_cross_entropy_with_logits(logits,targets.to(logits.dtype))

def aux_gradient_probe(model,pixels,targets):
    """One AUX-only backward, no optimizer or parameter/RNG change."""
    before=state_sha(model.state_dict());rng=torch.get_rng_state().clone()
    model.zero_grad(set_to_none=True)
    loss=aux_loss(model.forward_aux(pixels),targets);loss.backward()
    stem=model.core.base.head[0];w=stem.weight.grad;b=stem.bias.grad
    public=[p.grad for p in model.public_pixel.parameters()]
    if w is None or b is None or not torch.isfinite(w).all() or not torch.isfinite(b).all() or not torch.count_nonzero(w[:,:1154]) or not torch.count_nonzero(b) or torch.count_nonzero(w[:,1154:]):
        raise ValueError('AUX must reach shared visual1154+bias, never condition32 columns')
    forbidden=[*model.core.base.head[2].parameters(),*model.core.relational.parameters()]
    if any(p.grad is not None and torch.count_nonzero(p.grad) for p in forbidden) or any(g is None or not torch.isfinite(g).all() or not torch.count_nonzero(g) for g in public):
        raise ValueError('AUX gradient isolated from old output/relational; new scalar must learn')
    if before!=state_sha(model.state_dict()) or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Probe changed parameter/RNG state')
    receipt={'completed':True,'PASS':True,'shared_visual_weight_nonzero':True,'shared_bias_nonzero':True,'shared_condition32_gradient_nonzeros':0,'old_output_or_relational_gradient_nonzeros':0,'public_scalar_gradients_nonzero':True,'optimizer_updates':0,'state_before_after_sha256':before,'AUX_only':True}
    model.zero_grad(set_to_none=True)
    return receipt

def metadata_gate(e):
    codes,meta,payload=e['code_whitelist'],e['metadata_whitelist'],e['payload_whitelist']
    verify(e['code'],codes,(str(safe(__file__)),binding(__file__)['sha256']));verify(e['implementation_protocol'],codes)
    verify(e['original_trainer_code'],codes,ORIGINAL_CODE_PIN);verify(e['head_module'],codes,MODULE_PIN)
    root=read_metadata(e['root_protocol'],meta,AUX_ROOT_PIN);usage=read_metadata(e['pair_usage'],meta,PAIR_USAGE_PIN)
    scope=read_metadata(e['root_scope'],meta)
    if scope.get('approved_source_public_pixel_aux_training') is not True or scope.get('current_or_held_payloads_authorized') is not False:raise ValueError('Exact Root SOURCE-only fit read scope')
    packet=read_metadata(e['main_feature_complete'],meta);warm=read_metadata(e['warmstart_complete'],meta,WARM_COMPLETE_PIN)
    aux=read_metadata(e['aux_feature_complete'],meta)
    for obj,label in [(packet,'Original features incomplete'),(warm,'V239 core incomplete'),(aux,'AUX features incomplete')]:require_complete(obj,label)
    exact_binding(aux['root_protocol'],e['root_protocol']);exact_binding(aux['pair_usage'],e['pair_usage'])
    exact_binding(aux['features'],e['aux_features']);exact_binding(aux['samples'],e['aux_samples_npz'])
    if packet['counts']['known']!=7617 or packet['counts']['FIT']!=5876 or packet['counts']['DEV']!=1741 or packet['counts']['all_original_rows']!=15040 or packet['counts']['masked_target_minus1']!=7423:
        raise ValueError('Original source population unchanged')
    exact_binding(warm['feature_complete'],e['main_feature_complete']);exact_binding(warm['checkpoints'][OLD_ARM],e['warmstart_checkpoint']);exact_binding(warm['outputs'][OLD_ARM],e['baseline_CSV'])
    if e['warmstart_checkpoint']['sha256']!=WARM_CP_SHA or warm['new_optimizer_updates'][OLD_ARM]!=1000 or warm['head_parameters'][OLD_ARM]!=86054:raise ValueError('Exact frozen V239 final1000 core')
    for k in ('samples_npz','geometry_npz'):
        exact_binding(e[k],packet[k]);exact_binding(e[k],warm[k])
    for name in FEATURE_LAYOUT:exact_binding(e['main_features'][name],packet['features'][name])
    if packet['official_base']['sha256']!='f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb' or aux['official_base']['sha256']!=packet['official_base']['sha256'] or aux['asset_identity']!=packet['asset_identity']:
        raise ValueError('Same frozen official feature asset/identity')
    if aux['base_state_sha256_before']!=aux['base_state_sha256_after']:raise ValueError('AUX encoder frozen')
    expected=[*e['main_features'].values(),e['samples_npz'],e['geometry_npz'],e['warmstart_checkpoint'],e['baseline_CSV'],e['aux_features'],e['aux_samples_npz']]
    if len(payload)!=len(expected) or {identity(x) for x in payload}!={identity(x) for x in expected}:raise ValueError('Exact feature/nonpixel samples/warm CP/baselineCSV only; no RGB/certificates')
    return packet,warm,aux,pair_roles(usage)

def load_payload(e,roles):
    payload=e['payload_whitelist'];features={}
    for name,(dtype,shape) in FEATURE_LAYOUT.items():
        a=np.load(verify(e['main_features'][name],payload),mmap_mode='r',allow_pickle=False)
        if a.dtype!=dtype or a.shape!=shape:raise ValueError('Original canonical feature shape/dtype')
        features[name]=a
    with np.load(verify(e['samples_npz'],payload),allow_pickle=False) as z:
        if set(z.files)!=set(SAMPLE_KEYS):raise ValueError('Original main samples keys')
        data=validate_samples({k:z[k] for k in SAMPLE_KEYS})
    with np.load(verify(e['geometry_npz'],payload),allow_pickle=False) as z:
        if set(z.files)!=set(GEOMETRY_KEYS):raise ValueError('Original input geometry keys')
        geometry=validate_geometry({k:z[k] for k in GEOMETRY_KEYS},1767)
    ax=np.load(verify(e['aux_features'],payload),mmap_mode='r',allow_pickle=False)
    with np.load(verify(e['aux_samples_npz'],payload),allow_pickle=False) as z:
        if set(z.files)!=set(AUX_KEYS):raise ValueError('AUX sample exact nonpixel fields')
        ad={k:z[k] for k in AUX_KEYS}
    aq=validate_aux(ad,ax,roles)
    draws,eligible=fixed_draws(data);adraws,aeligible=aux_draws(ad)
    if len(data['target'])!=7617 or not data['available'].all() or eligible.sum()!=5876 or (~data['fit']).sum()!=1741 or hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()!=DRAW_SHA:
        raise ValueError('Original7617/5876/1741 and exact main1000 draws')
    return data,features,geometry,ad,ax,draws,adraws,eligible,aeligible,aq

def make_models(module,cp,device):
    if cp.get('arm')!=OLD_ARM or cp.get('updates')!=1000 or cp.get('head_parameters')!=86054 or cp.get('use_lora') is not False or cp.get('draws_sha256')!=DRAW_SHA:
        raise ValueError('V239 checkpoint typed identity')
    core_state=cp['head_state_dict'];warm_sha=state_sha(core_state);models={}
    for arm in ARMS:
        torch.manual_seed(17);core=module.RelationalQueryHead().float()
        core.load_state_dict(core_state,strict=True)
        models[arm]=PublicPixelAuxModel(core).float().to(device)
        if state_sha(models[arm].core.state_dict())!=warm_sha or sum(p.numel() for p in models[arm].core.parameters())!=86054 or sum(p.numel() for p in models[arm].parameters())!=86119:
            raise ValueError('Strict identical V239 warm core +65 new public scalar')
    if state_sha(models[CONTROL].state_dict())!=state_sha(models[TREAT].state_dict()) or not torch.count_nonzero(models[CONTROL].public_pixel.weight):raise ValueError('Same nonzero default seed17 public initialization')
    return models,warm_sha

def write_draw_ledger(out,draws,adraws):
    aux_sha=hashlib.sha256(adraws.astype('<i8').tobytes()).hexdigest();p=out/'COMMON_MAIN_AND_AUX_DRAW_LEDGER1000.jsonl'
    with p.open('x') as f:
        for i,(m,a) in enumerate(zip(draws,adraws),1):f.write(json.dumps({'update':i,'main_indices':m.tolist(),'aux_indices':a.tolist(),'main_draws_sha256':DRAW_SHA,'aux_draws_sha256':aux_sha,'aux_source':'SUPPORTED_AUX_FIT_ONLY','AUX_used_by_control':False})+'\n')
    return binding(p),aux_sha

def train_arm_aux(arm,model,data,features,geometry,ad,ax,draws,adraws,eligible,aeligible,device,out,e,aux_sha):
    initial=cpu_copy(model.state_dict());initial_sha=state_sha(initial);public_before=state_sha(model.public_pixel.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    if optimizer.state:raise ValueError('Reset optimizer')
    model.train();p=out/(arm+'_TRAIN1000.jsonl')
    with p.open('x') as f:
        for step,(indices,ai) in enumerate(zip(draws,adraws),1):
            if not eligible[indices].all() or not aeligible[ai].all():raise ValueError('Only unchanged main FIT/available and AUX_FIT/support in draws')
            x=torch.as_tensor(np.array(features['ORIGINAL_ALL_FEATURES.npy'][indices],copy=True),device=device)
            optimizer.zero_grad(set_to_none=True)
            logits=model.forward_main(x,*context_args(indices,features,geometry,device))
            main,parts=family_loss(logits,torch.as_tensor(data['target'][indices],device=device),torch.as_tensor(data['family'][indices],device=device))
            extra=None
            if arm==TREAT:
                pixels=torch.as_tensor(np.array(ax[ai],copy=True),device=device)
                extra=aux_loss(model.forward_aux(pixels),torch.as_tensor(ad['target'][ai],device=device))
            loss=main if extra is None else main+.25*extra
            if not torch.isfinite(loss):raise ValueError('Finite paired fixed objective')
            loss.backward()
            if arm==CONTROL and any(p.grad is not None for p in model.public_pixel.parameters()):raise ValueError('CONTROL has no AUX gradient/decay path')
            optimizer.step()
            f.write(json.dumps({'update':step,'main_indices':indices.tolist(),'aux_indices':ai.tolist(),'main_draws_sha256':DRAW_SHA,'aux_draws_sha256':aux_sha,'family0_BCE':float(parts[0].detach()),'family1_BCE':float(parts[1].detach()),'main_loss':float(main.detach()),'aux_BCE':None if extra is None else float(extra.detach()),'aux_weight':WEIGHTS[arm],'total_loss':float(loss.detach()),'main_family_counts':[16,16],'aux_class_counts':[16,16],'all_main_FIT_available_known':True,'all_aux_FIT_supported':True,'AUX_forward_and_backward':arm==TREAT},allow_nan=False)+'\n')
            if step%100==0:print(json.dumps({'stage':arm,'new_updates':step,'fixed_final':1000}),flush=True)
    expected=10 if arm==CONTROL else 12
    if len(optimizer.state)!=expected or any(float(s['step'])!=1000 or any(not torch.isfinite(v).all() for v in s.values() if isinstance(v,torch.Tensor)) for s in optimizer.state.values()):raise ValueError('Finite expected AdamW states@1000')
    if arm==CONTROL and (any(p in optimizer.state for p in model.public_pixel.parameters()) or state_sha(model.public_pixel.state_dict())!=public_before):raise ValueError('CONTROL public parameters not optimized/decayed')
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Finite frozen final parameters')
    path=out/(arm+'_FINAL1000.pt')
    torch.save({'arm':arm,'new_updates':1000,'prior_core_updates':1000,'cumulative_core_updates':2000,'total_parameters':86119,'core_parameters':86054,'public_parameters':65,'core_state_dict':cpu_copy(model.core.state_dict()),'public_aux_state_dict':cpu_copy(model.public_pixel.state_dict()),'model_state_dict':cpu_copy(model.state_dict()),'optimizer_state_dict':cpu_copy(optimizer.state_dict()),'initial_model_state_dict':initial,'initial_model_sha256':initial_sha,'warmstart_checkpoint':e['warmstart_checkpoint'],'main_draws_sha256':DRAW_SHA,'aux_draws_sha256':aux_sha,'aux_weight':WEIGHTS[arm],'public_optimizer_updates':0 if arm==CONTROL else 1000,'entry':e['_entry_binding'],'head_module':e['head_module'],'code':e['code'],'root_protocol':e['root_protocol'],'AUX_native_gold':False},path)
    return binding(path),binding(p),state_sha(model.state_dict()),initial_sha

def save_csv(path,cols,rows):
    with safe(path).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    return binding(path)

def main_outputs(model,data,features,geometry,device):
    scores={};model.eval().requires_grad_(False)
    with torch.no_grad():
        for start in range(0,len(data['target']),64):
            i=np.arange(start,min(start+64,len(data['target'])));i=i[data['available'][i]]
            if not len(i):continue
            x=torch.as_tensor(np.array(features['ORIGINAL_ALL_FEATURES.npy'][i],copy=True),device=device)
            z=model.forward_main(x,*context_args(i,features,geometry,device)).cpu().numpy()
            if not np.isfinite(z).all():raise ValueError('Finite frozen main logits')
            scores.update({int(k):v for k,v in zip(i,z)})
    rows=[]
    for i in range(len(data['target'])):
        family=int(data['family'][i]);z=scores.get(i);v=None if z is None else float(z[family])
        rows.append({'sample_index':i,'row_id':str(data['row_id'][i]),'family':family,'role':'FIT' if data['fit'][i] else 'DEV','available':bool(data['available'][i]),'target':int(data['target'][i]),'logit0':None if z is None else float(z[0]),'logit1':None if z is None else float(z[1]),'selected_family_logit':v,'prediction':'U' if v is None else 'E' if v>=0 else 'D'})
    return rows

def aux_outputs(model,d,x,device):
    scores={}
    with torch.no_grad():
        for start in range(0,len(d['target']),64):
            i=np.arange(start,min(start+64,len(d['target'])));i=i[d['available'][i]]
            if not len(i):continue
            z=model.forward_aux(torch.as_tensor(np.array(x[i],copy=True),device=device)).cpu().numpy()
            if not np.isfinite(z).all():raise ValueError('Finite frozen AUX logits')
            scores.update({int(j):float(v) for j,v in zip(i,z)})
    return [{'aux_index':i,'row_id':str(d['row_id'][i]),'pair_id':str(d['pair_id'][i]),'view':str(d['view'][i]),'original_source_split':str(d['original_source_split'][i]),'role':str(d['source_role'][i]),'available':bool(d['available'][i]),'target':int(d['target'][i]),'logit':scores.get(i),'prediction':'U' if i not in scores else 'E' if scores[i]>=0 else 'V'} for i in range(len(d['target']))]

def read_baseline_after_freeze(b,white,data):
    import io
    rows=list(csv.DictReader(io.StringIO(read(verify(b,white)).decode())))
    if len(rows)!=7617:raise ValueError('Exact frozen V239 baseline denominator')
    out=[]
    for i,r in enumerate(rows):
        if int(r['sample_index'])!=i or r['row_id']!=str(data['row_id'][i]) or int(r['family'])!=int(data['family'][i]) or int(r['target'])!=int(data['target'][i]) or r['role']!=('FIT' if data['fit'][i] else 'DEV') or r['available']!=str(bool(data['available'][i])):raise ValueError('Baseline original row identities unchanged')
        z=float(r['selected_family_logit'])
        if not np.isfinite(z) or r['prediction']!=('E' if z>=0 else 'D'):raise ValueError('Original frozen baseline scalar decoding')
        out.append({**r,'sample_index':i,'family':int(r['family']),'target':int(r['target']),'available':bool(data['available'][i]),'selected_family_logit':z})
    return out

def reduce_aux(rows):
    result={}
    for role in ('AUX_FIT','AUX_DEV'):
        rr=[r for r in rows if r['role']==role];cm=[[sum(r['target']==y and r['prediction']==p for r in rr) for p in ('V','E','U')] for y in (0,1)]
        item={'rows':len(rr),'CM_V_E_by_V_E_U':cm,'unsupported_U':sum(not r['available'] for r in rr),'native_gold':False}
        for name,j in [('V',0),('E',1)]:
            truth=sum(cm[j]);pred=cm[0][j]+cm[1][j];tp=cm[j][j]
            item[name]={'truth':truth,'predicted':pred,'precision':tp/pred if pred else None,'recall':tp/truth if truth else None}
        result[role]=item
    return result

def combined_gate(baseline,control,treat,aux_treat):
    legacy=source_gate(baseline,treat)
    def mean_br(m):
        values=[m[f'family{f}_DEV'][c]['recall'] for f in (0,1) for c in ('D','E')]
        return None if any(x is None for x in values) else sum(values)/4
    a,b=mean_br(control),mean_br(treat);gain=None if a is None or b is None else 100*(b-a)
    vr=aux_treat['AUX_DEV']['V']['recall'];er=aux_treat['AUX_DEV']['E']['recall']
    matched=gain is not None and gain>=1.;public=vr is not None and er is not None and vr>=.8 and er>=.8
    return {'PASS':bool(legacy['PASS'] and matched and public),'legacy_SOURCE_vs_V239':legacy,'matched_treatment_vs_control_gain_pp':gain,'matched_min_gain_pp':1.,'matched_PASS':matched,'AUX_DEV_V_recall':vr,'AUX_DEV_E_recall':er,'AUX_DEV_min_each':.8,'AUX_DEV_PASS':public,'empty_class_fill':False,'current_admission':False,'separate_current_evaluation_eligible':bool(legacy['PASS'] and matched and public),'automatic_current_execution':False}

def run(entry_path):
    e=json.loads(read(entry_path))
    if e.get('approved_source_public_pixel_aux_training') is not True or e.get('source_only') is not True or e.get('automatic_retry') is not False:raise PermissionError('Separate exact Root true SOURCE entry required')
    if e.get('fixed_recipe')!=RECIPE:raise ValueError('Exactly frozen matched AUX recipe')
    packet,warm,auxpacket,roles=metadata_gate(e)
    out=safe(e['output_directory'])
    if out.parent!=R/'V248_ROOT_PUBLIC_PIXEL_AUX' or out.name!='matched_source_public_pixel_aux_fit_v1' or out.exists():raise ValueError('Fresh fixed Root-owned SOURCE output only')
    out.mkdir();e['_entry_binding']=binding(entry_path);started=time.monotonic()
    write_json(out/'STARTED.json',{'entry':e['_entry_binding'],'automatic_retry':False,'source_only':True,'fixed_recipe':RECIPE})
    try:
        data,features,geometry,ad,ax,draws,adraws,eligible,aeligible,aq=load_payload(e,roles)
        globalQA=write_json(out/'GLOBAL_MAIN_AND_AUX_FEATURE_QA_BEFORE_OPTIMIZERS.json',{'completed':True,'PASS':True,'main':global_feature_QA(features,data,geometry),'AUX':aq,'main_all_rows':15040,'known':7617,'masked_S_U':7423,'FIT':5876,'DEV':1741,'before_optimizer':True,'new_DEV_logits':0})
        ledger,aux_sha=write_draw_ledger(out,draws,adraws)
        module=load_code(e['head_module'],e['code_whitelist'],'v248_frozen_relational',MODULE_PIN)
        configure_runtime()
        cp=torch.load(verify(e['warmstart_checkpoint'],e['payload_whitelist']),weights_only=True,map_location='cpu')
        if identity(cp['feature_complete'])!=identity(e['main_feature_complete']) or state_sha(cp['head_state_dict'])!=warm['final_head_sha256'][OLD_ARM]:raise ValueError('Warm checkpoint feature/state matches completed V239')
        models,warm_sha=make_models(module,cp,e['device']);del cp
        probe_i=adraws[0];probe_model=models[TREAT]
        probe=aux_gradient_probe(probe_model,torch.as_tensor(np.array(ax[probe_i],copy=True),device=e['device']),torch.as_tensor(ad['target'][probe_i],device=e['device']))
        probe_binding=write_json(out/'AUX_ONLY_SHARED_STEM_GRADIENT_QA_NO_OPTIMIZER.json',probe)
        if state_sha(models[CONTROL].state_dict())!=state_sha(models[TREAT].state_dict()):raise ValueError('Probe preserved matched initialization')
        checkpoints={};logs={};final={};initial={}
        for arm in ARMS:
            checkpoints[arm],logs[arm],final[arm],initial[arm]=train_arm_aux(arm,models[arm],data,features,geometry,ad,ax,draws,adraws,eligible,aeligible,e['device'],out,e,aux_sha)
        if initial[CONTROL]!=initial[TREAT]:raise ValueError('Matched initial core/public state')
        freeze=write_json(out/'BOTH_FINAL1000_AND_LOGS_FROZEN_BEFORE_ANY_DEV_OUTPUT.json',{'completed':True,'PASS':True,'checkpoints':checkpoints,'logs':logs,'new_updates':{a:1000 for a in ARMS},'cumulative_core_updates':{a:2000 for a in ARMS},'public_optimizer_updates':{CONTROL:0,TREAT:1000},'main_feature_QA':globalQA,'gradient_QA':probe_binding,'warmstart_checkpoint':e['warmstart_checkpoint'],'warm_core_state_sha256':warm_sha,'matched_initial_model_sha256':initial[CONTROL],'draw_ledger':ledger,'main_draws_sha256':DRAW_SHA,'aux_draws_sha256':aux_sha,'new_original_DEV_or_AUX_DEV_logits_before_bothfinalfreeze':0,'baseline_CSV_interpretation_before_freeze':0})
        # No baseline CSV body or new DEV head output is interpreted above.
        old=read_baseline_after_freeze(e['baseline_CSV'],e['payload_whitelist'],data)
        outputs={};auxoutputs={};reductions={};auxreductions={}
        for arm in ARMS:
            rows=main_outputs(models[arm],data,features,geometry,e['device']);arows=aux_outputs(models[arm],ad,ax,e['device'])
            outputs[arm]=save_csv(out/(arm+'_ALL_SOURCE_ROWS.csv'),MAIN_COLS,rows)
            auxoutputs[arm]=save_csv(out/(arm+'_ALL_PUBLIC_AUX_ROWS.csv'),AUX_COLS,arows)
            reductions[arm]=reduce_rows(rows);auxreductions[arm]=reduce_aux(arows)
            if final[arm]!=state_sha(models[arm].state_dict()):raise ValueError('Final model mutated during saved output')
        summary=write_json(out/'SOURCE_RESULTS.json',{'completed':True,'PASS':True,'arms':reductions,'public_AUX':auxreductions,'baseline_V239':reduce_rows(old),'SOURCE_gate':combined_gate(reduce_rows(old),reductions[CONTROL],reductions[TREAT],auxreductions[TREAT]),'native_and_conditional_pixel_referents_separate':True,'SOURCE_DEV_and_internal_PUBLIC_TRAIN_AUX_DEV_exposed_not_blind':True,'new_current_payload_or_inference':0,'current_admission':False})
        complete={'completed':True,'PASS':True,'schema':'V248_MATCHED_PUBLIC_PIXEL_AUX_SOURCE_TRIAL_V1','entry':e['_entry_binding'],'code':e['code'],'implementation_protocol':e['implementation_protocol'],'root_protocol':e['root_protocol'],'root_scope':e['root_scope'],'pair_usage':e['pair_usage'],'main_feature_complete':e['main_feature_complete'],'aux_feature_complete':e['aux_feature_complete'],'main_features':e['main_features'],'aux_features':e['aux_features'],'samples_npz':e['samples_npz'],'geometry_npz':e['geometry_npz'],'aux_samples_npz':e['aux_samples_npz'],'warmstart_complete':e['warmstart_complete'],'warmstart_checkpoint':e['warmstart_checkpoint'],'head_module':e['head_module'],'checkpoints':checkpoints,'logs':logs,'outputs':outputs,'aux_outputs':auxoutputs,'summary':summary,'freeze':freeze,'global_feature_QA':globalQA,'gradient_QA':probe_binding,'draw_ledger':ledger,'main_draws_sha256':DRAW_SHA,'aux_draws_sha256':aux_sha,'matched_initial_model_sha256':initial[CONTROL],'final_model_sha256':final,'new_optimizer_updates':{a:1000 for a in ARMS},'total_new_optimizer_updates':2000,'cumulative_core_updates':{a:2000 for a in ARMS},'core_parameters':86054,'public_parameters':65,'total_parameters':86119,'optimizer_state_tensors':{CONTROL:10,TREAT:12},'public_optimizer_updates':{CONTROL:0,TREAT:1000},'new_DEV_logits_before_bothfinalfreeze':0,'teacher_native_gold_created':False,'DINO_RGB_certificate_current_protected_payload_reads':0,'automatic_retry':False,'current_admission':False,'actual_seconds':time.monotonic()-started}
        write_json(out/'COMPLETE.json',complete);print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(out/'COMPLETE.json')}),flush=True)
    except BaseException:
        write_json(out/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc(),'partial_outputs_preserved':True});raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entry',required=True);args=parser.parse_args();run(args.entry)

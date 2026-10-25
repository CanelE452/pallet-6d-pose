"""V243 scene-CF data-only SOURCE head trial. Root true entry required.
No encoder, RGB, teacher reference, original mixed cache or private loader.
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
CONTROL='BASELINE_V239_CANONICAL_CONTEXT'
TREAT='TREAT_SCENE_CF_MATCHED_DATA'
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


BASELINE_COMPLETE_PIN=(str(R/'V239_ROOT_CANONICAL_CONTEXT/canonical_pair_source_fit_v1/COMPLETE.json'),'1fd6880528d1bd440376c68c7079b9cbee59ce736f006fdaef4244e3310a4df9')
OLD_ARM='TREAT_CANONICAL_CONTEXT'
INITIAL_FULL_SHA='196bf2dfb05f79658af04bbc753bbb377a3d5e51bb65eb4135da4d1c7faaea7d'
ORIGINAL_CODE_PIN=(str(R/'V239_CANONICAL_PAIRED_TRAINER_A/train_canonical_pair.py'),'ce3380b5938132717d349b7be27545e5347110da8663d2b698ec44c96dbfe419')
CF_FEATURE_CODE_PIN=(str(R/'V242_CF_FEATURE_PRODUCER_B/build_cf_features.py'),'688119df34a222f2a7a5903736d4bf5e6f8bd2e8ae0329db0eaf3d1eef872db5')
ROLES=('COVER','ADJACENT_SHAM')
CSV_COLUMNS=('sample_index','row_id','family','role','available','target','logit0','logit1','selected_family_logit','prediction')
RECIPE={'steps':1000,'seed':17,'draw_seed':17,'batch_size':32,'family_D_E':[8,8],
        'natural_first_D_E':[4,4],'matched_last_D_E':[4,4],
        'D_variant':'ADJACENT_SHAM','E_variant':'COVER','proof_failure':'KEEP_BOTH_ORIGINAL_NATURAL_SLOTS',
        'head_parameters':86054,'head_lr':.001,'weight_decay':.0001,'threshold_logit':0,
        'use_lora':False,'baseline_new_optimizer_updates':0,'treatment_new_optimizer_updates':1000,
        'new_random_CF_parent_choice':False,'teacher_features':False}
CF_LAYOUT={role+'_'+suffix+'.npy':(np.float32,shape) for role in ROLES for suffix,shape in
           [('BASE1186',(5406,1186)),('NODE_CLS',(5406,8,384)),('CONTEXT_CLS',(5406,384))]}
CF_LAYOUT.update({'RELATIVE_XY8.npy':(np.float32,(5406,8,2)),
                  'PARENT_INDICES.npy':(np.int64,(5406,)), 'CONSTRUCTED.npy':(np.bool_,(5406,))})


def require_complete(value, message):
    if value.get('completed') is not True or value.get('PASS') is not True:
        raise ValueError(message)


def exact_binding(a,b):
    if identity(a)!=identity(b):raise ValueError('Exact metadata graph identity mismatch')


def read_json_binding(b,white,pin=None):
    return json.loads(read(verify(b,white,pin)))


def copy_bound_opaque(b,white,destination):
    """Copy declared artifact bytes only. In particular, no baseline torch.load."""
    source=verify(b,white)
    with os.fdopen(os.open(source,os.O_RDONLY|os.O_NOFOLLOW),'rb') as src:
        with os.fdopen(os.open(safe(destination),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb') as dst:
            for block in iter(lambda:src.read(8*1024*1024),b''):dst.write(block)
    result=binding(destination)
    if result['sha256']!=b['sha256']:raise ValueError('Opaque baseline copy changed bytes')
    return result


def validate_packet_metadata(e,meta):
    """No feature/weight/CSV numerical decode occurs in the metadata graph gate."""
    baseline=read_json_binding(e['baseline_complete'],meta,BASELINE_COMPLETE_PIN)
    require_complete(baseline,'Published V239 baseline incomplete')
    if baseline['draws_sha256']!=DRAW_SHA or baseline['head_parameters'][OLD_ARM]!=86054 or baseline['new_optimizer_updates'][OLD_ARM]!=1000 or baseline['initial_head_sha256'][OLD_ARM]!=INITIAL_FULL_SHA or baseline['same_initial_base_sha256']!=INITIAL_BASE_SHA:
        raise ValueError('Frozen V239 baseline architecture/init/draw provenance')
    for kind,key in [('checkpoint','checkpoints'),('log','logs'),('CSV','outputs')]:
        exact_binding(e['baseline_'+kind],baseline[key][OLD_ARM])
    exact_binding(e['original_feature_complete'],baseline['feature_complete'])
    exact_binding(e['head_module'],baseline['head_module'])
    oldfreeze=read_json_binding(e['baseline_freeze'],meta)
    exact_binding(e['baseline_freeze'],baseline['freeze']);require_complete(oldfreeze,'Baseline final barrier missing')
    exact_binding(oldfreeze['checkpoints'][OLD_ARM],e['baseline_checkpoint'])
    exact_binding(oldfreeze['logs'][OLD_ARM],e['baseline_log'])
    if oldfreeze['all_final_updates'][OLD_ARM]!=1000 or oldfreeze['new_DEV_head_output_before_bothfinalfreeze']!=0:
        raise ValueError('Baseline already final before historical DEV output')
    original=read_json_binding(e['original_feature_complete'],meta)
    require_complete(original,'Canonical original producer incomplete')
    if original['counts']!={'groups':1767,'supported_nodes':13801,'known':7617,'FIT':5876,'DEV':1741,'all_original_rows':15040,'masked_target_minus1':7423,'context_supported_groups':1760}:
        raise ValueError('Original SOURCE population changed')
    for k in ('samples_npz','geometry_npz'):exact_binding(e[k],original[k]);exact_binding(e[k],baseline[k])
    for name in FEATURE_LAYOUT:
        exact_binding(e['original_features'][name],original['features'][name]);exact_binding(e['original_features'][name],baseline['features'][name])
    if not original['canonical_sharing_QA'].get('PASS') or original['canonical_sharing_QA']['known_rows']!=7617:
        raise ValueError('Original common-forward sharing QA')
    dataset=read_json_binding(e['CF_data_complete'],meta)
    dqa=read_json_binding(e['CF_data_QA'],meta)
    cf=read_json_binding(e['CF_feature_complete'],meta)
    fqa=read_json_binding(e['CF_feature_QA'],meta)
    for obj,label in [(dataset,'scene data'),(dqa,'independent scene data QA'),(cf,'scene features'),(fqa,'independent scene feature QA')]:require_complete(obj,label+' must complete and PASS')
    exact_binding(dqa['dataset_complete'],e['CF_data_complete'])
    exact_binding(fqa['feature_complete'],e['CF_feature_complete'])
    exact_binding(cf['data_complete'],e['CF_data_complete']);exact_binding(cf['data_QA'],e['CF_data_QA'])
    if dataset['schema']!='V242_SCENE_CONSISTENT_CF_DATA_V1' or dataset['counts']['attempted']!=5406 or dataset['all_original5406_attempted'] is not True:
        raise ValueError('All5406 parents; no old-success filtering')
    if cf['schema']!='V242_CANONICAL_CF_FEATURES_V1' or cf['rows']!=5406 or cf['all5406_original_order_retained'] is not True:
        raise ValueError('Canonical same-scene CF feature protocol')
    exact_binding(cf['code'],{'path':CF_FEATURE_CODE_PIN[0],'sha256':CF_FEATURE_CODE_PIN[1]})
    exact_binding(cf['samples_npz'],e['samples_npz'])
    if cf['base_state_sha256_before']!=cf['base_state_sha256_after'] or cf['official_base']['sha256']!=original['official_base']['sha256'] or cf['asset_identity']!=original['asset_identity']:
        raise ValueError('Same frozen official DINO asset; no adaptive encoder')
    for name in CF_LAYOUT:exact_binding(e['CF_features'][name],cf['features'][name])
    for name in ('PROOF_PASS.npy','PAIR_TARGETS.npy'):
        exact_binding(e['CF_proof_inputs'][name],dataset['outputs'][name])
    # Two producers wrote byte-identical .npy arrays, with independent paths.
    for name in ('PARENT_INDICES.npy','CONSTRUCTED.npy'):
        if e['CF_features'][name]['sha256']!=dataset['outputs'][name]['sha256']:
            raise ValueError('CF feature index/constructed bytes must match actual dataset')
    for role in ROLES:
        q=cf['canonical_sharing_QA'][role]
        if any(q[k] is not True for k in ('active_first384_byte_exact','original_condition32_byte_exact','known_fractions_byte_exact')) or q['UNCONSTRUCTED_feature_nonzeros']!=0:
            raise ValueError('CF shared active/node/condition/fractions producer proof')
    return baseline,original,dataset,cf


def load_array_set(bindings,layout,white):
    if set(bindings)!=set(layout):raise ValueError('Exact closed numerical feature set')
    result={}
    for name,(dtype,shape) in layout.items():
        value=np.load(verify(bindings[name],white),mmap_mode='r',allow_pickle=False)
        if value.dtype!=dtype or value.shape!=shape:raise ValueError('Frozen feature shape/dtype: '+name)
        result[name]=value
    return result


def validate_CF_features(cf,proof,targets,data,features,geometry):
    parents=np.asarray(cf['PARENT_INDICES.npy']);constructed=np.asarray(cf['CONSTRUCTED.npy'])
    if proof.dtype!=bool or proof.shape!=(5406,) or targets.dtype!=np.int8 or targets.shape!=(5406,2):raise ValueError('Exact scene proof/target headers')
    wanted=np.flatnonzero(data['fit']&data['available']&(data['target']==0))
    if len(wanted)!=5406 or len(np.unique(parents))!=5406 or not np.array_equal(np.sort(parents),wanted):
        raise ValueError('All and only original FIT available knownD5406 parents')
    if np.any(proof&~constructed) or not np.array_equal(targets[proof],np.tile(np.array([1,0],np.int8),(int(proof.sum()),1))) or np.any(targets[~proof]!=-1):
        raise ValueError('COVER E/SHAM D only from renewed proof; failureUNKNOWN retained')
    if not byte_equal(np.asarray(cf['RELATIVE_XY8.npy']),np.asarray(features['RELATIVE_XY8.npy'][parents])):
        raise ValueError('CF original sameq8 relative coordinates')
    slots=features['SAMPLE_NODE_INDEX.npy'][parents];groups=features['SAMPLE_GROUP_INDEX.npy'][parents]
    for role in ROLES:
        base=cf[role+'_BASE1186.npy'];node=cf[role+'_NODE_CLS.npy'];context=cf[role+'_CONTEXT_CLS.npy']
        for start in range(0,5406,64):
            i=np.arange(start,min(start+64,5406));use=i[constructed[i]];missing=i[~constructed[i]]
            if not all(np.isfinite(x[i]).all() for x in (base,node,context)):raise ValueError('Finite CF features')
            if any(np.count_nonzero(x[missing]) for x in (base,node,context)):raise ValueError('Only unconstructed features must remainzero')
            if not byte_equal(np.asarray(base[use,:384]),np.asarray(node[use,slots[use]])):raise ValueError('CF active/node common bytes')
            if not byte_equal(np.asarray(base[use,1154:]),data['condition'][parents[use]]) or not byte_equal(np.asarray(base[use,1152:1154]),data['fractions'][parents[use]]):raise ValueError('CF original condition/fractions unchanged')
            supported=geometry['node_available'][groups[use]]
            if np.count_nonzero(node[use][~supported]) or np.count_nonzero(context[use][geometry['context_fraction'][groups[use]]==0]):raise ValueError('Original unsupported descriptor zero')
    lookup=np.full(len(data['target']),-1,np.int64);lookup[parents]=np.arange(5406)
    return lookup,{'completed':True,'PASS':True,'all_original_FIT_D_parents':5406,'constructed':int(constructed.sum()),'proof_PASS':int(proof.sum()),'constructed_UNKNOWN':int((constructed&~proof).sum()),'unconstructed':int((~constructed).sum()),'all_original_coordinates_and_conditions_retained':True,'features_use_teacher_labels':False,'old_success_filter':False,'native_gold':False}


def step_plan(indices,data,lookup,proof):
    """No new RNG or success-pool selection: paired replacement follows each drawn D."""
    if indices.shape!=(32,):raise ValueError('Original32 draw required')
    parent=indices.copy();pair=np.full(32,-1,np.int64);role=np.full(32,-1,np.int8)
    for family in (0,1):
        start=family*16
        if not (data['family'][indices[start:start+16]]==family).all() or not np.array_equal(data['target'][indices[start:start+16]],np.array([0]*8+[1]*8,np.int8)):
            raise ValueError('Unchanged ordered original family8D8E')
        for j in range(4):
            d=start+4+j;e=start+12+j;p=int(lookup[indices[d]])
            if p<0:raise ValueError('Every drawn original D must have retained parent entry')
            if proof[p]:
                parent[d]=parent[e]=indices[d];pair[d]=pair[e]=p;role[d]=1;role[e]=0
    if not data['fit'][parent].all() or not data['available'][parent].all() or not np.array_equal(data['family'][parent],data['family'][indices]):raise ValueError('No DEV/UNKNOWN/family change in optimizer source')
    return parent,pair,role


def make_batch(indices,plan,data,features,cf,geometry,device):
    parents,pairs,roles=plan
    x=np.array(features['ORIGINAL_ALL_FEATURES.npy'][parents],copy=True)
    groups=np.asarray(features['SAMPLE_GROUP_INDEX.npy'][parents]);slots=np.asarray(features['SAMPLE_NODE_INDEX.npy'][parents])
    node=np.array(features['NODE_CLS.npy'][groups],copy=True);context=np.array(features['CONTEXT_CLS.npy'][groups],copy=True)
    relative=np.array(features['RELATIVE_XY8.npy'][parents],copy=True)
    for role_i,role in enumerate(ROLES):
        positions=np.flatnonzero(roles==role_i);p=pairs[positions]
        if len(p):
            x[positions]=cf[role+'_BASE1186.npy'][p];node[positions]=cf[role+'_NODE_CLS.npy'][p]
            context[positions]=cf[role+'_CONTEXT_CLS.npy'][p];relative[positions]=cf['RELATIVE_XY8.npy'][p]
    values=(x,node,context,np.array(slots,copy=True),geometry['original_available8'][groups],geometry['node_available'][groups],geometry['context_fraction'][groups]>0,relative)
    return tuple(torch.as_tensor(v,device=device) for v in values),torch.as_tensor(data['target'][indices],device=device),torch.as_tensor(data['family'][indices],device=device)


def plan_digest(draws,plans):
    h=hashlib.sha256();h.update(draws.astype('<i8').tobytes())
    for parent,pair,role in plans:h.update(parent.astype('<i8').tobytes());h.update(pair.astype('<i8').tobytes());h.update(role.tobytes())
    return h.hexdigest()


def train_new_head(head,data,features,cf,geometry,draws,plans,device,out,e,plan_sha):
    initial=cpu_copy(head.state_dict());initial_sha=state_sha(initial)
    if initial_sha!=INITIAL_FULL_SHA or state_sha(head.base.state_dict())!=INITIAL_BASE_SHA or sum(p.numel() for p in head.parameters())!=86054:
        raise ValueError('Fresh originalseed17 fullhead exact; no trained warmstart')
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
    if optimizer.state:raise ValueError('Reset optimizer required')
    log=out/(TREAT+'_TRAIN1000.jsonl');head.train()
    with log.open('x') as stream:
        for update,(indices,plan) in enumerate(zip(draws,plans),1):
            inputs,target,family=make_batch(indices,plan,data,features,cf,geometry,device)
            optimizer.zero_grad(set_to_none=True);logits=head(*inputs)[0]
            loss,parts=family_loss(logits,target,family)
            if not torch.isfinite(loss):raise ValueError('Finite SOURCE family objective')
            loss.backward();optimizer.step()
            stream.write(json.dumps({'update':update,'sample_indices':indices.tolist(),'source_parent_indices':plan[0].tolist(),'pair_indices':plan[1].tolist(),'source_roles':plan[2].tolist(),'draws_sha256':DRAW_SHA,'source_plan_sha256':plan_sha,'family0_BCE':float(parts[0].detach()),'family1_BCE':float(parts[1].detach()),'total_loss':float(loss.detach()),'family_counts':[16,16],'all_fit_available_known':True},allow_nan=False)+'\n')
            if update%100==0:print(json.dumps({'stage':TREAT,'updates':update,'fixed_final':1000}),flush=True)
    if len(optimizer.state)!=10 or any(float(s['step'])!=1000 or any(not torch.isfinite(v).all() for v in s.values() if isinstance(v,torch.Tensor)) for s in optimizer.state.values()):raise ValueError('All10 optimizer states finite@1000')
    if any(not torch.isfinite(p).all() for p in head.parameters()):raise ValueError('Finite finalhead')
    cp=out/(TREAT+'_FINAL1000.pt')
    torch.save({'arm':TREAT,'updates':1000,'head_parameters':86054,'use_lora':False,'adapter_parameters':0,'head_state_dict':cpu_copy(head.state_dict()),'optimizer_state_dict':cpu_copy(optimizer.state_dict()),'initial_head_state_dict':initial,'initial_head_sha256':initial_sha,'same_initial_base_sha256':INITIAL_BASE_SHA,'draws_sha256':DRAW_SHA,'source_plan_sha256':plan_sha,'entry':e['_entry_binding'],'original_feature_complete':e['original_feature_complete'],'CF_feature_complete':e['CF_feature_complete'],'CF_data_complete':e['CF_data_complete']},cp)
    return binding(cp),binding(log),state_sha(head.state_dict())


def read_baseline_rows(path,data):
    with safe(path).open(newline='') as stream:
        reader=csv.DictReader(stream)
        if tuple(reader.fieldnames)!=CSV_COLUMNS:raise ValueError('Frozen V239 baselineCSV columns')
        rows=[]
        for i,r in enumerate(reader):
            if int(r['sample_index'])!=i or r['row_id']!=str(data['row_id'][i]) or int(r['family'])!=data['family'][i] or r['role']!=('FIT' if data['fit'][i] else 'DEV') or r['available']!=str(bool(data['available'][i])) or int(r['target'])!=data['target'][i]:raise ValueError('Frozen baseline same original row/role/teacher identity')
            z=[None if not r[k] else float(r[k]) for k in ('logit0','logit1')];family=int(r['family']);v=z[family]
            if any(vv is not None and not np.isfinite(vv) for vv in z) or r['prediction']!=('U' if v is None else 'E' if v>=0 else 'D') or (None if not r['selected_family_logit'] else float(r['selected_family_logit']))!=v:raise ValueError('Frozen baseline scalar decode')
            rows.append({'sample_index':i,'row_id':r['row_id'],'family':family,'role':r['role'],'available':bool(data['available'][i]),'target':int(r['target']),'logit0':z[0],'logit1':z[1],'selected_family_logit':v,'prediction':r['prediction']})
    if len(rows)!=7617:raise ValueError('All7617 savedbaseline rows')
    return rows


def write_csv(path,rows,columns=None):
    with safe(path).open('x',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(columns or rows[0]));w.writeheader();w.writerows(rows)
    return binding(path)


def natural_outputs(head,data,features,geometry,device,out):
    rows=[];scores={};head.eval().requires_grad_(False)
    with torch.no_grad():
        available=np.flatnonzero(data['available'])
        for first in range(0,len(available),64):
            i=available[first:first+64];x=torch.as_tensor(np.array(features['ORIGINAL_ALL_FEATURES.npy'][i],copy=True),device=device)
            z=head(x,*context_args(i,features,geometry,device))[0].cpu().numpy()
            if not np.isfinite(z).all():raise ValueError('Finite final natural logits')
            scores.update({int(j):v for j,v in zip(i,z)})
    for i in range(len(data['target'])):
        f=int(data['family'][i]);z=scores.get(i);v=None if z is None else float(z[f])
        rows.append({'sample_index':i,'row_id':str(data['row_id'][i]),'family':f,'role':'FIT' if data['fit'][i] else 'DEV','available':bool(data['available'][i]),'target':int(data['target'][i]),'logit0':None if z is None else float(z[0]),'logit1':None if z is None else float(z[1]),'selected_family_logit':v,'prediction':'U' if v is None else 'E' if v>=0 else 'D'})
    return rows,write_csv(out/(TREAT+'_ALL_SOURCE_ROWS.csv'),rows,CSV_COLUMNS)


def CF_fit_outputs(head,data,features,cf,proof,geometry,device,out):
    """Separate FIT-only diagnostic. UNKNOWN constructed rows retained, never scored."""
    parent=np.asarray(cf['PARENT_INDICES.npy']);constructed=np.asarray(cf['CONSTRUCTED.npy']);rows=[]
    with torch.no_grad():
        for role_i,role in enumerate(ROLES):
            scores={};use=np.flatnonzero(constructed)
            for first in range(0,len(use),64):
                p=use[first:first+64];sample=parent[p];groups=features['SAMPLE_GROUP_INDEX.npy'][sample];slot=features['SAMPLE_NODE_INDEX.npy'][sample]
                arrays=(cf[role+'_BASE1186.npy'][p],cf[role+'_NODE_CLS.npy'][p],cf[role+'_CONTEXT_CLS.npy'][p],np.asarray(slot),geometry['original_available8'][groups],geometry['node_available'][groups],geometry['context_fraction'][groups]>0,cf['RELATIVE_XY8.npy'][p])
                z=head(*(torch.as_tensor(np.array(v,copy=True),device=device) for v in arrays))[0].cpu().numpy()
                if not np.isfinite(z).all():raise ValueError('Finite final CF FIT diagnostics')
                scores.update({int(j):v for j,v in zip(p,z)})
            for p,sample in enumerate(parent):
                f=int(data['family'][sample]);z=scores.get(p);v=None if z is None else float(z[f]);y=(1 if role_i==0 else 0) if proof[p] else -1
                rows.append({'pair_index':p,'parent_sample_index':int(sample),'row_id':str(data['row_id'][sample]),'family':f,'role':'FIT','variant':role,'constructed':bool(constructed[p]),'proof_PASS':bool(proof[p]),'target':y,'logit0':None if z is None else float(z[0]),'logit1':None if z is None else float(z[1]),'selected_family_logit':v,'prediction':'U' if v is None else 'E' if v>=0 else 'D','native_gold':False})
    metrics={}
    for f in (0,1):
        known=[r for r in rows if r['family']==f and r['target']>=0]
        cm=[[sum(r['target']==y and r['prediction']==p for r in known) for p in ('D','E','U')] for y in (0,1)]
        metrics['family'+str(f)]={'CM_D_E_by_D_E_U':cm,'known_rows':len(known),'UNKNOWN_rows':sum(r['family']==f and r['target']<0 for r in rows)}
    return write_csv(out/(TREAT+'_CF_FIT_ROWS.csv'),rows),metrics


def run(entry_path):
    e=json.loads(read(entry_path))
    if e.get('approved_scene_CF_source_training') is not True or e.get('source_only') is not True or e.get('automatic_retry') is not False:raise PermissionError('Separate Root actual SOURCE training entry required')
    if e.get('fixed_recipe')!=RECIPE:raise ValueError('Exactly frozen data-only recipe required')
    codes,meta,payload=e['code_whitelist'],e['metadata_whitelist'],e['payload_whitelist']
    verify(e['code'],codes,(str(safe(__file__)),binding(__file__)['sha256']));verify(e['implementation_protocol'],codes)
    verify(e['head_module'],codes,MODULE_PIN);verify(e['original_trainer_code'],codes,ORIGINAL_CODE_PIN)
    scope=read_json_binding(e['root_scope'],meta);read_json_binding(e['root_protocol'],meta)
    if scope.get('approved_SOURCE_scene_CF_training') is not True or scope.get('current_or_held_payloads_authorized') is not False:raise ValueError('Prospective SOURCE-only Root fit scope required')
    baseline,original,dataset,cfpacket=validate_packet_metadata(e,meta)
    expected=[*e['original_features'].values(),*e['CF_features'].values(),*e['CF_proof_inputs'].values(),e['samples_npz'],e['geometry_npz'],e['baseline_checkpoint'],e['baseline_log'],e['baseline_CSV']]
    if {identity(b) for b in payload}!={identity(b) for b in expected}:raise ValueError('Exact vectors/sample metadata + baseline opaquebytes only; no RGB/GT/DINO')
    out=safe(e['output_directory'])
    if out.parent!=R/'V243_ROOT_SCENE_CF_TRAINING' or out.name!='scene_CF_source_fit_v1' or out.exists():raise ValueError('Fresh fixed V243 output directory')
    out.mkdir();started=time.monotonic();e['_entry_binding']=binding(entry_path)
    write_json(out/'STARTED.json',{'entry':e['_entry_binding'],'automatic_retry':False,'source_only':True,'new_baseline_updates':0,'fixed_recipe':RECIPE})
    try:
        features=load_array_set(e['original_features'],FEATURE_LAYOUT,payload)
        cf=load_array_set(e['CF_features'],CF_LAYOUT,payload)
        with np.load(verify(e['samples_npz'],payload),allow_pickle=False) as z:
            if set(z.files)!=set(SAMPLE_KEYS):raise ValueError('Original samples exact nonpixel keys')
            data=validate_samples({k:z[k] for k in SAMPLE_KEYS})
        with np.load(verify(e['geometry_npz'],payload),allow_pickle=False) as z:geometry=validate_geometry({k:z[k] for k in GEOMETRY_KEYS},1767)
        proof=np.load(verify(e['CF_proof_inputs']['PROOF_PASS.npy'],payload),mmap_mode='r',allow_pickle=False)
        targets=np.load(verify(e['CF_proof_inputs']['PAIR_TARGETS.npy'],payload),mmap_mode='r',allow_pickle=False)
        draws,eligible=fixed_draws(data)
        if len(data['target'])!=7617 or not data['available'].all() or int(eligible.sum())!=5876 or int((~data['fit']).sum())!=1741:raise ValueError('Original known7617/FIT5876/DEV1741')
        if hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()!=DRAW_SHA:raise ValueError('Original1000draw ledger mismatch')
        originalQA=global_feature_QA(features,data,geometry)
        lookup,cfQA=validate_CF_features(cf,proof,targets,data,features,geometry)
        globalQA=write_json(out/'GLOBAL_SOURCE_AND_CF_FEATURE_QA_BEFORE_OPTIMIZER.json',{'completed':True,'PASS':True,'original':originalQA,'CF':cfQA,'before_optimizer':True,'new_DEV_head_output':0})
        plans=[step_plan(i,data,lookup,proof) for i in draws];plan_sha=plan_digest(draws,plans)
        ledger=out/'ORIGINAL_DRAW_AND_MATCHED_SOURCE_LEDGER1000.jsonl'
        with ledger.open('x') as stream:
            for update,(i,plan) in enumerate(zip(draws,plans),1):stream.write(json.dumps({'update':update,'sample_indices':i.tolist(),'source_parent_indices':plan[0].tolist(),'pair_indices':plan[1].tolist(),'source_roles':plan[2].tolist(),'draws_sha256':DRAW_SHA,'source_plan_sha256':plan_sha,'all_fit_available_known':True})+'\n')
        reuse={}
        for kind,suffix in [('checkpoint','FINAL1000.pt'),('log','TRAIN1000.jsonl'),('CSV','ALL_SOURCE_ROWS.csv')]:reuse[kind]=copy_bound_opaque(e['baseline_'+kind],payload,out/(CONTROL+'_'+suffix))
        reuse_receipt=write_json(out/'V239_BASELINE_BYTE_REUSE.json',{'completed':True,'PASS':True,'original_arm':OLD_ARM,'new_arm':CONTROL,'source_complete':e['baseline_complete'],'source_checkpoint':e['baseline_checkpoint'],'source_log':e['baseline_log'],'source_CSV':e['baseline_CSV'],'copied':reuse,'new_optimizer_updates':0,'numerical_checkpoint_loads':0,'baseline_scores_interpreted_before_new_finalfreeze':0})
        module=load_code(e['head_module'],codes,'v243_fixed_relational_head',MODULE_PIN)
        configure_runtime();torch.manual_seed(17);head=module.RelationalQueryHead().float().to(e['device'])
        cp,log,final_sha=train_new_head(head,data,features,cf,geometry,draws,plans,e['device'],out,e,plan_sha)
        checkpoints={CONTROL:reuse['checkpoint'],TREAT:cp};logs={CONTROL:reuse['log'],TREAT:log}
        freeze=write_json(out/'BASELINE_REUSED_AND_NEW_FINAL1000_FROZEN_BEFORE_DEV.json',{'completed':True,'PASS':True,'checkpoints':checkpoints,'logs':logs,'all_final_updates':{CONTROL:1000,TREAT:1000},'new_optimizer_updates':{CONTROL:0,TREAT:1000},'global_feature_QA':globalQA,'baseline_reuse':reuse_receipt,'draw_ledger':binding(ledger),'draws_sha256':DRAW_SHA,'source_plan_sha256':plan_sha,'initial_full_head_sha256':INITIAL_FULL_SHA,'new_DEV_head_output_before_finalfreeze':0,'new_model_or_threshold_selection':False})
        # This is the first interpretation of baseline saved scores and first new DEV head output.
        oldrows=read_baseline_rows(reuse['CSV']['path'],data)
        newrows,newCSV=natural_outputs(head,data,features,geometry,e['device'],out)
        pairCSV,pairmetrics=CF_fit_outputs(head,data,features,cf,proof,geometry,e['device'],out)
        if state_sha(head.state_dict())!=final_sha:raise ValueError('Frozen newhead mutated')
        reductions={CONTROL:reduce_rows(oldrows),TREAT:reduce_rows(newrows)}
        summary=write_json(out/'SOURCE_RESULTS.json',{'completed':True,'PASS':True,'arms':reductions,'paired_changes':paired_changes(oldrows,newrows),'SOURCE_gate':source_gate(reductions[CONTROL],reductions[TREAT]),'CF_FIT_only':pairmetrics,'pair_metrics_are_not_SOURCE_DEV_generalization':True,'two_SOURCE_referents_separate':True,'YCB_native_UNKNOWN':True,'all_original_rows':15040,'known_output_rows':7617,'masked_target_minus1_retained':7423,'current_admission':False,'automatic_retry':False})
        complete={'completed':True,'PASS':True,'schema':'V243_SCENE_CF_DATA_ONLY_HEAD_TRIAL_V1','entry':e['_entry_binding'],'code':e['code'],'implementation_protocol':e['implementation_protocol'],'root_protocol':e['root_protocol'],'root_scope':e['root_scope'],'baseline_complete':e['baseline_complete'],'baseline_reuse':reuse_receipt,'original_feature_complete':e['original_feature_complete'],'CF_data_complete':e['CF_data_complete'],'CF_data_QA':e['CF_data_QA'],'CF_feature_complete':e['CF_feature_complete'],'CF_feature_QA':e['CF_feature_QA'],'original_features':e['original_features'],'CF_features':e['CF_features'],'CF_proof_inputs':e['CF_proof_inputs'],'samples_npz':e['samples_npz'],'geometry_npz':e['geometry_npz'],'head_module':e['head_module'],'checkpoints':checkpoints,'logs':logs,'outputs':{CONTROL:reuse['CSV'],TREAT:newCSV},'CF_FIT_output':pairCSV,'summary':summary,'freeze':freeze,'global_feature_QA':globalQA,'draw_ledger':binding(ledger),'draws_sha256':DRAW_SHA,'source_plan_sha256':plan_sha,'same_initial_base_sha256':INITIAL_BASE_SHA,'initial_full_head_sha256':INITIAL_FULL_SHA,'final_head_sha256':{CONTROL:baseline['final_head_sha256'][OLD_ARM],TREAT:final_sha},'new_optimizer_updates':{CONTROL:0,TREAT:1000},'head_parameters':{CONTROL:86054,TREAT:86054},'new_optimizer_state_tensors':{CONTROL:0,TREAT:10},'new_DEV_head_output_before_finalfreeze':0,'DINO_LoRA_RGB_GT_protected_payload_reads':0,'baseline_checkpoint_numerical_loads':0,'native_gold':False,'automatic_retry':False,'current_admission':False,'actual_seconds':time.monotonic()-started}
        write_json(out/'COMPLETE.json',complete);print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(out/'COMPLETE.json')}),flush=True)
    except BaseException:
        write_json(out/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)

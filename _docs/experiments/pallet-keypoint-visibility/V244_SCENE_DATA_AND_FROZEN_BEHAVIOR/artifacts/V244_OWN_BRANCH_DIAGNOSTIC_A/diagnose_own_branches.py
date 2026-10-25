"""V244 own frozen TREAT branch behavior, SOURCE only. No optimizer/encoder.
Exact Root entry is required before checkpoint/vector/sample access.
"""
import argparse,csv,hashlib,json,os,sys,time,traceback,types
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).absolute().parent.parent
MODULE_PIN=(str(R/'V237_RELATIONAL_QUERY_DESIGN_B/relational_query_head.py'),'dfd26ab7ccfa7037f0c3dca548d59c8574aa8b78c50699f70f01e03bf48935a9')
REPLAY_SOURCE_PIN=(str(R/'verify_v239_saved_head_cpu.py'),'08d0cab9b2f24e92c8d2f776aeadd91c1debf20a76c01de05b77e1de818f9624')
FIT_COMPLETE_PIN=(str(R/'V239_ROOT_CANONICAL_CONTEXT/canonical_pair_source_fit_v1/COMPLETE.json'),'1fd6880528d1bd440376c68c7079b9cbee59ce736f006fdaef4244e3310a4df9')
ROOT_PROTOCOL_PIN=(str(R/'V244_ROOT_FROZEN_BEHAVIOR/PROTOCOL_BEFORE_FROZEN_SOURCE_DIAGNOSTICS.json'),'fb283db25a4c8d73d0a90b8e7cd9ee4c7ba53ca1665bd10a9e1c691974cc4de4')
ARM='TREAT_CANONICAL_CONTEXT'
ATOL=RTOL=1e-5
GEOMETRY_KEYS=('node_available','node_fraction','context_fraction','q8','L8','original_available8','box_xyxy')
FEATURE_LAYOUT={'ORIGINAL_ALL_FEATURES.npy':(np.float32,(7617,1186)),'NODE_CLS.npy':(np.float32,(1767,8,384)),'CONTEXT_CLS.npy':(np.float32,(1767,384)),'SAMPLE_GROUP_INDEX.npy':(np.int64,(7617,)),'SAMPLE_NODE_INDEX.npy':(np.int64,(7617,)),'RELATIVE_XY8.npy':(np.float32,(7617,8,2))}
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

def state_sha(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(a.dtype).encode())
        digest.update(json.dumps(list(a.shape)).encode());digest.update(a.tobytes())
    return digest.hexdigest()

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


def checked_json(b,white,pin=None):
    return json.loads(read(verify(b,white,pin)))


def exact(a,b):
    if identity(a)!=identity(b):raise ValueError('Exact metadata binding mismatch')


def require_PASS(value):
    if value.get('completed') is not True or value.get('PASS') is not True:raise ValueError('Upstream completion/PASS required')


def read_saved_rows(path,data):
    with safe(path).open(newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=7617:raise ValueError('Same all7617 SOURCE denominator')
    for i,r in enumerate(rows):
        if int(r['sample_index'])!=i or r['row_id']!=str(data['row_id'][i]) or int(r['family'])!=int(data['family'][i]) or int(r['target'])!=int(data['target'][i]) or r['role']!=('FIT' if data['fit'][i] else 'DEV') or r['available']!='True':raise ValueError('Saved TOTAL same row/family/role/target/input availability')
        f=int(r['family']);z=np.array([float(r['logit0']),float(r['logit1'])],np.float32)
        if not np.isfinite(z).all() or float(r['selected_family_logit'])!=float(r['logit'+str(f)]) or r['prediction']!=('E' if z[f]>=0 else 'D'):raise ValueError('Saved TOTAL two binary outputs; no argmax')
    return rows,np.asarray([[float(r['logit0']),float(r['logit1'])] for r in rows],np.float32)


def own_branches(model,active,args):
    """The three quantities belong to ONE frozen checkpoint. No separate CONTROL."""
    ownbase=model.base(active)
    residual,detail=model.relational(active,*args)
    total=ownbase+residual
    for value in (ownbase,residual,total):
        if value.dtype!=torch.float32 or value.shape!=(len(active),2) or not torch.isfinite(value).all():raise ValueError('Finite FP32 two binary family logits')
    return ownbase,residual,total,detail


def rows_from_branches(data,saved,base,residual,total):
    if any(x.dtype!=np.float32 or x.shape!=(len(saved),2) or not np.isfinite(x).all() for x in (base,residual,total)):raise ValueError('Finite full SOURCE branch arrays')
    if not byte_equal(base+residual,total):raise ValueError('Exact FP32 additive reconstruction')
    rows=[]
    for i,r in enumerate(saved):
        f=int(data['family'][i]);y=int(data['target'][i]);s=1 if y==1 else -1
        b,delta,z=map(float,(base[i,f],residual[i,f],total[i,f]));pbase='E' if b>=0 else 'D';pres='E' if delta>=0 else 'D';ptotal='E' if z>=0 else 'D';truth='E' if y==1 else 'D'
        cross='D_TO_E' if pbase=='D' and ptotal=='E' else 'E_TO_D' if pbase=='E' and ptotal=='D' else 'NONE'
        effect='RECOVERED' if pbase!=truth and ptotal==truth else 'DAMAGED' if pbase==truth and ptotal!=truth else 'UNCHANGED_CORRECT' if ptotal==truth else 'UNCHANGED_WRONG'
        rows.append({'sample_index':i,'row_id':str(data['row_id'][i]),'family':f,'role':'FIT' if data['fit'][i] else 'DEV','available':bool(data['available'][i]),'target':y,
          'own_base_logit0':float(base[i,0]),'own_base_logit1':float(base[i,1]),'own_residual_logit0':float(residual[i,0]),'own_residual_logit1':float(residual[i,1]),'own_total_logit0':float(total[i,0]),'own_total_logit1':float(total[i,1]),
          'selected_own_base_logit':b,'selected_own_residual_logit':delta,'selected_own_total_logit':z,'own_base_prediction':pbase,'own_residual_sign_prediction':pres,'own_total_prediction':ptotal,
          'saved_total_logit':float(r['selected_family_logit']),'saved_total_prediction':r['prediction'],'total_replay_delta':z-float(r['selected_family_logit']),
          'target_aligned_residual':s*delta,'base_target_margin':s*b,'total_target_margin':s*z,'threshold_crossing':cross,'target_effect':effect,'base_correct':pbase==truth,'total_correct':ptotal==truth})
    return rows


def distribution(values):
    v=np.asarray(values,np.float64)
    if not len(v):return {'N':0,'mean':None,'std':None,'min':None,'p25':None,'median':None,'p75':None,'max':None,'negative':0,'zero':0,'positive':0}
    q=np.quantile(v,[0,.25,.5,.75,1])
    return {'N':len(v),'mean':float(v.mean()),'std':float(v.std()),'min':float(q[0]),'p25':float(q[1]),'median':float(q[2]),'p75':float(q[3]),'max':float(q[4]),'negative':int((v<0).sum()),'zero':int((v==0).sum()),'positive':int((v>0).sum())}


def summarize(rows):
    metrics={};strata=[]
    for branch,field in [('own_base','own_base_prediction'),('own_residual_sign_only','own_residual_sign_prediction'),('own_total','own_total_prediction')]:
        rr=[{'family':r['family'],'role':r['role'],'target':r['target'],'available':r['available'],'prediction':r[field]} for r in rows]
        metrics[branch]=reduce_rows(rr)
    for f in (0,1):
        for role in ('FIT','DEV'):
            for y in (0,1):
                rr=[r for r in rows if r['family']==f and r['role']==role and r['target']==y]
                item={'family':f,'role':role,'target':y,'rows':len(rr),'own_base_correct':sum(r['base_correct'] for r in rr),'own_total_correct':sum(r['total_correct'] for r in rr)}
                for key in ('selected_own_base_logit','selected_own_residual_logit','selected_own_total_logit','target_aligned_residual','base_target_margin','total_target_margin'):item[key]=distribution([r[key] for r in rr])
                item['crossings']={key:sum(r['threshold_crossing']==key for r in rr) for key in ('D_TO_E','E_TO_D','NONE')}
                item['effects']={key:sum(r['target_effect']==key for r in rr) for key in ('RECOVERED','DAMAGED','UNCHANGED_CORRECT','UNCHANGED_WRONG')}
                strata.append(item)
    return metrics,strata


def write_csv(path,rows):
    with safe(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return binding(path)


def flat_stats(strata):
    out=[]
    for item in strata:
        row={k:item[k] for k in ('family','role','target','rows','own_base_correct','own_total_correct')}
        for key in ('selected_own_base_logit','selected_own_residual_logit','selected_own_total_logit','target_aligned_residual','base_target_margin','total_target_margin'):
            row.update({key+'_'+k:v for k,v in item[key].items()})
        row.update({'crossing_'+k:v for k,v in item['crossings'].items()});row.update({'effect_'+k:v for k,v in item['effects'].items()});out.append(row)
    return out


def run(entry_path):
    e=json.loads(read(entry_path))
    if e.get('approved_frozen_SOURCE_own_branch_diagnosis') is not True or e.get('source_only') is not True or e.get('automatic_retry') is not False:raise PermissionError('Separate explicit Root frozen SOURCE diagnosis entry required')
    if e.get('device')!='cpu' or e.get('batch_size')!=64 or e.get('atol')!=ATOL or e.get('rtol')!=RTOL:raise ValueError('Existing CPU replay/batch64/atol=rtol1e-5 unchanged')
    codes,meta,payload=e['code_whitelist'],e['metadata_whitelist'],e['payload_whitelist']
    verify(e['code'],codes,(str(safe(__file__)),binding(__file__)['sha256']));verify(e['implementation_protocol'],codes)
    verify(e['head_module'],codes,MODULE_PIN);verify(e['reference_replay_code'],codes,REPLAY_SOURCE_PIN)
    checked_json(e['root_protocol'],meta,ROOT_PROTOCOL_PIN)
    scope=checked_json(e['root_scope'],meta)
    if scope.get('approved_frozen_SOURCE_own_branch_diagnosis') is not True or scope.get('new_training_or_current_forward') is not False:raise ValueError('Root SOURCE-only scope before checkpoint/vector read')
    complete=checked_json(e['fit_complete'],meta,FIT_COMPLETE_PIN);require_PASS(complete)
    packet=checked_json(e['feature_complete'],meta);require_PASS(packet)
    freeze=checked_json(e['frozen_both_heads'],meta);require_PASS(freeze)
    exact(e['feature_complete'],complete['feature_complete']);exact(e['frozen_both_heads'],complete['freeze'])
    exact(e['checkpoint'],complete['checkpoints'][ARM]);exact(e['checkpoint'],freeze['checkpoints'][ARM]);exact(e['saved_total_CSV'],complete['outputs'][ARM])
    exact(e['head_module'],complete['head_module'])
    if complete['head_parameters'][ARM]!=86054 or complete['new_optimizer_updates'][ARM]!=1000 or freeze['all_final_updates'][ARM]!=1000 or complete['new_DEV_head_output_before_bothfinalfreeze']!=0:raise ValueError('Already frozen final1000 TREAT only')
    for k in ('samples_npz','geometry_npz'):exact(e[k],complete[k]);exact(e[k],packet[k])
    if set(e['features'])!=set(FEATURE_LAYOUT):raise ValueError('Exact six frozen feature arrays')
    for name in FEATURE_LAYOUT:exact(e['features'][name],complete['features'][name]);exact(e['features'][name],packet['features'][name])
    expected=[*e['features'].values(),e['checkpoint'],e['saved_total_CSV'],e['samples_npz'],e['geometry_npz']]
    if {identity(b) for b in payload}!={identity(b) for b in expected}:raise ValueError('Only declared SOURCE head/vector/sample payloads; no image/DINO/GT/CONTROL')
    out=safe(e['output_directory'])
    if out.parent!=R/'V244_OWN_BRANCH_DIAGNOSTIC_A' or out.name!='actual_diagnosis_v1' or out.exists():raise ValueError('Fresh own output only; no retries')
    out.mkdir();start=time.monotonic();write_json(out/'STARTED.json',{'entry':binding(entry_path),'automatic_retry':False,'source_only':True})
    try:
        features={}
        for name,(dtype,shape) in FEATURE_LAYOUT.items():
            value=np.load(verify(e['features'][name],payload),mmap_mode='r',allow_pickle=False)
            if value.dtype!=dtype or value.shape!=shape:raise ValueError('Canonical feature dtype/shape')
            features[name]=value
        with np.load(verify(e['samples_npz'],payload),allow_pickle=False) as z:
            if set(z.files)!=set(SAMPLE_KEYS):raise ValueError('Exact original SOURCE sample fields')
            data=validate_samples({k:z[k] for k in SAMPLE_KEYS})
        with np.load(verify(e['geometry_npz'],payload),allow_pickle=False) as z:geometry=validate_geometry({k:z[k] for k in GEOMETRY_KEYS},1767)
        if len(data['target'])!=7617 or not data['available'].all() or int(data['fit'].sum())!=5876 or int((~data['fit']).sum())!=1741:raise ValueError('Original known7617/FIT5876/DEV1741; no filtering')
        globalQA=write_json(out/'CANONICAL_SOURCE_INPUT_QA.json',global_feature_QA(features,data,geometry))
        saved,expected_total=read_saved_rows(verify(e['saved_total_CSV'],payload),data)
        torch.set_num_threads(1)
        module=load_code(e['head_module'],codes,'v244_frozen_own_TREAT',MODULE_PIN)
        cp=torch.load(verify(e['checkpoint'],payload),weights_only=True,map_location='cpu')
        if cp['arm']!=ARM or cp['updates']!=1000 or cp['head_parameters']!=86054 or cp['adapter_parameters']!=0 or cp['use_lora'] is not False:raise ValueError('Exact final frozen SOURCE checkpoint')
        state=cp['head_state_dict']
        if len(state)!=10 or any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state.values()) or sum(v.numel() for v in state.values())!=86054:raise ValueError('Typed finite full10tensor86054 state')
        model=module.RelationalQueryHead().cpu().eval().requires_grad_(False);model.load_state_dict(state,strict=True)
        before=state_sha(model.state_dict())
        if before!=complete['final_head_sha256'][ARM] or any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('Frozen ownhead matches source final state')
        base=np.empty((7617,2),np.float32);residual=np.empty_like(base);total=np.empty_like(base)
        with torch.no_grad():
            for first in range(0,7617,64):
                ix=np.arange(first,min(first+64,7617));x=torch.as_tensor(np.array(features['ORIGINAL_ALL_FEATURES.npy'][ix],copy=True))
                b,r,t,detail=own_branches(model,x,context_args(ix,features,geometry,'cpu'))
                base[ix]=b.numpy();residual[ix]=r.numpy();total[ix]=t.numpy()
                # Same masked read invariant used by the existing V239 CPU replay.
                gi=features['SAMPLE_GROUP_INDEX.npy'][ix];si=features['SAMPLE_NODE_INDEX.npy'][ix]
                other=geometry['original_available8'][gi]&geometry['node_available'][gi];other[np.arange(len(ix)),si]=False
                a=detail['attention_other7'].numpy()
                if np.count_nonzero(a[~other]) or not np.allclose(a.sum(1),other.any(1).astype(np.float32),atol=1e-6,rtol=0):raise ValueError('Saved masked relational read invariant')
        if not np.allclose(total,expected_total,atol=ATOL,rtol=RTOL):raise ValueError('TOTAL replay failed unchanged1e-5; maxabs='+str(float(np.max(np.abs(total-expected_total)))))
        selected=total[np.arange(7617),data['family'].astype(np.int64)];pred=np.where(selected>=0,'E','D')
        if pred.tolist()!=[r['prediction'] for r in saved]:raise ValueError('TOTAL all7617 predicted signs must match exactly')
        after=state_sha(model.state_dict())
        if after!=before:raise ValueError('Frozen head changed during diagnosis')
        replay=write_json(out/'TOTAL_REPLAY_QA.json',{'completed':True,'PASS':True,'rows':7617,'logits_checked':15234,'atol':ATOL,'rtol':RTOL,'max_abs':float(np.max(np.abs(total-expected_total))),'predictions_exact':7617,'additive_FP32_sum_byte_exact':byte_equal(base+residual,total),'state_sha256_before':before,'state_sha256_after':after,'own_checkpoint_only':True,'independent_CONTROL_used':False,'global_input_QA':globalQA})
        rows=rows_from_branches(data,saved,base,residual,total);metrics,strata=summarize(rows)
        rowfile=out/'ALL7617_OWN_BRANCH_ROWS.jsonl'
        with rowfile.open('x') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        rowCSV=write_csv(out/'ALL7617_OWN_BRANCH_ROWS.csv',rows);strataCSV=write_csv(out/'FAMILY_ROLE_TARGET_BRANCH_STATS.csv',flat_stats(strata))
        limits=['Within-checkpoint additive behavioral decomposition only; not proof of visual occlusion cause or information impossibility.','OWN_BASE is this TREAT.base, not the independently trained CONTROL_CANONICAL_LOCAL.','Residual sign-only confusion is a diagnostic of one additive term, not a separately trained or calibrated model.','Family0 saved intended-native witness0 and family1 YCB associated-r surface referents remain separate; no new physical native gold.','SOURCE DEV is exposed/reused and no current/held performance or model/threshold choice is made.','No encoder/raw RGB/GT/reference/pose recomputation, fitting, intervention, coordinate recentering or PnP.']
        summary=write_json(out/'SUMMARY.json',{'completed':True,'PASS':True,'rows':7617,'FIT':5876,'DEV':1741,'branch_metrics':metrics,'family_role_target_stats':strata,'replay_QA':replay,'threshold':0,'two_binary_outputs_not_argmax':True,'target_aligned_residual_definition':'(2*target-1)*own_residual_selected_logit','threshold_crossing_definition':'Ownbase sign to owntotal sign at fixed >=0 E boundary','limits':limits,'new_training_updates':0,'new_current_inference':0,'current_admission':False})
        text='''# Frozen TREAT 내부 분기 SOURCE 진단\n\n같은 최종 checkpoint의 `model.base(active_features)`와 `model.relational(동일 입력)`을 각각 계산한 뒤 FP32 합으로 TOTAL을 만들었습니다. 이는 별도 CONTROL과의 비교가 아닙니다. 두 family는 각자의 단일 logit에 0 문턱을 적용하며 두 출력을 argmax하지 않습니다.\n\n저장 TOTAL의 7617행/15234 logits을 기존 CPU 재생의 atol=rtol=1e-5로 검산하고, 7617개 예측 부호를 정확히 확인해야 완료됩니다. 원 q/L/native/role/target/available는 유지했습니다.\n\n행별 표의 target_aligned_residual은 정답 방향의 잔차 부호·크기이고, threshold crossing은 ownbase와 total의 문턱 교차입니다. 잔차의 단독 sign 분류는 추가항의 행동 진단이며 별도 훈련·보정된 모델이 아닙니다. 수치분해는 시각적 원인·정답 native geometry·인과관계·해결불가능성의 증명이 아닙니다. SOURCE DEV는 이미 노출·재사용됐습니다.\n\n[SUMMARY.json](SUMMARY.json) · [전체 행 CSV](ALL7617_OWN_BRANCH_ROWS.csv) · [family/role/target 통계](FAMILY_ROLE_TARGET_BRANCH_STATS.csv) · [TOTAL 재생 QA](TOTAL_REPLAY_QA.json)\n'''
        report=out/'KO.md'
        with report.open('x') as f:f.write(text)
        complete_out={'completed':True,'PASS':True,'entry':binding(entry_path),'code':e['code'],'protocol':e['implementation_protocol'],'fit_complete':e['fit_complete'],'feature_complete':e['feature_complete'],'frozen_both_heads':e['frozen_both_heads'],'checkpoint':e['checkpoint'],'head_module':e['head_module'],'saved_total_CSV':e['saved_total_CSV'],'features':e['features'],'samples_npz':e['samples_npz'],'geometry_npz':e['geometry_npz'],'reference_replay_code':e['reference_replay_code'],'input_QA':globalQA,'replay_QA':replay,'summary':summary,'rows_JSONL':binding(rowfile),'rows_CSV':rowCSV,'strata_CSV':strataCSV,'report':binding(report),'new_optimizer_updates':0,'encoder_RGB_GT_or_protected_payload_reads':0,'independent_CONTROL_checkpoint_used':False,'automatic_retry':False,'new_current_inference':0,'actual_seconds':time.monotonic()-start}
        write_json(out/'COMPLETE.json',complete_out);print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(out/'COMPLETE.json')}),flush=True)
    except BaseException:
        write_json(out/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)

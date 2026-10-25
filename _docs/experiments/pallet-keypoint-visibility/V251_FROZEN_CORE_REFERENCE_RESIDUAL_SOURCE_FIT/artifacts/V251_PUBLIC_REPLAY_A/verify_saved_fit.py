"""V251 verify-only public CPU replay. No optimizer, training or bank fitting."""
import argparse, ast, csv, hashlib, json, os, stat, sys, types
from pathlib import Path, PurePosixPath
sys.dont_write_bytecode=True
PROTOCOL_SHA='630e3916466b0521a86eb8885853cc47c49600058b3a1fec2d46efc0a10b398c'
PROTOCOL_SIZE=13699
MAP_SHA='15312e86fa05481da23982369895c88330fe6fdac6c2b86be22e955fb9d94a12'
MAP_SIZE=10154
IMPLEMENTATION_MAP_SHA='3f39283c361ba6c23a7804bdb52a377e1e2841428db6902bf3b3c97076e94fe8'
IMPLEMENTATION_MAP_SIZE=950
PRODUCER_SHA='34709709750ed2ee8f30a91fe490bfde16fcf95b6d1e6758ea8f07d23c9ecbc3'
PRODUCER_SIZE=27578
ARMS=('CONTROL_NO_REFERENCE','TREAT_REFERENCE')
MODELS=('BASE_FROZEN_V239',*ARMS)
CORE_STATE_SHA='5988479c4ccfa82a50b91ddc82d235c0009cc78280845830d0ddc4868d53cbf4'
ARTIFACT_KEYS={'producer_code','fit_complete','final_freeze','reference_banks','CONTROL_checkpoint','TREAT_checkpoint','BASE_CSV','CONTROL_CSV','TREAT_CSV'}
CSV_FIELDS=('sample_index','row_id','family','role','target','available','logit0','logit1','selected_family_E_logit','prediction')


def lexical(value):
    s=str(value)
    if any(t in s.casefold() for t in ('v112','cache','.pyc','current223','publicheld','public_held','oldval','oldhold','/hold/','source_holdout','source_val','sourceval','v177','v178','v182','private_evaluation')) or '\x00' in s or '..' in Path(s).parts:
        raise ValueError('Forbidden path before filesystem access')
    return s


def guard(value,exists=False,regular=False):
    s=lexical(value)
    if not Path(s).is_absolute():raise ValueError('Explicit absolute path required')
    p=Path(os.path.abspath(s))
    for parent in reversed((p,*p.parents)):
        try:info=os.lstat(parent)
        except FileNotFoundError:
            if exists:raise
            continue
        if stat.S_ISLNK(info.st_mode):raise ValueError('Symlink artifact/ancestor')
    if regular and not stat.S_ISREG(os.lstat(p).st_mode):raise ValueError('Regular file required')
    return p


def raw(p):
    with os.fdopen(os.open(guard(p,True,True),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()


def binding(p):
    p=guard(p,True,True);h=hashlib.sha256();n=0
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b);n+=len(b)
    return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':n}


def verify(p,record):
    actual=binding(p)
    if any(actual[k]!=record[k] for k in ('sha256','size_bytes')):raise ValueError('SHA/size mismatch: '+str(p))
    return actual


def relative(root,value):
    text=lexical(value);p=PurePosixPath(text)
    if not p.parts or p.is_absolute() or text!=p.as_posix() or '\\' in text or any(x in ('.','..') for x in p.parts):raise ValueError('Canonical relative POSIX path required')
    out=guard(root/Path(*p.parts),True,True)
    if not out.is_relative_to(root):raise ValueError('Outside declared root')
    return out


def jread(p):return json.loads(raw(p))


def jwrite(p,value):
    p=guard(p);data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb')as f:f.write(data)
    return binding(p)


def content_same(actual,expected):
    if any(actual[k]!=expected[k]for k in ('sha256','size_bytes')):raise ValueError('Saved artifact binding identity mismatch')


def pure_module(path,names,namespace):
    tree=ast.parse(raw(path),filename=str(path));selected=[]
    for name in names:
        matches=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name]
        if len(matches)!=1:raise ValueError('Exact pure CODE definition required: '+name)
        selected.extend(matches)
    tree=ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]))
    module=types.ModuleType('v251_public_pure');module.__dict__.update(namespace)
    exec(compile(tree,str(path),'exec'),module.__dict__)
    return module


def authorize(repo,package,entry_path,out):
    repo=guard(repo,True);package=guard(package,True);out=guard(out)
    if not repo.is_dir()or not package.is_dir()or out.exists():raise ValueError('Existing roots and fresh output required')
    here=guard(__file__,True,True).parent
    if not here.is_relative_to(package):raise ValueError('Run wrapper from declared final package')
    e=jread(entry_path)
    if e.get('approved_saved_SOURCE_checkpoint_replay')is not True or e.get('automatic_retry')is not False:raise ValueError('False entry; Root later exact replay entry required')
    if e.get('output_directory')!=str(out):raise ValueError('Exact fresh output binding')
    verify(__file__,e['wrapper_code'])
    protocol_path=here/'ORIGINAL_PROTOCOL.json';verify(protocol_path,{'sha256':PROTOCOL_SHA,'size_bytes':PROTOCOL_SIZE});protocol=jread(protocol_path)
    mp=here/'dependency_map.json';verify(mp,{'sha256':MAP_SHA,'size_bytes':MAP_SIZE});m=jread(mp)
    ip=here/'implementation_dependency.json';verify(ip,{'sha256':IMPLEMENTATION_MAP_SHA,'size_bytes':IMPLEMENTATION_MAP_SIZE});im=jread(ip)
    if set(m['by_key'])!=set(protocol['inputs'])or len(m['by_key'])!=14:raise ValueError('Exact original protocol14 runtime inputs')
    inputs={};verified={}
    for key,r in m['by_key'].items():
        expected=protocol['inputs'][key]
        if expected!={'path':r['original_local_path'],'sha256':r['sha256'],'size_bytes':r['size_bytes']}:raise ValueError('Map/protocol mismatch')
        p=relative(repo,r['repository_relative_path']);verified[key]=verify(p,r);inputs[key]=p
    p=relative(repo,im['repository_relative_path']);verified['utility_code']=verify(p,im);inputs['utility_code']=p
    if set(e['package_bindings'])!=ARTIFACT_KEYS:raise ValueError('Closed final checkpoint/bank/CSV/CODE/artifact bindings')
    artifacts={};artifact_proofs={}
    for key,r in e['package_bindings'].items():
        p=relative(package,r['package_relative_path']);artifact_proofs[key]=verify(p,r);artifacts[key]=p
    if artifact_proofs['producer_code']['sha256']!=PRODUCER_SHA or artifact_proofs['producer_code']['size_bytes']!=PRODUCER_SIZE:raise ValueError('Pinned B V2 API CODE')
    complete=jread(artifacts['fit_complete']);freeze=jread(artifacts['final_freeze'])
    if complete.get('completed')is not True or complete.get('PASS')is not True or freeze.get('completed')is not True or freeze.get('PASS')is not True:raise ValueError('Both-final fit/freeze required')
    content_same(artifact_proofs['producer_code'],complete['code']);content_same(artifact_proofs['final_freeze'],complete['freeze']);content_same(artifact_proofs['reference_banks'],complete['reference_banks'])
    if complete['root_protocol']['sha256']!=PROTOCOL_SHA or complete['core_optimizer_updates']!=0 or complete['new_DEV_core_reference_or_head_outputs_before_bothfinalfreeze']!=0:raise ValueError('Original frozen-core/final-before-DEV contract')
    if complete['optimizer_updates']!={a:1000 for a in ARMS}or freeze['new_updates_each']!=1000 or freeze['core_updates']!=0 or freeze['new_DEV_core_reference_or_head_outputs_before_freeze']!=0:raise ValueError('Both final1000 barrier')
    for arm,tag in zip(ARMS,('CONTROL','TREAT')):
        content_same(artifact_proofs[tag+'_checkpoint'],complete['checkpoints'][arm]);content_same(artifact_proofs[tag+'_checkpoint'],freeze['checkpoints'][arm]);content_same(artifact_proofs[tag+'_CSV'],complete['outputs'][arm])
    content_same(artifact_proofs['BASE_CSV'],complete['BASE_FROZEN_V239_CSV']);content_same(verified['warmstart_checkpoint'],freeze['core_checkpoint']);content_same(artifact_proofs['reference_banks'],freeze['bank'])
    for key in ('head_module','selector_code','utility_code'):content_same(verified[key],complete[key])
    out.mkdir(parents=True)
    admission=jwrite(out/'EXACT_SAVED_REPLAY_ADMISSION.json',{'completed':True,'PASS':True,'entry':binding(entry_path),'wrapper':binding(__file__),'original_protocol':binding(protocol_path),'runtime_inputs':verified,'package_inputs':artifact_proofs,'all_bytes_verified_before_numeric_or_checkpoint_decode':True,'original_absolute_pointer_followed':False,'training_bank_fit_or_optimizer_allowed':False})
    return inputs,artifacts,verified,artifact_proofs,admission,out


def replay(repo,package,entry_path,out):
    inputs,artifacts,verified,proofs,admission,out=authorize(repo,package,entry_path,out)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(17)
        ns={'np':np,'torch':torch,'nn':torch.nn,'hashlib':hashlib,'json':json}
        utility=pure_module(inputs['utility_code'],('state_sha','context_args'),ns)
        producer=pure_module(artifacts['producer_code'],('ReferenceResidual','normalize','reference_inputs'),ns)
        selector=pure_module(inputs['selector_code'],('unit','select_reference_bank'),ns)
        module=types.ModuleType('v251_public_core');exec(compile(raw(inputs['head_module']),str(inputs['head_module']),'exec'),module.__dict__)
        cp=torch.load(inputs['warmstart_checkpoint'],weights_only=True,map_location='cpu')
        if cp['arm']!='TREAT_CANONICAL_CONTEXT' or cp['updates']!=1000 or utility.state_sha(cp['head_state_dict'])!=CORE_STATE_SHA:raise ValueError('Frozen exact V239 core')
        core=module.RelationalQueryHead().float().eval().requires_grad_(False);core.load_state_dict(cp['head_state_dict'],strict=True);del cp
        if sum(p.numel()for p in core.parameters())!=86054:raise ValueError('Frozen86054 core')
        models={};before={}
        for arm,tag in zip(ARMS,('CONTROL','TREAT')):
            cp=torch.load(artifacts[tag+'_checkpoint'],weights_only=True,map_location='cpu')
            if cp['arm']!=arm or cp['updates']!=1000 or cp['core_optimizer_updates']!=0 or cp['trainable_parameters']!=2434 or cp['frozen_core_state_sha256']!=CORE_STATE_SHA or cp['frozen_core_checkpoint_sha256']!=verified['warmstart_checkpoint']['sha256']:raise ValueError('Final residual identity')
            content_same(proofs['reference_banks'],cp['reference_banks'])
            model=producer.ReferenceResidual().float().eval().requires_grad_(False);model.load_state_dict(cp['residual_state_dict'],strict=True)
            if sum(p.numel()for p in model.parameters())!=2434 or any(not torch.isfinite(p).all()for p in model.parameters()):raise ValueError('Finite2434 residual')
            models[arm]=model;before[arm]=utility.state_sha(model.state_dict());del cp
        fkeys={'ORIGINAL_ALL_FEATURES.npy':'original_features','NODE_CLS.npy':'node_cls','CONTEXT_CLS.npy':'context_cls','SAMPLE_GROUP_INDEX.npy':'sample_group_index','SAMPLE_NODE_INDEX.npy':'sample_node_index','RELATIVE_XY8.npy':'relative_xy8'}
        features={k:np.load(inputs[v],mmap_mode='r',allow_pickle=False)for k,v in fkeys.items()}
        x=features['ORIGINAL_ALL_FEATURES.npy'];node=features['NODE_CLS.npy'];group=features['SAMPLE_GROUP_INDEX.npy'];slot=features['SAMPLE_NODE_INDEX.npy']
        shapes={'ORIGINAL_ALL_FEATURES.npy':('float32',(7617,1186)),'NODE_CLS.npy':('float32',(1767,8,384)),'CONTEXT_CLS.npy':('float32',(1767,384)),'SAMPLE_GROUP_INDEX.npy':('int64',(7617,)),'SAMPLE_NODE_INDEX.npy':('int64',(7617,)),'RELATIVE_XY8.npy':('float32',(7617,8,2))}
        for key,(dtype,shape)in shapes.items():
            if features[key].dtype!=np.dtype(dtype)or features[key].shape!=shape:raise ValueError('Canonical original shape/dtype')
        if not ((group>=0)&(group<1767)&(slot>=0)&(slot<8)).all():raise ValueError('Original group/slot index')
        if np.ascontiguousarray(x[:,:384]).tobytes()!=np.ascontiguousarray(node[group,slot]).tobytes():raise ValueError('Original active/node byte sharing')
        with np.load(inputs['samples'],allow_pickle=False)as z:data={k:z[k]for k in ('target','family','fit','available','row_id','condition')}
        with np.load(inputs['geometry'],allow_pickle=False)as z:geometry={k:z[k]for k in ('original_available8','node_available','context_fraction')}
        if any(data[k].shape!=(7617,)for k in ('target','family','fit','available','row_id'))or data['condition'].shape!=(7617,32)or not np.array_equal(data['condition'],x[:,1154:]):raise ValueError('Original sample identity/condition32')
        if not np.isin(data['family'],[0,1]).all()or not np.isin(data['target'],[0,1]).all()or data['available'].dtype!=bool or data['fit'].dtype!=bool:raise ValueError('Original known-source typed metadata')
        syn=data['family']==0;onehot=np.eye(8,dtype=np.float32)[slot[syn]]
        if not np.array_equal(data['condition'][syn,:8],onehot):raise ValueError('Original SYN input address, not teacher canonical part')
        with np.load(artifacts['reference_banks'],allow_pickle=False)as z:banks={k:z[k]for k in ('SYN_prototypes','SYN_supported','YCB_prototypes','YCB_supported','view')}
        if banks['SYN_prototypes'].shape!=(3,8,384)or banks['SYN_supported'].shape!=(3,8)or banks['YCB_prototypes'].shape!=(3,4,384)or banks['YCB_supported'].shape!=(3,4)or tuple(map(str,banks['view']))!=('full_CLS','fixed_centre_Q','centre_CLS'):raise ValueError('Saved fixed bank schema')
        with np.load(inputs['YCB_prototypes'],allow_pickle=False)as z:
            if any(z[k].dtype!=banks[bk].dtype or z[k].shape!=banks[bk].shape or z[k].tobytes()!=banks[bk].tobytes() for k,bk in (('prototypes','YCB_prototypes'),('supported','YCB_supported'))):raise ValueError('Original YCB frozen bank bytes reused')
        predictions={name:np.zeros((7617,2),np.float32)for name in MODELS}
        with torch.no_grad():
            for start in range(0,7617,64):
                ids=np.arange(start,min(start+64,7617));xx=torch.as_tensor(np.array(x[ids],copy=True))
                base=core(xx,*utility.context_args(ids,features,geometry,'cpu'))[0];hidden=core.base.head[1](core.base.head[0](xx))
                ref,_=producer.reference_inputs(ids,x,node,group,slot,geometry,banks['SYN_prototypes'],banks['SYN_supported'],banks['YCB_prototypes'],banks['YCB_supported'],selector.select_reference_bank,data['family'],data['available'])
                predictions['BASE_FROZEN_V239'][ids]=base.numpy()
                for arm in ARMS:
                    rr=torch.zeros_like(torch.as_tensor(ref))if arm==ARMS[0]else torch.as_tensor(ref)
                    predictions[arm][ids]=(base+models[arm](hidden,rr)).numpy()
        results={}
        for name,tag in zip(MODELS,('BASE','CONTROL','TREAT')):
            with artifacts[tag+'_CSV'].open(newline='',encoding='utf-8')as f:
                reader=csv.DictReader(f)
                if tuple(reader.fieldnames)!=CSV_FIELDS:raise ValueError('Exact saved CSV schema')
                rows=list(reader)
            if len(rows)!=7617:raise ValueError('Original7617 CSV denominator')
            for i,r in enumerate(rows):
                if (int(r['sample_index'])!=i or r['row_id']!=str(data['row_id'][i])or int(r['family'])!=int(data['family'][i])or r['role']!=('FIT'if data['fit'][i]else'DEV')or int(r['target'])!=int(data['target'][i])or r['available']!=str(bool(data['available'][i]))):raise ValueError('Saved CSV original row identity')
            expected=np.asarray([[float(r['logit0']),float(r['logit1'])]for r in rows],np.float32);actual=predictions[name]
            if not np.isfinite(actual).all()or not np.isfinite(expected).all()or not np.allclose(actual,expected,atol=1e-5,rtol=1e-5):raise ValueError('Fixed prospective CPU/GPU logit tolerance: '+name)
            selected=actual[np.arange(7617),data['family']];saved_selected=np.asarray([float(r['selected_family_E_logit'])for r in rows],np.float32)
            if not np.allclose(selected,saved_selected,atol=1e-5,rtol=1e-5):raise ValueError('Selected-family raw E-logit')
            pred=np.where(data['available'],np.where(selected>=0,'E','D'),'U')
            if pred.tolist()!=[r['prediction']for r in rows]:raise ValueError('Predictions must be exact, threshold remains0')
            results[name]={'rows':7617,'both_logits_checked':15234,'max_abs':float(np.max(np.abs(actual-expected))),'atol':1e-5,'rtol':1e-5,'same_prediction_rows':7617}
        if utility.state_sha(core.state_dict())!=CORE_STATE_SHA or any(utility.state_sha(models[a].state_dict())!=before[a]for a in ARMS):raise ValueError('Frozen model state unchanged')
        for key,p in artifacts.items():verify(p,proofs[key])
        result={'completed':True,'PASS':True,'mode':'PUBLIC_SAVED_SOURCE_CPU_INFERENCE_ONLY','admission':admission,'models':results,'core86054_and_residual2434_each':True,'SYN_bank_refit':0,'optimizer_updates':0,'training':0,'encoder_or_RGB_GT_current_held':0,'reference_query_GTID':False,'SOURCE_FIT_supervised_bank_reused':True,'SOURCE_DEV_reused_exposed_diagnostic':True,'arrays_nonpixel_labels_and_frozen_cp_decoded':True,'new_model_or_threshold_or_view_choice':0,'same_predictions_not_float_byteexact':True,'wrapper_runtime_versions':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__},'limits':'Verifies saved frozen SOURCE feature/head outputs; not raw-image feature extraction, annotation/GT replay, training reproduction, physical native gold or independent deployment accuracy.'}
        jwrite(out/'COMPLETE.json',result);print(json.dumps(result,ensure_ascii=False))
    except BaseException as exc:
        jwrite(out/'FAILURE.json',{'completed':False,'PASS':False,'error_type':type(exc).__name__,'error':str(exc),'admission':admission,'automatic_retry':False,'partial_outputs_preserved':True});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('public-repo','package','entry','output-dir'):p.add_argument('--'+name,required=True)
    a=p.parse_args();replay(a.public_repo,a.package,a.entry,a.output_dir)

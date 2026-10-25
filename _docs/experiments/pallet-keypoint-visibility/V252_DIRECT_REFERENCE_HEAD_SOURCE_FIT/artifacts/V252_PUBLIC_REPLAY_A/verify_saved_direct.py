"""V252 final-package saved direct41 head CPU verification; no training/bank fit."""
import argparse,ast,csv,hashlib,json,os,stat,sys,types
from pathlib import Path,PurePosixPath
sys.dont_write_bytecode=True
PROTOCOL_SHA='35a49f0628e48f35fbacdbfb40f44aca5fd22d47f58d9cf82dbc9dcff2034a71'
PROTOCOL_SIZE=13136
MAP_SHA='24ae673d44cd5306df61b7eb8990e4f6c1ae590e0e5c654fcd1a47ddf78be6d6'
MAP_SIZE=6360
PRODUCER_SHA='9617992ad39d196d215c4bf9cc3424b470d01c019f1eacca9153559cde59d575'
PRODUCER_SIZE=23925
ARMS=('CONTROL_GENERIC_REFERENCE','TREAT_TARGET_REFERENCE')
ARTIFACT_KEYS={'producer_code','fit_complete','final_freeze','generic_bank','CONTROL_checkpoint','TREAT_checkpoint','CONTROL_CSV','TREAT_CSV'}
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
    if not repo.is_dir()or not package.is_dir()or out.exists():raise ValueError('Existing explicit roots and fresh output required')
    here=guard(__file__,True,True).parent
    if not here.is_relative_to(package):raise ValueError('Use wrapper within declared final package')
    e=jread(entry_path)
    if e.get('approved_saved_SOURCE_direct_head_replay')is not True or e.get('automatic_retry')is not False:raise ValueError('False entry refused before payload')
    if e['output_directory']!=str(out):raise ValueError('Fresh exact output directory')
    verify(__file__,e['wrapper_code'])
    protocol_path=here/'ORIGINAL_PROTOCOL.json';verify(protocol_path,{'sha256':PROTOCOL_SHA,'size_bytes':PROTOCOL_SIZE});protocol=jread(protocol_path)
    mp=here/'runtime_dependencies.json';verify(mp,{'sha256':MAP_SHA,'size_bytes':MAP_SIZE});m=jread(mp)
    if set(m['by_key'])!=set(protocol['inputs'])or len(m['by_key'])!=9:raise ValueError('Exact nine protocol inputs')
    inputs={};verified={}
    for key,r in m['by_key'].items():
        if any(r[k]!=protocol['inputs'][key][k]for k in ('original_local_path','repository_relative_path','sha256','size_bytes')):raise ValueError('Original frozen protocol/map equality')
        p=relative(repo,r['repository_relative_path']);verified[key]=verify(p,r);inputs[key]=p
    if set(e['package_bindings'])!=ARTIFACT_KEYS:raise ValueError('Closed final artifacts only')
    artifacts={};proofs={}
    for key,r in e['package_bindings'].items():
        p=relative(package,r['package_relative_path']);proofs[key]=verify(p,r);artifacts[key]=p
    if proofs['producer_code']['sha256']!=PRODUCER_SHA or proofs['producer_code']['size_bytes']!=PRODUCER_SIZE:raise ValueError('Pinned direct-head pure API code')
    complete=jread(artifacts['fit_complete']);freeze=jread(artifacts['final_freeze'])
    if any(d.get('completed')is not True or d.get('PASS')is not True for d in (complete,freeze)):raise ValueError('Completed both-final1000 required')
    content_same(proofs['producer_code'],complete['code']);content_same(proofs['final_freeze'],complete['freeze']);content_same(proofs['generic_bank'],complete['generic_banks'])
    if complete['root_protocol']['sha256']!=PROTOCOL_SHA or complete['parameters_each']!=1410 or complete['input_width']!=41 or complete['optimizer_updates']!={arm:1000 for arm in ARMS}or complete['old_core_CP_module_hidden_logits_or_preds_forward_inputs']!=0 or complete['new_DEV_head_outputs_before_bothfinalfreeze']!=0:raise ValueError('Fixed direct41 no-core final checkpoint contract')
    if freeze['updates_each']!=1000 or freeze['new_DEV_head_outputs_before_freeze']!=0 or freeze['BASE_CSV_bytes_or_interpretation_before_freeze']!=0:raise ValueError('Both-final-before-DEV/BASE barrier')
    for key in inputs:content_same(verified[key],complete['inputs'][key])
    for arm,tag in zip(ARMS,('CONTROL','TREAT')):
        content_same(proofs[tag+'_checkpoint'],complete['checkpoints'][arm]);content_same(proofs[tag+'_checkpoint'],freeze['checkpoints'][arm]);content_same(proofs[tag+'_CSV'],complete['outputs'][arm])
    content_same(proofs['generic_bank'],freeze['generic_bank']);content_same(verified['BASE_FROZEN_V239_CSV'],complete['BASE_FROZEN_V239_CSV'])
    out.mkdir(parents=True)
    admission=jwrite(out/'EXACT_SAVED_REPLAY_ADMISSION.json',{'completed':True,'PASS':True,'entry':binding(entry_path),'wrapper':binding(__file__),'original_protocol':binding(protocol_path),'runtime_inputs':verified,'package_inputs':proofs,'all_bytes_verified_before_numeric_or_checkpoint_decode':True,'no_old_core_CP_or_forward':True,'training_or_bankfit':False,'original_absolute_input_pointer_followed':False})
    return inputs,artifacts,verified,proofs,complete,admission,out


def replay(repo,package,entry_path,out):
    inputs,artifacts,verified,proofs,complete,admission,out=authorize(repo,package,entry_path,out)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        ns={'np':np,'torch':torch,'nn':torch.nn,'hashlib':hashlib,'json':json}
        producer=pure_module(artifacts['producer_code'],('DirectReferenceHead','build_model','assemble_inputs','normalize','generic_reference_inputs'),ns)
        utility=pure_module(inputs['utility_code'],('state_sha',),ns)
        models={};before={};initial={}
        for arm,tag in zip(ARMS,('CONTROL','TREAT')):
            cp=torch.load(artifacts[tag+'_checkpoint'],weights_only=True,map_location='cpu')
            if cp['arm']!=arm or cp['updates']!=1000 or cp['parameters']!=1410 or cp['input_width']!=41 or cp['old_core_used']is not False or cp['threshold_raw_logit']!=0 or cp['use_lora']is not False:raise ValueError('Direct1410 final identity')
            content_same(proofs['generic_bank'],cp['generic_bank'])
            model=producer.build_model().float().eval().requires_grad_(False);model.load_state_dict(cp['model_state_dict'],strict=True)
            initial_model=producer.build_model().float().eval().requires_grad_(False);initial_model.load_state_dict(cp['initial_model_state_dict'],strict=True)
            if sum(p.numel()for p in model.parameters())!=1410 or len(model.state_dict())!=4 or any(not torch.isfinite(p).all()for p in model.parameters()):raise ValueError('Finite1410 parameter/4tensor head')
            if torch.count_nonzero(initial_model.net[-1].weight)or torch.count_nonzero(initial_model.net[-1].bias):raise ValueError('Zero final layer initialization')
            initial[arm]=utility.state_sha(initial_model.state_dict())
            if initial[arm]!=cp['initial_model_sha256']or initial[arm]!=complete['initial_model_sha256'][arm]:raise ValueError('Saved exact initial state')
            before[arm]=utility.state_sha(model.state_dict())
            if before[arm]!=complete['final_model_sha256'][arm]:raise ValueError('Saved exact final state')
            models[arm]=model;del initial_model,cp
        if initial[ARMS[0]]!=initial[ARMS[1]]:raise ValueError('Matched same initial state')
        x=np.load(inputs['original_features'],mmap_mode='r',allow_pickle=False);targetref=np.load(inputs['V251_saved_reference_inputs'],mmap_mode='r',allow_pickle=False)
        if x.dtype!=np.float32 or x.shape!=(7617,1186)or targetref.dtype!=np.float32 or targetref.shape!=(7617,9):raise ValueError('Original descriptors/targetref shape')
        with np.load(inputs['samples'],allow_pickle=False)as z:data={k:z[k]for k in ('condition','family','target','fit','available','row_id')}
        if any(data[k].shape!=(7617,)for k in ('family','target','fit','available','row_id'))or data['condition'].shape!=(7617,32)or data['condition'].dtype!=np.float32 or data['condition'].tobytes()!=np.ascontiguousarray(x[:,1154:]).tobytes():raise ValueError('Original all7617 source condition/row metadata')
        if not np.isin(data['family'],[0,1]).all()or not np.isin(data['target'],[0,1]).all()or data['available'].dtype!=bool or data['fit'].dtype!=bool:raise ValueError('Original SOURCE typed family/target/roles')
        with np.load(artifacts['generic_bank'],allow_pickle=False)as z:proto=z['prototypes'];supported=z['supported'];views=z['view'];families=z['family']
        if proto.dtype!=np.float64 or proto.shape!=(2,3,384)or supported.dtype!=bool or supported.shape!=(2,3)or tuple(map(str,views))!=('full_CLS','fixed_centre_Q','centre_CLS')or not np.array_equal(families,[0,1]):raise ValueError('Saved generic2family x3views bank schema')
        actual={arm:np.zeros((7617,2),np.float32)for arm in ARMS}
        with torch.no_grad():
            for start in range(0,7617,64):
                ids=np.arange(start,min(start+64,7617));generic=producer.generic_reference_inputs(ids,x,data['family'],data['available'],proto,supported)
                for arm,ref in ((ARMS[0],generic),(ARMS[1],np.array(targetref[ids],copy=True))):
                    inp=producer.assemble_inputs(ref,np.array(data['condition'][ids],copy=True));actual[arm][ids]=models[arm](torch.as_tensor(inp)).numpy()
        results={}
        for arm,tag in zip(ARMS,('CONTROL','TREAT')):
            with artifacts[tag+'_CSV'].open(newline='',encoding='utf-8')as src:
                reader=csv.DictReader(src)
                if tuple(reader.fieldnames)!=CSV_FIELDS:raise ValueError('Fixed saved CSV fields')
                rows=list(reader)
            if len(rows)!=7617:raise ValueError('All original known7617 rows')
            for i,r in enumerate(rows):
                if int(r['sample_index'])!=i or r['row_id']!=str(data['row_id'][i])or int(r['family'])!=int(data['family'][i])or int(r['target'])!=int(data['target'][i])or r['role']!=('FIT'if data['fit'][i]else'DEV')or r['available']!=str(bool(data['available'][i])):raise ValueError('Saved CSV original identity/role/target/U')
            expected=np.asarray([[float(r['logit0']),float(r['logit1'])]for r in rows],np.float32);z=actual[arm]
            if not np.isfinite(z).all()or not np.isfinite(expected).all()or not np.allclose(z,expected,atol=1e-5,rtol=1e-5):raise ValueError('Prospective fixed CPU/GPU logits1e-5')
            selected=z[np.arange(7617),data['family']];saved_selected=np.asarray([float(r['selected_family_E_logit'])for r in rows],np.float32)
            if not np.allclose(selected,saved_selected,atol=1e-5,rtol=1e-5):raise ValueError('Selected-family raw E-logit')
            pred=np.where(data['available'],np.where(selected>=0,'E','D'),'U')
            if pred.tolist()!=[r['prediction']for r in rows]:raise ValueError('All7617 predictions exact')
            if utility.state_sha(models[arm].state_dict())!=before[arm]:raise ValueError('Frozen checkpoint state unchanged')
            results[arm]={'rows':7617,'both_logits_checked':15234,'max_abs':float(np.max(np.abs(z-expected))),'atol':1e-5,'rtol':1e-5,'same_prediction_rows':7617}
        # Saved BASE is a row-join comparator only, never an old core model/input.
        with inputs['BASE_FROZEN_V239_CSV'].open(newline='',encoding='utf-8')as src:base=list(csv.DictReader(src))
        if len(base)!=7617 or any(int(r['sample_index'])!=i or r['row_id']!=str(data['row_id'][i])or int(r['family'])!=int(data['family'][i])or int(r['target'])!=int(data['target'][i])or r['role']!=('FIT'if data['fit'][i]else'DEV')or r['available']!=str(bool(data['available'][i]))for i,r in enumerate(base)):raise ValueError('Pinned BASE CSV same original rows only')
        for key,p in artifacts.items():verify(p,proofs[key])
        result={'completed':True,'PASS':True,'mode':'V252_PUBLIC_SAVED_DIRECT_HEAD_CPU_INFERENCE_ONLY','admission':admission,'models':results,'parameters_each':1410,'input_width':41,'initial_state_same_and_final_layer_zero':True,'final_checkpoint_state_unchanged':True,'BASE_saved_row_join_count':7617,'BASE_model_forward':0,'old_core_CP_or_module':0,'generic_or_target_bank_refit':0,'optimizer_or_training':0,'RGB_GT_DINO_or_current_held':0,'query_true_objectID':False,'SOURCE_supervised_reference_statistics_reused':True,'SOURCE_DEV_exposed_reused_not_blind':True,'same_prediction_not_float_byteexact':True,'versions':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__},'limits':'Saved SOURCE feature/reference/head replay only; not image/GT/teacher/training reproduction or physical native/deployment success. Generic-reference duplication is not certified wrong target.'}
        jwrite(out/'COMPLETE.json',result);print(json.dumps(result,ensure_ascii=False))
    except BaseException as exc:
        jwrite(out/'FAILURE.json',{'completed':False,'PASS':False,'error_type':type(exc).__name__,'error':str(exc),'admission':admission,'automatic_retry':False});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('public-repo','package','entry','output-dir'):p.add_argument('--'+key,required=True)
    a=p.parse_args();replay(a.public_repo,a.package,a.entry,a.output_dir)

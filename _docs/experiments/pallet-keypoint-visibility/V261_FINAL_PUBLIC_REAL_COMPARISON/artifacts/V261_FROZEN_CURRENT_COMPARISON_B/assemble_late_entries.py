"""Small Root-only late entry assembler; never launches inference or gallery.

Current stage: after BOTH final1000 metadata proof, read the original input-only
manifest, resolve only its declared aliases and hash encoded bytes (no decode).
Gallery stage: after saved comparison, derive bindings from generated metadata.
"""
import argparse, ast, hashlib, json, os, stat, sys, types
from pathlib import Path
import numpy as np

sys.dont_write_bytecode=True
HERE=Path(__file__).absolute().parent;R=HERE.parent;ROOT=R/'V261_ROOT_FINAL_PUBLIC_REAL_COMPARISON'
CURRENT_TEMPLATE={'path':str(HERE/'ENTRY_FALSE_TEMPLATE.json'),'sha256':'3b549078f8ba9d069bef4c7fc4031388a501ac9eb4353507d3db532e40d18009','size_bytes':13806}
GALLERY_TEMPLATE={'path':str(R/'V261_FINAL_CROP_GALLERY_B/ENTRY_FALSE_TEMPLATE.json'),'sha256':'32dd4dd4a7320e14ebf1511e11ece6d2d621f70f6b2754b51d3a865940dd4bb1','size_bytes':2835}
ARMS=('CONTROL_ORIGINAL_SOURCE_FIT','TREAT_ORIGINAL_PLUS_PUBLIC_SURFACE_AUX')

def need(ok,message):
    if not bool(ok):raise ValueError(message)

def lexical(value):
    s=str(value);need(not any(x in s.casefold()for x in('v112','cache','.pyc','source_holdout')),'Forbidden literal before FS')
    p=Path(s);need(p.is_absolute()and'..'not in p.parts,'Closed absolute path');return p

def regular(value,exists=True):
    p=lexical(value)
    for a in [*reversed(p.parents),p]:
        try:mode=os.lstat(a).st_mode
        except FileNotFoundError:need(not exists,'Missing exact artifact');continue
        need(not stat.S_ISLNK(mode),'Regular artifact and ancestors required')
    return p

def checked(b):
    p=regular(b['path']);raw=p.read_bytes()
    need(hashlib.sha256(raw).hexdigest()==b['sha256']and len(raw)==b['size_bytes'],'Exact SHA/size');return raw

def binding(p):
    raw=regular(p).read_bytes();return{'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}

def missing_size_stat_only(b):
    """No body/hash read: old SHA remains authoritative until child's late phase."""
    out=dict(b);p=regular(out['path'])
    if out['size_bytes']is None:out['size_bytes']=os.stat(p,follow_symlinks=False).st_size
    return out

def write_json(path,obj):
    p=regular(path,False);need(not p.exists(),'Fresh entry/scope; no overwrite')
    with p.open('x',encoding='utf-8')as f:json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return binding(p)

def unique(bindings):return list({(b['path'],b['sha256'],b['size_bytes']):b for b in bindings}.values())

def frozen_training(complete_binding,expected_policy):
    complete=json.loads(checked(complete_binding))
    need(complete.get('completed')is True and complete.get('PASS')is True and complete['policy']==expected_policy,'Completed final-one training')
    need(complete['new_updates']=={a:1000 for a in ARMS}and complete['head_parameters_each']==76098,'Both final1000 old-head architecture')
    freeze=json.loads(checked(complete['freeze']))
    need(freeze.get('completed')is True and freeze.get('PASS')is True and freeze['updates']=={a:1000 for a in ARMS}and freeze['DEV_or_current_or_v61_head_forwards_before_barrier']==0,'Both final heads/logs/draws metadata barrier')
    need({a:freeze['checkpoints_and_logs'][a]['checkpoint']for a in ARMS}==complete['checkpoints'],'Final checkpoint bindings agree')
    return complete,freeze

def load_declared_alias_resolver(binding_):
    """Reuse only the literal old resolver AST; aliases are an explicit exception."""
    node=next(n for n in ast.parse(checked(binding_)).body if isinstance(n,ast.FunctionDef)and n.name=='safe_original_image_path')
    def guarded_lstat(value):return os.lstat(lexical(value))
    def guarded_readlink(value):
        target=os.readlink(lexical(value))
        # Relative '..' is transported by the frozen normalizer, never blindly
        # followed. Every resulting absolute component passes guarded_lstat.
        need(not any(x in target.casefold()for x in('v112','cache','.pyc','source_holdout')),'Forbidden alias target before expansion')
        return target
    def encoded_sha(value):
        p=regular(value);h=hashlib.sha256()
        with p.open('rb')as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
        return h.hexdigest()
    ns={'os':types.SimpleNamespace(path=os.path,lstat=guarded_lstat,readlink=guarded_readlink),'stat':stat,'Path':Path,'sha':encoded_sha}
    exec(compile(ast.Module(body=[node],type_ignores=[]),binding_['path'],'exec'),ns)
    return ns['safe_original_image_path']

def current_entry(config):
    e=json.loads(checked(CURRENT_TEMPLATE));complete,_=frozen_training(config['training_complete'],e['policy'])
    # Above is the gate: no current metadata/image/GT/CP body is touched first.
    e['training_complete']=config['training_complete'];e['training_freeze']=complete['freeze'];e['checkpoints']=complete['checkpoints']
    e['training_metadata_whitelist']=[e['policy'],e['dependency_closure'],e['training_complete'],e['training_freeze']]
    e['input_manifest']=missing_size_stat_only(e['input_manifest'])
    data=json.loads(checked(e['input_manifest']))
    e['input_metadata_whitelist']=[e['input_manifest'],e['input_QA']]
    contract=e['additional_code']['V229_input_contract'];alias=e['additional_code']['alias_resolver']
    node=next(n for n in ast.parse(checked(contract)).body if isinstance(n,ast.FunctionDef)and n.name=='validate_current')
    ns={'np':np,'require':need,'identity':lambda b:(str(Path(os.path.abspath(b['path']))),b['sha256']),'ALIAS_CODE':Path(alias['path']),'ALIAS_SHA':alias['sha256']}
    exec(compile(ast.Module(body=[node],type_ignores=[]),contract['path'],'exec'),ns)
    frames=ns['validate_current'](data,np);resolve=load_declared_alias_resolver(alias);mappings=[]
    for frame in frames:
        original=frame['source_fullRGB']['binding'];lexical(original['path'])
        path=regular(resolve(original))
        size=os.stat(path,follow_symlinks=False).st_size
        if type(original.get('size_bytes'))is int:need(size==original['size_bytes'],'Original encoded image size')
        mappings.append({'frame_index':frame['frame_index'],'alias_binding':original,'canonical_binding':{'path':str(path),'sha256':original['sha256'],'size_bytes':size}})
    need(len(mappings)==223,'Only the223 explicitly declared aliases')
    e['encoded_alias_transport']['canonical_mappings']=mappings
    official=json.loads(checked(e['dependency_closure']))['model_loader_config']['official_state']
    e['payload_whitelist']=unique([official,*e['checkpoints'].values(),*[m['canonical_binding']for m in mappings]])
    # Cohort annotations: size only here; body/hash is deferred until the child
    # freezes BOTH new1784 prediction/feature artifacts. No baseline read here.
    e['cohort_flags']=missing_size_stat_only(e['cohort_flags'])
    e['late_evaluation_whitelist']=[e['evaluation_ledger'],e['cohort_flags']]
    e['approved_actual_final_current_comparison']=True
    return e,{'current_manifest_rows':1784,'declared_aliases_resolved':223,'encoded_image_files_hashed':223,'RGB_images_decoded':0,'GT_or_baseline_body_reads':0,'checkpoint_or_weight_body_reads':0,'child_executed':False}

def gallery_entry(config):
    e=json.loads(checked(GALLERY_TEMPLATE));complete=json.loads(checked(config['comparison_complete']))
    need(complete.get('completed')is True and complete.get('PASS')is True and complete['policy']==e['policy'],'Completed final fixed current comparison')
    freeze=json.loads(checked(complete['prediction_freeze']))
    need(freeze.get('completed')is True and freeze.get('PASS')is True and freeze['rows_each']==1784 and freeze['GT_cohort_or_v61_baseline_body_reads_before_this_freeze']==0,'Both current prediction/feature artifacts frozen')
    e['comparison_complete']=config['comparison_complete'];e['prediction_freeze']=complete['prediction_freeze']
    e['ledger']=complete['saved_evaluation_outputs']['ALL1784_FINAL_COMPARISON_LEDGER'];e['input_facts']=complete['input_facts']
    facts=json.loads(checked(e['input_facts']))['frames'];need(len(facts)==223,'Original223 input-fact records')
    e['metadata_whitelist']=[e['policy'],e['comparison_complete'],e['prediction_freeze'],e['ledger'],e['input_facts']]
    e['image_whitelist']=unique([f['canonical_encoded_binding']for f in facts]);e['approved_actual_final_gallery']=True
    return e,{'input_fact_records':223,'GT_ledger_body_reads':0,'RGB_images_hashed_or_decoded':0,'checkpoint_or_weight_body_reads':0,'case_selection':'Deferred to child; child writes selection before selectedRGB decode','child_executed':False}

def run(config_path):
    config_raw=regular(config_path).read_bytes();config=json.loads(config_raw)
    need(config.get('approved_late_entry_assembly')is True and config.get('automatic_child_execution')is False,'Root late true config required; no automatic child launch')
    need(config['code']==binding(Path(__file__).absolute()),'Exact assembler CODE')
    stage=config['stage'];need(stage in('current','gallery'),'Two fixed assembly modes only')
    entry_path=regular(config['entry_output'],False);scope_path=regular(config['scope_output'],False)
    need(entry_path.parent==scope_path.parent==ROOT and not entry_path.exists()and not scope_path.exists(),'Fresh direct Root entry/scope paths')
    e,counters=current_entry(config)if stage=='current'else gallery_entry(config)
    flag='approved_actual_final_current_comparison'if stage=='current'else'approved_actual_final_gallery'
    scope={flag:True,'entry_contract':{k:v for k,v in e.items()if k!='root_scope'},'assembly_config':{'path':str(config_path),'sha256':hashlib.sha256(config_raw).hexdigest(),'size_bytes':len(config_raw)},'assembly_CODE':config['code'],'counters':counters}
    e['root_scope']=write_json(scope_path,scope);entry=write_json(entry_path,e)
    print(json.dumps({'completed':True,'PASS':True,'stage':stage,'entry':entry,'scope':e['root_scope'],'counters':counters}),flush=True)
    return entry,e['root_scope']

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);run(ap.parse_args().entry)

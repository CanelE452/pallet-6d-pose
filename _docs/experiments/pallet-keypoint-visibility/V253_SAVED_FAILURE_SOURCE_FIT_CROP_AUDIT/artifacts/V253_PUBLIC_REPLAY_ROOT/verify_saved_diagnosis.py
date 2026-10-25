"""Recompute V253 saved-score diagnosis from six exact public SOURCE inputs."""
import argparse, hashlib, json, os, types
from pathlib import Path, PurePosixPath
import numpy as np

PROTOCOL_SHA='e9f2a2217930ce38fde59a8c1ebd922cd97f7ad8fe2db02cfd76dafa8511b513'
MAP_SHA='21190dc91292c536e1126e10ace8c6af523b737f671f0726773d8ea9e9324e90'
CODE_SHA='a8c5fc5d2173ce323e7a2aba0dfdec3e2001a9d11ed77eabf2159cf7278f14c2'

def safe(value):
    p=Path(value)
    if not p.is_absolute() or '..' in p.parts or any(t in str(p).casefold() for t in ('v112','cache','.pyc','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation')):raise ValueError('Forbidden explicit artifact path')
    for a in reversed([p,*p.parents]):
        if a.is_symlink():raise ValueError('Artifact symlink/ancestor')
    return p

def raw(p):return safe(p).read_bytes()
def digest(b):return hashlib.sha256(b).hexdigest()
def bind(p):
    p=safe(p);b=raw(p);return {'path':str(p),'sha256':digest(b),'size_bytes':len(b)}
def check(p,b):
    a=bind(p)
    if any(a[k]!=b[k] for k in ('sha256','size_bytes')):raise ValueError('Exact byte binding mismatch')
    return safe(p)
def relative(root,rel):
    p=PurePosixPath(rel)
    if not p.parts or p.is_absolute() or '..' in p.parts or str(p)!=rel or '\\' in rel:raise ValueError('Canonical relative path')
    return safe(root/Path(*p.parts))
def jread(p):return json.loads(raw(p))
def jwrite(p,obj):
    p=safe(p)
    with p.open('xb') as f:f.write((json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())

def run(repo,package,entry_path,out):
    repo=safe(repo);package=safe(package);out=safe(out);entry=jread(entry_path)
    if entry.get('approved_public_saved_SOURCE_diagnosis_replay') is not True or entry.get('automatic_retry') is not False or entry['output_directory']!=str(out) or out.exists():raise ValueError('Exact fresh replay entry')
    check(__file__,entry['wrapper_code'])
    here=safe(Path(__file__).absolute().parent)
    if not here.is_relative_to(package):raise ValueError('Execute final packaged verifier')
    mp=here/'runtime_dependencies.json';mb=raw(mp)
    if digest(mb)!=MAP_SHA:raise ValueError('Frozen six-input public map')
    mapping=json.loads(mb);artifacts={}
    for key,b in entry['package_bindings'].items():artifacts[key]=check(relative(package,b['package_relative_path']),b)
    protocol=jread(artifacts['root_protocol'])
    if bind(artifacts['root_protocol'])['sha256']!=PROTOCOL_SHA or set(mapping['by_key'])!=set(protocol['diagnostic_numeric_inputs']):raise ValueError('Frozen protocol exact six inputs')
    paths={};verified={}
    for key,b in mapping['by_key'].items():
        original=protocol['inputs'][key]
        if b['original_local_path']!=original['path'] or any(b[k]!=original[k] for k in ('sha256','size_bytes')):raise ValueError('Public map original identity')
        paths[key]=check(relative(repo,b['repository_relative_path']),b);verified[key]=bind(paths[key])
    complete=jread(artifacts['diagnosis_complete'])
    if complete.get('completed') is not True or complete.get('PASS') is not True or complete['code']['sha256']!=CODE_SHA or complete['new_forward_training_CP_RGB_GT_or_protected_reads']!=0:raise ValueError('Completed frozen saved-score diagnosis')
    if bind(artifacts['producer_code'])['sha256']!=CODE_SHA:raise ValueError('Pinned pure diagnostic CODE')
    for key,p in paths.items():check(p,complete['inputs'][key])
    outputs={}
    for name,b in complete['outputs'].items():
        key='saved_output/'+name
        if key not in artifacts:raise ValueError('Missing exact output binding')
        check(artifacts[key],b);outputs[name]=artifacts[key]
    # All new package leaves are checked; serialized provenance paths stay inert.
    manifest=jread(package/'MANIFEST.json')
    if manifest.get('PASS') is not True or manifest.get('completed') is not True:raise ValueError('Completed package manifest')
    for row in manifest['files']:check(relative(package,row['relative_path']),row)
    out.mkdir(parents=True)
    jwrite(out/'ADMISSION.json',{'completed':True,'PASS':True,'entry':bind(entry_path),'package_manifest':bind(package/'MANIFEST.json'),'public_source_inputs':verified,'original_absolute_pointer_followed':False,'model_or_training_or_bankfit':0,'full_rawRGB_or_GT_archive_read':0})
    try:
        module=types.ModuleType('v253_public_pinned_diagnostic');module.__file__=str(artifacts['producer_code'])
        exec(compile(raw(artifacts['producer_code']),str(artifacts['producer_code']),'exec'),module.__dict__)
        with np.load(paths['samples'],allow_pickle=False) as z:samples=module.validate_samples({k:z[k] for k in module.SAMPLE_KEYS})
        generic=np.load(paths['generic_ref9'],mmap_mode='r',allow_pickle=False);target=np.load(paths['target_ref9'],mmap_mode='r',allow_pickle=False)
        saved=module.load_saved_rows(paths,samples);rows=module.make_diag_rows(saved,generic,target,samples);tables=module.summarize(rows);selection=module.make_selection(rows)
        rebuilt=module.write_results(out/'recomputed',rows,tables,selection)
        if set(rebuilt)!=set(outputs):raise ValueError('Recomputed output names')
        comparisons={}
        for name,b in rebuilt.items():
            actual=bind(outputs[name])
            if any(actual[k]!=b[k] for k in ('sha256','size_bytes')):raise ValueError('Numerical/selection output not byteexact: '+name)
            comparisons[name]={'sha256':actual['sha256'],'size_bytes':actual['size_bytes'],'byteexact':True}
        result={'completed':True,'PASS':True,'mode':'PUBLIC_SAVED_SOURCE_SCORE_DIAGNOSIS_RECOMPUTATION_ONLY','known_rows':len(rows),'public_inputs':len(paths),'recomputed_outputs':comparisons,'selected_FIT_rows':selection['unique_selected_rows'],'selection_cells':len(selection['groups']),'empty_cells':selection['empty_cells'],'model_forward_CP_training_or_bankfit':0,'new_threshold_or_model_choice':False,'current_or_held_reads':0,'full_raw_RGB_or_GT_archive_read':0,'source_teacher_derived_labels_and_conditions_read':True,'original_absolute_fallback':False,'limits':'Replays saved-score joins/statistics and deterministic FIT selection. Published selected FIT pixel files are package-hash checked; original full Source RGB/teacher generation and model training are not reproduced. No deployment or private transfer success claim.','numpy_version':np.__version__}
        jwrite(out/'COMPLETE.json',result);print(json.dumps(result,ensure_ascii=False))
    except BaseException as exc:
        jwrite(out/'FAILED.json',{'completed':False,'PASS':False,'type':type(exc).__name__,'error':str(exc),'automatic_retry':False});raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('public-repo','package','entry','output-dir'):p.add_argument('--'+k,required=True)
    a=p.parse_args();run(a.public_repo,a.package,a.entry,a.output_dir)

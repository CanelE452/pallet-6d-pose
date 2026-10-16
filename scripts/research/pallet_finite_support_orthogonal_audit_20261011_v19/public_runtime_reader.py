"""Portable location-only adapter for the unchanged V19 stdlib clock audit.

Producer JSON and all clock/counter/candidate/metadata formulas remain literal.
Only paths() is dispatched differently: its exact code runs in copied globals
whose declared_path resolves explicit manifest origins before the old private
root check. Direct external-readonly declarations remain unverified metadata.
The unchanged original checker still requires its own fresh freeze/run.
"""
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType,SimpleNamespace

CHECKER_SHA='729c3797b04e27d696fc5d343e1a7ea1908f0d08962215dc8e83e9baed800842'
RESOLVER_SHA='c12701e12dcd3e983c69033652958f9c4e26671c3e12670bd17c6ccf9c8411ad'

def load_resolver():
    path=Path(__file__).absolute().with_name('public_stats_reader.py')
    if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('missing/symlink explicit resolver source')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=RESOLVER_SHA:
        raise ValueError('exact adjacent resolver source SHA required')
    spec=importlib.util.spec_from_file_location('_v19_public_runtime_resolver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,path

def main():
    helper,resolver_source=load_resolver();p=helper.parser(__doc__)
    p.add_argument('--validation-input',type=Path,required=True);p.add_argument('--runtime-input',type=Path,required=True)
    args=p.parse_args();repo=helper.safe_path(args.repo);artifact=helper.safe_path(args.artifact_root)
    for name in ('input','output','validation_input','runtime_input'):
        helper.require(getattr(args,name).is_absolute(),'explicit absolute '+name+' required')
    folder=helper.safe_path(args.input);validation=helper.safe_path(args.validation_input)
    runtime=helper.safe_path(args.runtime_input);out=helper.safe_path(args.output)
    helper.require(all(path.is_relative_to(artifact) for path in (folder,validation,runtime,out)),'copied artifact inputs and new checker output subtree required')
    relocation=helper.Relocations(repo,artifact,args.manifest)
    reader=relocation.resolve(__file__);resolver=relocation.resolve(resolver_source)
    helper.require(reader==helper.safe_path(__file__) and resolver==resolver_source,'executing reader and resolver must themselves be in the copied root')
    checker=relocation.resolve(args.checker or Path(__file__).with_name('runtime_check.py'))
    namespace=helper.load_checker(checker,CHECKER_SHA);original_paths=namespace['paths'];original_declared=namespace['declared_path']
    def declared_path(a,key,value):
        origin=value.get('origin')
        if origin in ('external_readonly_dependency','public_repository'):
            target=original_declared(a,key,value)
            return None if target is None else relocation.resolve(target)
        source=Path(value['path'])
        helper.require(origin is None and source.is_absolute() and '..' not in source.parts,'explicit unchanged absolute declaration '+key)
        return relocation.resolve(source)
    path_globals=dict(namespace);path_globals['declared_path']=declared_path
    relocated_paths=FunctionType(original_paths.__code__,path_globals,original_paths.__name__,original_paths.__defaults__,original_paths.__closure__)
    helper.require(relocated_paths.__code__ is original_paths.__code__,'whole original paths code must remain unchanged')
    def paths(a):
        enumeration=SimpleNamespace(**vars(a));enumeration.private=relocation.original_artifact_root
        result={key:relocation.resolve(path) for key,path in relocated_paths(enumeration).items()}
        result.update(public_reader=reader,public_reader_resolver=resolver,relocation_manifest=relocation.manifest,original_checker=checker)
        return result
    namespace['paths']=paths
    if args.stage=='freeze':out.mkdir(parents=True,exist_ok=True)
    helper.dispatch(namespace,[str(checker),args.stage,'--repo',str(repo),'--private',str(artifact),
        '--input',str(folder),'--validation-input',str(validation),'--runtime-input',str(runtime),'--output',str(out)])

if __name__=='__main__':main()

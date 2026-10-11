"""Relocate the unchanged V19 stdlib geometry audit using explicit byte origins.

The whole original checker AST, metadata, independent scalar formulas, cusp
rules and saved failure history remain unchanged. Only input_paths() is routed
to copied repo/artifact locations; producer JSON paths are not rewritten. This
adapter does not turn the historical V16 full FAIL into PASS or run production.
"""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

CHECKER_SHA='0456bbc94b039e32fa689e09b60b349626427634c2be80067b80f53d5629e4a4'
RESOLVER_SHA='c12701e12dcd3e983c69033652958f9c4e26671c3e12670bd17c6ccf9c8411ad'

def load_resolver():
    # An absolute adjacent source load also works under python -I -S -B;
    # neither the process search path nor an ambient module is consulted.
    path=Path(__file__).absolute().with_name('public_stats_reader.py')
    if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('missing/symlink explicit resolver source')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=RESOLVER_SHA:
        raise ValueError('exact adjacent resolver source SHA required')
    spec=importlib.util.spec_from_file_location('_v19_public_geometry_resolver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,path

def main():
    helper,resolver_source=load_resolver();p=helper.parser(__doc__);p.add_argument('--private',type=Path)
    args=p.parse_args();repo=helper.safe_path(args.repo);artifact=helper.safe_path(args.artifact_root)
    helper.require(args.input.is_absolute() and args.output.is_absolute(),'explicit absolute input/output required')
    folder=helper.safe_path(args.input);out=helper.safe_path(args.output);private=helper.safe_path(args.private or artifact)
    helper.require(folder.is_relative_to(artifact) and private.is_relative_to(artifact) and out!=folder,'copied geometry/private root and distinct output required')
    relocation=helper.Relocations(repo,artifact,args.manifest)
    reader=relocation.resolve(__file__);resolver=relocation.resolve(resolver_source)
    helper.require(reader==helper.safe_path(__file__) and resolver==resolver_source,'executing reader and resolver must themselves be in the copied root')
    checker=relocation.resolve(args.checker or Path(__file__).with_name('validation.py'))
    namespace=helper.load_checker(checker,CHECKER_SHA);original_paths=namespace['input_paths']
    def input_paths(a):
        enumeration=SimpleNamespace(**vars(a));enumeration.private=relocation.original_artifact_root
        result={key:relocation.resolve(path) for key,path in original_paths(enumeration).items()}
        result.update(public_reader=reader,public_reader_resolver=resolver,
            relocation_manifest=relocation.manifest,original_checker=checker)
        return result
    namespace['input_paths']=input_paths
    # The unchanged geometry driver expects its own output directory to exist.
    if args.stage=='freeze':out.mkdir(parents=True,exist_ok=True)
    helper.dispatch(namespace,[str(checker),args.stage,'--repo',str(repo),'--input',str(folder),'--output',str(out),'--private',str(private)])

if __name__=='__main__':main()

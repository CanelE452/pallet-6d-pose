"""Relocate the unchanged V19 stdlib statistics reader for a copied public root.

No producer JSON is edited. The original checker module AST is executed in an
isolated namespace, and only paths() is dispatched through an explicit manifest.
The checker keeps its original metadata, moments, bootstrap and failure rules.
This is a reader adapter; a portable PASS requires its own actual freeze/run.

Manifest schema: observed_frame_public_relocation_manifest_v19. Each record in
records has original_path (absolute), root ('repo' or 'artifact'), relative_path,
bytes, and sha256. Optional original_artifact_root declares the historical
private prefix used only when enumerating geometry/runtime dependency names.
Historical declarations remain literal in copied JSON.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

SCHEMA='observed_frame_public_relocation_manifest_v19'
CHECKER_SHA='4cad27af50af1711ac1817e3c24cfddf73386b535f1cf8c30950950ef2ab1b0c'

def require(value,message):
    if not value:raise ValueError(message)

def safe_path(path):
    path=Path(path).absolute()
    require(not any(p.is_symlink() for p in (path,*path.parents)),'symlink: '+str(path))
    return path

def binding(path):
    path=safe_path(path);require(path.is_file(),'required file missing: '+str(path));h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())

class Relocations:
    """Explicit declared-origin lookup; no existence-based fallback or search."""
    def __init__(self,repo,artifact,manifest):
        self.roots={key:safe_path(path) for key,path in (('repo',repo),('artifact',artifact))}
        require(all(p.is_dir() for p in self.roots.values()),'existing public roots required')
        self.manifest=safe_path(manifest);self.records={};self.cache={};self.locations={}
        require(any(self.manifest.is_relative_to(p) for p in self.roots.values()),'manifest must be inside a copied public root')
        value=json.loads(self.manifest.read_bytes())
        require(value['schema']==SCHEMA and isinstance(value['records'],list),'explicit relocation manifest required')
        self.original_artifact_root=Path(value.get('original_artifact_root',str(self.roots['artifact'])))
        require(self.original_artifact_root.is_absolute() and '..' not in self.original_artifact_root.parts,'explicit original artifact prefix')
        for record in value['records']:
            require(set(record)=={'original_path','root','relative_path','bytes','sha256'},'exact relocation fields')
            old=record['original_path'];relative=Path(record['relative_path']);root=record['root']
            require(isinstance(old,str) and Path(old).is_absolute() and '..' not in Path(old).parts,'absolute declared origin')
            require(root in self.roots and not relative.is_absolute() and relative.parts and '..' not in relative.parts,'safe explicit relative location')
            require(type(record['bytes']) is int and record['bytes']>=0 and isinstance(record['sha256'],str) and len(record['sha256'])==64 and all(c in '0123456789abcdef' for c in record['sha256']),'exact byte/SHA declaration')
            require(old not in self.records,'duplicate declared origin: '+old)
            physical=safe_path(self.roots[root]/relative)
            expected=(record['bytes'],record['sha256'])
            require(physical not in self.locations or self.locations[physical]==expected,'conflicting copied byte declarations')
            self.locations[physical]=expected;self.records[old]=(physical,expected)

    def resolve(self,path):
        source=Path(path).absolute();item=self.records.get(str(source))
        if item is not None:
            physical,expected=item
        else:
            # CLI folders and fixed repo-relative code are already explicit
            # copied locations. An unrelated historical absolute path is fatal.
            require(any(source.is_relative_to(root) for root in self.roots.values()),'unmapped historical origin: '+str(source))
            physical=safe_path(source);expected=self.locations.get(physical)
        actual=self.cache.get(physical)
        if actual is None:actual=binding(physical);self.cache[physical]=actual
        if expected is not None:require((actual['bytes'],actual['sha256'])==expected,'relocated bytes differ: '+str(source))
        return physical

def load_checker(path,expected_sha):
    path=safe_path(path);source=path.read_bytes()
    require(hashlib.sha256(source).hexdigest()==expected_sha,'unchanged checker source SHA required')
    tree=ast.parse(source,filename=str(path));before=ast.dump(tree,include_attributes=False)
    namespace=dict(__name__='_v19_public_reader_isolated_checker',__file__=str(path),__package__=None)
    exec(compile(tree,str(path),'exec'),namespace)
    require(ast.dump(tree,include_attributes=False)==before,'whole checker AST changed')
    return namespace

def dispatch(namespace,argv):
    previous=sys.argv
    try:sys.argv=list(argv);namespace['main']()
    finally:sys.argv=previous

def parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('stage',choices=('freeze','run'));p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--artifact-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checker',type=Path);return p

def main():
    args=parser(__doc__).parse_args();repo=safe_path(args.repo);artifact=safe_path(args.artifact_root)
    require(args.input.is_absolute() and args.output.is_absolute(),'explicit absolute input/output required')
    folder=safe_path(args.input);out=safe_path(args.output)
    require(folder.is_relative_to(artifact) and out!=folder,'input must be inside the copied artifact root; output separate')
    relocation=Relocations(repo,artifact,args.manifest)
    reader=relocation.resolve(__file__)
    require(reader==safe_path(__file__),'executing reader must itself be in the copied root')
    checker=relocation.resolve(args.checker or Path(__file__).with_name('stats_check.py'))
    namespace=load_checker(checker,CHECKER_SHA);original_paths=namespace['paths']
    def paths(a):
        result={key:relocation.resolve(path) for key,path in original_paths(a).items()}
        result.update(public_reader=reader,relocation_manifest=relocation.manifest,original_checker=checker)
        return result
    namespace['paths']=paths
    dispatch(namespace,[str(checker),args.stage,'--repo',str(repo),'--input',str(folder),'--output',str(out)])

if __name__=='__main__':main()

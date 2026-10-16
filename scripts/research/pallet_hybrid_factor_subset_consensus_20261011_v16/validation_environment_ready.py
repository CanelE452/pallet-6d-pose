"""Separate V16 checker run: add only ast to an isolated helper namespace.

Original source, own freeze, STARTED and zero-arithmetic failure are preserved.
The complete checker AST and all scalar statements remain unchanged otherwise.
"""
import ast
import runpy
from pathlib import Path

CODE=Path(__file__).resolve().parent
AUTHORITY=CODE/'validation_authority_ready.py'

def main():
    a=runpy.run_path(str(AUTHORITY));tree,marker=a['source_proof']()
    env=dict(__file__=str(a['ORIGINAL']),__name__='isolated_environment_ready_scalar_kernel')
    exec(compile(tree,str(a['ORIGINAL'])+' [isolated ast environment only]','exec'),env)
    original_paths,original_metadata,original_write=env['input_paths'],env['metadata'],env['write']
    extras=('ready_checker','authority_code','original_validation_protocol','original_validation_started','original_failure')
    def paths(args):
        f=original_paths(args);f.update(ready_checker=Path(__file__),authority_code=AUTHORITY,
            original_validation_protocol=args.input/'VALIDATION_PROTOCOL.json',
            original_validation_started=args.input/'VALIDATION_STARTED.json',
            original_failure=args.input/'FIRST_SCALAR_ENVIRONMENT_FAILURE.json');return f
    def metadata(f,b):
        old=a['failure_proof'](f['original_failure'].parent)
        narrowed={k:v for k,v in b.items() if k not in extras}
        a['require'](narrowed==old['inputs'],'all original frozen checker inputs unchanged')
        return original_metadata({k:v for k,v in f.items() if k not in extras},narrowed)
    def write(path,value):
        value.update(environment_ready=marker,ready_checker=env['binding'](Path(__file__)),authority_code=env['binding'](AUTHORITY))
        original_write(path,value)
    env.update(FILES=a['FILES'],input_paths=paths,metadata=metadata,write=write)
    env['main']()

if __name__=='__main__':main()

"""Explicit output relocation after the original quiet preflight failed.

The original fixed fresh runtime and failed preflight remain byte-identical.
ROOT alone freezes, preflights and measures. Only the output folder changes;
inputs and JSON envelopes additionally disclose this adapter and old failure.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE/'runtime.py'
ORIGINAL_SHA='d840c6963002c1f6e6776efdf34211a5e3b6183faccff8ccf3530db76f0fc31b'

def require(v,s):
    if not v:raise ValueError(s)
def binding(p):
    p=Path(p).resolve();require(p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)),'existing nonsymlink input');d=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):d.update(block)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=d.hexdigest())
def read(p):return json.loads(Path(p).read_bytes())
def canonical(n):return ast.dump(n,include_attributes=False)

def prepare():
    raw=ORIGINAL.read_bytes();require(hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA,'immutable original runtime source')
    original=ast.parse(raw);tree=copy.deepcopy(original);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    sites=[n for n in ast.walk(main) if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div) and isinstance(n.left,ast.Attribute) and isinstance(n.left.value,ast.Name) and n.left.value.id=='a' and n.left.attr=='private' and isinstance(n.right,ast.Constant) and n.right.value=='runtime']
    require(len(sites)==1,'exact one output-folder expression');sites[0].right.value='runtime_ready';revised=copy.deepcopy(tree);sites[0].right.value='runtime'
    require(canonical(tree)==canonical(original),'inverse whole runtime AST equality')
    proto,failed=HERE/'runtime/RUNTIME_PROTOCOL.json',HERE/'runtime/RUNTIME_PREFLIGHT.json';p,first=read(proto),read(failed)
    require(p['schema']=='fresh_whole_U_point_closure_runtime_protocol_v21' and p['configured_calls']==150 and p['method']=='HYBRID_POINT_CONSENSUS_WHOLE_U_CLOSURE' and p['inputs']['code']['sha256']==ORIGINAL_SHA,'original fixed protocol preserved')
    require(first['passed'] is False and first['actual_model_fit_decode_calls']==0 and first['resource']['quiet'] is False and all(first['protocol'][k]==binding(proto)[k] for k in ('bytes','sha256')),'actual initial quiet preflight failed before scientific calls')
    bound=dict(adapter=binding(Path(__file__)),original_runtime=binding(ORIGINAL),original_protocol=binding(proto),first_failed_preflight=binding(failed))
    proof=dict(schema='explicit_whole_U_runtime_output_relocation_v21',bindings=bound,output_folder_AST_changes=1,inverse_whole_original_runtime_AST_equal=True,
        original_output_folder='runtime',new_output_folder='runtime_ready',original_runtime_source_identity_retained=True,
        original_failed_preflight_preserved=True,first_attempt_model_fit_decode_calls=0,scheduler_math_resource_or_threshold_changes=0,
        fresh_attempt_explicitly_authorized=True,automatic_retry=False)
    env=dict(__file__=str(ORIGINAL),__name__='isolated_v21_runtime_output_relocation',__package__=None)
    exec(compile(ast.fix_missing_locations(revised),str(ORIGINAL),'exec'),env)
    old_inputs,old_write=env['inputs'],env['write']
    def inputs(a,V7):
        result=old_inputs(a,V7);require(not any(k.startswith('runtime_output_review:') for k in result),'new explicit binding aliases')
        current={k:binding(v['path']) for k,v in bound.items()};require(current==bound,'original and adapter inputs unchanged')
        result.update({'runtime_output_review:'+k:v for k,v in bound.items()});return result
    def write(path,value):
        if value.get('schema') in ('fresh_whole_U_point_closure_runtime_protocol_v21','fresh_whole_U_point_closure_runtime_receipt_v21'):
            require('runtime_output_review' not in value,'one review marker');value=dict(value,runtime_output_review=proof)
        return old_write(path,value)
    env.update(inputs=inputs,write=write,runtime_output_review=proof)
    return env

def main():prepare()['main']()
if __name__=='__main__':main()

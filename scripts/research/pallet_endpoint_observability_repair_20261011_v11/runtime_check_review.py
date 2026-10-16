"""Own-frozen stdlib review of one first-warmup detector accounting rule.

Preserve the unchanged checker and both failed fresh runtime attempts. Reuse
its clock, statistics, pose and primitive audit unchanged. Exactly one model
expectation and one metadata input-closure predicate are adapted in its AST.
No inference, timing, image, GT, solver, optimizer, Jacobian or SVD calls.
"""
import argparse
import ast
import hashlib
from pathlib import Path

ORIGINAL_SHA='0f024b6332ccb4c10c3369e3f4aee04fdd8b86f898fe805ba62ffd4dcffb5e11'
FAILURE_FILES=('RUNTIME_PROTOCOL.json','RUNTIME_PREFLIGHT.json','RUNTIME_STARTED.json',
    'RUNTIME_RECEIPT.json','RUNTIME_RESOURCES.json','INTERRUPTED_RUNTIME_ROWS.jsonl.gz',
    'INTERRUPTED_RUNTIME_PACKET.json')
FILES=('FORWARD_RUNTIME_CHECK_PROTOCOL.json','FORWARD_RUNTIME_CHECK_STARTED.json','FORWARD_RUNTIME_CHECKS.json')
REVIEW=dict(original_checker_changed=False,original_checker_previously_executed=False,
    numerical_core_or_saved_runtime_rows_changed=False,AST_per_row_model_expectation_changes=1,
    AST_runtime_input_closure_changes=1,first_warmup_detector_forwards=2,
    other_prediction_detector_forwards=1,actual_detector_forwards=151,actual_N3_forwards=150,
    detector_internal_initialization_forwards=1,failed_original_runtime_serialized_rows=0,
    failed_metadata_runtime_serialized_rows=1,clock_statistics_pose_counter_audit_changed=False)


def load():
    original=Path(__file__).with_name('runtime_check.py');raw=original.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ORIGINAL_SHA:raise ValueError('unchanged original runtime checker required')
    tree=ast.parse(raw.decode());model=ast.parse("a.eq(d['model'],dict(detector=1,N3=1),label+'.models')",mode='eval').body
    imports={'argparse','collections','copy','gzip','hashlib','json','math','os','pathlib','traceback'}
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[v.name for v in node.names] if isinstance(node,ast.Import) else [node.module]
            if any(n not in imports for n in names):raise ValueError('stdlib-only original checker required')
    changes=dict(model=0,closure=0)
    for function in tree.body:
        if not isinstance(function,ast.FunctionDef):continue
        for node in ast.walk(function):
            if function.name=='run' and isinstance(node,ast.Call) and ast.dump(node)==ast.dump(model):
                node.args[1]=ast.parse('dict(detector=(2 if i==0 else 1),N3=1)',mode='eval').body;changes['model']+=1
            if function.name=='metadata' and isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='require' and len(node.args)==2 and isinstance(node.args[1],ast.Constant) and node.args[1].value=='unchanged runtime closure or metadata supplement differs':
                expected="set(proto['inputs'])==set(old['inputs'])|{'metadata_review:'+k for k in extra} and all(proto['inputs'][k]==v for k,v in old['inputs'].items()) and all(same(proto['inputs']['metadata_review:'+k],b[v]) for k,v in extra.items())"
                if ast.dump(node.args[0])!=ast.dump(ast.parse(expected,mode='eval').body):raise ValueError('exact original metadata predicate required')
                node.args[0]=ast.parse('forward_closure(proto,old,extra,b,files)',mode='eval').body;changes['closure']+=1
    if changes!=dict(model=1,closure=1):raise ValueError('exactly one model and one closure predicate change required')
    namespace={'__file__':str(original),'__name__':'_own_frozen_forward_runtime_scalar_checker'}
    exec(compile(ast.fix_missing_locations(tree),str(original),'exec'),namespace)
    original_paths,original_metadata,original_write=namespace['paths'],namespace['metadata'],namespace['write']
    def paths(parent,timing):
        result=original_paths(parent,timing);result.update(review_checker=Path(__file__).resolve(),
            forward_code=original.with_name('runtime_forward_review.py'),forward_contract=parent/'RUNTIME_FORWARD_REVIEW_CONTRACT_KO.md')
        result.update({'second:'+n:parent/'runtime_review'/n for n in FAILURE_FILES});return result
    def forward_closure(proto,old,extra,b,files):
        require,read,same=namespace['require'],namespace['read'],namespace['same']
        metadata_inputs={'metadata_review:'+k:b[v] for k,v in extra.items()}
        forward=dict(review_code='forward_code',review_contract='forward_contract')
        forward.update({'preserved_second:'+n:'second:'+n for n in FAILURE_FILES})
        second,failed,pre,journal,resources=[read(files['second:'+n]) for n in ('RUNTIME_PROTOCOL.json','RUNTIME_RECEIPT.json','RUNTIME_PREFLIGHT.json','RUNTIME_STARTED.json','RUNTIME_RESOURCES.json')]
        def joined(actual,base,added,label):
            require(set(actual)==set(base)|set(added) and all(actual[k]==v for k,v in base.items()),label+' unchanged original closure and exact added keys')
            require(all(same(actual[k],v) and actual[k]['path']==v['path'] and actual[k].get('origin')=='public_repository' for k,v in added.items()),label+' public input bytes and paths')
        joined(second['inputs'],old['inputs'],metadata_inputs,'second metadata review')
        joined(proto['inputs'],second['inputs'],{'forward_review:'+k:b[v] for k,v in forward.items()},'forward review')
        require(same(failed['protocol'],b['second:RUNTIME_PROTOCOL.json']),'preserved second frozen protocol differs')
        require(failed['complete'] is False and failed['status']=='FAILED_RUNTIME' and failed['serialized_rows']==1 and failed['summaries'] is None and failed['statistics_official'] is False and failed['execution']==dict(image_decode_calls=26,image_decode_returns=26,model_initializations_started=1,model_initializations_complete=1,calls_started=1) and failed['error']['message']=='SAME_OBSERVATION_GUARD: actual fresh detector/N3 forward count differs','specific first internal warmup guard failure required')
        require(failed['model_forwards_after_loading']==dict(detector=0,N3=0) and failed['model_forwards']==dict(detector=2,N3=1,detector_internal_initialization_calls=1) and failed['actual_base_counts']['detector_calls']==1,'preserved second actual forward witness differs')
        require(not failed['cleanup_errors'] and all(failed['cleanup'][k] for k in ('instruments_restored','opencv_threads_restored','head_hooks_removed','pipeline_closed','environment_restored')),'preserved second cleanup required')
        require(pre['passed'] is True and same(pre['protocol'],b['second:RUNTIME_PROTOCOL.json']) and journal['status']=='FAILED_RUNTIME' and journal['execution']==failed['execution'] and same(journal['receipt'],b['second:RUNTIME_RECEIPT.json']),'second preflight/journal differs')
        require(same(failed['resources'],b['second:RUNTIME_RESOURCES.json']) and same(failed['raw_rows'],b['second:INTERRUPTED_RUNTIME_ROWS.jsonl.gz']) and same(resources['protocol'],b['second:RUNTIME_PROTOCOL.json']) and resources['snapshots']==failed['resource_snapshots'],'preserved second prefix/resources differ')
        return True
    def metadata(files,b):
        proto,receipt,resources=original_metadata(files,b)
        namespace['require'](receipt['model_forwards_after_loading']==dict(detector=0,N3=0) and receipt['model_forwards']==dict(detector=151,N3=150,detector_internal_initialization_calls=1),'complete actual 151/150/one initialization forward required')
        return proto,receipt,resources
    def write(path,value):
        if path.name in FILES:value=dict(value,review=REVIEW)
        original_write(path,value)
    namespace.update(paths=paths,metadata=metadata,forward_closure=forward_closure,write=write,FILES=FILES)
    return namespace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    root=Path(__file__).resolve().parents[3]/'_docs/experiments/pallet_endpoint_observability_repair_20261011_v11'
    p.add_argument('--input',type=Path,default=root);p.add_argument('--runtime-input',type=Path,default=root/'runtime_forward_review');p.add_argument('--output',type=Path,default=root)
    args=p.parse_args();namespace=load();namespace['freeze' if args.stage=='freeze' else 'run'](args)


if __name__=='__main__':main()

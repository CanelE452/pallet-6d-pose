"""Location-only public reader for saved runtime rows with failed cleanup.

No producer JSON, receipt status, clock/predicate formula or failure rule is
changed. ROOT alone runs a fresh reader freeze/run. Row success still leaves
overall passed=False and unofficial timing summaries in the resulting checks.
"""
import ast
import copy
import hashlib
from pathlib import Path

SOURCE_READER = Path(__file__).resolve().with_name('public_runtime_reader.py')
READER_SHA = '4d6eebacfd7a1232678f83270ce36640f5417e3918fc72fa982fcbf4f6c52124'
READY_SHA = '5472bec7b8b7b38133019796955133f3511cce8bfb8f39e976cb68b978f4173c'
ORIGINAL_CHECKER_SHA = '729c3797b04e27d696fc5d343e1a7ea1908f0d08962215dc8e83e9baed800842'


def require(value, message):
    if not value: raise ValueError(message)


def canonical(node):
    return ast.dump(node, include_attributes=False)


def binding(path):
    raw = Path(path).read_bytes()
    return dict(path=str(Path(path).resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def install_location_writer(namespace,proof):
    previous=namespace['write']
    def write(path,value):
        return previous(path,dict(value,public_runtime_location_review=copy.deepcopy(proof)))
    namespace['write']=write


def prepare():
    require(not any(p.is_symlink() for p in (SOURCE_READER,*SOURCE_READER.parents)), 'explicit nonsymlink original reader')
    bound=binding(SOURCE_READER); require(bound['sha256']==READER_SHA, 'unchanged original public reader SHA')
    original=ast.parse(SOURCE_READER.read_bytes()); tree=copy.deepcopy(original);changes=[]
    def matching_span(root,nodes):
        result=[];expected=[canonical(n) for n in nodes]
        for owner in ast.walk(root):
            for _,items in ast.iter_fields(owner):
                if isinstance(items,list):
                    for i in range(len(items)-len(nodes)+1):
                        span=items[i:i+len(nodes)]
                        if all(isinstance(n,ast.AST) for n in span) and [canonical(n) for n in span]==expected:
                            result.append((items,i))
        require(len(result)==1,'one exact location-dispatch source span');return result[0]
    def change(before,after,label):
        old,new=ast.parse(before).body,ast.parse(after).body
        items,index=matching_span(tree,old);items[index:index+len(old)]=copy.deepcopy(new)
        changes.append(dict(label=label,old=old,new=new))

    change("checker=relocation.resolve(args.checker or Path(__file__).with_name('runtime_check.py'))",
           "checker=relocation.resolve(args.checker or Path(__file__).with_name('runtime_check_ready.py'))",
           'explicit saved-row checker default only')
    change("namespace=helper.load_checker(checker,CHECKER_SHA)","""
original_checker=relocation.resolve(checker.with_name('runtime_check.py'))
source_reader=relocation.resolve(SOURCE_READER)
helper.require(helper.binding(original_checker)['sha256']==ORIGINAL_CHECKER_SHA,'exact adjacent original mathematical checker')
helper.require(source_reader==helper.safe_path(SOURCE_READER),'executing original reader source must be copied')
loaded=helper.load_checker(checker,CHECKER_SHA)
namespace=loaded['prepare']()
install_location_writer(namespace,PUBLIC_READER_PROOF)
""",'load whole saved-row adapter then its explicit original-checker namespace')
    change("""
path_globals=dict(namespace)
path_globals['declared_path']=declared_path
relocated_paths=FunctionType(original_paths.__code__,path_globals,original_paths.__name__,original_paths.__defaults__,original_paths.__closure__)
helper.require(relocated_paths.__code__ is original_paths.__code__,'whole original paths code must remain unchanged')
""","namespace['declared_path']=declared_path",
        'route declarations in compiled namespace used by the ready paths closure')
    change("result={key:relocation.resolve(path) for key,path in relocated_paths(enumeration).items()}",
           "result={key:relocation.resolve(path) for key,path in original_paths(enumeration).items()}",
           'resolve every exact ready path result without changing its function globals')
    change("result.update(public_reader=reader,public_reader_resolver=resolver,relocation_manifest=relocation.manifest,original_checker=checker)",
           "result.update(public_reader=reader,public_reader_resolver=resolver,relocation_manifest=relocation.manifest,rows_checker=checker,original_mathematical_checker=original_checker,original_public_reader=source_reader)",
           'bind reader resolver ready original mathematical checker and original reader')
    undo=copy.deepcopy(tree)
    for edit in reversed(changes):
        items,index=matching_span(undo,edit['new']);items[index:index+len(edit['new'])]=copy.deepcopy(edit['old'])
    require(canonical(undo)==canonical(original),'whole original public-reader AST recovered')
    proof=dict(schema='saved_runtime_rows_public_location_dispatch_proof_v19',
        original_reader=bound,reader=binding(__file__),ready_checker_sha256=READY_SHA,
        original_mathematical_checker_sha256=ORIGINAL_CHECKER_SHA,
        inverse_whole_original_reader_AST_equal=True,AST_source_span_changes=len(changes),
        changes=[c['label'] for c in changes],producer_JSON_paths_or_bytes_rewritten=False,
        mathematical_or_metadata_failure_rules_changed=False,
        only_checker_selection_and_declared_location_dispatch_changed=True,
        compiled_namespace_declared_path_replaces_closed_original_paths_lookup=True,
        non_AST_dispatch_overrides=['CHECKER_SHA: exact ready source','write: location witness only','module docstring: rows-only scope'],
        overall_runtime_cleanup_FAIL_remains_FAIL=True,reader_source_is_not_actual_portable_PASS=True)
    namespace=dict(__name__='isolated_public_runtime_saved_rows_reader',__file__=str(Path(__file__).resolve()),
        SOURCE_READER=SOURCE_READER,ORIGINAL_CHECKER_SHA=ORIGINAL_CHECKER_SHA,
        install_location_writer=install_location_writer,PUBLIC_READER_PROOF=proof)
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE_READER)+':explicit ready location dispatch','exec'),namespace)
    namespace['CHECKER_SHA']=READY_SHA;namespace['__doc__']=__doc__
    return namespace,proof


if __name__=='__main__':prepare()[0]['main']()

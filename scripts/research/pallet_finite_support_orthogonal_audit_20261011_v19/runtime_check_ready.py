"""Audit saved 150 runtime rows while preserving the failed cleanup contract.

ROOT alone freezes/runs this stdlib source. No measurement is repeated. The
original receipt, journal, interrupted gzip and checker remain unchanged.
CLI zero means the saved-row audit passed; overall runtime passed stays False.
"""
import ast
import copy
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / 'runtime_check.py'
ORIGINAL_SHA = '729c3797b04e27d696fc5d343e1a7ea1908f0d08962215dc8e83e9baed800842'
PROTOCOL_SCHEMA = 'independent_observed_frame_runtime_rows_protocol_v19'
CHECKS_SCHEMA = 'independent_observed_frame_runtime_rows_checks_v19'
REVIEW = dict(
    schema='saved_runtime_rows_with_preserved_cleanup_failure_review_v19',
    original_runtime_complete=False, original_runtime_statistics_official=False,
    original_runtime_preserved=True, original_runtime_checker_modified=False,
    original_runtime_cleanup_passed=False, automatic_remeasurement=False,
    final_input_bytes_proof_executed_in_original_cleanup=False,
    final_input_bytes_independently_checked_by_this_rows_audit=True,
    current_process_environment_restoration_certified=False,
    source_namespace_restoration_in_original_process_certified=False,
    summaries_are_posthoc_saved_clocks_not_official_runtime_statistics=True,
    CLI_exit_zero_means_saved_rows_only_not_original_runtime_contract_PASS=True,
    new_measurement_model_RGB_GT_fit_projection_J_SVD_calls=0,
)


def require(value, message):
    if not value: raise ValueError(message)


def canonical(node):
    return ast.dump(node, include_attributes=False)


def binding(path):
    raw = Path(path).read_bytes()
    return dict(path=str(Path(path).resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def prepare():
    bound = binding(ORIGINAL)
    require(bound['sha256'] == ORIGINAL_SHA, 'immutable original runtime checker changed')
    original = ast.parse(ORIGINAL.read_bytes()); tree = copy.deepcopy(original); changes = []
    functions = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    def one(function, predicate, label):
        matches = [n for n in ast.walk(functions[function]) if predicate(n)]
        require(len(matches)==1, 'unique source site '+label); return matches[0]
    def replacement(old, new, label):
        old_dump, new_dump = canonical(old), canonical(new)
        require(old_dump != new_dump, 'nonempty review change '+label)
        owners=[name for name,function in functions.items() if any(node is old for node in ast.walk(function))]
        require(len(owners)==1,'one source function '+label)
        class Replace(ast.NodeTransformer):
            count = 0
            def generic_visit(self, node):
                if node is old:
                    self.count += 1; return copy.deepcopy(new)
                return super().generic_visit(node)
        visitor = Replace(); visitor.visit(tree); require(visitor.count==1, 'one replacement '+label)
        changes.append(dict(label=label,function=owners[0],old=copy.deepcopy(old),new=copy.deepcopy(new)))

    old = one('paths', lambda n:isinstance(n,ast.Constant) and n.value=='RUNTIME_ROWS.jsonl.gz', 'interrupted rows filename')
    replacement(old, ast.Constant(value='INTERRUPTED_RUNTIME_ROWS.jsonl.gz'), 'interrupted rows filename only')
    old = one('metadata', lambda n:isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require'
              and len(n.args)==2 and isinstance(n.args[1],ast.Constant) and n.args[1].value=='actual full150 runtime required', 'historical receipt gate')
    new = ast.parse("""require(
        r['schema']=='fresh_observed_frame_primary_runtime_receipt_v19'
        and r['complete'] is False and r['status']=='FAILED_RUNTIME_CLEANUP'
        and r['method']==METHOD and r['configured_calls']==150
        and r['statistics_official'] is False and r['summaries'] is None
        and r['serialized_rows']==150 and r.get('error') is None
        and r['execution']==dict(image_decode_calls=26,model_initializations=1,calls_started=150,calls_complete=150)
        and r['cleanup']==dict(instruments_entered=True,instrument_restoration_errors=[],
            instruments_restored=True,opencv_threads_restored=True,head_hooks_removed=True,
            pipeline_closed=True,environment_restored=False)
        and r['cleanup_errors']==[dict(phase='restoration',type='RuntimeError',message='V19_RUNTIME: environment restoration')]
        and r['protected_before']==r['protected_after']
        and r['all_recorded_resource_snapshots_quiet'] is True,
        'exact complete saved150 rows and sole preserved cleanup environment failure')""").body[0].value
    replacement(old,new,'strict historical failed-cleanup receipt gate')
    old = one('metadata', lambda n:isinstance(n,ast.Compare) and isinstance(n.left,ast.Subscript)
              and canonical(n.left)==canonical(ast.parse("journal['status']",mode='eval').body), 'failed journal status')
    new = ast.parse("journal['status']=='FAILED_RUNTIME_CLEANUP'",mode='eval').body
    replacement(old,new,'preserved failed journal status')
    old = one('run', lambda n:isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='k'
              and isinstance(n.iter,ast.Name) and n.iter.id=='TIMES' and len(n.body)==1
              and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Call)
              and isinstance(n.body[0].value.func,ast.Attribute) and n.body[0].value.func.attr=='near', 'original saved summary comparison')
    new = ast.parse("""for k in TIMES:
        summaries[k]=env['describe'](timings[k])
        a.eq(set(summaries[k]),{'n','mean','sample_variance','sample_std','median','P90','maximum','unit','ddof'},'posthoc.summary_fields.'+k)
        a.eq(summaries[k]['n'],130,'posthoc.measured130.'+k)
        a.eq((summaries[k]['unit'],summaries[k]['ddof']),('ms',1),'posthoc.units_ddof.'+k)
        require(all(type(summaries[k][field]) in (int,float) and math.isfinite(summaries[k][field])
            for field in ('n','mean','sample_variance','sample_std','median','P90','maximum')),'nonfinite posthoc saved-clock summary')""").body[0]
    replacement(old,new,'same original describe over saved clocks, no nonexistent official summary comparison')
    old = one('run', lambda n:isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='eq'
              and len(n.args)==3 and canonical(n.args[0])==canonical(ast.parse("r['cleanup'][k]",mode='eval').body), 'recorded cleanup accounting')
    new = copy.deepcopy(old); new.args[1]=ast.parse("False if k=='environment_restored' else True",mode='eval').body
    replacement(old,new,'recount recorded environment failure without classifying it PASS')
    old = one('run', lambda n:isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict'
              and any(k.arg=='schema' and isinstance(k.value,ast.Constant) and k.value.value=='independent_observed_frame_runtime_checks_v19' for k in n.keywords), 'rows result envelope')
    new = copy.deepcopy(old)
    for keyword in new.keywords:
        if keyword.arg=='schema': keyword.value=ast.Constant(value=CHECKS_SCHEMA)
        if keyword.arg=='passed': keyword.value=ast.Constant(value=False)
    additions = ast.parse("dict(scientific_rows_passed=passed,operational_harness_cleanup_passed=False,expected_historical_cleanup_failure_count=1,overall_contract_failure_count=a.failure_count+1,summaries=summaries,summaries_unofficial_due_to_historical_cleanup_failure=True,originalruntime_statistics_official=False,runtime_cleanup_review=READY_REVIEW,checker_source_review=READY_PROOF)",mode='eval').body.keywords
    new.keywords.extend(additions); replacement(old,new,'explicit rows PASS versus overall cleanup FAIL output')
    old = one('run', lambda n:isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='print'
              and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='OBSERVED_FRAME_RUNTIME_CHECK', 'CLI status caption')
    new = ast.parse("print('OBSERVED_FRAME_RUNTIME_ROWS_CHECK','ROWS_PASS_CLEANUP_FAIL' if passed else 'ROWS_FAIL_CLEANUP_FAIL',a.checks,flush=True)").body[0].value
    replacement(old,new,'explicit CLI rows-only audit status')
    old = one('main', lambda n:isinstance(n,ast.Constant) and n.value=='runtime_checks', 'separate output directory')
    replacement(old,ast.Constant(value='runtime_checks_ready'),'separate output directory')
    for label in ('freeze','run'):
        matches = [n for n in ast.walk(functions[label]) if isinstance(n,ast.Constant) and n.value=='independent_observed_frame_runtime_protocol_v19']
        require(len(matches)==1,'one protocol schema '+label)
        replacement(matches[0],ast.Constant(value=PROTOCOL_SCHEMA),'rows protocol schema '+label)

    # Extra local state has no numerical formula or producer receipt mutation.
    run = functions['run']; insertion = ast.parse('summaries={}').body[0]; run.body.insert(0,insertion)
    undo = copy.deepcopy(tree)
    undo_run = next(n for n in undo.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    require(canonical(undo_run.body[0])==canonical(insertion),'summary state one insertion');undo_run.body.pop(0)
    for change in reversed(changes):
        target = canonical(change['new'])
        scope=next(n for n in undo.body if isinstance(n,ast.FunctionDef) and n.name==change['function'])
        matches = [n for n in ast.walk(scope) if canonical(n)==target]
        require(len(matches)==1,'unique inverse '+change['label'])
        old, new = matches[0], copy.deepcopy(change['old'])
        class Undo(ast.NodeTransformer):
            def generic_visit(self,node):
                return new if node is old else super().generic_visit(node)
        Undo().visit(undo)
    require(canonical(undo)==canonical(original),'full original checker AST recovered')
    proof=dict(schema='runtime_saved_rows_cleanup_scope_AST_review_v19',original_checker=bound,
        ready_checker=binding(__file__),inverse_whole_AST_equal=True,
        AST_replacement_count=len(changes),AST_local_state_insertions=1,
        changes=[c['label'] for c in changes],clock_describe_Audit_candidate_counter_parity_formulas_changed=False,
        non_AST_dispatch_overrides=['paths: bind original mathematical checker','write: explicit review markers','LIMITS: metadata scope only'],
        original_runtime_or_checker_or_receipt_or_rows_changed=False,new_measurements_or_scientific_fits=0)
    namespace=dict(__name__='isolated_saved_runtime_rows_review_v19',__file__=str(Path(__file__).resolve()),READY_REVIEW=copy.deepcopy(REVIEW),READY_PROOF=proof)
    exec(compile(ast.fix_missing_locations(tree),str(ORIGINAL)+':explicit saved-row cleanup-scope review','exec'),namespace)
    old_paths,old_write=namespace['paths'],namespace['write']
    def paths(args):
        result=old_paths(args);result['original_mathematical_checker']=ORIGINAL;return result
    def write(path,value):
        value=dict(value,runtime_cleanup_review=copy.deepcopy(REVIEW),checker_source_review=copy.deepcopy(proof))
        return old_write(path,value)
    namespace.update(paths=paths,write=write)
    namespace['LIMITS']=dict(namespace['LIMITS'],historical_runtime_operational_cleanup_passed=False,
        summaries_unofficial_due_to_historical_cleanup_failure=True,current_environment_or_original_namespace_cleanup_certified=False)
    return namespace


if __name__=='__main__':prepare()['main']()

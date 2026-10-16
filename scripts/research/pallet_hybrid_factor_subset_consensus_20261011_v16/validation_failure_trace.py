"""Own-frozen complete log of the already failed V16 scalar validation.

Retain all 564 failures instead of the original Audit's first 100. Every
scalar predicate, tolerance and candidate body remains unchanged. A complete
matching trace is separate from a geometry PASS; the original geometry FAIL
is preserved and this command retains its nonzero scalar-validation exit.
"""
import ast
import copy
import hashlib
from pathlib import Path
import runpy

CODE=Path(__file__).resolve().parent
AUTHORITY=CODE/'validation_authority_ready.py'
READY=CODE/'validation_environment_ready.py'
READY_SHA='5320cb208a561f996150f2eaee709ce69f0e042efd77c25c10ab13118d31eef5'
AUDIT_SOURCE=CODE.parent/'pallet_sparse_local_line_20261010_v9/validation_checks.py'
AUDIT_SHA='4319477832a599c70c015fdb2a8538a51cb0f19077591e352c75e15d7860ec1e'
FILES=('TRACE_VALIDATION_PROTOCOL.json','TRACE_VALIDATION_STARTED.json','TRACE_VALIDATION_CHECKS.json')

def logging_proof(a):
    a['require'](a['binding'](READY)['sha256']==READY_SHA and a['binding'](AUDIT_SOURCE)['sha256']==AUDIT_SHA,'immutable READY and scalar Audit source')
    tree=ast.parse(AUDIT_SOURCE.read_text());original=ast.dump(tree,include_attributes=False)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Audit')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='ok')
    target=ast.parse('len(self.failures)<100',mode='eval').body
    sites=[n for n in ast.walk(fn) if isinstance(n,ast.If) and ast.dump(n.test,include_attributes=False)==ast.dump(target,include_attributes=False)]
    a['require'](len(sites)==1,'exact original capped failure-log predicate')
    site=sites[0];site.test=ast.Constant(True);derived=ast.dump(tree,include_attributes=False);patched=copy.deepcopy(cls)
    site.test=target;a['require'](ast.dump(tree,include_attributes=False)==original,'whole scalar source canonical restoration')
    proof=dict(logging_guard_AST_changes=1,old_guard='len(self.failures)<100',new_guard='True',
        scalar_formula_threshold_or_acceptance_AST_changes=0,whole_original_scalar_source_restore_proven=True,
        original_Audit_source_sha256=AUDIT_SHA,original_full_AST_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_logging_full_AST_sha256=hashlib.sha256(derived.encode()).hexdigest(),
        original_READY_source_sha256=READY_SHA,original_failure_preserved=True,
        geometry_solver_or_operating_rows_modified=False,trace_is_not_geometry_PASS=True)
    return ast.fix_missing_locations(ast.Module(body=[patched],type_ignores=[])),proof

def main():
    a=runpy.run_path(str(AUTHORITY));tree,environment_marker=a['source_proof']();audit_tree,trace_marker=logging_proof(a)
    env=dict(__file__=str(a['ORIGINAL']),__name__='isolated_complete_failure_trace')
    exec(compile(tree,str(a['ORIGINAL'])+' [same READY environment]','exec'),env)
    original_paths,original_metadata,original_write,original_helpers=env['input_paths'],env['metadata'],env['write'],env['helpers']
    ready_extras=('ready_checker','authority_code','original_validation_protocol','original_validation_started','original_failure')
    trace_extras=('trace_checker','completed_failed_ready_protocol','completed_failed_ready_started','completed_failed_ready_checks','ready_environment_source')
    def paths(args):
        f=original_paths(args);f.update(ready_checker=READY,authority_code=AUTHORITY,
            original_validation_protocol=args.input/'VALIDATION_PROTOCOL.json',original_validation_started=args.input/'VALIDATION_STARTED.json',
            original_failure=args.input/'FIRST_SCALAR_ENVIRONMENT_FAILURE.json',trace_checker=Path(__file__),ready_environment_source=READY,
            completed_failed_ready_protocol=args.input/'VALIDATION_READY_PROTOCOL.json',completed_failed_ready_started=args.input/'VALIDATION_READY_STARTED.json',completed_failed_ready_checks=args.input/'VALIDATION_READY_CHECKS.json');return f
    def metadata(f,b):
        old=a['failure_proof'](f['original_failure'].parent)
        ready,started,failed=[a['read'](f[k]) for k in ('completed_failed_ready_protocol','completed_failed_ready_started','completed_failed_ready_checks')]
        a['require'](failed['complete'] is True and failed['passed'] is False and failed['checks']==198134007 and failed['failure_count']==564 and len(failed['failures'])==100 and failed['exception'] is None and failed['new_production_calls']==0,'preserved full-run scalar failure, not a geometry PASS')
        narrowed={k:v for k,v in b.items() if k not in trace_extras}
        a['require'](ready['inputs']==started['inputs']==failed['inputs']==narrowed and ready['environment_ready']==started['environment_ready']==failed['environment_ready']==environment_marker and a['same'](failed['own_protocol'],b['completed_failed_ready_protocol']) and a['same'](started['own_protocol'],b['completed_failed_ready_protocol']),'exact completed failed READY closure')
        a['require'](all(a['same'](v['ready_checker'],b['ready_checker']) and a['same'](v['authority_code'],b['authority_code']) for v in (ready,started,failed)),'original READY current source identity')
        original_inputs={k:v for k,v in narrowed.items() if k not in ready_extras}
        a['require'](original_inputs==old['inputs'],'all original frozen scalar inputs unchanged')
        return original_metadata({k:v for k,v in f.items() if k not in (*ready_extras,*trace_extras)},original_inputs)
    def helpers(info):
        h,n,policy=original_helpers(info);isolated=dict(h)
        exec(compile(audit_tree,str(AUDIT_SOURCE)+' [uncapped logging only]','exec'),isolated)
        h['Audit']=isolated['Audit'];return h,n,policy
    def write(path,value):
        value.update(environment_ready=environment_marker,failure_trace=trace_marker,ready_checker=env['binding'](READY),authority_code=env['binding'](AUTHORITY),trace_checker=env['binding'](Path(__file__)))
        if path.name==FILES[2]:
            old=a['read'](env['public_trace_failed_ready_path']);failures=value['failures']
            comparisons=dict(scalar_complete_unchanged=value['complete']==old['complete'],scalar_failed_unchanged=value['passed']==old['passed']==False,
                checks_unchanged=value['checks']==old['checks']==198134007,failure_count_unchanged=value['failure_count']==old['failure_count']==564,
                first100_failures_unchanged=failures[:100]==old['failures'],all_failures_retained=len(failures)==value['failure_count'],
                exact_max_scalar_difference_unchanged=value['max_scalar_absolute_difference']==old['max_scalar_absolute_difference'],
                no_scalar_exception=value['exception'] is None,counts_unchanged=value['counts']==old['counts'],native_counts_unchanged=value['native_counts']==old['native_counts'],cusp_events_unchanged=value['cusp_reconciliations']==old['cusp_reconciliations'])
            value.update(schema='hybrid_factor_subset_scalar_failure_trace_checks_v16',trace_complete=value['complete'],trace_passed=all(comparisons.values()),trace_checks=comparisons,scalar_validation_passed=value['passed'],all_failed_predicates_retained=comparisons['all_failures_retained'])
        original_write(path,value)
    original_input_paths=paths
    def paths_with_failure(args):
        result=original_input_paths(args);env['public_trace_failed_ready_path']=result['completed_failed_ready_checks'];return result
    env.update(FILES=FILES,input_paths=paths_with_failure,metadata=metadata,write=write,helpers=helpers)
    env['main']()

if __name__=='__main__':main()

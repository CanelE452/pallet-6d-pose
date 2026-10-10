"""Preserve the failed inspector and separately check its one optional field.

The original stdlib checker AST is used with exactly one change: an unavailable
result may omit best_candidate, so that one assertion uses dict.get. Original
checker __file__, method source, input rows and first failure remain immutable.
Own freeze precedes the new full scalar pass; no production solver is executed.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

CODE=Path(__file__).resolve().parent
ORIGINAL=CODE/'validation_checks.py'
ORIGINAL_SHA='2f1318af78ddaf2be533e01587618dc74591d2c241acc5185c7aa803e64c9220'
REPO=CODE.parents[2]
DOC=REPO/'_docs/experiments/pallet_endpoint_observability_repair_20261011_v11'
NAMES=('REVIEW_VALIDATION_PROTOCOL.json','REVIEW_VALIDATION_STARTED.json','REVIEW_VALIDATION_CHECKS.json')
REVIEW=dict(original_checker_changed=False,numerical_core_or_saved_rows_changed=False,
    AST_optional_best_candidate_access_changes=1)


def require(value,message):
    if not value:raise ValueError(message)


def extracted():
    source=ORIGINAL.read_bytes();require(hashlib.sha256(source).hexdigest()==ORIGINAL_SHA,'immutable original checker required')
    module=ast.parse(source);original_ast=ast.dump(module,include_attributes=False)
    packet=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='packet')
    target=ast.parse("audit.eq(s['best_candidate'],None,label+'.no_viable_selected_candidate')",mode='eval').body
    expected=ast.dump(target,include_attributes=False);changed=0
    class OneOptionalAccess(ast.NodeTransformer):
        def visit_Call(self,node):
            nonlocal changed
            if ast.dump(node,include_attributes=False)==expected:
                changed+=1;node.args[0]=ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id='s',ctx=ast.Load()),
                    attr='get',ctx=ast.Load()),args=[ast.Constant(value='best_candidate')],keywords=[]),node.args[0])
                return node
            return self.generic_visit(node)
    OneOptionalAccess().visit(packet);require(changed==1,'exact unique optional absent-best assertion required')
    ast.fix_missing_locations(module)
    imports={n.module for n in module.body if isinstance(n,ast.ImportFrom)}|{v.name for n in module.body if isinstance(n,ast.Import) for v in n.names}
    require(imports=={'__future__','argparse','ast','collections','copy','gzip','hashlib','json','math','os','pathlib','struct','time','traceback'},'stdlib-only original imports differ')
    witness=dict(original_code_sha256=ORIGINAL_SHA,original_module_AST_sha256=hashlib.sha256(original_ast.encode()).hexdigest(),
        reviewed_module_AST_sha256=hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest(),
        function='packet',original_expression="s['best_candidate']",reviewed_expression="s.get('best_candidate')",
        assertion_label='no_viable_selected_candidate',AST_optional_best_candidate_access_changes=changed)
    return module,witness


def namespace():
    module,witness=extracted()
    env=dict(__name__='preserved_original_scalar_review',__file__=str(ORIGINAL))
    exec(compile(module,str(ORIGINAL)+':one_optional_access_review','exec'),env)
    original_paths,original_write=env['input_paths'],env['write'];env['OUTPUTS']=NAMES
    def paths(args):
        result=original_paths(args)
        result.update(review_checker=Path(__file__).resolve(),original_failure=args.input/'VALIDATION_CHECKS.json',
            original_validation_protocol=args.input/'VALIDATION_PROTOCOL.json')
        return result
    def write(path,value):
        value=dict(value,review=REVIEW,AST_review=witness,review_checker=env['binding'](Path(__file__).resolve()))
        original_write(path,value)
    env.update(input_paths=paths,write=write)
    return env,witness


def history(env,args,witness):
    paths=env['input_paths'](args);failed=env['read'](paths['original_failure']);first=env['read'](paths['original_validation_protocol'])
    require(failed['schema']=='supplemental_endpoint_observability_scalar_checks_v11' and failed['passed'] is False and
        failed['complete'] is False and failed['checks']==95 and failed['failure_count']==0 and
        failed['exception']['type']=='KeyError' and failed['exception']['message']=="'best_candidate'",
        'preserved specific original inspector failure required')
    require(failed['inputs']==first['inputs'] and env['same'](failed['own_protocol'],env['binding'](paths['original_validation_protocol'])) and
        env['same'](failed['checker'],env['binding'](ORIGINAL)),'original failed attempt byte authority differs')
    current={k:env['binding'](p) for k,p in paths.items()}
    require(set(current)==set(first['inputs'])|{'review_checker','original_failure','original_validation_protocol'} and
        all(current[k]==v for k,v in first['inputs'].items()),'original complete checker inputs changed')
    own=args.output/NAMES[0]
    if args.stage=='run':
        record=env['read'](own)
        require(record['review']==REVIEW and record['AST_review']==witness and
            record['review_checker']==current['review_checker'],'own optional-field review freeze differs')
    return failed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    for name,default in (('input',DOC),('output',DOC),('protocol',DOC/'PROTOCOL.json'),
        ('parent',REPO/'_docs/experiments/pallet_three_head_observation_20261010_v7'),
        ('audit',REPO/'_docs/experiments/pallet_unused_line_endpoint_audit_20261011_v10'),
        ('cohort',REPO/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/COHORT.json')):
        p.add_argument('--'+name,type=Path,default=default)
    args=p.parse_args();env,witness=namespace();history(env,args,witness)
    env[args.stage](args)


if __name__=='__main__':main()

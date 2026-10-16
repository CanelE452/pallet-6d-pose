"""Reporter authority adapter after the saved UNKNOWN-label audit failed.

ROOT alone runs. The immutable reporter's tables, values and science are
unchanged. Only the memory-source join and routing output default change;
the new label proof and both failed diagnostic attempts are separately bound.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

ORIGINAL = Path(__file__).resolve().with_name('report_results.py')
ORIGINAL_SHA = 'e4f0ad376116ed5a90d008ff993355ef2a13cbb439edabf4662071901bab3940'
FAILED_PREFREEZE_SOURCE = Path(__file__).resolve().with_name('report_results_prefreeze_metrics_alias.py')
FAILED_PREFREEZE_SHA = 'b04c11fc539a6dc4dab61bd7b14383b2d6832d3bbe1297f079bd37e73a54c811'

def require(value, message):
    if not value: raise ValueError(message)

def binding(path):
    path=Path(path).absolute();require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)), 'required nonsymlink source/evidence')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())

def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def read(path):return json.loads(Path(path).read_bytes())
def canonical(node):return ast.dump(node,include_attributes=False)

def prepare():
    source=binding(ORIGINAL);require(source['sha256']==ORIGINAL_SHA,'immutable completed reporter source')
    original=ast.parse(ORIGINAL.read_bytes());tree=copy.deepcopy(original);changes=[]
    replacements=(("same(memory['ready_source'],b['routing_code'])", "same(memory['ready_source'],b['routing_memory_adapter'])"),
        ("ROUTING/'audit_ready'", "ROUTING/'audit_labels_ready'"),
        ("s['inputs']['METRICS']", "s['inputs']['METRICS.json']"))
    for before,after in replacements:
        old,new=[ast.parse(value,mode='eval').body for value in (before,after)]
        found=[(parent,field,index) for parent in ast.walk(tree) for field,value in ast.iter_fields(parent)
            for index,item in (enumerate(value) if isinstance(value,list) else ((None,value),))
            if isinstance(item,ast.AST) and canonical(item)==canonical(old)]
        require(len(found)==1,'one exact reporter metadata expression')
        parent,field,index=found[0]
        if index is None:setattr(parent,field,copy.deepcopy(new))
        else:getattr(parent,field)[index]=copy.deepcopy(new)
        changes.append((old,new))
    undo=copy.deepcopy(tree)
    for old,new in reversed(changes):
        found=[(parent,field,index) for parent in ast.walk(undo) for field,value in ast.iter_fields(parent)
            for index,item in (enumerate(value) if isinstance(value,list) else ((None,value),))
            if isinstance(item,ast.AST) and canonical(item)==canonical(new)]
        require(len(found)==1,'one inverse reporter metadata expression');parent,field,index=found[0]
        if index is None:setattr(parent,field,copy.deepcopy(old))
        else:getattr(parent,field)[index]=copy.deepcopy(old)
    require(canonical(undo)==canonical(original),'inverse whole reporter AST equality')
    failed_source=binding(FAILED_PREFREEZE_SOURCE);require(failed_source['sha256']==FAILED_PREFREEZE_SHA,'preserved first metadata prefreeze source')
    proof=dict(schema='saved_whole_U_report_routing_label_authority_review_v21', original_reporter=source,
        reporter_adapter=binding(__file__),exact_metadata_AST_expression_changes=3,inverse_whole_original_reporter_AST_equal=True,
        first_failed_prefreeze_source=failed_source,statistics_METRICS_json_alias_matches_immutable_producer=True,
        tables_values_metric_or_moment_or_CI_formulas_changed=False,original_reporter_source_modified=False,
        memory_proof_source_and_current_label_producer_distinguished=True,original_failed_diagnostic_attempts_preserved=True,
        large_gzip_manifest_family_from_explicit_declared_root=True,
        Markdown_large_gzip_target_metadata_changed=True,numeric_table_values_and_statistical_formulas_unchanged=True)
    env=dict(__name__='_isolated_label_ready_reporter',__file__=str(Path(__file__).resolve()),__package__=None)
    exec(compile(ast.fix_missing_locations(tree),str(ORIGINAL)+':metadata-only authority adapter','exec'),env)
    old_paths,old_gate,old_write,old_link=env['paths'],env['gate'],env['write'],env['link']

    def paths(a):
        f=old_paths(a);protocol=read(f['r:PROTOCOL.json'])
        f['original_reporter']=ORIGINAL
        f['first_failed_prefreeze_source']=FAILED_PREFREEZE_SOURCE
        f['routing_memory_adapter']=Path(protocol['inputs']['memory_ready_source']['path'])
        for key in ('memory_ready_source','memory_attempt_protocol','memory_attempt_started','memory_attempt_checks',
                    'memory_attempt_partial_rows','producer_evaluator_source','producer_posthoc_source'):
            f['routing_labels:'+key]=Path(protocol['inputs'][key]['path'])
        return f

    def gate(a,f,b):
        evidence=old_gate(a,f,b)
        p,s,c,summary=[read(f['r:'+name]) for name in ('PROTOCOL.json','STARTED.json','CHECKS.json','SUMMARY.json')]
        review=c['diagnostic_label_review']
        require(review==p['diagnostic_label_review']==s['diagnostic_label_review']==summary['diagnostic_label_review']
            and review['schema']=='routing_saved_proxy_label_review_v22' and review['old_unknown_literal']=='UNKNOWN'
            and review['producer_unknown_literal']=='UNKNOWN_REFERENCE_OR_INPUT' and review['diagnostic_semantic_literal_changes']==1
            and review['output_folder_AST_changes']==2 and review['inverse_whole_memory_ready_diagnostic_AST_equal'] is True
            and review['memory_adapter_compile_dispatch_AST_changes']==1 and review['inverse_whole_memory_ready_source_AST_equal'] is True
            and review['memory_only_review_four_span_proof_unchanged'] is True
            and review['scientific_score_projection_ranking_candidate_or_settings_changes']==0
            and review['original_and_memory_attempt_outputs_modified'] is False and review['automatic_retry'] is False
            and same(review['ready_source'],b['routing_code']) and same(review['original_memory_ready_source'],b['routing_memory_adapter']),
            'actual typed-label source authority; unchanged memory proof and science')
        for key in ('producer_evaluator_source','producer_posthoc_source'):
            require(same(review[key],b['routing_labels:'+key]),'exact immutable quality producer '+key)
        for key,value in review['failed_memory_attempt_bindings'].items():
            require(same(value,b['routing_labels:'+key]),'preserved failed label attempt '+key)
        protocol,started,failed=[read(f['routing_labels:'+key]) for key in ('memory_attempt_protocol','memory_attempt_started','memory_attempt_checks')]
        require(failed['schema']=='saved_point_closure_routing_checks_v22' and failed['complete'] is failed['passed'] is False
            and failed['checks']==154 and failed['failure_count']==0 and failed['new_production_calls']==0
            and failed['exception']['type']=='ValueError' and failed['exception']['message']=='eight saved proxy labels'
            and failed['inputs']==protocol['inputs']==started['inputs']
            and same(failed['own_protocol'],b['routing_labels:memory_attempt_protocol'])
            and same(started['protocol'],failed['own_protocol'])
            and same(protocol['inputs']['code'],b['routing_memory_adapter'])
            and failed['memory_only_review']==c['memory_only_review'], 'actual previous complete=False label failure stays failed')
        require(review['first_memory_attempt_checks']==154 and review['first_memory_attempt_numerical_failures']==0
            and review['first_memory_attempt_complete'] is review['first_memory_attempt_passed'] is False,
            'preserved diagnostic label failure; no accuracy retry')
        return evidence

    def write(path,value):
        if Path(path).name in env['FILES'][:3]:value=dict(value,reporter_routing_label_review=proof)
        return old_write(path,value)

    def link(path,a,text=None,line=None):
        path=Path(path).resolve()
        if path.suffix!='.gz' or path.stat().st_size<=40*1024*1024:return old_link(path,a,text,line)
        family=None
        for root,name in ((a.private,a.family),(a.history,'pallet_finite_support_orthogonal_audit_20261011_v19'),
                          (a.routing_private,a.routing_family)):
            if path.is_relative_to(root):family=name;break
        if family is None:
            public_docs=a.repo/'_docs/experiments'
            require(path.is_relative_to(public_docs),'large gzip must have an explicit public document family')
            parts=path.relative_to(public_docs).parts;require(len(parts)>=2,'public gzip family/file path');family=parts[0]
        manifest=Path('_docs/experiments')/family/'ARCHIVE_MANIFEST.json'
        target=env['os'].path.relpath(manifest,Path('_docs/experiments')/a.family)
        return '['+(text or env['label'](path,a))+']('+target+('#L'+str(line) if line is not None else '')+')'

    env.update(paths=paths,gate=gate,write=write,link=link)
    return env

if __name__=='__main__':prepare()['main']()

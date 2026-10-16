"""Narrow postseal dispatch of frozen V16 stages to explicit READY authority.

Only three validation filenames, current wrapper filenames and the two local
evaluation imports change. Whole canonical AST restoration proves every
arithmetic, fit, timing, display and diagnostic statement is preserved.
"""
import ast
from collections import Counter
import copy
import hashlib
from pathlib import Path
import runpy

CODE=Path(__file__).resolve().parent
AUTHORITY=CODE/'validation_authority_ready.py'
SOURCES={
 'evaluation':('989bd126b35d155b6899d9b6d17e8b868af38058f475fdae8484a0354374c436','evaluation_inputs','bindings'),
 'verify':('f28571e1a0e31bb8d1194c02cf4699870ea4083b0cfad95d5499898dd40b2ce3','input_paths','paths'),
 'runtime':('641003379dbf419ed0b3be000f9eef954b4910fc2fde5a5d03ca210516747538','inputs','bindings'),
 'runtime_check':('b20d617b7ec110c08668add547b453f5164b02c4b6c835db11af77a56723e73b','paths','paths'),
 'render':('187311ef48e147ad5b02818e415d856d4401e3df4cd2aff68f7f8312a63b495a','inputs','paths'),
 'saved_correspondence_diagnostic':('e85578662c6693be9f80c610ee9ec187b689faaa245d195dc3eb5f5a8d3be058','inputs','bindings')}

def derive(stage,wrapper):
    a=runpy.run_path(str(AUTHORITY));proof=a['geometry_authority']();require=a['require'];binding=a['binding']
    sha,input_name,kind=SOURCES[stage];source=CODE/(stage+'.py');require(binding(source)['sha256']==sha,'immutable original stage '+stage)
    before=ast.parse(source.read_text());canonical=ast.dump(before,include_attributes=False);tree=copy.deepcopy(before)
    aliases=dict(a['ALIASES'])
    # Current-stage pointers are explicit; original mathematical checker stays
    # validation_checks.py and is separately distinguished from READY code.
    for name in ('evaluation','verify','runtime','runtime_check','render','saved_correspondence_diagnostic'):
        aliases[name+'.py']=name+'_ready.py'
    hits=Counter();imports=Counter();source_lookups=Counter()
    original_lookup=ast.parse('Path(__file__).read_text()',mode='eval').body
    ready_lookup=ast.parse('_ready_original_stage.read_text()',mode='eval').body
    class Replace(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if stage=='runtime_check' and node.name=='extraction':
                class Lookup(ast.NodeTransformer):
                    def visit_Call(self,call):
                        if ast.dump(call,include_attributes=False)==ast.dump(original_lookup,include_attributes=False):source_lookups['immutable_predicate_source_lookup']+=1;return copy.deepcopy(ready_lookup)
                        return self.generic_visit(call)
                node=Lookup().visit(node)
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in aliases:
                hits[node.value]+=1;return ast.copy_location(ast.Constant(aliases[node.value]),node)
            return node
        def visit_ImportFrom(self,node):
            if stage in ('runtime','render') and node.level==1 and node.module is None and len(node.names)==1 and node.names[0].name=='evaluation' and node.names[0].asname=='E':
                imports['evaluation_import']+=1;node.names[0].name='evaluation_ready'
            return self.generic_visit(node)
    tree=Replace().visit(tree);require(all(hits[k]>0 for k in a['ALIASES']),'all three explicit READY filename joins')
    require(dict(imports)==({'evaluation_import':1} if stage in ('runtime','render') else {}),'exact local authority import change')
    require(dict(source_lookups)==({'immutable_predicate_source_lookup':1} if stage=='runtime_check' else {}),'exact original predicate self-source lookup')
    derived=ast.dump(tree,include_attributes=False);undo=Counter();reverse={v:k for k,v in aliases.items()}
    class Undo(ast.NodeTransformer):
        def visit_Call(self,node):
            if stage=='runtime_check' and ast.dump(node,include_attributes=False)==ast.dump(ready_lookup,include_attributes=False):return copy.deepcopy(original_lookup)
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in reverse:
                undo[reverse[node.value]]+=1;return ast.copy_location(ast.Constant(reverse[node.value]),node)
            return node
        def visit_ImportFrom(self,node):
            if stage in ('runtime','render') and node.level==1 and node.module is None and len(node.names)==1 and node.names[0].name=='evaluation_ready' and node.names[0].asname=='E':node.names[0].name='evaluation'
            return self.generic_visit(node)
    require(ast.dump(Undo().visit(copy.deepcopy(tree)),include_attributes=False)==canonical and undo==hits,'whole original stage canonical restoration')
    witness=dict(original_source=binding(source),filename_AST_changes=dict(hits),local_import_AST_changes=dict(imports),source_lookup_AST_changes=dict(source_lookups),
        original_full_AST_sha256=hashlib.sha256(canonical.encode()).hexdigest(),ready_full_AST_sha256=hashlib.sha256(derived.encode()).hexdigest(),
        whole_original_stage_canonical_restore_proven=True,scalar_fit_timing_display_formula_AST_changes=0,
        original_checker_did_not_PASS=True,actual_environment_ready_PASS=True)
    env=dict(__file__=str(Path(wrapper).resolve()),__name__='isolated_'+stage+'_ready_dispatch',
        __package__='scripts.research.pallet_hybrid_factor_subset_consensus_20261011_v16',_ready_original_stage=source)
    exec(compile(ast.fix_missing_locations(tree),str(source)+' [explicit READY authority aliases]','exec'),env)
    inherited=env[input_name]
    extra_paths={k:CODE.parent.parent.parent/v['path'] for k,v in proof['bindings'].items()}
    extra_paths.update(dispatch=Path(__file__),wrapper=Path(wrapper),original_stage=source)
    def inputs(*args,**kwargs):
        current=a['geometry_authority']();require(current['bindings']==proof['bindings'] and current['marker']==proof['marker'],'READY source/receipts changed')
        result=inherited(*args,**kwargs)
        if stage=='render':
            require(isinstance(result,tuple) and len(result)==2 and all(isinstance(v,dict) for v in result),'exact render paths/bindings pair')
            paths,bindings=result
            for k,p in extra_paths.items():
                key='environment_ready:'+k;require(key not in paths and key not in bindings,'exclusive READY render input alias')
                paths[key]=p;bindings[key]=env['C'].binding(p)
            return paths,bindings
        for k,p in extra_paths.items():
            key='environment_ready:'+k;require(key not in result,'exclusive READY input alias')
            result[key]=dict(binding(p),origin='public_repository') if kind=='bindings' else p
        return result
    env[input_name]=inputs
    # verify.main replaces its input_paths through a closure; it will resolve
    # this new global at the same original call site. Other bodies likewise
    # call the wrapped current-stage input function before their own freeze.
    env['READY_DISPATCH_WITNESS']=witness;env['ENVIRONMENT_READY_AUTHORITY']=proof
    if stage in ('runtime_check','saved_correspondence_diagnostic'):
        original_write=env['write']
        def write(path,value):
            value.update(environment_ready_dispatch=witness,original_failed_checker_did_not_PASS=True,
                successful_geometry_authority_is_separate_environment_ready=True)
            original_write(path,value)
        env['write']=write
    if stage=='verify':
        original_kernel=env['kernel']
        def kernel():
            h,w=original_kernel();w['environment_ready_dispatch']=witness;return h,w
        env['kernel']=kernel
    return env

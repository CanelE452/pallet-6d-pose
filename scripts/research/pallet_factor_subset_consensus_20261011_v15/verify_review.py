"""Preserve external declarations without opening them; unchanged V15 math.

The original freeze failed before a protocol or arithmetic was created. This
separate wrapper skips explicitly external evaluation bindings during public
file enumeration. Their declarations remain recorded and protected by the
whole evaluation-protocol byte binding. They are never treated as verified
public files. No metric, bootstrap, status or verdict formula changes.
"""
import ast
import hashlib
from pathlib import Path

CODE=Path(__file__).resolve().parent
SOURCE=CODE/'verify.py'
SOURCE_SHA='e624d465fdb05c845716d94219f9bf4f0c171ee821eb6b0f1480f8737b0272c3'
FILES=('REVIEW_STATISTICS_VERIFICATION_PROTOCOL.json','REVIEW_STATISTICS_VERIFICATION_STARTED.json','REVIEW_VERIFICATION.json')


def require(value,message):
    if not value:raise ValueError(message)


def is_public(item):
    if item.get('origin')=='external_readonly_dependency':return False
    p=Path(item['path'])
    require(item.get('origin')=='public_repository' and not p.is_absolute()
        and '..' not in p.parts and p.parts and p.parts[0] in ('scripts','_docs'),
        'explicit public repository binding required')
    return True


def load():
    require(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA,'immutable original V15 verifier')
    tree=ast.parse(SOURCE.read_text());original=ast.dump(tree,include_attributes=False)
    before=ast.parse("for key,value in evaluation['inputs'].items():f['evaluation:'+key]=h['public_path'](value)").body[0]
    after=ast.parse("for key,value in evaluation['inputs'].items():\n if review_is_public(value):f['evaluation:'+key]=h['public_path'](value)").body[0]
    hits=0
    class Replace(ast.NodeTransformer):
        def visit(self,node):
            nonlocal hits
            if ast.dump(node,include_attributes=False)==ast.dump(before,include_attributes=False):hits+=1;return after
            return super().visit(node)
    tree=Replace().visit(tree);require(hits==1,'exactly one public-enumeration AST site')
    revised=ast.dump(tree,include_attributes=False)
    class Undo(ast.NodeTransformer):
        def visit(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(after,include_attributes=False):return before
            return super().visit(node)
    require(ast.dump(Undo().visit(ast.parse(ast.unparse(tree))),include_attributes=False)==original,
        'full original verifier AST restoration')
    hits=0;tree=ast.fix_missing_locations(Replace().visit(ast.parse(SOURCE.read_text())))
    require(hits==1,'same one compilation site')
    witness=dict(original_source_sha256=SOURCE_SHA,public_enumeration_AST_changes=1,
        original_full_AST_sha256=hashlib.sha256(original.encode()).hexdigest(),
        revised_full_AST_sha256=hashlib.sha256(revised.encode()).hexdigest(),
        full_canonical_original_restore_proven=True,statistic_formula_AST_changes=0,
        original_checker_changed=False,external_assets_opened=False,
        external_declared_bytes_are_not_current_asset_verification=True)
    env=dict(__file__=str(SOURCE),__name__='isolated_v15_public_binding_review',review_is_public=is_public)
    exec(compile(tree,str(SOURCE)+' [public metadata enumeration only]','exec'),env)
    return env,witness


def main():
    env,witness=load();old_kernel,old_paths,old_metadata=env['kernel'],env['input_paths'],env['metadata']
    env['FILES']=FILES
    def paths(args,h):
        f=old_paths(args,h);f.update(review_verifier=Path(__file__).resolve(),
            original_metadata_failure=args.input/'STATISTICS_METADATA_FIRST_FAILURE.json')
        return f
    def metadata(f,b,h):
        failure=h['read'](f['original_metadata_failure'])
        require(failure['schema']=='preserved_factor_subset_statistics_metadata_first_failure_v15'
            and failure['complete'] is False and failure['CLI_stage']=='verify.freeze'
            and failure['failed_before_own_protocol'] is True and failure['failed_before_STARTED'] is True
            and failure['own_statistic_arithmetic_calls']==failure['new_GT_models_fits_projection_SVD_seed_calls']==0
            and failure['missing_external_target_opened'] is False and failure['original_checker_modified'] is False
            and failure['error']['type']=='ValueError'
            and failure['error']['message'].startswith('missing/symlink input ')
            and failure['error']['message'].endswith('/TARGETS.json')
            and h['same'](failure['original_checker'],h['binding'](SOURCE)),
            'preserved original pre-protocol external-binding failure')
        evaluation=h['read'](f['evaluation_protocol']);declared={k:v for k,v in evaluation['inputs'].items() if not is_public(v)}
        require(declared and all(v['origin']=='external_readonly_dependency' for v in declared.values()),
            'explicit external declarations, never public file paths')
        require(all('evaluation:'+k not in f for k in declared),'external paths were enumerated')
        # Original metadata compares byte declarations. Only public entries
        # were independently hashed; external entries are unchanged declared
        # metadata inside the independently bound evaluation protocol.
        augmented=dict(b);augmented.update({'evaluation:'+k:v for k,v in declared.items()})
        return old_metadata(f,augmented,h)
    def kernel():
        h,topology=old_kernel();old_write=h['write']
        def write(path,value):
            evaluation=h['read'](h['public_path'](value['inputs']['evaluation_protocol']))
            declared={k:v for k,v in evaluation['inputs'].items() if not is_public(v)}
            value['public_binding_review']=dict(witness,original_metadata_failure=value['inputs']['original_metadata_failure'],
                review_verifier=value['inputs']['review_verifier'],external_declared_metadata=declared,
                verified_public_file_count=len(value['inputs']),external_declared_metadata_count=len(declared),
                original_freeze_failed_before_protocol_or_arithmetic=True)
            old_write(path,value)
        h['write']=write;return h,topology
    env.update(kernel=kernel,input_paths=paths,metadata=metadata);env['main']()


if __name__=='__main__':main()

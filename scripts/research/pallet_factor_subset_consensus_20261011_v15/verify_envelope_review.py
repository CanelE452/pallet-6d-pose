"""Supply the frozen input envelope to STARTED; unchanged statistic math.

The previous public-binding review froze successfully but its STARTED writer
expected an inputs field not supplied by the original verifier. That attempt
failed before STARTED or arithmetic. This separate wrapper inserts one envelope
conditional, preserves both earlier attempts, and changes no arithmetic AST.
"""
import ast
import hashlib
from pathlib import Path

CODE=Path(__file__).resolve().parent;SOURCE=CODE/'verify_review.py'
SOURCE_SHA='cffc16974ec9976acc63e0cb86ec23a7dd5866963a18d7f27875d561b1deb51b'
FILES=('ENVELOPE_STATISTICS_PROTOCOL.json','ENVELOPE_STATISTICS_STARTED.json','ENVELOPE_VERIFICATION.json')


def require(value,message):
    if not value:raise ValueError(message)


def load():
    require(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA,'immutable first public-binding review')
    tree=ast.parse(SOURCE.read_text());original=ast.dump(tree,include_attributes=False)
    marker=ast.parse("evaluation=h['read'](h['public_path'](value['inputs']['evaluation_protocol']))").body[0]
    inserted=ast.parse("if 'inputs' not in value:value['inputs']=h['read'](path.parent/FILES[0])['inputs']").body[0]
    matches=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef) and node.name=='write':
            for index,statement in enumerate(node.body):
                if ast.dump(statement,include_attributes=False)==ast.dump(marker,include_attributes=False):matches.append((node,index))
    require(len(matches)==1,'one inherited envelope hook site');node,index=matches[0];node.body.insert(index,inserted)
    revised=ast.dump(tree,include_attributes=False);node.body.pop(index)
    require(ast.dump(tree,include_attributes=False)==original,'whole original review AST restores exactly');node.body.insert(index,inserted)
    witness=dict(original_public_binding_review_sha256=SOURCE_SHA,envelope_AST_insertions=1,
        full_canonical_original_restore_proven=True,original_full_AST_sha256=hashlib.sha256(original.encode()).hexdigest(),
        revised_full_AST_sha256=hashlib.sha256(revised.encode()).hexdigest(),statistic_formula_AST_changes=0,
        original_verifier_or_public_binding_review_changed=False,original_two_failed_attempts_preserved=True,
        only_missing_STARTED_inputs_are_supplied_from_own_frozen_protocol=True)
    env=dict(__file__=str(SOURCE),__name__='isolated_v15_started_envelope_review')
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE)+' [STARTED envelope only]','exec'),env)
    env['FILES']=FILES;return env,witness


def main():
    env,witness=load();old_load=env['load']
    def inherited_load():
        verifier,public_witness=old_load();old_paths=verifier['input_paths'];old_metadata=verifier['metadata'];old_kernel=verifier['kernel']
        def paths(args,h):
            f=old_paths(args,h);f.update(envelope_review=Path(__file__).resolve(),
                inherited_public_review=SOURCE,original_public_review_protocol=args.input/'REVIEW_STATISTICS_VERIFICATION_PROTOCOL.json',
                original_envelope_failure=args.input/'REVIEW_ENVELOPE_FIRST_FAILURE.json')
            return f
        def metadata(f,b,h):
            failure=h['read'](f['original_envelope_failure']);prior=h['read'](f['original_public_review_protocol'])
            require(failure['schema']=='preserved_factor_subset_statistics_started_envelope_failure_v15'
                and failure['CLI_stage']=='verify_review.run' and failure['complete'] is False and failure['error']['type']=='KeyError'
                and failure['error']['message'] in ('inputs',"'inputs'")
                and failure['failed_before_STARTED'] is True and failure['own_statistic_arithmetic_calls']==failure['new_GT_models_fits_projection_SVD_seed_calls']==0
                and failure['original_checker_or_review_modified'] is False
                and h['same'](failure['original_review'],b['inherited_public_review'])
                and h['same'](failure['existing_review_protocol'],b['original_public_review_protocol']),
                'preserved public-review STARTED envelope failure')
            require(prior['schema']=='independent_factor_subset_statistics_protocol_v15'
                and prior['frozen_after_full_geometry_cleanup_PASS_scoring_statistics_before_own_arithmetic'] is True,
                'previous successful freeze retained')
            for key,value in prior['inputs'].items():require(key in b and h['same'](value,b[key]),'original successful freeze input preserved '+key)
            require(prior['public_binding_review']['public_enumeration_AST_changes']==1
                and prior['public_binding_review']['statistic_formula_AST_changes']==0
                and prior['public_binding_review']['external_assets_opened'] is False
                and h['same'](prior['public_binding_review']['review_verifier'],b['inherited_public_review']),
                'previous external-origin-only review authority')
            return old_metadata(f,b,h)
        def kernel():
            h,topology=old_kernel();old_write=h['write']
            def write(path,value):
                value['STARTED_envelope_review']=dict(witness,review_verifier=h['binding'](__file__),
                    original_public_review=h['binding'](SOURCE))
                old_write(path,value)
            h['write']=write;return h,topology
        verifier.update(input_paths=paths,metadata=metadata,kernel=kernel);return verifier,public_witness
    env['load']=inherited_load;env['main']()


if __name__=='__main__':main()

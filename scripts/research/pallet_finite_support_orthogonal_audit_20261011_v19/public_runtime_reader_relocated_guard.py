"""Copied-root output guard for the unchanged saved-runtime rows reader.

Only the repository-ancestor output restriction is removed. Every relocated
required input, source and manifest remains disjoint from the new output.
The historical cleanup failure, unofficial timing and all arithmetic stay
literal. ROOT alone freezes/runs; this source is not a portable PASS.
"""
import argparse
import ast
import copy
import hashlib
from pathlib import Path
import sys

READY = Path(__file__).resolve().with_name('public_runtime_reader_ready.py')
READY_SHA = 'f45b75d84edf23a65f073f984bb207148ba7343eb6dc861bdeb66148ef48877b'
CHECKER_SHA = '729c3797b04e27d696fc5d343e1a7ea1908f0d08962215dc8e83e9baed800842'

def require(value, message):
    if not value: raise ValueError(message)

def canonical(node): return ast.dump(node, include_attributes=False)

def binding(path):
    path = Path(path).absolute()
    require(path.is_file() and not any(q.is_symlink() for q in (path, *path.parents)), 'missing/symlink source')
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

def prepare(first_failure):
    first_failure = Path(first_failure)
    require(first_failure.is_absolute() and '..' not in first_failure.parts,
            'explicit manifest-mapped historical preflight note required')
    ready_binding = binding(READY)
    require(ready_binding['sha256'] == READY_SHA, 'exact immutable ready public reader')
    original = ast.parse(READY.read_bytes()); before = canonical(original)
    env = dict(__name__='_isolated_public_runtime_relocated_guard', __file__=str(Path(__file__).resolve()))
    exec(compile(original, str(READY) + ':unchanged reader source', 'exec'), env)
    require(canonical(original) == before, 'whole ready-reader AST unchanged')
    previous_install = env['install_location_writer']

    def install(namespace, location_proof):
        previous_install(namespace, location_proof)
        checker = Path(namespace['__file__'])
        raw = checker.with_name('runtime_check.py')
        raw_binding = binding(raw)
        require(raw_binding['sha256'] == CHECKER_SHA, 'exact immutable original runtime checker')
        module = ast.parse(raw.read_bytes())
        originals = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'guard']
        require(len(originals) == 1, 'one original directory guard')
        old = originals[0]; changed = copy.deepcopy(old)
        old_tuple = ast.parse('(args.repo,args.input,args.validation_input,args.runtime_input)', mode='eval').body
        new_tuple = ast.parse('(args.input,args.validation_input,args.runtime_input)', mode='eval').body
        matches = []
        for owner in ast.walk(changed):
            for field, value in ast.iter_fields(owner):
                if isinstance(value, ast.AST) and canonical(value) == canonical(old_tuple): matches.append((owner, field))
        require(len(matches) == 1, 'one repository ancestor restriction')
        setattr(matches[0][0], matches[0][1], copy.deepcopy(new_tuple))
        undo = copy.deepcopy(changed); reverted = 0
        for owner in ast.walk(undo):
            for field, value in ast.iter_fields(owner):
                if isinstance(value, ast.AST) and canonical(value) == canonical(new_tuple):
                    setattr(owner, field, copy.deepcopy(old_tuple)); reverted += 1
        require(reverted == 1 and canonical(undo) == canonical(old), 'whole original guard inverse AST proof')
        guard_env = dict(namespace)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[changed], type_ignores=[])),
                     str(raw) + ':copied output guard', 'exec'), guard_env)
        directory_guard = guard_env['guard']; old_paths = namespace['paths']

        def paths(args):
            values = old_paths(args)
            values.update(relocated_guard_ready_reader=READY, first_portable_guard_preflight_note=first_failure)
            return values

        namespace['paths'] = paths
        proof = dict(schema='public_saved_runtime_copied_output_guard_review_v19',
            reader=binding(__file__), immutable_ready_reader=ready_binding, immutable_original_checker=raw_binding,
            whole_ready_reader_AST_unchanged=True, original_guard_inverse_whole_AST_equal=True,
            guard_AST_tuple_replacements=1, repository_ancestor_output_rule_removed=True,
            inputs_validation_runtime_directory_disjoint_rules_unchanged=True,
            all_manifest_resolved_required_files_disjoint_from_output=True,
            historical_preflight_note_is_producer_receipt=False,
            original_failure_or_arithmetic_or_runtime_status_changed=False,
            overall_runtime_cleanup_FAIL_remains_FAIL=True, actual_portable_run_claimed=False)

        def guard(args, stage):
            directory_guard(args, stage)
            require(args.output != args.private and args.output != args.repo,
                    'new checker output must be a proper artifact subdirectory')
            # At this point the unchanged public main has installed its explicit
            # manifest resolver on paths(); no historical-location fallback.
            resolved = namespace['paths'](args)
            for key, path in resolved.items():
                path = Path(path).absolute()
                require(path.is_file() and not any(q.is_symlink() for q in (path, *path.parents)),
                        'required resolved input missing/symlink: ' + key)
                require(not path.is_relative_to(args.output), 'checker output contains required input: ' + key)

        previous_write = namespace['write']
        def write(path, value):
            return previous_write(path, dict(value, public_runtime_relocated_guard_review=copy.deepcopy(proof)))
        namespace.update(guard=guard, write=write)

    env['install_location_writer'] = install
    namespace, proof = env['prepare']()
    return namespace, proof

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--first-guard-failure', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    namespace, _ = prepare(args.first_guard_failure)
    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__).resolve()), *remaining]
        namespace['main']()
    finally: sys.argv = previous

if __name__ == '__main__': main()

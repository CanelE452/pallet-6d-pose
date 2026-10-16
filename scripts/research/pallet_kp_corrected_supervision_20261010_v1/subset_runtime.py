"""Fresh complete routes on a frozen easy/medium panel; no cached timing.

Only population guards and the panel loader change. The original benchmark
body, models, decoder, solver, timing boundaries and parity checks are reused.
All changes to the old stdlib prerequisite function are exported as an AST
audit. Importing this module performs no inference or pose fitting.
"""
import argparse
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_kp_corrected_supervision_20261010_v1'


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def require(value, message):
    if not value:
        raise RuntimeError('SUBSET_RUNTIME_PENDING: ' + message)


def guarded_prereqs(args, A):
    """Replay every original stored-data guard with the declared population.

    Full319 membership is checked first. The old function is compiled alone,
    with exact constant edits recorded; no model/CV2 imports are executed.
    """
    scope = read(args.scope_protocol)
    require(scope['evaluation_scope'] == 'easy_medium_only', 'scope protocol differs')
    for bound in scope['fixed_inputs'].values():
        A.bound_file(REPO / bound['path'], bound, 'frozen subset input')
    cohort = read(args.cohort)
    ids = cohort['ids']
    n = len(ids)
    full = read(A.ORIGINAL / 'INPUTS.json')
    by_id = {r['id']: r for r in full['frames']}
    require(len(by_id) == len(full['frames']) == 319, 'original319 identity changed')
    require(len(ids) == len(set(ids)) and n > 0 and set(ids) <= set(by_id), 'cohort membership invalid')
    selected = [by_id[fid] for fid in ids]
    sessions = len({r['session'] for r in selected})
    require(cohort['count'] == n, 'cohort count differs')
    for filename, stage in [(args.infer_receipt, 'infer'), (args.evaluate_receipt, 'evaluate')]:
        receipt = read(filename)
        require(receipt['stage'] == stage, 'stage linkage invalid')
        require(receipt['cohort']['sha256'] == A.sha(args.cohort), 'cohort receipt linkage differs')
    source = Path(A.__file__).read_text()
    original = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'prereqs')
    function = copy.deepcopy(original)
    changes = []
    numbers = {319: n, 957: 3 * n, 1914: 6 * n, 1595: 5 * n, 13: sessions}
    schema_before = 'original_path_supervision_repair_adapter_v1'
    schema_after = 'subset_original_path_supervision_repair_adapter_v1'
    class Scope(ast.NodeTransformer):
        def visit_Constant(self, node):
            value = node.value
            if type(value) is int and value in numbers:
                changes.append(dict(line=node.lineno, original=value, scoped=numbers[value]))
                return ast.copy_location(ast.Constant(numbers[value]), node)
            if value == schema_before:
                changes.append(dict(line=node.lineno, original=value, scoped=schema_after))
                return ast.copy_location(ast.Constant(schema_after), node)
            return node
    Scope().visit(function)
    def selected_read(path):
        if Path(path).resolve() == (A.ORIGINAL / 'INPUTS.json').resolve():
            return dict(full, frames=selected)
        return A.read(path)
    namespace = dict(A.__dict__, read=selected_read)
    isolated = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(isolated, str(Path(A.__file__)) + ':scoped_prereqs', 'exec'), namespace)
    proof = namespace['prereqs'](args)
    proof.pop('original319_input_identity', None)
    proof.update(original_full_population_membership_checked=319,
                 cohort=A.binding(args.cohort), selected_frames=n,
                 excluded_frames=319-n, scoped_prerequisite_AST_changes=changes,
                 original_prerequisite_source=A.binding(Path(A.__file__)),
                 original_function_AST_sha256=hashlib.sha256(ast.dump(original, include_attributes=False).encode()).hexdigest(),
                 scoped_function_AST_sha256=hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest())
    return proof


def preserved(A, path):
    data = read(path)
    require(data['protected_prior_files'] == len(data['files']) == 331, 'prior331 population differs')
    for bound in data['files']:
        location = (REPO / bound['path']).resolve()
        require(location.is_relative_to(REPO), 'prior path escapes repository')
        A.bound_file(location, bound, 'protected prior ' + bound['path'])
    return data


def measure(args):
    from scripts.research.pallet_kp_repair_runtime_20261010_v1 import adapter as A
    proof = guarded_prereqs(args, A)
    output, scratch = A.output_guards(args)
    panel = read(args.panel)
    require(panel['cohort']['sha256'] == A.sha(args.cohort), 'runtime panel cohort binding differs')
    selected = panel['frames']
    require(len(selected) == len({r['frame_id'] for r in selected}) == 26, 'runtime needs26 distinct eligible images')
    require({r['frame_id'] for r in selected} <= set(read(args.cohort)['ids']), 'runtime panel contains excluded image')
    before = preserved(A, args.prior_bindings)
    fixed = {name: A.binding(path) for name, path in {
        'adapter': Path(__file__), 'original_runtime_adapter': Path(A.__file__),
        'original_benchmark': A.OLD_CODE / 'benchmark.py', 'cohort': Path(args.cohort),
        'runtime_panel': Path(args.panel), 'prior_publications': Path(args.prior_bindings)
    }.items()}
    write_new(output / 'RUNTIME_ADAPTER_STARTED.json', dict(
        schema='easy_medium_runtime_attempt_start_v1', prerequisites=proof,
        fixed_inputs=fixed, configured_pipeline_calls=600, actual_pipeline_calls=0,
        parameter_changes=False, original_timing_body_unchanged=True))
    scratch.mkdir(parents=True, exist_ok=True)
    metadata = A.checkpoint_metadata(args)
    with A.original_context(args, output, scratch) as (C, L):
        from scripts.research.pallet_observation_refiner_20261009_v1 import benchmark as B
        from scripts.research.pallet_n3_subpix_20261008_v1 import runtime as R
        import cv2
        import numpy as np
        original_panel, original_write = R._panel, C.write
        def scoped_panel():
            images, predictions, bindings = [], [], [A.binding(args.panel), A.binding(args.cohort)]
            base = R.C.read(R.BASE_CACHE)
            require(base['complete'], 'original candidate cache incomplete')
            bindings.append(R.binding(R.BASE_CACHE))
            for row in selected:
                path = C.ROOT / row['image_key']
                A.bound_file(path, row['image'], 'eligible runtime RGB')
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                require(image is not None and list(image.shape[:2]) == row['original_hw'], 'eligible RGB shape differs')
                require(image.dtype == np.uint8 and image.shape[2] == 3, 'eligible RGB format differs')
                candidates = base['frames'][row['image_key']]
                index = int(np.argmax([c['score'] for c in candidates])) if candidates else None
                predictions.append(dict(candidates=copy.deepcopy(candidates), selected_index=index))
                images.append(image)
            return copy.deepcopy(selected), images, predictions, bindings
        def scoped_write(path, value):
            if Path(path).name == 'RUNTIME.json':
                value = dict(value, panel_sessions=len({r['session_id'] for r in selected}),
                             evaluation_scope='easy_medium_only', cohort=A.binding(args.cohort),
                             panel_selection=A.binding(args.panel),
                             metadata_scope_adapter=A.binding(Path(__file__)),
                             original_timed_pipeline_body_unchanged=True)
            return original_write(path, value)
        R._panel, C.write = scoped_panel, scoped_write
        try:
            result = B.run('scripts.research.pallet_observation_refiner_20261009_v1.benchmark:benchmark_adapter',
                           learned_reference=Path(args.geometry).resolve(), remaining_seconds=args.remaining_seconds)
        finally:
            R._panel, C.write = original_panel, original_write
    require(preserved(A, args.prior_bindings) == before, 'prior331 changed')
    write_new(output / 'RUNTIME_ADAPTER_RECEIPT.json', dict(
        schema='easy_medium_corrected_original_runtime_adapter_v1', complete=result['complete'],
        status=result['status'], prerequisites=proof, fixed_inputs=fixed,
        checkpoint_metadata_checks=metadata, original_prior_files_preserved=331,
        old_context_restored=True, cached_coordinate_timing=False, new_training_updates=0,
        runtime=A.binding(output/'RUNTIME.json'), raw_rows=A.binding(output/'RUNTIME_ROWS.jsonl.gz')))
    return result


def main():
    from scripts.research.pallet_kp_repair_runtime_20261010_v1 import adapter as A
    parser = A.parser()
    parser.add_argument('--cohort', default=str(DOC/'COHORT.json'))
    parser.add_argument('--panel', default=str(DOC/'RUNTIME_PANEL.json'))
    parser.add_argument('--scope-protocol', default=str(DOC/'SUBSET_PROTOCOL.json'))
    args = parser.parse_args()
    args.completion = args.completion or str(Path(args.fits)/'TRAINING_COMPLETION.json')
    if args.stage == 'preflight':
        A.output_guards(args)
        print(json.dumps(guarded_prereqs(args, A), ensure_ascii=False, indent=2))
    elif args.stage == 'measure':
        result = measure(args)
        if not result['complete']:
            raise SystemExit(1)
    else:
        raise SystemExit('Only scoped preflight/measure are supported')


if __name__ == '__main__':
    main()

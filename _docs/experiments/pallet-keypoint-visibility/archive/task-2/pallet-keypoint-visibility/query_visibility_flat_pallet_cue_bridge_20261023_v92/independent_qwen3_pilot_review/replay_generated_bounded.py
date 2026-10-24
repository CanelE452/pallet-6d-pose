"""Independent generated-only replay. Never calls freeze/authorize-success/run."""
from pathlib import Path
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parent / 'qwen3_source_pilot/run_qwen_source_once.py'
EXPECTED = 'd16ee9b45cb61e08940ceb412e7b948cda90e1c85a269939baef532008e7da12'

def binding(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)

def main():
    assert binding(CODE)['sha256'] == EXPECTED
    spec = importlib.util.spec_from_file_location('reviewed_generated_only', CODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Imported module has no top-level model/data execution; only generated() is called.
    with tempfile.TemporaryDirectory(prefix='v92_independent_generated_') as temporary:
        module.ROOT = Path(temporary)
        with contextlib.redirect_stdout(io.StringIO()):
            module.generated()
        generated = json.loads((module.ROOT / 'GENERATED_QWEN_SOURCE_TESTS.json').read_text())
        assert generated['PASS'] and generated['code']['sha256'] == EXPECTED
        combinations = 0
        for values in itertools.product(*(sorted(module.ENUMS[k]) for k in module.ENUMS)):
            good = dict(zip(module.ENUMS, values), note='Generated visible-content fixture.')
            assert module.parsed_output(json.dumps(good)) == (good, None)
            combinations += 1
        bad = dict(local_object='PALLET_STRUCTURE', visible_distinct_pallet_cue='YES', target_binding='CLEAR', note='x')
        invalid_count = 0
        for value in ['', 'x' * 401, None, 7, [], {}]:
            assert module.parsed_output(json.dumps(dict(bad, note=value)))[0] is None
            invalid_count += 1
        for key in module.ENUMS:
            for value in ['D', 'E', 'S', None, 1]:
                assert module.parsed_output(json.dumps(dict(bad, **{key: value})))[0] is None
                invalid_count += 1
        flags = np.ones(160, dtype=bool)
        flags[np.random.default_rng(20261023).choice(160, 38, replace=False)] = False
        selected = module.calls_for_flags(flags)
        assert len(selected) == 122 and selected == np.flatnonzero(flags).tolist()
        # Fail-closed early negatives: these never reach source, asset, or model reads.
        negatives = 0
        base = dict(schema='V92_ROOT_APPROVED_QWEN3_SOURCE_FORWARD_ENTRY_V1', approved=False, source_only=True)
        fixtures = [base, dict(base, approved=True, current=True, fit=False, PnP=False),
                    dict(base, approved=True, current=False, fit=False, PnP=False, code=dict(path=str(CODE), sha256='0' * 64))]
        for i, entry in enumerate(fixtures):
            path = module.ROOT / f'negative_{i}.json'
            path.write_text(json.dumps(entry))
            try:
                module.authorize(path)
            except AssertionError:
                negatives += 1
            else:
                raise AssertionError('Malformed entry unexpectedly accepted')
        assert negatives == 3
    result = dict(completed=True, PASS=True, code=binding(CODE), reviewer_code=binding(Path(__file__)),
                  builtin_generated_independently_rerun=True, enum_combinations=combinations,
                  extra_invalid_parser_fixtures=invalid_count, noncontiguous_valid122_invalid38=True,
                  early_gate_negative_fixtures=negatives, generated_result=generated,
                  actual_source_RGB_GT_weight_body_reads=0, model_constructor_forward_fit_current_PnP=0,
                  scope='Bounded static and generated API checks; not actual model or source execution.')
    output = ROOT / 'INDEPENDENT_GENERATED_REPLAY.json'
    assert not output.exists()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(PASS=True, output=binding(output))))

if __name__ == '__main__':
    main()

"""Bounded format-only review: code/protocol and generated arrays, never source/model."""
from pathlib import Path
import ast, contextlib, hashlib, importlib.util, io, json, tempfile

ROOT = Path(__file__).resolve().parent
V92 = ROOT.parent
V1 = V92 / 'qwen3_source_pilot'
V2 = V92 / 'qwen3_source_pilot_v2_schema'

def binding(p):
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)

def load_code(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    c1, c2 = V1 / 'run_qwen_source_once.py', V2 / 'run_qwen_source_once.py'
    assert binding(c1)['sha256'] == 'd16ee9b45cb61e08940ceb412e7b948cda90e1c85a269939baef532008e7da12'
    assert binding(c2)['sha256'] == '298f456f8c8b28a8919307ec78c1224ddb72c519227287a0fca61ffd0d02d7b4'
    trees = [ast.parse(p.read_text()) for p in [c1, c2]]
    functions = [{n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)} for tree in trees]
    assert functions[0].keys() == functions[1].keys()
    changed = [k for k in functions[0] if functions[0][k] != functions[1][k]]
    assert changed == ['freeze']
    m1, m2 = load_code(c1, 'review_v1_code_only'), load_code(c2, 'review_v2_code_only')
    prefix1 = m1.PROMPT.split('Return only one JSON object with exactly these fields: ')[0]
    prefix2 = m2.PROMPT.split('Return exactly one valid JSON object with these four keys: ')[0]
    assert prefix1 == prefix2 and m1.ENUMS == m2.ENUMS
    example = m2.PROMPT.split('Valid JSON example: ')[1].split(' No Markdown fences')[0]
    parsed, error = m2.parsed_output(example)
    assert parsed is not None and error is None and set(parsed) == set(m2.ENUMS) | {'note'}
    assert 'Enum alternatives are values, never keys.' in m2.PROMPT
    # The reported V1 extra-key failure remains invalid: no relaxed parser.
    for key in ['OTHER_OBJECT', 'BACKGROUND', 'semantic_state']:
        assert m2.parsed_output(json.dumps(dict(parsed, **{key: 'NO'})))[0] is None
    p1, p2 = [json.loads((p / 'SOURCE_PILOT_PROTOCOL_BEFORE_MODEL_WEIGHTS.json').read_text()) for p in [V1, V2]]
    allowed_changes = {'code', 'generated', 'prompt', 'output', 'format_revision', 'prior_V1_code', 'prior_V1_protocol', 'prior_V1_inference_seal', 'semantic_task_prefix_sha256', 'format_only_scope'}
    changes = sorted(k for k in p1.keys() | p2.keys() if p1.get(k) != p2.get(k))
    assert set(changes) <= allowed_changes
    assert p2['code'] == binding(c2) and p2['prior_V1_code'] == binding(c1)
    assert p2['format_revision'] == 2
    assert p2['valid_inference_slots'] == 122 and p2['no_call_U'] == 38
    assert p2['semantic_task_prefix_sha256'] == hashlib.sha256(prefix2.encode()).hexdigest()
    original_generated = json.loads((V2 / 'GENERATED_QWEN_SOURCE_TESTS.json').read_text())
    assert original_generated['PASS'] and original_generated['code'] == binding(c2)
    assert p2['generated'] == binding(V2 / 'GENERATED_QWEN_SOURCE_TESTS.json')
    with tempfile.TemporaryDirectory(prefix='v92_v2_generated_only_') as temporary:
        m2.ROOT = Path(temporary)
        with contextlib.redirect_stdout(io.StringIO()):
            m2.generated()
        generated = json.loads((m2.ROOT / 'GENERATED_QWEN_SOURCE_TESTS.json').read_text())
        assert generated == original_generated
        # Entry still fails before source/assets on unapproved V2 input.
        entry = m2.ROOT / 'unapproved.json'
        entry.write_text(json.dumps(dict(schema='V92_ROOT_APPROVED_QWEN3_SOURCE_FORWARD_ENTRY_V1', approved=False, source_only=True)))
        try:
            m2.authorize(entry)
        except AssertionError:
            pass
        else:
            raise AssertionError('Unapproved entry accepted')
    result = dict(completed=True, PASS=True, V1_code=binding(c1), code=binding(c2), reviewer_code=binding(Path(__file__)),
                  changed_function_AST=changed, identical_function_AST=[k for k in functions[0] if k not in changed],
                  semantic_task_prefix_byte_identical=True, semantic_task_prefix_sha256=hashlib.sha256(prefix2.encode()).hexdigest(),
                  protocol_changed_keys=changes, protocol_changes_within_declared_allowlist=True,
                  strict_example_accepted=True, three_extra_key_variants_still_rejected=True,
                  builtin_generated_rerun_exact=True, unapproved_entry_rejected=True,
                  model_constructor_forward_source_pixels_weights_current_fit_PnP=0)
    output = ROOT / 'INDEPENDENT_DIFF_AND_GENERATED_REPLAY.json'
    assert not output.exists()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(PASS=True, output=binding(output))))

if __name__ == '__main__':
    main()

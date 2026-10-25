"""In-memory selection/AST checks only; no public payload or encoder import."""
from pathlib import Path
import ast
import copy
import hashlib
import json

H = Path(__file__).absolute().parent
R = H.parent
def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc'))
    for ancestor in reversed([p, *p.parents]):
        assert not ancestor.is_symlink(), str(ancestor)
    return p
p = safe(H / 'build_public_aux_features.py')
source = p.read_bytes()
tree = ast.parse(source)
names = {'validate_reference', 'select_indices', 'authorize'}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
ns = {'hashlib': hashlib, 'SALT': 'V261_PUBLIC_POINT_STRATIFIED64_V1|',
      'REF_KEYS': set('reference_grid_index r_xy r_derived_label r_derived_state r_depth_eligible r_known r_raw_observed_minus_render_m r_raw_residual_finite r_exclusion_reasons native_slot SELF_state'.split())}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'), ns)
rows = []
for i in range(4800):
    label = 0 if i < 100 else 1 if i < 200 else -1
    row = {'reference_grid_index': i, 'r_xy': [4+8*(i%80),4+8*(i//80)],
           'r_derived_label': label, 'r_derived_state': {-1:'U',0:'D',1:'E'}[label],
           'r_depth_eligible': label >= 0, 'r_known': label >= 0,
           'r_raw_observed_minus_render_m': None if label < 0 else 0. if label == 0 else -.2,
           'r_raw_residual_finite': label >= 0,
           'r_exclusion_reasons': ['invented_exclusion'] if label < 0 else [],
           'native_slot': None, 'SELF_state': 'UNKNOWN'}
    ns['validate_reference'](row, i)
    rows.append(row)
path = 'INVENTED_group/0000'
chosen, ranks = ns['select_indices'](rows, path)
expected = set()
for label in (0, 1):
    candidates = [(hashlib.sha256(('V261_PUBLIC_POINT_STRATIFIED64_V1|'+path+'|'+str(label)+'|'+str(i)).encode()).hexdigest(),i)
                  for i,r in enumerate(rows) if r['r_derived_label'] == label]
    expected.update(i for _,i in sorted(candidates)[:64])
assert chosen == sorted(expected) and len(chosen) == 128
assert sum(rows[i]['r_derived_label'] == 0 for i in chosen) == 64
assert sum(rows[i]['r_derived_label'] == 1 for i in chosen) == 64
assert all(rows[i]['r_derived_label'] >= 0 for i in chosen)
modified = copy.deepcopy(rows)
for row in modified:
    row['r_raw_observed_minus_render_m'] = 123.
    row['r_exclusion_reasons'] = ['changed_sidecar_not_selection_input']
assert ns['select_indices'](modified, path)[0] == chosen
assert ns['select_indices'](rows, path)[0] == chosen
assert ns['select_indices'](rows, path+'different')[0] != chosen
try:
    ns['authorize']({'execution_authorized': False, 'encoder_only': True})
except ValueError:
    pass
else:
    raise AssertionError('False entry failed to stop before I/O')
oldp = safe(R / 'V260_SINGLE_QUERY_FEATURE_TRANSPORT_B/build_single_query_features.py')
old = oldp.read_bytes()
assert hashlib.sha256(old).hexdigest() == 'c7fd59a252cde3f535b4a1b6286fd4867888486e3beddc8319d0e7091be8e1c2'
helpers = {'lexical','guard','binding_key','checked','bind','json_write','module','pure'}
def functions(t):
    return {n.name: ast.dump(n,include_attributes=False) for n in ast.parse(t).body if isinstance(n,ast.FunctionDef)}
a,b = functions(source),functions(old)
assert all(a[n] == b[n] for n in helpers)
result = {'completed': True, 'PASS': True, 'code_sha256': hashlib.sha256(source).hexdigest(),
          'checks': ['original4800grid schema validation', 'independent hex-SHA rank matches64/class policy',
                     'U excluded without replacement/filling', 'depth/reasons metadata cannot change selection',
                     'determinism and publisher-path salt separation', 'false entry rejects before filesystem/model access',
                     'all8 reused transport helper functions AST exact'],
          'synthetic_in_memory_records_only': True, 'public_payload_model_weights_or_Git_access': 0}
with safe(H / 'PREPARATION_CHECKS.json').open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(result, ensure_ascii=False))

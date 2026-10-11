"""Memory-bounded saved routing audit; immutable first attempt is preserved.

Only retained actual_pose objects become canonical streaming SHA identities.
The same whole value is checked against each geometry row; no pose field,
score, gate, ranking, outcome, summary or threshold is changed. Exit137 is
preserved as execution history and is not assigned a cause by this source.
ROOT alone freezes/runs; authoring performs no helper/data/math execution.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

ORIGINAL = Path(__file__).resolve().with_name('diagnose.py')
ORIGINAL_SHA = '70e610fbab06d8e7c1153a02fb11344dd1114582046e479336577f8504ec67f6'
PRIVATE = Path('/tmp/pallet-consensus-routing-private-20261011-v22')

def require(value, message):
    if not value: raise ValueError(message)

def canonical(node): return ast.dump(node, include_attributes=False)

def binding(path):
    path = Path(path).absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'required source/history missing or symlink')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())

def actual_pose_identity(value):
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
    for chunk in encoder.iterencode(value): digest.update(chunk.encode())
    return digest.hexdigest()

def prepare():
    bound = binding(ORIGINAL); require(bound['sha256'] == ORIGINAL_SHA, 'exact immutable first diagnostic source')
    original = ast.parse(ORIGINAL.read_bytes()); tree = copy.deepcopy(original); edits = []

    def span(root, expected):
        found = []; keys = [canonical(n) for n in expected]
        for owner in ast.walk(root):
            for _, items in ast.iter_fields(owner):
                if isinstance(items, list):
                    for i in range(len(items) - len(expected) + 1):
                        chunk = items[i:i + len(expected)]
                        if all(isinstance(n, ast.AST) for n in chunk) and [canonical(n) for n in chunk] == keys:
                            found.append((items, i))
        require(len(found) == 1, 'one exact memory-only AST source span')
        return found[0]

    def change(before, after, label):
        old, new = ast.parse(before).body, ast.parse(after).body
        items, i = span(tree, old); items[i:i + len(old)] = copy.deepcopy(new)
        edits.append(dict(old=old, new=new, label=label))

    change("""
fields=('id','session','method','output_status','new_pose_estimated','fallback_used','pose_available',
        'input_points','original_native_N3_points','K','xyz','raw_hw','hidden_initial','head_checkpoint_sha256',
        'calibration_binding','selected_corner_ids','actual_pose','native_points')
result={k:copy.deepcopy(row.get(k)) for k in fields}
""", """
fields=('id','session','method','output_status','new_pose_estimated','fallback_used','pose_available',
        'input_points','original_native_N3_points','K','xyz','raw_hw','hidden_initial','head_checkpoint_sha256',
        'calibration_binding','selected_corner_ids','native_points')
result={k:copy.deepcopy(row.get(k)) for k in fields}
result['actual_pose_identity']=actual_pose_identity(row.get('actual_pose')) if row['method']==PRIMARY else None
""", 'retained full actual_pose replaced by canonical streaming SHA only')
    change("""
for key in ('output_status','new_pose_estimated','fallback_used','pose_available','actual_pose','native_points'):
    require(scores[PRIMARY][key]==row[key],'new score/geometry immutable output '+key)
""", """
for key in ('output_status','new_pose_estimated','fallback_used','pose_available','native_points'):
    require(scores[PRIMARY][key]==row[key],'new score/geometry immutable output '+key)
require(scores[PRIMARY]['actual_pose_identity']==actual_pose_identity(row['actual_pose']),
        'new score/geometry immutable whole actual_pose canonical SHA')
""", 'same complete score/geometry pose identity checked without retaining pose packets')
    change("p.add_argument('--stats-checker',type=Path,default=V21/'stats_check.py');p.add_argument('--output',type=Path,default=PRIVATE/'audit')",
           "p.add_argument('--stats-checker',type=Path,default=V21/'stats_check.py');p.add_argument('--output',type=Path,default=PRIVATE/'audit_ready')",
           'exclusive separate ready output default')
    change("require(a.output==PRIVATE/'audit' and not any(x.is_symlink() for x in (a.output,*a.output.parents)),'exclusive fixed private22 output')",
           "require(a.output==PRIVATE/'audit_ready' and not any(x.is_symlink() for x in (a.output,*a.output.parents)),'exclusive fixed private22 ready output')",
           'exclusive separate ready output guard')
    undo = copy.deepcopy(tree)
    for edit in reversed(edits):
        items, i = span(undo, edit['new']); items[i:i + len(edit['new'])] = copy.deepcopy(edit['old'])
    require(canonical(undo) == canonical(original), 'whole original diagnostic inverse AST equality')
    proof = dict(schema='routing_saved_pose_memory_only_review_v22', original_source=bound,
        ready_source=binding(__file__), inverse_whole_original_AST_equal=True,
        AST_source_span_changes=len(edits), changes=[e['label'] for e in edits],
        retained_full_actual_pose_packets=False,
        whole_actual_pose_identity='SHA256 canonical sorted-key JSON, same full value on both sides',
        actual_pose_identity_scope='PRIMARY_ONLY; other saved control pose packets are neither retained nor hashed',
        streaming_JSON_encoder=True, scientific_score_gate_routing_outcome_summary_formulas_changed=False,
        parent_candidate_poses_or_metrics_or_settings_changed=False,
        original_exit137_cause_confirmed=False, original_attempt_outputs_modified=False,
        actual_new_run_or_memory_recovery_claimed=False)
    namespace = dict(__name__='_isolated_memory_bounded_routing', __file__=str(Path(__file__).resolve()),
                     actual_pose_identity=actual_pose_identity)
    exec(compile(ast.fix_missing_locations(tree), str(ORIGINAL) + ':memory-only identity adapter', 'exec'), namespace)
    old_prepare, old_write = namespace['prepare'], namespace['write']

    def ready_prepare(args):
        f, metrics, old_proof, ae, scalar = old_prepare(args)
        f.update(original_diagnostic_source=ORIGINAL,
            original_attempt_protocol=PRIVATE/'audit/PROTOCOL.json',
            original_attempt_started=PRIVATE/'audit/STARTED.json',
            original_attempt_exit_note=PRIVATE/'FIRST_RUN_EXIT_NOTE.json')
        return f, metrics, dict(old_proof, memory_only_review=copy.deepcopy(proof)), ae, scalar

    def write(path, value):
        return old_write(path, dict(value, memory_only_review=copy.deepcopy(proof)))

    namespace.update(prepare=ready_prepare, write=write)
    return namespace, proof

if __name__ == '__main__': prepare()[0]['main']()

"""Typed-label-only routing adapter; both failed attempts remain immutable.

ROOT alone freezes/runs. One saved proxy UNKNOWN literal matches the existing
producer; two output spans select a new folder. The memory-only four-span proof
is preserved. No score, projection, ranking, gate, fit or setting is changed.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

PRIVATE = Path('/tmp/pallet-consensus-routing-private-20261011-v22')
MEMORY_SOURCE = Path(__file__).resolve().with_name('diagnose_ready.py')
MEMORY_SHA = '01593e5f1dd4f94f652493a7d4966a168bc583a343f88315ef6b63b8b0c298e4'
EVALUATOR = Path('/tmp/pallet-whole-u-closure-private-20261011-v21/evaluate.py')
EVALUATOR_SHA = '3938edcc3491e9c19eae99adc8bd1af04595a8c9e42be02dd968a6c5fbae52d1'
PRODUCER_RELATIVE = 'scripts/research/pallet_three_head_observation_20261010_v7/statistics.py'
PRODUCER_SHA = 'fffa7422b3020c6f61386912eb4910a8099ca5a814c771c11896919dbd4e4f4f'
OLD_UNKNOWN = 'UNKNOWN'
PRODUCER_UNKNOWN = 'UNKNOWN_REFERENCE_OR_INPUT'

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

def same(a, b): return all(a.get(k) == b.get(k) for k in ('bytes', 'sha256'))

def span(root, expected):
    matches = []; keys = [canonical(n) for n in expected]
    for owner in ast.walk(root):
        for _, items in ast.iter_fields(owner):
            if isinstance(items, list):
                for i in range(len(items) - len(expected) + 1):
                    chunk = items[i:i + len(expected)]
                    if all(isinstance(n, ast.AST) for n in chunk) and [canonical(n) for n in chunk] == keys:
                        matches.append((items, i))
    require(len(matches) == 1, 'one exact label/output AST source span')
    return matches[0]

def producer_proof(repo):
    producer = Path(repo) / PRODUCER_RELATIVE
    eb, pb = binding(EVALUATOR), binding(producer)
    require(eb['sha256'] == EVALUATOR_SHA and pb['sha256'] == PRODUCER_SHA, 'exact existing evaluation/posthoc producer sources')
    evaluator_tree, producer_tree = ast.parse(EVALUATOR.read_bytes()), ast.parse(producer.read_bytes())
    posthoc = [n for n in evaluator_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'posthoc']
    require(len(posthoc) == 1 and canonical(posthoc[0].body[0]) == canonical(ast.parse(
        "result = S.posthoc(dict(row, method='IMAGE_ROLE_CORNERWISE_HYBRID'), n3, observation)").body[0]), 'unchanged evaluator delegates saved quality to S7 posthoc')
    expected = ast.parse("quality = ('UNKNOWN_REFERENCE_OR_INPUT' if errors['input'] is None else 'CORRECT_WITHIN8PX' if errors['input'] <= 8 else 'INCORRECT_OVER8PX')").body[0]
    source_posthoc = [n for n in producer_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'posthoc']
    require(len(source_posthoc) == 1 and sum(canonical(n) == canonical(expected) for n in ast.walk(source_posthoc[0])) == 1,
        'exact existing input_quality UNKNOWN conditional and unchanged fixed8px branches')
    return dict(producer_evaluator_source=eb, producer_posthoc_source=pb,
        producer_quality_assignment_AST_sha256=hashlib.sha256(canonical(expected).encode()).hexdigest())

def prepare():
    memory_bound = binding(MEMORY_SOURCE)
    require(memory_bound['sha256'] == MEMORY_SHA, 'exact immutable memory-only ready source')
    original = ast.parse(MEMORY_SOURCE.read_bytes()); outer_tree = copy.deepcopy(original); label_proof = {}

    def adapt_memory_tree(tree):
        adapted = copy.deepcopy(tree); edits = []
        def change(before, after, label):
            old, new = ast.parse(before).body, ast.parse(after).body
            items, i = span(adapted, old); items[i:i + len(old)] = copy.deepcopy(new)
            edits.append(dict(old=old, new=new, label=label))
        change("GOOD,BAD,UNKNOWN='CORRECT_WITHIN8PX','INCORRECT_OVER8PX','UNKNOWN'",
               "GOOD,BAD,UNKNOWN='CORRECT_WITHIN8PX','INCORRECT_OVER8PX','UNKNOWN_REFERENCE_OR_INPUT'",
               'diagnostic UNKNOWN exactly matches existing saved input_quality producer')
        change("p.add_argument('--stats-checker',type=Path,default=V21/'stats_check.py');p.add_argument('--output',type=Path,default=PRIVATE/'audit_ready')",
               "p.add_argument('--stats-checker',type=Path,default=V21/'stats_check.py');p.add_argument('--output',type=Path,default=PRIVATE/'audit_labels_ready')",
               'exclusive distinct label-ready output default')
        change("require(a.output==PRIVATE/'audit_ready' and not any(x.is_symlink() for x in (a.output,*a.output.parents)),'exclusive fixed private22 ready output')",
               "require(a.output==PRIVATE/'audit_labels_ready' and not any(x.is_symlink() for x in (a.output,*a.output.parents)),'exclusive fixed private22 label-ready output')",
               'exclusive distinct label-ready output guard')
        undo = copy.deepcopy(adapted)
        for edit in reversed(edits):
            items, i = span(undo, edit['new']); items[i:i + len(edit['new'])] = copy.deepcopy(edit['old'])
        require(canonical(undo) == canonical(tree), 'whole memory-ready diagnostic inverse AST equality')
        label_proof.update(schema='routing_saved_proxy_label_review_v22', original_memory_ready_source=memory_bound,
            ready_source=binding(__file__), old_unknown_literal=OLD_UNKNOWN, producer_unknown_literal=PRODUCER_UNKNOWN,
            diagnostic_semantic_literal_changes=1, output_folder_AST_changes=2,
            inverse_whole_memory_ready_diagnostic_AST_equal=True,
            memory_adapter_compile_dispatch_AST_changes=1, inverse_whole_memory_ready_source_AST_equal=True,
            changes=[edit['label'] for edit in edits], memory_only_review_four_span_proof_unchanged=True,
            scientific_score_projection_ranking_candidate_or_settings_changes=0,
            original_and_memory_attempt_outputs_modified=False, automatic_retry=False)
        return adapted

    old = ast.parse("exec(compile(ast.fix_missing_locations(tree), str(ORIGINAL) + ':memory-only identity adapter', 'exec'), namespace)").body
    new = ast.parse("exec(compile(ast.fix_missing_locations(adapt_memory_tree(tree)), str(ORIGINAL) + ':label-only diagnostic adapter', 'exec'), namespace)").body
    items, i = span(outer_tree, old); items[i:i + len(old)] = copy.deepcopy(new)
    undo = copy.deepcopy(outer_tree); items, i = span(undo, new); items[i:i + len(new)] = copy.deepcopy(old)
    require(canonical(undo) == canonical(original), 'whole immutable memory adapter inverse AST equality')
    outer = dict(__name__='_isolated_typed_label_routing_adapter', __file__=str(MEMORY_SOURCE), adapt_memory_tree=adapt_memory_tree)
    exec(compile(ast.fix_missing_locations(outer_tree), str(MEMORY_SOURCE) + ':single isolated compile dispatch', 'exec'), outer)
    namespace, memory_proof = outer['prepare']()
    require(label_proof and same(memory_proof['ready_source'], memory_bound), 'original memory-only proof source identity retained')
    namespace['__file__'] = str(Path(__file__).resolve())
    old_prepare, old_write = namespace['prepare'], namespace['write']
    attempts = dict(memory_ready_source=MEMORY_SOURCE,
        memory_attempt_protocol=PRIVATE/'audit_ready/PROTOCOL.json', memory_attempt_started=PRIVATE/'audit_ready/STARTED.json',
        memory_attempt_checks=PRIVATE/'audit_ready/CHECKS.json', memory_attempt_partial_rows=PRIVATE/'audit_ready/ROWS.jsonl.gz')

    def labels_prepare(args):
        f, metrics, prior_proof, ae, scalar = old_prepare(args)
        source_proof = producer_proof(args.repo); bound = {k:binding(v) for k,v in attempts.items()}
        protocol, started, failed = [json.loads(attempts[key].read_bytes()) for key in ('memory_attempt_protocol','memory_attempt_started','memory_attempt_checks')]
        require(protocol['schema'] == 'saved_point_closure_routing_protocol_v22'
            and failed['schema'] == 'saved_point_closure_routing_checks_v22' and failed['complete'] is failed['passed'] is False
            and failed['exception']['type'] == 'ValueError' and failed['exception']['message'] == 'eight saved proxy labels'
            and failed['checks'] == 154 and failed['failure_count'] == 0 and failed['new_production_calls'] == 0,
            'preserved actual typed-label mismatch before numerical predicate failure')
        require(protocol['inputs'] == started['inputs'] == failed['inputs']
            and same(protocol['inputs']['code'], memory_bound) and same(failed['own_protocol'], bound['memory_attempt_protocol'])
            and same(started['protocol'], bound['memory_attempt_protocol'])
            and protocol['memory_only_review'] == started['memory_only_review'] == failed['memory_only_review'] == memory_proof,
            'exact failed memory-only source/protocol/STARTED/checks authority')
        f.update(attempts)
        f.update(producer_evaluator_source=EVALUATOR, producer_posthoc_source=Path(args.repo)/PRODUCER_RELATIVE)
        label_proof.update(source_proof, failed_memory_attempt_bindings=bound,
            first_memory_attempt_checks=154, first_memory_attempt_numerical_failures=0,
            first_memory_attempt_complete=False, first_memory_attempt_passed=False)
        return f, metrics, dict(prior_proof, diagnostic_label_review=copy.deepcopy(label_proof)), ae, scalar

    def write(path, value):
        return old_write(path, dict(value, diagnostic_label_review=copy.deepcopy(label_proof)))

    namespace.update(prepare=labels_prepare, write=write)
    return namespace, dict(memory_only_review=memory_proof, diagnostic_label_review=label_proof)

if __name__ == '__main__': prepare()[0]['main']()

"""Exercise the production postfreeze AST with invented features/mock heads.

No SOURCE pixels, metadata rows, checkpoints, prefix, GT or pretrained model.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import numpy as np
import torch

HERE = Path(__file__).absolute().parent


def own_read(name):
    path = HERE/name
    assert 'v112' not in str(path).lower() and '__pycache__' not in str(path).lower() and not name.endswith('.pyc')
    for part in reversed((path, *path.parents)):
        assert not part.is_symlink()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        return stream.read()


def production_slice(raw):
    run = next(x for x in ast.parse(raw).body if isinstance(x, ast.FunctionDef) and x.name == 'run')
    body = next(x for x in run.body if isinstance(x, ast.Try)).body
    begin = next(i for i, x in enumerate(body) if isinstance(x, ast.Assign) and
                 any(isinstance(t, ast.Attribute) and t.attr == 'final_barrier' for t in x.targets))
    end = next(i for i, x in enumerate(body) if isinstance(x, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id == 'treatment_rows' for t in x.targets))
    return body[begin:end]


def exercise(statements, directory, require_success):
    n = 7617
    index = np.arange(n, dtype=np.int64)
    float_index = index.astype(np.float32)
    expected = np.stack((float_index, -float_index), 1)
    state = {'global_barrier': False, 'source_batches': [], 'head_batches': [],
             'QA_names': [], 'DEV_checked': 0, 'new_head_calls_before_barrier': 0}

    class Fluent:
        def eval(self): return self
        def requires_grad_(self, flag): return self
        def float(self): return self
        def to(self, device): return self
        def load_state_dict(self, value, strict): assert strict

    class Control(Fluent):
        def __call__(self, x):
            return torch.stack((x[:, 0], -x[:, 0]), 1)

    class Treatment(Fluent):
        def __call__(self, x, *unused):
            if not state['global_barrier']:
                state['new_head_calls_before_barrier'] += 1
                raise AssertionError('New head reached before full global QA barrier')
            state['head_batches'].append(x[:, 0].to(torch.int64).tolist())
            return torch.stack((x[:, 0]+.25, -x[:, 0]-.25), 1), {}

    def features_for(indices, *unused):
        assert not state['global_barrier'], 'No prefix recomputation in second pass'
        state['source_batches'].append(indices.tolist())
        x = torch.zeros(len(indices), 1186, dtype=torch.float32)
        x[:, 0] = torch.from_numpy(indices.astype(np.float32))
        x[:, 1] = torch.from_numpy((indices % 11).astype(np.float32))
        return x

    def align(module, x, args):
        state['DEV_checked'] += len(x)
        return {'checked': len(x), 'max_abs': 0.}

    def bind(path):
        raw = Path(path).read_bytes()
        return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}

    def write_json(path, obj):
        name = Path(path).name
        if name == 'DEV_ACTIVE_DESCRIPTOR_QA_AFTER_FREEZE.json':
            assert obj['checked'] == state['DEV_checked'] == 1741
        elif name == 'CONTROL_ALL_SOURCE_REPLAY_QA.json':
            assert obj['rows'] == 7617 and obj['both_logits_byte_exact']
            assert obj['actual_FP32_bytes_sha256'] == obj['original_CSV_FP32_bytes_sha256']
        elif name == 'ALL_SOURCE_QA_PASSED_BEFORE_NEW_HEAD_DEV_FORWARD.json':
            assert state['QA_names'] == ['DEV_ACTIVE_DESCRIPTOR_QA_AFTER_FREEZE.json', 'CONTROL_ALL_SOURCE_REPLAY_QA.json']
            assert obj['DEV_checked'] == 1741 and obj['CONTROL_replayed'] == 7617
            assert state['new_head_calls_before_barrier'] == 0
            state['global_barrier'] = True
        else:
            raise AssertionError('Unexpected production QA file')
        state['QA_names'].append(name)
        return {'path': str(path), 'sha256': 'toy-receipt-only'}

    namespace = {
        'np': np, 'torch': torch, 'hashlib': hashlib,
        'store': types.SimpleNamespace(final_barrier=False), 'head': Treatment(),
        'crop': types.SimpleNamespace(CropFamilyHead=Control), 'control': {'head_state_dict': {}},
        'source': types.SimpleNamespace(features_for=features_for), 'model': None, 'runtime': None,
        'relational': None, 'node_args': lambda *args: (), 'alignment': align,
        'group': None, 'slot': None, 'relative': None, 'geometry': None, 'node': None, 'context': None,
        'data': {'target': np.zeros(n, np.int8), 'available': np.ones(n, bool),
                 'fit': np.r_[np.ones(5876, bool), np.zeros(1741, bool)]},
        'e': {'device': 'cpu'}, 'out': Path(directory), 'expected': expected,
        'write_json': write_json, 'safe': lambda path: path, 'binding': bind,
        'features': {}, 'freeze': {'path': 'toy-existing-final-freeze', 'sha256': 'toy-only'},
    }
    try:
        exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), '<production-postfreeze-AST-only>', 'exec'), namespace)
    except AssertionError:
        if require_success:
            raise
        assert state['new_head_calls_before_barrier'] == 1
        return {'expected_first_call_order_failure_reproduced': True}
    if not require_success:
        raise AssertionError('V1 should exhibit the ordering defect')
    assert state['global_barrier'] and state['DEV_checked'] == 1741
    assert len(state['source_batches']) == len(state['head_batches']) == 120
    assert state['source_batches'] == state['head_batches']
    assert len(namespace['scores']) == 7617
    for i, z in namespace['scores'].items():
        assert np.array_equal(z, np.array([i+.25, -i-.25], np.float32))
    return {'DEV_checked': 1741, 'CONTROL_replayed': 7617,
            'new_head_before_global_QA': 0, 'same_batch64_feature_replay_batches': 120,
            'new_prefix_work_after_global_barrier': 0,
            'saved_features_used_for_all_second_pass_logits': 7617}


def run():
    torch.set_num_threads(1)
    v1 = own_read('train_context_residual.py')
    v2 = own_read('train_context_residual_v2.py')
    def functions(raw):
        return {x.name: ast.dump(x, include_attributes=False) for x in ast.parse(raw).body
                if isinstance(x, (ast.FunctionDef, ast.ClassDef))}
    a, b = functions(v1), functions(v2)
    assert set(a) == set(b) and [k for k in a if a[k] != b[k]] == ['run']
    with tempfile.TemporaryDirectory(prefix='toy_v2_old_', dir=HERE) as first:
        negative = exercise(production_slice(v1), first, False)
    with tempfile.TemporaryDirectory(prefix='toy_v2_fixed_', dir=HERE) as second:
        positive = exercise(production_slice(v2), second, True)
    result = {'completed': True, 'PASS': True, 'fixture_groups': 3,
              'V1_code_sha256': hashlib.sha256(v1).hexdigest(), 'V2_code_sha256': hashlib.sha256(v2).hexdigest(),
              'unchanged_function_AST_count': 16, 'only_changed_function': 'run',
              'V1_expected_ordering_failure': negative, 'V2_global_QA_then_saved_features': positive,
              'production_AST_exercised': True, 'invented_numpy_torch_features_and_parameter_free_mock_heads_only': True,
              'optimizer_steps': 0, 'DINO_or_actual_source_weight_feature_RGB_prefix_reads': 0}
    with open(HERE/'TOY_V2_BARRIER_RESULT.json', 'x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    run()

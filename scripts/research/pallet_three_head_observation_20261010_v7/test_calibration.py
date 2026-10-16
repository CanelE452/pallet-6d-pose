"""Stdlib-only source-CAL provenance and original head masking-prefix checks.

This runner never imports Torch, NumPy, the model module or the v2 numerical
calibrator. It executes only the original forward's clone/mask prefix against
a small list-backed tensor fixture. This is not a full head forward. Parent
executes once after static review; failed receipts are preserved.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace

from . import calibration as C


def rejects(fn):
    try:
        fn()
    except (AssertionError, FileExistsError):
        return
    raise AssertionError('required rejection did not occur')


def metadata_fixture():
    rows, checkpoints = [], []
    for arm in C.ARMS:
        b = dict(path=arm + '.pt', sha256=C.CHECKPOINT_SHAS[arm], bytes=28068 if arm == 'IMAGE_ROLE' else 28104)
        rows.append(dict(arm=arm, steps=3000, parameters=5890, config=dict(channels=28, width=32),
                         initial_state_sha256='same_initial', batch_order_sha256='same_order',
                         protocol_sha256='same_training_protocol', repaired_target_sha256='same_targets', checkpoint=b))
        checkpoints.append(dict(arm=arm, updates=3000, checkpoint=dict(b)))
    completion = dict(complete=True, formal_updates=9000, total_updates=9000,
                      throwaway_updates=0, seed=1, batch=16, checkpoints=checkpoints)
    return dict(rows=rows), completion


def metadata_last_fixed():
    meta, completed = metadata_fixture()
    result = C.validate_metadata(meta, completed)
    assert set(result) == set(C.ARMS)
    bad = copy.deepcopy(meta); bad['rows'][0]['steps'] = 2000
    rejects(lambda: C.validate_metadata(bad, completed))
    bad = copy.deepcopy(meta); bad['rows'][1]['checkpoint'] = dict(bad['rows'][2]['checkpoint'])
    rejects(lambda: C.validate_metadata(bad, completed))
    bad = copy.deepcopy(completed); bad['formal_updates'] = 8999
    rejects(lambda: C.validate_metadata(meta, bad))
    return dict(last_steps=3000, all_three_checkpoints_required=True, altered_steps_and_arm_rejected=True)


def identical_training_provenance():
    meta, completed = metadata_fixture()
    for key in ('config', 'initial_state_sha256', 'batch_order_sha256', 'protocol_sha256', 'repaired_target_sha256'):
        bad = copy.deepcopy(meta)
        bad['rows'][1][key] = dict(channels=27) if key == 'config' else 'different'
        rejects(lambda: C.validate_metadata(bad, completed))
    return dict(equal_config_initial_order_training_protocol_targets=True, unequal_provenance_rejected=True)


def source_fixture():
    return [dict(index=i, id='fixed_' + str(i), family='family_' + str(i), partition='calibration',
                 queries=[dict(edge=q // 7) for q in range(84)]) for i in C.INDICES]


def source_partition_gate():
    good = source_fixture(); C.validate_source(good)
    bad = copy.deepcopy(good); bad[0]['partition'] = 'source_test'
    rejects(lambda: C.validate_source(bad))
    bad = copy.deepcopy(good); bad[0]['index'] = 767
    rejects(lambda: C.validate_source(bad))
    bad = copy.deepcopy(good); bad[-1]['queries'].pop()
    rejects(lambda: C.validate_source(bad))
    rejects(lambda: C.validate_source(good[:-1]))
    return dict(indices=[768, 895], families=128, queries_per_frame=84, train_and_test_rejected=True)


def source_family_gate():
    good = source_fixture()
    bad = copy.deepcopy(good); bad[-1]['family'] = bad[0]['family']
    rejects(lambda: C.validate_source(bad))
    bad = copy.deepcopy(good); bad[-1]['id'] = bad[0]['id']
    rejects(lambda: C.validate_source(bad))
    return dict(unique_families_and_ids_required=True)


def feature_row_alignment():
    source = source_fixture()
    records = [None] * 1024
    families = [None] * 1024
    for row in source:
        row['frozen_selected_points'] = [[c, c + 1] for c in range(9)]
        records[row['index']] = dict(index=row['index'], id=row['id'], family=row['family'],
             partition='calibration', selected_points=copy.deepcopy(row['frozen_selected_points']),
             variant=0, composition=dict(kind='existing_P0_original', generated=False))
        families[row['index']] = dict(id=row['id'], family=row['family'], partition='calibration')
    manifest = dict(complete=True, records=records, specs=dict(features=dict(shape=[1024, 84, 28, 65], dtype='float16')))
    split = dict(records=families, counts=dict(train=768, calibration=128, source_test=128))
    C.validate_cache_sources(source, manifest, split)
    bad = copy.deepcopy(manifest); bad['records'][768]['selected_points'][0][0] += 1
    rejects(lambda: C.validate_cache_sources(source, bad, split))
    bad = copy.deepcopy(split); bad['records'][768]['family'] = 'other'
    rejects(lambda: C.validate_cache_sources(source, manifest, bad))
    return dict(CAL_source_cache_family_alignment=True, altered_query_or_family_rejected=True)


class TensorFixture:
    shape = (1, 84, 28, 65)

    def __init__(self, values=None):
        self.values = values if values is not None else [[1000 * (c + 1) + b + 1 for b in range(65)] for c in range(28)]

    def clone(self):
        return TensorFixture(copy.deepcopy(self.values))

    def __setitem__(self, key, value):
        assert len(key) == 3 and key[0] is Ellipsis and key[2] == slice(None)
        for channel in range(*key[1].indices(28)):
            self.values[channel] = [value] * 65


def original_mask_prefix():
    model = C.REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/model.py'
    module = ast.parse(model.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'CorrespondenceHead')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    body = []
    for statement in fn.body:
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'shape' for t in statement.targets):
            break
        body.append(copy.deepcopy(statement))
    assert len(body) == 4 and [type(n).__name__ for n in body] == ['Assert', 'Assign', 'If', 'If']
    assert not any(isinstance(n, ast.Attribute) and n.attr in ('body', 'score', 'none') for s in body for n in ast.walk(s))
    prefix = ast.FunctionDef(name='prefix', args=copy.deepcopy(fn.args),
                             body=body + [ast.Return(value=ast.Name(id='x', ctx=ast.Load()))],
                             decorator_list=[])
    scope = dict(ARMS=C.ARMS)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix], type_ignores=[])), '<original-head-mask-prefix>', 'exec'), scope)
    return scope['prefix'], C.binding(model)


def mode_exact_prefix():
    prefix, model_binding = original_mask_prefix()
    original = TensorFixture(); before = copy.deepcopy(original.values)
    for arm in C.ARMS:
        masked = prefix(None, original, arm)
        assert original.values == before and masked is not original
        zero = set(range(19)) if arm == 'GEOMETRY_ONLY' else set(range(25, 28)) if arm == 'IMAGE_NO_ROLE' else set()
        for c in range(28):
            assert masked.values[c] == ([0] * 65 if c in zero else before[c])
    rejects(lambda: prefix(None, original, 'GEOMETRY_ONLY_WITHOUT_ROLE'))
    return dict(model=model_binding, prefix_clone_and_mask_only=True,
                GEO_retains_role25_28=True, NO_ROLE_retains_image0_19=True,
                original_tensor_preserved=True, actual_head_forwards=0)


def provenance_only_adaptation():
    old = dict(protocol_sha256='old', confidence=dict(enabled=True, threshold=.75),
               uncertainty=dict(query_scale=1.2), supported_edges=[0, 2, 4])
    before = copy.deepcopy(old)
    result = C.annotate_calibration(old, 'GEOMETRY_ONLY', dict(sha256='new'), dict(sha256='old'), dict(sha256='weights'))
    assert old == before and result['source_algorithm_protocol_sha256'] == 'old' and result['protocol_sha256'] == 'new'
    for key in ('confidence', 'uncertainty', 'supported_edges'):
        assert result[key] == before[key]
    rejects(lambda: C.annotate_calibration(old, 'IMAGE_ROLE', dict(sha256='new'), dict(sha256='old'), {}))
    rejects(lambda: C.annotate_calibration(old, 'IMAGE_NO_ROLE', dict(sha256='new'), dict(sha256='incorrect'), {}))
    return dict(numeric_fields_preserved=True, old_protocol_retained=True, ROLE_relabelling_rejected=True)


def byte_reuse_and_guards():
    with tempfile.TemporaryDirectory(prefix='pallet-v7-cal-contract-', dir='/dev/shm') as temp:
        root = Path(temp); source = root / 'old.bin'; source.write_bytes(b'exact\x00frozen\xffbytes')
        expected = C.binding(source); target = root / 'copy.bin'
        actual = C.copy_exact(source, target, expected)
        assert actual['sha256'] == expected['sha256'] and target.read_bytes() == source.read_bytes()
        rejects(lambda: C.copy_exact(source, target, expected))
        link = root / 'link.bin'; link.symlink_to(source)
        rejects(lambda: C.copy_exact(source, link, expected))
        directory_link = root / 'directory_link'; directory_link.symlink_to(root, target_is_directory=True)
        rejects(lambda: C.reject_symlinks(directory_link / 'new.json'))
    return dict(byte_exact=True, existing_output_and_symlink_rejected=True, new_ROLE_forwards=0)


def no_global_output_monkeypatch():
    module = ast.parse(Path(C.__file__).read_text())
    for node in ast.walk(module):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assert not any(isinstance(t, ast.Attribute) and t.attr == 'DOC' for t in targets)
    run = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    calls = [n for n in ast.walk(run) if isinstance(n, ast.Call)]
    assert sum(isinstance(n.func, ast.Attribute) and n.func.attr == 'calibrate' for n in calls) == 1
    assert any(isinstance(n.func, ast.Name) and n.func.id == 'head' and isinstance(n.args[-1], ast.Name)
               and n.args[-1].id == 'arm' for n in calls)
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in ('inputs', 'initial_geometry', 'predict', 'solve') for n in calls)
    return dict(original_algorithm_called_without_global_mutation=True,
                head_call_uses_arm_parameter=True, detector_and_pose_calls_in_run=False)


CHECKS = (metadata_last_fixed, identical_training_provenance, source_partition_gate,
          source_family_gate, feature_row_alignment, mode_exact_prefix, provenance_only_adaptation,
          byte_reuse_and_guards, no_global_output_monkeypatch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(C.DOC))
    args = parser.parse_args()
    output = Path(args.output).absolute(); C.reject_symlinks(output)
    resolved = output.resolve()
    C.require(resolved == C.DOC.resolve() or resolved.is_relative_to(C.PRIVATE.resolve()), 'test output must be new DOC/private subtree')
    names = ('SOURCE_CALIBRATION_CONTRACT_STARTED.json', 'SOURCE_CALIBRATION_CONTRACT_CHECKS.json')
    C.require(all(not (resolved / n).exists() for n in names), 'preserve test attempt')
    resolved.mkdir(parents=True, exist_ok=True)
    code = dict(test=C.binding(Path(__file__)), calibration=C.binding(Path(C.__file__)))
    C.write_new(resolved / names[0], dict(utc=datetime.now(timezone.utc).isoformat(), code=code, actual_head_calls=0))
    rows = []
    for check in CHECKS:
        try:
            rows.append(dict(name=check.__name__, passed=True, details=check()))
        except BaseException as exc:
            rows.append(dict(name=check.__name__, passed=False, error=dict(type=type(exc).__name__, message=str(exc))))
    result = dict(schema='source_CAL_and_mask_prefix_contract_checks_v7', passed=all(r['passed'] for r in rows),
                  code=code, checks=rows, check_groups=len(rows),
                  actual_model_imports=0, actual_head_calls=0, source_images_or_feature_rows_loaded=0,
                  PnP_calls=0, rays=0, training_updates=0,
                  limits='Original clone/mask prefix and scalar/list fixtures only; full head numeric parity not claimed')
    C.write_new(resolved / names[1], result)
    print('SOURCE_CALIBRATION_CONTRACT_CHECKS', result['passed'], len(rows), flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()

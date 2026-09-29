"""새 반복 어댑터가 기존 기록/비용/짝지은 계약을 보존하는지 검사한다."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from . import training as T


def decision():
    return dict(approved=True, locked_before_fit=True, kind='SAME_RECIPE_COMPLETION',
                condition='CLEAR', seed=43, why='무학습 대조 이후 누락된 반복을 완성',
                evidence=[dict(path='new/results.json', sha256='hash')])


def test_explicit_decision_and_no_hyperparameter_change():
    assert T.validate_decision(decision(), 'CLEAR_S43') == ('CLEAN_RAW_CLEAR', 'CLEAN_REF_CLEAR')
    for key, value in [('approved', False), ('locked_before_fit', False), ('seed', 44), ('evidence', []), ('lr', 1e-4)]:
        row = decision()
        row[key] = value
        with pytest.raises(AssertionError):
            T.validate_decision(row, 'CLEAR_S43')


def test_context_preserves_old_outputs_and_cumulative_budget(monkeypatch):
    calls = []
    monkeypatch.setattr(T.C, 'read', lambda p: calls.append(p) or dict(totals=dict(student_fits=6)))
    monkeypatch.setattr(T.C, 'resource', lambda name, **kw: calls.append((name, kw)))
    context = T.Context()
    assert context.read(context.DOC/'RESOURCE_LEDGER.json')['totals']['student_fits'] == 6
    assert calls[0] == T.C.DOC/'RESOURCE_LEDGER.json'
    context.resource('FIT_RAW', fits=1, updates=320)
    assert calls[1][0] == 'CLEAR_S43::FIT_RAW'
    for path in (T.C.OLD.DOC/'REPORT_KO.md', context.DOC/'RESOURCE_LEDGER.json'):
        with pytest.raises(AssertionError):
            context.save(path, {})
    with pytest.raises(AssertionError):
        T.Context('../OLD')


def test_scoped_module_restore_after_failure():
    initial = object()
    modules = [SimpleNamespace(C=initial) for _ in range(4)]
    context = T.Context()
    with pytest.raises(RuntimeError):
        with T.scoped_context(context, modules):
            assert all(m.C is context for m in modules)
            raise RuntimeError('test failure')
    assert all(m.C is initial for m in modules)


def traces(seed, condition):
    rows = []
    for i in range(8):
        real = dict(name='real.png', role='REAL', recording='recording',
                    plan=dict(seed=seed, scheduled=True, applied=True),
                    geometric_demotions=0, before_image=f'base{seed}',
                    after_image=f'masked{seed}' if condition == 'OCC' else f'base{seed}', applied=condition == 'OCC')
        syn = dict(name='syn__x.png', role='SOURCE', recording='source', plan=None,
                   geometric_demotions=0, before_image=f'source{seed}', after_image=f'source{seed}', applied=False)
        names = [real['name'], syn['name']] if seed == 42 else [syn['name'], real['name']]
        rows.append(dict(names=names, before_images=[real['before_image'], syn['before_image']],
            images=f'images{seed}{condition}', boxes='box', support='support', coordinates='coord',
            batch_idx='index', roles='roles', transfer=[real, syn]))
    return rows


def test_clear_occ_only_changes_applied_real_RGB():
    clear, occ = traces(43, 'CLEAR'), traces(43, 'OCC')
    result = T.cross_condition_checks(clear, occ)
    assert result['passed'] and result['applied_real_occurrences'] == 8
    for key in ('coordinates', 'support', 'boxes', 'before_images'):
        changed = deepcopy(occ)
        changed[0][key] = 'changed'
        with pytest.raises(AssertionError):
            T.cross_condition_checks(clear, changed)


def test_actual_seed_change_requires_same_identity_augmentation():
    result = T.seed_difference(traces(42, 'CLEAR'), traces(43, 'CLEAR'))
    assert result['matched_identity_base_RGB_changed'] == 1
    with pytest.raises(AssertionError):
        T.seed_difference(traces(42, 'CLEAR'), traces(42, 'CLEAR'))
    second = traces(43, 'CLEAR')
    for row in second:
        row['transfer'][0]['before_image'] = 'base42'
    with pytest.raises(AssertionError):
        T.seed_difference(traces(42, 'CLEAR'), second)

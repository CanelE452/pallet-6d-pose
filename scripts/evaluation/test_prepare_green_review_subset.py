import pytest

from scripts.evaluation.prepare_green_review_subset import evenly_spaced, select, stratified_sample


def row(i, session='s', groups=()):
    return dict(id=f'{session}__{i:06d}', image=f'{session}/rgb/{i:06d}.png',
                image_sha256=f'{session}_{i}', session=session, groups=list(groups), split='eval')


def inventory():
    return ([row(i, 'trunc', ['truncation', 'handheld_multiview']) for i in range(47)]
            + [row(i, 'handheld', ['handheld_multiview']) for i in range(100)]
            + [row(i, 'bright', ['evening_capture_candidate']) for i in range(44)]
            + [row(i, 'dusk', ['evening_capture_candidate', 'darker_dusk_candidate']) for i in range(7)])


def test_spacing_unique_stable_and_spread():
    rows = [row(i) for i in range(100)]
    chosen = evenly_spaced(rows, 10)
    assert [int(r['id'].split('__')[1]) for r in chosen] == list(range(5, 100, 10))
    assert chosen == evenly_spaced(list(reversed(rows)), 10)
    assert evenly_spaced(rows, 0) == []
    with pytest.raises(ValueError):
        evenly_spaced(rows, 101)


def test_proportional_quota():
    selected = stratified_sample([row(i, 'a') for i in range(28)] + [row(i, 'b') for i in range(16)], 23)
    assert sum(r['session'] == 'a' for r in selected) == 15
    assert sum(r['session'] == 'b' for r in selected) == 8


def test_disjoint_150_and_no_mutation():
    rows = inventory()
    import copy
    original = copy.deepcopy(rows)
    chosen = select(rows)
    assert rows == original
    assert len(chosen) == len({r['id'] for r in chosen}) == 150
    assert sum(r['selection_bucket'] == 'truncation' for r in chosen) == 47
    assert sum(r['selection_bucket'] == 'evening' for r in chosen) == 30


def test_train_excluded_and_never_silently_refilled():
    rows = inventory()
    rows[0]['split'] = 'train'
    with pytest.raises(ValueError, match='47 truncation'):
        select(rows)


def test_duplicate_rejected():
    rows = inventory()
    with pytest.raises(ValueError, match='Duplicate IDs'):
        select(rows + [rows[0]])
    rows[1]['image_sha256'] = rows[0]['image_sha256']
    with pytest.raises(ValueError, match='Duplicate image'):
        select(rows)

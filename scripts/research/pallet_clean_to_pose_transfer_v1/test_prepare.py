from pathlib import Path

import numpy as np
import pytest

from .prepare import digest, label_path, pair_summary, sample_real_names


def test_source_label_path_preserves_validation_subfolder():
    assert label_path('/tmp/dataset/images/val/a.png') == Path('/tmp/dataset/labels/val/a.txt')
    assert label_path('/tmp/dataset/images/a.png') == Path('/tmp/dataset/labels/a.txt')
    with pytest.raises(AssertionError):
        label_path('/tmp/dataset/a.png')


def test_sampling_same_membership_permutation_and_target():
    names = [f'PLASTIC__{i:03d}.png' for i in range(78)]
    a = sample_real_names(names)
    assert a == sample_real_names(list(reversed(names)))
    assert len(a) == 512 and set(a) <= set(names)
    assert a == np.random.default_rng(9021).choice(sorted(names), 512, replace=True).tolist()


def test_sampling_duplicate_ids_rejected():
    with pytest.raises(AssertionError):
        sample_real_names(['same.png','same.png'])


def test_export_pair_coordinates_only(tmp_path):
    labels = {}
    for target, x in [('RAW',.25),('REF',.3)]:
        path = tmp_path / (target+'.txt')
        points = [[x,.3,2]]*8 + [[.5,.5,1]]
        path.write_text(' '.join(map(str,[0,.5,.5,.8,.8]+np.array(points).ravel().tolist()))+'\n')
        labels[target] = {'one.png':path}
    result = pair_summary(labels,['one.png','one.png'])
    assert result['RAW']['supervised'] == 16 and result['RAW']['ignored'] == 2
    assert result['RAW']['boxes'] == result['REF']['boxes']
    assert result['RAW']['support'] == result['REF']['support']
    assert result['RAW']['coordinates'] != result['REF']['coordinates']


def test_export_pair_support_difference_fails(tmp_path):
    labels = {}
    for target, v in [('RAW',2),('REF',1)]:
        path = tmp_path / (target+'.txt')
        path.write_text(' '.join(map(str,[0,.5,.5,.8,.8]+[.2,.3,v]*9))+'\n')
        labels[target] = {'one.png':path}
    with pytest.raises(AssertionError,match='support'):
        pair_summary(labels,['one.png'])


def test_digest_key_order_stable():
    assert digest(dict(a=1,b=2)) == digest(dict(b=2,a=1))

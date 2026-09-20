import numpy as np
from .stability_review import restored, order


def test_inverse_coordinates_and_missing():
    p=dict(selected_index=0,candidates=[dict(keypoints_xy=[[10.,20.]]*9)])
    m=np.diag([2.,2.,1.])
    np.testing.assert_array_equal(restored(p,m),[[5.,10.]]*9)
    assert p['candidates'][0]['keypoints_xy']==[[10.,20.]]*9
    assert restored(dict(selected_index=None,candidates=[]),m) is None


def test_rank_suspect_first_not_quality_label():
    rows=[dict(id='a',score=.1),dict(id='c',score=None),dict(id='b',score=.2)]
    assert [r['id'] for r in order(rows)]==['c','b','a']
    assert all('accepted' not in r for r in order(rows))

import numpy as np
from scripts.evaluation.green_saved_labels_v1 import annotation_arrays


def test_manual_mask_excludes_projected_without_altering_coordinates():
    entries = [dict(xy=[i, i], source='manual_click' if i < 4 else 'pnp_projected', visibility=2) for i in range(9)]
    doc = dict(objects=[dict(keypoint_annotations=entries)])
    gt, manual = annotation_arrays(doc, True)
    gt2, all_known = annotation_arrays(doc)
    assert manual.sum() == 4 and all_known.sum() == 9
    np.testing.assert_array_equal(gt, gt2)


def test_unknown_and_sentinel_not_scored():
    entries = [dict(xy=[i, i], source='manual_click', visibility=2) for i in range(9)]
    entries[0]['xy'] = None
    entries[1]['xy'] = [-1, -1]
    entries[2]['visibility'] = 0
    entries[3]['xy'] = [float('nan'), 2]
    _, valid = annotation_arrays(dict(objects=[dict(keypoint_annotations=entries)]), True)
    assert valid.tolist() == [False] * 4 + [True] * 5

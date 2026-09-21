import copy
import inspect
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts.research.pallet_occlusion_refiner_transfer_v1 import run as R


def test_no_gt_inference_interface():
    assert list(inspect.signature(R.N.C.predict).parameters)==['model','bgr','prediction','cap_fraction']
    p={'candidates':[], 'selected_index':None, 'gt':'must not pass'}
    assert 'gt' not in R.predictions_only(p)


def test_transform_roundtrip():
    matrix=R.N.C.axis_aligned_crop_matrix(np.array([-20,13,640,479]))
    points=np.arange(18,dtype=float).reshape(9,2)*29
    q=R.N.C.transform_points(points,matrix)
    np.testing.assert_allclose(R.N.C.transform_points(q,np.linalg.inv(matrix)),points,atol=1e-10)


def test_oracle_single_whole_object():
    rows={'A':{'errors':[0,50],'frame_mean_px':25},'B':{'errors':[50,0],'frame_mean_px':25}}
    name=R.best_candidate(rows)
    assert name=='A'
    assert sum(x<=10 for x in rows[name]['errors'])==1  # never combine A and B corners


def test_detection_failure_denominator():
    gt=np.arange(18,dtype=float).reshape(9,2)
    row=R.M.measure(np.zeros((9,2)),gt,np.ones(9,bool),[list(range(9))],[480,640],False,False)
    assert row['corners']==8 and row['errors']==[800.]*8
    assert not row['matched']


def test_whole_object_symmetry():
    gt=np.array([[0,0],[100,0],[100,100],[0,100],[0,200],[100,200],[100,300],[0,300],[50,150.]])
    perms=[list(range(9)),[1,0,3,2,5,4,7,6,8]]
    bad=gt.copy();bad[:2]=bad[[1,0]]
    r=R.M.measure(bad,gt,np.ones(9,bool),perms,[480,640])
    assert sum(r['errors'])>0


def test_completed_artifact_protection():
    with tempfile.TemporaryDirectory() as folder:
        dest=Path(folder)/'file.json'
        with patch.object(R,'DOC',Path(folder)):
            R.freeze(dest,{'x':1});R.freeze(dest,{'x':1})
            with pytest.raises(AssertionError):R.freeze(dest,{'x':2})
        assert R.read(dest)=={'x':1}


def test_unverified_clean_not_eligible():
    pool=R.read(R.DOC/'E1_PSEUDOLABEL_POOL.json')
    assert len({r['image']['sha256'] for r in pool['records']})==len(pool['records'])
    for r in pool['records']:
        if r['eligible']:assert r['verified_clean'] and not r['overlap']
        if not r['verified_conditions']:assert not r['eligible']


def test_self_occlusion_preserves_visible_and_center():
    assert all(R.S.tests().values())

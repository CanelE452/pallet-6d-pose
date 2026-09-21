"""Pilot pretraining integrity checks; actual arm-update parity checked after fits."""
import inspect
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from scripts.research.pallet_occlusion_refiner_transfer_v2 import run as C
from scripts.research.pallet_occlusion_refiner_transfer_v2 import pilot as P


def test_no_gt_inference_contract():
    assert list(inspect.signature(C.N.C.predict).parameters)==['model','bgr','prediction','cap_fraction']
    lock=C.read(C.DOC/'E2_INPUT_LOCK.json')
    assert all('annotation' not in r for r in lock['records'])


def test_coordinate_and_common_target_mask_parity():
    lock=C.read(C.DOC/'E2_INPUT_LOCK.json')
    for b in lock['paired_files']:
        C.verify(b);data=torch.load(C.ROOT/b['path'],map_location='cpu',weights_only=False)
        clean,occ=data['pair']['CLEAN'],data['pair']['OCC'];meta=data['metadata'];mask=np.array(meta['mask'])
        np.testing.assert_array_equal(clean['target_valid'],occ['target_valid'])
        for x in (clean,occ):
            np.testing.assert_array_equal(x['original_gt'],meta['target_original'])
            np.testing.assert_array_equal(x['target_valid'],mask)
            q=C.N.C.transform_points(x['target'],np.linalg.inv(x['matrix']))
            np.testing.assert_allclose(q[mask],np.array(meta['target_original'])[mask],atol=1e-4,rtol=0)


def test_actual_occluded_R0_not_clean_copy():
    rows=C.read(C.DOC/'E2_INPUT_LOCK.json')['records'];changed=[r for r in rows if r['plans']]
    assert changed and all(r['R0_actually_rerun'] for r in rows)
    assert all(r['clean_RGB_sha']!=r['occluded_RGB_sha'] for r in changed)
    assert all(r['input_points_max_change_px']>0 for r in changed)


def test_detection_center_confidence_preservation():
    for r in C.read(C.RAW/'PSEUDOLABEL_MANIFEST.json'):C.assert_preserved(r['raw'],r['refined'])


def test_shared_real_order_and_source_not_extra_real():
    protocol=C.read(C.DOC/'E2_PROTOCOL.json');order=torch.load(C.RAW/'REAL_ORDER.pt',weights_only=True)
    assert order.shape==(300,8) and protocol['real_exposures']==2400
    assert protocol['arms']==['A00','A01','A10','A11'] and not protocol['compute_matched']


def test_train_eval_identity_and_session_disjoint():
    role=C.read(C.DOC/'DATA_ROLE_MANIFEST.json');protocol=C.read(C.DOC/'E2_PROTOCOL.json')
    hashes={r['image']['sha256'] for r in role['train_candidates']};aliases=set(role['excluded_recording_aliases'])
    for r in protocol['eval_records']:
        assert r['image']['sha256'] not in hashes
        assert str(Path(r['image']['path']).parent.parent) not in aliases
        assert r['session'] not in protocol['teacher_sessions_excluded']


def test_no_pointwise_oracle():
    gt=np.arange(18,dtype=float).reshape(9,2)*20
    pred=gt.copy();pred[:2]=pred[[1,0]]
    out=C.V.M.measure(pred,gt,np.ones(9,bool),[list(range(9)),[1,0,3,2,5,4,7,6,8]],[480,640])
    assert sum(out['errors'])>0


def test_original_models_hashes():
    for b in C.read(C.DOC/'DATA_ROLE_MANIFEST.json')['checkpoint_bindings'].values():C.verify(b)


def test_artifact_overwrite_forbidden():
    with tempfile.TemporaryDirectory() as d:
        with patch.object(C,'DOC',Path(d)):
            p=Path(d)/'test.json';C.freeze(p,{'a':1})
            with pytest.raises(AssertionError):C.freeze(p,{'a':2})


def test_recipe_masks_and_determinism():
    assert all(P.A.tests().values())

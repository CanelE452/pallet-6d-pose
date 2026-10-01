"""Invented-only CPU checks for the ResNet18 image/grid coordinate contract."""
from __future__ import annotations

import json
import cv2
import numpy as np
import torch

import input_data as I


def run():
    torch.set_num_threads(1); cv2.setNumThreads(1)
    rng = np.random.default_rng(20261001)
    count = 0
    for h, w in ((480, 640), (480, 720), (540, 960), (560, 560), (333, 517)):
        raw = rng.integers(0, 256, (h, w, 3), np.uint8)
        prepared_image = cv2.copyMakeBorder(raw, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
        real = I.prepare(raw)
        source = I.prepare(prepared_image, True)
        assert torch.equal(real['tensor'], source['tensor'])
        assert real['tensor'].shape == (3, 384, 512) and real['tensor'].dtype == torch.float32
        points = rng.uniform(0., 1., (9, 2)) * [w, h]
        rn = I.transform_points(points, real['affine_input_to_net'])
        sn = I.transform_points(points + 100, source['affine_input_to_net'])
        np.testing.assert_allclose(rn, sn, rtol=0., atol=1e-12)
        np.testing.assert_allclose(I.transform_points(rn, real['affine_net_to_input']), points, rtol=0., atol=1e-12)
        rh, rw = source['resized_hw']; left, top, right, bottom = source['padding_ltrb']
        assert rh+top+bottom == 384 and rw+left+right == 512
        assert right-left in (0, 1) and bottom-top in (0, 1)
        expected = cv2.resize(prepared_image, (rw, rh), interpolation=cv2.INTER_LINEAR)
        expected = cv2.copyMakeBorder(expected, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128,128,128))
        expected = (expected[..., ::-1].astype(np.float32) / np.float32(255.) - I.MEAN) / I.STD
        assert np.array_equal(real['tensor'].numpy(), expected.transpose(2,0,1))
        count += 1
    # Continuous Gaussian, its support and exact boundary/missing masks.
    grid = np.array([[12.25, 17.75], [0.,0.], [127.9,95.9], [128.,95.],
                     [-.01,4.], [50.,96.], [np.nan,np.nan], [30.,20.], [60.,40.]])
    eligible = np.ones(9, bool); eligible[7] = False
    heatmaps, valid, centers, inside = I.gaussian_targets(grid*4., eligible)
    assert valid.tolist() == [True,True,True,False,False,False,False,False,True]
    assert not heatmaps[~valid].any()
    for k in np.flatnonzero(valid):
        for y,x in ((0,0),(18,12),(17,12),(95,127),(40,60)):
            dx,dy = x-grid[k,0],y-grid[k,1]
            value = np.exp(-(dx*dx+dy*dy)/8.) if abs(dx)<=6 and abs(dy)<=6 else 0.
            assert heatmaps[k,y,x] == np.float32(value)
    # Full 9-channel/full-frame denominator including an entirely invalid row.
    pred = torch.ones((2,9,96,128), requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.zeros((2,9), dtype=torch.bool); mask[0,0] = True
    loss = I.masked_mse(pred, target, mask)
    torch.testing.assert_close(loss, torch.tensor(1./36), rtol=0, atol=0)
    loss.backward()
    assert not pred.grad[1].any() and not pred.grad[0,1:].any()
    torch.testing.assert_close(pred.grad[0,0], torch.full((96,128),1./(2*9*96*128)), rtol=0, atol=0)
    # Decoder tie, quarter-pixel, boundary and predeclared inclusive threshold.
    maps = np.zeros((9,96,128), np.float32)
    maps[0,10,20] = 1.; maps[0,10,21] = .8; maps[0,9,20] = .7
    maps[1,0,0] = .1
    maps[2,30,40] = .099
    maps[3,40,60] = .5; maps[3,40,61] = .5
    d = I.decode(maps, real)
    assert np.array_equal(d['grid_points'][0], [20.25,9.75])
    assert np.array_equal(d['grid_points'][1], [0.,0.])
    assert not d['valid'][2] and np.isnan(d['points_original'][2]).all()
    assert np.array_equal(d['grid_points'][3], [60.25,40.])
    assert d['valid'].sum() == 3 and d['bbox_original'] is not None
    np.testing.assert_allclose(I.transform_points(d['points_original'], real['affine_input_to_net']),d['points_net'],rtol=0,atol=1e-12,equal_nan=True)
    empty = I.decode(np.zeros_like(maps), real)
    assert not empty['valid'].any() and np.isnan(empty['points_original']).all()
    assert empty['bbox_original'] is None and empty['bbox_net'] is None
    zero, invalid, _, _ = I.gaussian_targets(np.full((9,2), np.nan), np.zeros(9,bool))
    assert not zero.any() and not invalid.any()
    # Invented normalized source metadata: no real labels/images are opened.
    k = np.tile([.5,.5,2.],(9,1)); k[1,2]=0; k[2]=[-1.,-1.,2.]; k[3]=[3.,3.,2.]
    record = dict(prepared_shape_hw=source['input_hw'],targets=[dict(keypoints_normalized=k.tolist())])
    t = I.targets(record, source)
    assert t['counts'] == dict(total_channels=9,visible_channels=8,visible_finite_channels=7,
        outside_support_channels=1,supervised_channels=6,all_masked_frames=0,denominator_channels=9)
    assert not t['target_valid'][1:4].any()
    result = dict(PASS=True, invented_only=True, image_shapes=count,
        actual_images_read=0, actual_labels_read=0, model_forwards=0, GPU_calls=0,
        coordinate_roundtrip=True, padding_source_real_tensor_exact=True,
        continuous_gaussian=True, full_denominator_loss_and_gradient=True,
        quarter_pixel_and_threshold=True)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()

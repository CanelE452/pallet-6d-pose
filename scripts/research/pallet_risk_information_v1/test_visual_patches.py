"""Coordinate/missingness tests use analytic RGB, no model or optimizer fits."""
import unittest

import numpy as np

from .visual_patches import EDGES, original_model_queries, project_queries, sample


class VisualCoordinateContract(unittest.TestCase):
    def setUp(self):
        self.xyz = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]

    def test_original_adjacency_and_native_midpoints(self):
        queries = original_model_queries(self.xyz)
        self.assertEqual(queries.shape, (20, 3))
        np.testing.assert_array_equal(queries[:8], self.xyz)
        for index, (a, b) in enumerate(EDGES):
            np.testing.assert_array_equal(queries[index + 8],
                (np.asarray(self.xyz[a]) + np.asarray(self.xyz[b])) / 2)

    def test_native_registration_and_camera(self):
        Q = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        pose = dict(R_physical=np.eye(3).tolist(), centroid=[0, 0, 8])
        K = np.array([[10, 0, 20], [0, 11, 30], [0, 0, 1]])
        geometry = dict(native_xyz=self.xyz, physical_registration_Q=Q.tolist(), K=K.tolist())
        uv, z = project_queries(pose, geometry)
        camera = original_model_queries(self.xyz) @ Q + [0, 0, 8]
        np.testing.assert_allclose(uv, (camera @ K.T)[:, :2] / camera[:, 2:], rtol=0, atol=0)
        np.testing.assert_array_equal(z, camera[:, 2])

    def test_exact_bilinear_linear_ramp(self):
        y, x = np.mgrid[:20, :20]
        image = np.stack((x + 2 * y, x, y), axis=-1).astype(np.uint8)
        centers = np.repeat([[8.25, 9.5]], 20, axis=0)
        patches, validity, query = sample(image, centers, np.ones(20), 4.)
        offset = ((np.arange(16) + .5) / 16 - .5) * 4
        dx, dy = np.meshgrid(offset, offset)
        np.testing.assert_allclose(patches[0, 0],
            (8.25 + dx + 2 * (9.5 + dy)) / 255, rtol=0, atol=3e-8)
        np.testing.assert_array_equal(validity, 1)
        self.assertTrue(query.all())

    def test_outside_queries_not_clipped_to_border(self):
        image = np.full((8, 8, 3), 255, np.uint8)
        centers = np.repeat([[-10., 4.]], 20, axis=0)
        patches, validity, _ = sample(image, centers, np.ones(20), 2.)
        self.assertFalse(patches.any())
        self.assertFalse(validity.any())

    def test_partial_canvas_mask_and_bad_depth(self):
        image = np.full((8, 8, 3), 255, np.uint8)
        centers = np.repeat([[0., 4.]], 20, axis=0)
        depths = np.ones(20)
        depths[1] = -1
        patches, validity, query = sample(image, centers, depths, 2.)
        self.assertEqual(validity[0], .5)
        self.assertFalse(query[1])
        self.assertFalse(patches[1].any())
        np.testing.assert_array_equal(patches[:, 0], patches[:, 3])

    def test_missing_box_is_absent_not_perfect(self):
        image = np.full((8, 8, 3), 255, np.uint8)
        patches, validity, query = sample(image, np.ones((20, 2)), np.ones(20), None)
        self.assertFalse(patches.any())
        self.assertFalse(validity.any())
        self.assertFalse(query.any())


if __name__ == "__main__":
    unittest.main()

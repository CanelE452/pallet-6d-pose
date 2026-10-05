import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit
import run_pipeline


class PlanAndGateTests(unittest.TestCase):
    def rows(self,times):
        return [{'camera_sensor_timestamp_ms':t} for t in times]

    def test_midpoint_tie_earlier_and_repeats_are_time_fixed(self):
        picked = audit.choose_samples(self.rows(list(range(61))),30)
        self.assertEqual([p['saved_frame_index'] for p in picked],list(range(1,61,2)))
        self.assertEqual([p['bin_index'] for p in picked if p['repeat_review']],[4,9,14,19,24,29])
        self.assertEqual(audit.choose_samples(self.rows([0,2]),1)[0]['saved_frame_index'],0)

    def test_duplicate_selection_keeps_first_no_neighbour_substitution(self):
        selected = audit.choose_samples(self.rows([0,0,1,1]),30)
        self.assertEqual([s['saved_frame_index'] for s in selected],[0,2])

    def test_backward_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.choose_samples(self.rows([0,2,1]))

    def test_frozen_plan_drift_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan.json'
            audit.dump(p,{'a':1},frozen=True)
            audit.dump(p,{'a':1},frozen=True)
            with self.assertRaises(RuntimeError):
                audit.dump(p,{'a':2},frozen=True)

    def test_real_plan_complete_indices_and_image_hashes(self):
        plan=json.loads((audit.OUT/'LIFTER_EVALUATION_PLAN.json').read_text('utf-8'))
        manifest=json.loads((audit.OUT/'review/MANIFEST.json').read_text('utf-8'))
        self.assertEqual(plan['stored_frame_count'],8910)
        self.assertEqual(len({f['frame_id'] for f in plan['frames']}),8910)
        self.assertEqual(len(manifest['frames']),120)
        self.assertEqual(sum(f['repeat_review'] for f in manifest['frames']),24)
        self.assertEqual(manifest['plan_sha256'],audit.sha(audit.OUT/'LIFTER_EVALUATION_PLAN.json'))
        for f in manifest['frames']:
            self.assertEqual(f['image_sha256'],audit.sha(audit.OUT/'review'/f['image_path']))
            self.assertEqual(f['frame_i'],f['saved_frame_index']+36)

    def test_missing_fixed_assets_block_before_model_load(self):
        with tempfile.TemporaryDirectory() as d:
            _,lr=audit.modules()
            class CheckpointsOnly:
                LEGACY_BINDINGS = [entry for entry in lr.LEGACY_BINDINGS
                                   if entry[0] in run_pipeline.ASSET_NAMES]
            isolated=Path(d)/'isolated_checkout'
            with mock.patch.object(run_pipeline,'ISO',isolated):
                records=run_pipeline.stage_verified_assets(CheckpointsOnly,Path(d)/'empty_assets')
            self.assertEqual({r['name'] for r in records
                              if r['status']=='BLOCKED_CONTRACT'},
                             {'n3_checkpoint','yolo_r0'})
            self.assertFalse(any(isolated.rglob('*.pt')))


if __name__=='__main__':
    unittest.main()

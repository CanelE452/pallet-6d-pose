import json
from pathlib import Path
import subprocess
import sys
import unittest

from . import calibration as K
from . import common as C


class CalibrationTests(unittest.TestCase):
    def test_fixed_generators(self):
        self.assertEqual(K.ARMS,('CLEAN_RAW_CLEAR','CLEAN_REF_CLEAR'))

    def test_existing_fixed_source_population(self):
        split=C.read(K.OLD_SOURCE/'SYNTHETIC_SPLIT_LOCK.json')
        rows=K.select_rows(C.read(C.ROOT/split['inputs']['path']))
        self.assertEqual(len(rows),5120)
        self.assertNotIn('TEST',{row['split'] for row in rows})

    def test_guard_blocks_real_outcomes(self):
        code='from scripts.research.pallet_clean_pose_minimal_v1.calibration import guard;guard(True);open("/tmp/CONTROL_RESULTS.json")'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0); self.assertIn('CALIBRATION_READ_DENIED',result.stderr)

    def test_generation_guard_blocks_synthetic_labels(self):
        code='from scripts.research.pallet_clean_pose_minimal_v1.calibration import guard;guard();open("/tmp/SYNTH_LABELS.npz")'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0); self.assertIn('CALIBRATION_READ_DENIED',result.stderr)

    def test_new_outputs(self):
        for path in (K.PROTOCOL,K.FEATURE_LOCK,K.LABEL_LOCK,K.FIT):
            self.assertTrue(path.is_relative_to(C.DOC))
        self.assertTrue(K.PRIVATE.is_relative_to(C.RAW))

    def test_fit_is_single_linear_recipe(self):
        source=Path(K.__file__).read_text()
        self.assertIn('M.fit(x,y,train,val,linear=True)',source)
        self.assertIn("assert not attempt.exists()",source)
        self.assertNotIn('M.fit(x,y,train,val,linear=False)',source)


if __name__=='__main__': unittest.main()

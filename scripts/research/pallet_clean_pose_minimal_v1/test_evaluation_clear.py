import unittest
import subprocess
import sys
import tempfile
from pathlib import Path
from . import evaluation_clear as E
from . import common as C


class ClearEvaluationTests(unittest.TestCase):
    def test_stage_and_two_clear_arms(self):
        self.assertEqual(E.CTX.stage,'CLEAR_S43')
        self.assertEqual(E.ARMS,('CLEAN_RAW_CLEAR','CLEAN_REF_CLEAR'))
        self.assertTrue(E.PRIVATE.is_relative_to(C.RAW))

    def test_all_three_selectors_get_matched_contrasts(self):
        pairs=E.pairs();self.assertEqual(len(pairs),len(set(pairs)))
        for selector in ('D9','GEO','NEWGEO'):
            self.assertIn(('RAW_CLEAR_S43_'+selector,'REF_CLEAR_S43_'+selector),pairs)
            self.assertIn(('REF_CLEAR_S43_'+selector,'REF_OCC_S43_'+selector),pairs)
            self.assertIn(('RAW_CLEAR_S42_'+selector,'RAW_CLEAR_S43_'+selector),pairs)

    def test_exact_rgb_whitelist_does_not_allow_neighbor_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'data/evaluation/session/rgb';folder.mkdir(parents=True)
            rgb=folder/'approved.png';rgb.touch()
            other=folder/'unlisted.png';other.touch()
            label=folder/'POSE_METRICS.json';label.touch()
            prefix='from scripts.research.pallet_clean_pose_minimal_v1.evaluation_clear import guard_reference_reads;'
            prefix+=f'guard_reference_reads([{str(rgb)!r}]);'
            for path,allowed in [(rgb,True),(other,False),(label,False)]:
                done=subprocess.run([sys.executable,'-c',prefix+f'open({str(path)!r}).read()'],capture_output=True,text=True)
                self.assertEqual(done.returncode==0,allowed,done.stderr)
                if not allowed:self.assertIn('REFERENCE_READ_DENIED',done.stderr)


if __name__=='__main__':unittest.main()

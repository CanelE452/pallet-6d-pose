"""Meaningful invariants for the two pre-existing cap definitions and I/O split."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from . import posefix_replay as P


class PoseFixContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo=Path(__file__).resolve().parents[3]
        cls.sub=P.functions(cls.repo/P.SOURCES[1],['_inputs','cap_points','correct'],
            dict(np=np,cv2=cv2,Counter=Counter,METHODS=('SUBPIX','CVRANK'),_configuration=lambda:{'SUBPIX':P.SUBPIX}))

    def test_fixed_anchors_are_distinct_and_preserve_missing_and_center(self):
        q0=np.full((9,2),50.);q0[7]=[-1,-1];support=np.ones(9,bool);support[7]=False
        pf=q0.copy();pf[:7,0]+=30;native=pf.copy();native[:7,0]+=20
        a=self.sub['cap_points'](pf,native,100,100,support)
        b=self.sub['cap_points'](q0,native,100,100,support)
        limit=.01*np.hypot(100,100)
        self.assertTrue(np.allclose(np.linalg.norm(a[:7]-pf[:7],axis=1),limit))
        self.assertTrue(np.allclose(np.linalg.norm(b[:7]-q0[:7],axis=1),limit))
        self.assertTrue(np.all(np.linalg.norm(a[:7]-q0[:7],axis=1)>30))
        self.assertTrue(np.array_equal(a[7:],q0[7:]))
        self.assertTrue(np.array_equal(b[7:],q0[7:]))
        self.assertFalse(np.array_equal(a,b))

    def test_subpix_cannot_query_unsupported_or_outside_points(self):
        q=np.full((9,2),10.);q[0]=[-5,10];q[1]=[-1,-1]
        support=np.ones(9,bool);support[1]=False
        result,diag=self.sub['correct'](np.zeros((30,30),np.uint8),q,support,'SUBPIX')
        self.assertEqual(diag['corner_records'][0]['status'],'outside_initial')
        self.assertEqual(diag['corner_records'][1]['status'],'unsupported_prediction')
        self.assertFalse(diag['corner_records'][0]['attempted'])
        self.assertFalse(diag['corner_records'][1]['attempted'])
        self.assertTrue(np.array_equal(result[[0,1,8]],q[[0,1,8]]))

    def test_answer_reads_and_external_writes_fail_closed(self):
        with tempfile.TemporaryDirectory()as folder:
            out=Path(folder)/'owned';out.mkdir()
            with P.input_boundary(self.repo,out,{})as receipt:
                for name in ('FIXED_REFERENCES.json','TARGETS.json','ISOLATED_CAL_SCORING.json'):
                    with self.assertRaises(PermissionError):Path(folder,name).read_bytes()
                with self.assertRaises(PermissionError):Path(folder,'outside').write_text('x')
                Path(out,'scientific_output.json').write_text('{}')
            self.assertEqual(receipt['scientific_answer_reads'],0)
            self.assertEqual(receipt['forbidden_probe_denied'],6)


if __name__=='__main__':unittest.main()

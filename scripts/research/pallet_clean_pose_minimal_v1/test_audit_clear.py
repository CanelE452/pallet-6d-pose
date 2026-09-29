import unittest
import numpy as np
from .audit_clear import direct_errors


class ClearAuditTests(unittest.TestCase):
    def truth(self):return dict(order=2,R=np.eye(3),t=[0,0,1])

    def test_translation_m_to_cm(self):
        value=direct_errors(dict(available=True,centroid=[.03,.04,1],R_physical=np.eye(3)),self.truth())
        self.assertAlmostEqual(value[0],5);self.assertAlmostEqual(value[1],0)

    def test_half_turn_is_equivalent(self):
        _,r=direct_errors(dict(available=True,centroid=[0,0,1],R_physical=np.diag([-1.,1.,-1.])),self.truth())
        self.assertAlmostEqual(r,0)

    def test_quarter_turn_is_not_equivalent(self):
        _,r=direct_errors(dict(available=True,centroid=[0,0,1],R_physical=[[0,0,1],[0,1,0],[-1,0,0]]),self.truth())
        self.assertAlmostEqual(r,90)


if __name__=='__main__':unittest.main()

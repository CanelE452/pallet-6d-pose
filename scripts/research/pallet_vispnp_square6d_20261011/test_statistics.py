"""Semantic checks for binary seed aggregation, thresholds and fixed draws."""
import unittest
from . import verdict as V

class ContractTests(unittest.TestCase):
    def test_exclusive_success_and_inclusive_yaw_edges(self):
        p=dict(available=True,translation_cm=5.,rotation_deg=4.,yaw_deg=0.)
        self.assertEqual(V.indicators(p)['success_rate'],0)
        p.update(translation_cm=4.,rotation_deg=5.)
        self.assertEqual(V.indicators(p)['success_rate'],0)
        p.update(rotation_deg=45.,yaw_deg=60.)
        self.assertEqual(V.indicators(p)['confusion_rate'],0)
        p.update(rotation_deg=45.001,yaw_deg=-60.)
        self.assertEqual(V.indicators(p)['confusion_rate'],1)

    def test_binary_before_seed_mean(self):
        errors=[(4.,4.),(4.,4.),(6.,6.)]
        rates=[V.indicators(dict(available=True,translation_cm=t,rotation_deg=r,yaw_deg=0))['success_rate'] for t,r in errors]
        self.assertEqual(sum(rates)/3,2/3)
        averaged=dict(available=True,translation_cm=sum(t for t,_ in errors)/3,rotation_deg=sum(r for _,r in errors)/3,yaw_deg=0)
        self.assertEqual(V.indicators(averaged)['success_rate'],1)

    def verdict(self,confusion,success,coverage=True):
        primary=dict(coverage_complete=coverage,confusion_rate=dict(CI95=confusion,improved_seeds=3),
            success_rate=dict(CI95=success,improved_seeds=3))
        return V.evaluate({'seed_mean':{'ALL':{V.PRIMARY_CONTRAST:primary}}},phase='A1')

    def test_harm_overrides_improvement_and_zero_is_not_excluded(self):
        self.assertEqual(self.verdict([.01,.03],[.01,.05])['verdict'],'WORSENED')
        self.assertFalse(self.verdict([-.05,-.01],[-.03,-.01])['continue_to_A2'])
        self.assertEqual(self.verdict([-.01,0.],[0.,.01])['verdict'],'UNRESOLVED')
        self.assertEqual(self.verdict([-.04,-.01],[-.01,.03])['verdict'],'SUPPORTED')

    def test_missing_pose_cannot_be_confusion_recovery(self):
        self.assertEqual(self.verdict([-.04,-.01],[.01,.03],False)['verdict'],'NOT_ESTIMABLE')
        self.assertFalse(self.verdict([-.04,-.01],[.01,.03],False)['continue_to_A2'])

    def test_exact_existing_real_draw_digest(self):
        from .statistics import Draws
        self.assertEqual(Draws(list(range(13)),'cluster').sha256,
            '63e288a51d7b0612616beefac28fcc625e76e8c5d7b5a0ecc5e8b85c73048fa5')

if __name__=='__main__':unittest.main()

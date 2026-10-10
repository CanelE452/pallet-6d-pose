"""Test semantic boundaries that affect scientific gates and denominators."""
import unittest
import numpy as np
from . import verdict as V
from .statistics import Draws, average_vectors, vectors, auc


def primary(conf=(-.02,.01), success=(-.01,.01), delta=-.003, seeds=2):
    return {'confusion_rate':{'CI95':list(conf),'delta':delta,'improved_seeds':seeds},
            'success_rate':{'CI95':list(success),'delta':0.,'improved_seeds':0}}


class ContractTests(unittest.TestCase):
    def test_binary_before_seed_average(self):
        poses=[dict(available=True,translation_cm=t,rotation_deg=r,yaw_deg=0,
                    ADDsym_m=.1,IoU3D=.5) for t,r in ((4,8),(6,2),(4,2))]
        values=average_vectors([vectors([pose]) for pose in poses])
        self.assertAlmostEqual(values['success_rate'][0],1/3)
        self.assertLess(values['T_cm'][0],5);self.assertLess(values['R_deg'][0],5)

    def test_fixed_thirteen_session_draw(self):
        draw=Draws([str(i) for i in range(13)],'cluster')
        self.assertEqual(draw.sha256,V.definitions()['real_cluster_draw_sha256'])

    def test_auc_ties(self):
        self.assertEqual(auc([0,0,0,0],[0,1,0,1])['AUC'],.5)
        self.assertEqual(auc([0,1,2,3],[0,0,1,1])['AUC'],1.)

    def test_success_gain_alone_is_not_supported(self):
        self.assertEqual(V.evaluate(primary(),primary(conf=(-.01,.02),success=(.01,.03)))['verdict'],'UNRESOLVED')

    def test_supported_requires_synthetic_seed_agreement(self):
        real=primary(conf=(-.02,-.001))
        self.assertEqual(V.evaluate(primary(seeds=1),real)['verdict'],'UNRESOLVED')
        self.assertEqual(V.evaluate(primary(seeds=2),real)['verdict'],'SUPPORTED')

    def test_synthetic_harm_stops_real(self):
        self.assertFalse(V.synth_gate(primary(conf=(.001,.02)))['run_real'])
        self.assertFalse(V.synth_gate(primary(success=(-.02,-.001)))['run_real'])
        self.assertTrue(V.synth_gate(primary(conf=(0,.02),success=(-.02,0)))['run_real'])

    def test_s3_is_feasibility_only(self):
        self.assertEqual(V.evaluate(primary(),primary(),rule='S3')['verdict'],'FEASIBILITY_ONLY')

    def test_missing_pose_descriptive_binary_false(self):
        self.assertEqual(V.indicators({'available':False})['success_rate'],0.)
        self.assertEqual(V.indicators({'available':False})['confusion_rate'],0.)


if __name__=='__main__':unittest.main()

import unittest
from .audit_calibration import validate_curve


class AuditCalibrationTests(unittest.TestCase):
    def test_earliest_best_with_exact_patience(self):
        curve=[dict(epoch=i+1,val_accuracy=.6) for i in range(6)]
        self.assertEqual(validate_curve(dict(curve=curve,epochs=6,best_epoch=1,best_val_accuracy=.6),32),192)

    def test_reject_later_tied_checkpoint(self):
        curve=[dict(epoch=i+1,val_accuracy=.6) for i in range(6)]
        with self.assertRaises(AssertionError):validate_curve(dict(curve=curve,epochs=6,best_epoch=2,best_val_accuracy=.6),32)

    def test_reject_unjustified_early_stop(self):
        curve=[dict(epoch=i+1,val_accuracy=.6) for i in range(4)]
        with self.assertRaises(AssertionError):validate_curve(dict(curve=curve,epochs=4,best_epoch=1,best_val_accuracy=.6),32)


if __name__=='__main__':unittest.main()

import unittest
from . import calibration_eval as E
from . import common as C


class CalibrationEvalTests(unittest.TestCase):
    def test_every_existing_base_receives_same_scorer(self):
        self.assertEqual(len(E.MODELS),8)
        self.assertEqual(set(E.MODELS),{'R0','OLD_REF','RAW_CLEAR_S42','REF_CLEAR_S42',
            'RAW_OCC_S42','REF_OCC_S42','RAW_OCC_S43','REF_OCC_S43'})

    def test_unique_contrasts_and_no_fabricated_clear_seed(self):
        pairs=E.contrasts();self.assertEqual(len(pairs),len(set(pairs)))
        self.assertTrue(all('CLEAR_S43' not in before+after for before,after in pairs))
        self.assertIn(('OLD_REF_GEO','R0_NEWGEO'),pairs)
        self.assertIn(('RAW_CLEAR_S42_NEWGEO','REF_CLEAR_S42_NEWGEO'),pairs)

    def test_new_outputs_only(self):
        self.assertTrue(E.PRIVATE.is_relative_to(C.RAW))
        self.assertTrue(E.LOCK.is_relative_to(C.DOC))
        self.assertTrue(E.RESULT.is_relative_to(C.DOC))


if __name__=='__main__':unittest.main()

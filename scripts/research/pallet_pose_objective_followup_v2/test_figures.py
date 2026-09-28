import unittest
from .figures import direction,select_examples


class FigureSelectionTests(unittest.TestCase):
    def test_joint_direction(self):
        self.assertEqual(direction(-1,-2),'improved');self.assertEqual(direction(1,2),'worsened')
        self.assertEqual(direction(-1,2),'tradeoff_or_single_axis')
        self.assertEqual(direction(0,1),'tradeoff_or_single_axis');self.assertEqual(direction(1e-8,-1e-8),'unchanged')

    def test_no_unapproved_fallback_and_symmetric_ranking(self):
        ids=['a','b','c','d'];before={i:dict(available=True,translation_cm=5.,rotation_deg=5.) for i in ids}
        after={i:dict(available=True,translation_cm=t,rotation_deg=r) for i,t,r in [('a',4.,4.),('b',6.,6.),('c',.1,.1),('d',5.,6.)]}
        result,counts=select_examples({'a','b','d'},ids,before,after)
        self.assertEqual([r.get('id') for r in result],['a','b',None])
        self.assertEqual(result[-1]['status'],'NA');self.assertEqual(counts['tradeoff_or_single_axis'],1)


if __name__=='__main__':unittest.main()

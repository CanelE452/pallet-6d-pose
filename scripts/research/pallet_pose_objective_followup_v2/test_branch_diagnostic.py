import unittest
from .branch_diagnostic import decompose


class BranchDecompositionTest(unittest.TestCase):
    def test_telescope(self):
        self.assertEqual(decompose(7., 9., 4.), dict(within_baseline_branch=2., branch_switch=-5., total=-3.))

    def test_same_branch(self):
        self.assertEqual(decompose(7., 4., 4.)['branch_switch'], 0.)


if __name__ == '__main__':
    unittest.main()

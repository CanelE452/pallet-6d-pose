import unittest
import torch
from .probe import stable_sign, vector_override


class DirectionContracts(unittest.TestCase):
    def test_sign_requires_all_epsilon_agreement(self):
        self.assertEqual(stable_sign([.1,.2,.15]),'TOWARD')
        self.assertEqual(stable_sign([-.1,-.2,-.15]),'AWAY')
        self.assertEqual(stable_sign([0.,1e-5,-1e-5]),'NEAR_ZERO')
        self.assertEqual(stable_sign([-.1,.1,.01]),'EPS_SENSITIVE')

    def test_override_leaves_original_unchanged(self):
        p=torch.nn.Parameter(torch.ones(2)); q=torch.nn.Parameter(torch.ones(1)*2)
        result=vector_override(['p','q'],[p,q],torch.tensor([1.,2.,3.]),.1)
        self.assertTrue(torch.equal(p,torch.ones(2)))
        self.assertTrue(torch.allclose(result['p'],torch.tensor([1.1,1.2])))
        self.assertTrue(torch.allclose(result['q'],torch.tensor([2.3])))


if __name__=='__main__':unittest.main()
def test_fixture_alias_uses_matching_label_stem():
    from .probe import fixture_image_name
    assert fixture_image_name({'image': {'path': 'source/G__f1.png'},
                               'label': {'path': 'labels/syn__G38__G__f1.txt'}}) == 'syn__G38__G__f1.png'

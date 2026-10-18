import unittest
from . import independent_training as I

class TrainingContract(unittest.TestCase):
    def test_GO_before_loading(self):
        with self.assertRaises(PermissionError):I.run('/not/input','/not/output')
    def test_epoch_order_reproducible_and_renaming_independent(self):
        self.assertEqual(I.expected_orders(20261018,3),I.expected_orders(20261018,3))
        self.assertNotEqual(I.expected_orders(20261018,1),I.expected_orders(20261019,1))
        self.assertNotEqual(*I.expected_orders(20261018,2))
    def test_stop_strict_first_minimum(self):
        h=[dict(epoch=i+1,validation=dict(loss=v))for i,v in enumerate([2.,1.,1.,1.,1.,1.,1.])]
        self.assertTrue(I.stop_contract(h))
        self.assertFalse(I.stop_contract(h[:-1]))
        self.assertFalse(I.stop_contract(h+[dict(epoch=8,validation=dict(loss=.5))]))
    def test_initialization_hashes_match_architecture_without_forward(self):
        import torch
        from .risk_models import RiskModel
        for visual in(False,True):
            names=('V_ORDER_NULL','V_ORDER_RGB')if visual else('R_ORDER_F1_ZERO','R_ORDER_F1')
            for arm in names:
                digest,count=I.initialization(20261018,visual=visual)
                torch.manual_seed(20261018);model=RiskModel(arm,288)
                import hashlib
                h=hashlib.sha256()
                for key,tensor in model.state_dict().items():h.update(key.encode());h.update(tensor.numpy().tobytes())
                self.assertEqual(digest,h.hexdigest());self.assertEqual(count,sum(x.numel()for x in model.parameters()))

if __name__=='__main__':unittest.main()

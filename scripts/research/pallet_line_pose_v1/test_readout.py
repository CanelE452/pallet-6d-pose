"""Generated CPU data only: frozen-head parity and synthetic selection math."""
import inspect
import unittest

import numpy as np
import torch

from model import PalletLinePoseHead
from readout import readout, prepare_readout, apply_readout
from select_synthetic import frame_metrics, summarize, choose_temperature, choose_rule


torch.set_num_threads(1)


def fixture():
    p = torch.tensor([[[40., 40.], [120., 40.], [120., 100.], [40., 100.],
                       [60., 20.], [140., 20.], [140., 80.], [60., 80.], [90., 60.]]])
    return p, torch.tensor([[30., 10., 150., 110.]]), torch.ones(1, 9, dtype=torch.bool), torch.tensor([[160., 192.]])


class ReadoutTests(unittest.TestCase):
    def test_temperature_one_matches_full_head(self):
        torch.manual_seed(83)
        head = PalletLinePoseHead(4, 6, hidden=4).eval()
        p,b,v,s = fixture()
        # A second image's lines lie outside the input rectangle; coverage must
        # disable them in both paths, despite valid finite endpoints and boxes.
        p,b,v,s = p.repeat(2,1,1),b.repeat(2,1),v.repeat(2,1),s.repeat(2,1)
        p[1] += 1000
        b[1] += 1000
        with torch.no_grad():
            expected = head(torch.randn(2,4,20,24),torch.randn(2,6,10,12),p,b,v,s,lam=.25)
            actual = readout(expected["logits"],p,b,v,s,temperature=1,lam=.25)
        for name in ("candidate_h","line_valid","sample_coverage","h_mean","moment","null_probability","points"):
            torch.testing.assert_close(actual[name],expected[name],atol=0,rtol=0)
        self.assertFalse(actual["line_valid"][1].any())

    def test_lambda_zero_cap_missing_and_center(self):
        p,b,v,s = fixture()
        p[:,0] = float("nan")
        v[:,1] = False
        p[:,4] = -1
        logits = torch.linspace(-3,3,222)[None,None].expand(1,8,-1)
        q = readout(logits,p,b,v,s,temperature=4,lam=0,cap=0)["points"]
        torch.testing.assert_close(q,p,atol=0,rtol=0,equal_nan=True)
        q = readout(logits,p,b,v,s,lam=4,cap=.01)["points"]
        torch.testing.assert_close(q[:,[0,1,4,8]],p[:,[0,1,4,8]],atol=0,rtol=0,equal_nan=True)
        good = torch.tensor([2,3,5,6,7])
        self.assertLessEqual((q[:,good]-p[:,good]).norm(dim=-1).max().item(),.01001)

    def test_temperature_null_and_cap_scale(self):
        p,b,v,s = fixture()
        logits = torch.full((1,8,222),-5.)
        logits[:,:,6*17+16] = 8.
        cold = prepare_readout(logits,p,b,v,s,.5)
        hot = prepare_readout(logits,p,b,v,s,4.)
        self.assertGreater(hot["entropy_normalized"].mean().item(),cold["entropy_normalized"].mean().item())
        q = apply_readout(cold,4.,torch.tensor([.5]))["points"]
        lengths = (q[:,:8]-p[:,:8]).norm(dim=-1)
        self.assertLessEqual(lengths.max().item(),.50001)
        self.assertGreater(lengths.max().item(),.4999)
        # All-null is the same exact baseline regardless of nominal strength.
        logits.fill_(-1000)
        logits[:,:,-1] = 1000
        q = readout(logits,p,b,v,s,lam=4)["points"]
        torch.testing.assert_close(q,p,atol=0,rtol=0)

    def test_gt_free_signatures_and_invalid_arguments(self):
        for function in (readout,prepare_readout,apply_readout):
            self.assertFalse(any("gt" in name for name in inspect.signature(function).parameters))
        p,b,v,s = fixture()
        logits = torch.zeros(1,8,222)
        for temperature in (0.,-1.,float("nan")):
            with self.assertRaises(ValueError):
                readout(logits,p,b,v,s,temperature=temperature)
        with self.assertRaises(ValueError):
            readout(logits,p,b,v,s,cap=-1)


class SelectionMathTests(unittest.TestCase):
    @staticmethod
    def record(index, targets=1, visible=8):
        target = dict(keypoints_normalized=[[.1,.2,2. if k<visible else 0.] for k in range(9)])
        return dict(index=index,id=f"generated_{index}",raw_shape_hw=[300,400],targets=[target for _ in range(targets)])

    def test_all_gt_denominators_missing_unmatched_and_other_instances(self):
        records = [self.record(0),self.record(1),self.record(2,targets=2),self.record(3,visible=0)]
        gt = np.zeros((4,9,2)); q = np.zeros_like(gt)
        q[0,:,0] = 10 # gain2 ->5 original pixels; diagonal500 ->.01
        pv = np.ones((4,9),bool); gv = np.ones_like(pv)
        pv[0,0] = False # seven observed + one missing -> (1+.07)/8
        gv[1] = False
        q[1] = np.nan
        rows = frame_metrics(q,pv,gt,gv,np.array([2.,1.,1.,1.]),records,
                             [True,False,True,True],[True,False,True,True],[0,-1,1,0])
        self.assertAlmostEqual(rows[0]["score"],1.07/8)
        self.assertEqual(rows[0]["pck10_hits"],7)
        self.assertEqual(rows[1]["score"],1.)
        self.assertEqual(rows[1]["gt_corners"],8)
        self.assertEqual(rows[2]["score"],.5) # one of two GT objects is unpredicted
        self.assertEqual(rows[2]["gt_corners"],16)
        self.assertEqual(rows[3]["score"],1.)
        summary = summarize(rows)
        self.assertEqual(summary["gt_corners"],32)
        self.assertEqual(summary["observed_corners"],15)
        self.assertEqual(summary["missing_corners"],17)
        self.assertAlmostEqual(summary["pck10_all_gt"],15/32)

    def test_diagonal_cap_and_observed_only_bias_are_explicit(self):
        records = [self.record(0),self.record(1)]
        gt = np.zeros((2,9,2)); q = np.zeros_like(gt)
        q[0,:,0] = 5000
        v = np.ones((2,9),bool)
        rows = frame_metrics(q,v,gt,v,np.ones(2),records,[True,False],[True,True],[0,-1])
        self.assertEqual(rows[0]["score"],1.)
        self.assertEqual(rows[1]["score"],1.)
        self.assertEqual(summarize(rows)["frame_score_mean"],1.)

    def test_missing_sentinel_is_a_failure_even_if_cache_valid_is_true(self):
        records = [self.record(0)]
        gt = np.zeros((1,9,2)); q = np.zeros_like(gt)
        q[0,0] = -1
        v = np.ones((1,9),bool)
        row = frame_metrics(q,v,gt,v,np.ones(1),records,[True],[True],[0])[0]
        self.assertEqual(row["observed_corners"],7)
        self.assertEqual(row["missing_corners"],1)
        self.assertEqual(row["score"],1/8)

    def test_predeclared_ties(self):
        rows = [dict(temperature=t,score=3.) for t in (.5,1.,2.,4.)]
        self.assertEqual(choose_temperature(rows)["temperature"],1.)
        self.assertEqual(choose_temperature([rows[0],rows[2]])["temperature"],.5)
        rows = [dict(lam=lam,max_move_image_diagonal_fraction=cap,score=.25)
                for lam in (0.,.0625,.25,1.,4.) for cap in (None,.01)]
        rule = choose_rule(rows)
        self.assertEqual(rule["lam"],0.)
        self.assertEqual(rule["max_move_image_diagonal_fraction"],.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import unittest
import numpy as np
import torch
from scripts.research.pallet_posefix_replay_v1 import core as N
from scripts.research.pallet_posefix_replay_v1.train import combined_micro_loss,real_rows
from scripts.research.pallet_posefix_large_error_v1.train import corrupted
from scripts.research.pallet_posefix_replay_v1.decision import decide


class ReplayContracts(unittest.TestCase):
    def gate_fixture(self):
        import copy
        protocol=dict(promotion_gate=dict(source_clean_PCK10_drop_max_pp=1.,source_clean_P90_ratio_max=1.10,
            raw_good5_to_bad10_rate_max=.01,raw_PCK10_vs_N2_drop_max_pp=.5,raw_P90_vs_N2_ratio_max=1.05,
            DEV_raw_min_recovered_hard_corners=5,populations=['DEV72','GREEN150_MANUAL']))
        before=dict(PRIOR_SOURCE=dict(clean=dict(PCK10=.9,P90_px=10.)))
        fit=dict(trainability_pass=True,source_probe=dict(clean=dict(PCK10=.9,P90_px=10.)))
        base=dict(PCK={'10':.8},matched_pooled_corner8_P90_px=20.)
        raw=dict(PCK={'10':.8},matched_pooled_corner8_P90_px=20.,matched_recovery_damage=dict(damage_rate=0.,recovered=5))
        r=dict(results={ds:dict(POSEFIX_RAW=copy.deepcopy(raw),A_N2=copy.deepcopy(base)) for ds in ('DEV72','GREEN150_MANUAL')})
        return protocol,before,fit,r

    def test_gate_requires_all_source_and_real_checks(self):
        p,b,f,r=self.gate_fixture();self.assertTrue(decide(p,b,f,r)['gate_pass'])
        f['source_probe']['clean']['PCK10']=.88
        d=decide(p,b,f,r);self.assertFalse(d['gate_pass']);self.assertIn('source_clean_PCK10_delta_pp',d['failed_checks'])

    def test_gate_does_not_choose_safe_cap_or_ignore_damage(self):
        p,b,f,r=self.gate_fixture()
        r['results']['DEV72']['POSEFIX_RAW']['matched_recovery_damage']['damage_rate']=.02
        r['results']['DEV72']['POSEFIX_CAP8']=r['results']['DEV72']['A_N2']
        self.assertFalse(decide(p,b,f,r)['gate_pass'])
        r['results']['DEV72']['POSEFIX_RAW']['matched_recovery_damage']['damage_rate']=None
        self.assertFalse(decide(p,b,f,r)['gate_pass'])

    def test_L2_once_and_real_weight_unchanged(self):
        p=torch.tensor(2.,requires_grad=True)
        parts=dict(heatmap=p,coordinate=p*2,L2=p*3)
        real=sum(combined_micro_loss(parts,'real') for _ in range(4))
        source=sum(combined_micro_loss(parts,'source') for _ in range(4))
        self.assertEqual(float(real),12);self.assertEqual(float(source),6)
        (real+source).backward();self.assertEqual(float(p.grad),9)

    def test_real_order_exact_original_interleaved_rng(self):
        real=[dict(points=np.ones((9,2),np.float32)*i,valid=np.ones(9,bool),target=np.ones((9,2),np.float32)*20,
            target_valid=np.array([1,1,1,1,0,0,0,0,0],bool),matrix=np.eye(3),bbox_diagonal=100.) for i in range(9)]
        order=N.make_real_order(real);rng=np.random.default_rng(6401)
        for step in range(300):
            ii=rng.integers(0,9,8);expected=[corrupted(real[int(i)],rng) for i in ii]
            actual=real_rows(real,order,step)
            np.testing.assert_array_equal(ii,order['real_rows'][step])
            for a,b in zip(actual,expected):
                np.testing.assert_array_equal(a['points'],b['points']);np.testing.assert_array_equal(a['valid'],b['valid'])
                np.testing.assert_array_equal(a['target'],b['target'])
        before=order['real_points'].copy();other=np.random.default_rng(7103);other.normal(size=100000)
        np.testing.assert_array_equal(order['real_points'],before)


if __name__=='__main__':unittest.main()

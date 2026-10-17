from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from . import learned_head as L


def candidate(i,value,eligible=True):
    return dict(candidate_id=str(i),eligible=eligible,features=[float(value)]*31+[1.]*31,actual_pose=dict(available=True,R_physical=np.eye(3).tolist(),centroid=[float(i),0,2.]))


def frame():return dict(id='arbitrary',candidates=[candidate(0,1),candidate(1,3)],fallback=dict(available=False))


class HeadContracts(unittest.TestCase):
    def setUp(self):L.cpu_config();torch.manual_seed(20261018);self.r=frame();self.n=L.fit_normalizer([self.r]);self.model=L.make_model('L_HEAD',188).eval()
    def test_candidate_permutation_and_identity_metadata_do_not_change_scores(self):
        # 3*62+2=188; the score is frame/candidate permutation equivariant.
        x,_=L.model_inputs(self.r,self.n);r=deepcopy(self.r);r['id']='different';r['session']='forbidden metadata';r['candidates'].reverse();y,_=L.model_inputs(r,self.n)
        a=self.model(x).detach().numpy().ravel();b=self.model(y).detach().numpy().ravel()
        np.testing.assert_allclose(a,b[[1,0,2]],rtol=0,atol=1e-7)
    def test_ineligible_candidate_cannot_change_pool_or_selected_distribution(self):
        x,_=L.model_inputs(self.r,self.n);r=deepcopy(self.r);r['candidates'].append(candidate(9,1e12,False));y,indices=L.model_inputs(r,self.n)
        self.assertEqual(indices,[0,1]);self.assertTrue(torch.equal(x,y))
    def test_missing_is_different_from_actual_zero(self):
        r=frame();r['candidates'][0]['features'][0]=0.;r['candidates'][1]['features'][0]=0.;r['candidates'][1]['features'][31]=0.;n=L.fit_normalizer([r]);x,_=L.model_inputs(r,n)
        self.assertEqual(float(x[0,31]),1.);self.assertEqual(float(x[1,31]),0.)
    def test_padded_logits_have_no_target_or_softmax_mass(self):
        model=L.make_model('L_LINEAR',188);x,_=L.model_inputs(self.r,self.n);target=torch.tensor([[.5,.5,0.]]);mask=torch.ones((1,3),dtype=torch.bool)
        a=L.loss_value(model,(x[None],mask,target));pad=torch.cat([x,torch.full((3,188),1e8)],0);b=L.loss_value(model,(pad[None],torch.tensor([[True,True,True,False,False,False]]),torch.tensor([[.5,.5,0.,0.,0.,0.]])))
        self.assertTrue(torch.allclose(a,b,atol=1e-7,rtol=0))
    def test_prediction_reference_reads_fail_closed(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'SYNTH_GT_LABELS.jsonl.gz'
            with L.prediction_no_refs():
                with self.assertRaises(PermissionError):p.read_bytes()


if __name__=='__main__':unittest.main()

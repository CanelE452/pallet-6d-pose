"""Constructed scalar audits only; no feature archive/target/model reads."""
import unittest
from pathlib import Path
import tempfile
import numpy as np
from .f1_information_summary import Moments,Population,analysis_guard,sha


class InformationSummaryTests(unittest.TestCase):
    def test_masked_moments_omit_absence_and_keep_real_zero(self):
        m=Moments(2);m.add(np.array([[0.,100.],[2.,999.]]),np.array([[1,1],[1,0]],dtype=bool))
        self.assertEqual(m.scalar(0),dict(n=2,minimum=0.,maximum=2.,mean=1.,population_SD=1.))
        self.assertEqual(m.scalar(1)['mean'],100.);self.assertEqual(m.scalar(1)['n'],1)

    def test_streaming_merge_matches_population_sd(self):
        x=np.array([[0.,3.],[2.,5.],[4.,7.],[8.,9.]])
        m=Moments(2)
        for chunk in (x[:1],x[1:3],x[3:]):m.add(chunk,np.ones(chunk.shape,dtype=bool))
        np.testing.assert_allclose(m.mean,x.mean(axis=0));np.testing.assert_allclose(np.sqrt(m.m2/m.n),x.std(axis=0))

    def test_unsupported_scalar_stays_numeric_zero_validity_not_rewritten(self):
        norm=dict(mean=[0.,0.],scale=[1.,1.],valid_train_counts=[1,0]);p=Population(2,norm)
        row=dict(id='r',actions=[dict(candidate_id='N0',extra_values=[0.,12.],extra_validity=[1,1]),
            dict(candidate_id='X1',extra_values=[2.,0.],extra_validity=[1,0])])
        p.add(row,2)
        self.assertEqual(p.normalized.scalar(1),dict(n=1,minimum=0.,maximum=0.,mean=0.,population_SD=0.))
        self.assertEqual(p.raw.n.tolist(),[2,1]);self.assertEqual(p.valid_zero.tolist(),[1,0]);self.assertEqual(p.frame_present.tolist(),[1,1])
        self.assertEqual(p.frame_varied.tolist(),[1,0])

    def test_all_missing_returns_null_moments(self):
        m=Moments(1);m.add(np.array([[0.]]),np.array([[False]]))
        self.assertEqual(m.scalar(0),dict(n=0,minimum=None,maximum=None,mean=None,population_SD=None))

    def test_nonbinary_validity_and_duplicate_action_rejected(self):
        norm=dict(mean=[0.],scale=[1.],valid_train_counts=[1])
        for acts in ([dict(candidate_id='N0',extra_values=[0.],extra_validity=[2])],
            [dict(candidate_id='N0',extra_values=[0.],extra_validity=[1])]*2):
            with self.assertRaises(ValueError):Population(1,norm).add(dict(id='r',actions=acts),1)

    def test_guard_can_hash_own_output_but_rejects_unregistered_reference(self):
        with tempfile.TemporaryDirectory()as directory:
            base=Path(directory);output=base/'new';output.mkdir();forbidden=base/'reference.json';forbidden.write_text('private')
            guard=analysis_guard([],base,output)
            with guard:
                target=output/'summary.json';target.write_text('{}')
                self.assertEqual(len(sha(target)),64)
                with self.assertRaises(PermissionError):forbidden.read_text()
            self.assertEqual(len(guard.blocked),1)


if __name__=='__main__':unittest.main()

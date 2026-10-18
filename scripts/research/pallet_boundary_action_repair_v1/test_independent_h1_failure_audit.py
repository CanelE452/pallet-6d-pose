"""Constructed-data checks; never open real references or campaign payloads."""
from __future__ import annotations
import ast
from copy import deepcopy
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from . import independent_h1_failure_audit as A


def pose(x=0., angle=0.):
    c, s = math.cos(angle), math.sin(angle)
    return dict(available=True, R_physical=[[c,0.,s],[0.,1.,0.],[-s,0.,c]],centroid=[x,0.,0.])


def side(t, r, available=True):
    return dict(actual_pose_present=True, actual_pose_available=available,
        metric_present=True, metric_pose_available=available, success=False,
        metric_error=None, translation_cm=t, rotation_deg=r)


def detail(selected, baseline):
    return A.one_detail(dict(id='toy', partition='TEST', session='s', cluster_id='c'), 'h1', 'N0',
        selected, baseline, dict(selected_candidate_id='a', candidate_count=2, available_n=2),
        dict(role='OTHER', aliases=[]), dict(role='N0', aliases=['N0']),
        dict(fallback=False, status='ACTUAL_ACTION_SELECTED'), dict(fallback=False, status='BASELINE'))


class IndependentH1FailureTests(unittest.TestCase):
    def test_fifteen_declared_H1_arms(self):
        self.assertEqual(len(A.ARMS), 15)
        self.assertTrue(all('_H1_' in a for a in A.ARMS))
        self.assertEqual(len(A.POPULATIONS)*len(A.ARMS)*len(A.COMPARATORS),180)

    def test_go_required_before_any_input_read(self):
        with patch.object(A.I, 'read', side_effect=AssertionError('decoded')):
            with self.assertRaises(PermissionError):
                A.guard(Path('/missing'),Path('/missing'),Path('/missing'),Path('/missing'),False)

    def test_separate_metric_source_and_no_production_imports(self):
        source=Path(A.__file__).read_text(); tree=ast.parse(source)
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(x and any(z in x for z in ('action_data','action_models','evaluate','evaluator','all_failure_analysis'))for x in imports))
        self.assertIn('scripts.research.pallet_statewise_end_to_end_recovery_v1',imports)

    def test_exact_identity_duplicate_and_missing_are_reported(self):
        problems=[]; A.exact_identities([dict(id='a'),dict(id='a')],('id',),[('a',),('b',)],'toy',problems)
        self.assertEqual(len(problems),1); self.assertEqual(problems[0]['missing'],[['b']])
        self.assertEqual(problems[0]['duplicates'],[dict(identity=['a'],n=2)])

    def test_unique_catalog_actual_pose_resolution(self):
        feature=dict(candidates=[dict(candidate_id='a',actual_pose=pose(.1))],fallback=pose(),e0_regenerated=pose(.2))
        catalog=dict(actions=[dict(candidate_id='N0'),dict(candidate_id='a'),dict(candidate_id='E0_REGENERATED')])
        self.assertEqual([x[0]for x in A.resolve_catalog(catalog,feature)],['N0','a','E0_REGENERATED'])

    def test_virtual_none_and_missing_ID_rejected(self):
        feature=dict(candidates=[],fallback=pose(),e0_regenerated=pose(.2))
        for cid in ('H0_NONE','missing'):
            with self.assertRaises(ValueError): A.resolve_catalog(dict(actions=[dict(candidate_id=cid)]),feature)

    def test_duplicate_physical_actions_rejected(self):
        feature=dict(candidates=[dict(candidate_id='a',actual_pose=pose())],fallback=pose(),e0_regenerated=pose(.2))
        with self.assertRaises(ValueError):
            A.resolve_catalog(dict(actions=[dict(candidate_id='a'),dict(candidate_id='N0')]),feature)

    def test_same_pose_does_not_collapse_C2(self):
        self.assertFalse(A.same_pose(pose(),pose(angle=math.pi),1.))

    def test_N0_priority_with_shared_E0_aliases(self):
        p=pose(); result=A.source_identity(p,dict(N0=p,E0=p,E0_REGENERATED=p),1.)
        self.assertEqual(result,dict(role='N0',aliases=['N0','E0_REGENERATED','E0']))

    def test_raw_strict_tradeoff_counts_both(self):
        row=detail(side(20.,8.),side(10.,12.))
        self.assertTrue(row['worse_either']);self.assertTrue(row['improved_either'])
        self.assertEqual((row['T_delta'],row['R_delta']),(10.,-4.))
        summary=A.summarize([row],'P','h1','N0',192)
        self.assertEqual(summary['worseEither_n'],1);self.assertEqual(summary['improvedEither_n'],1)
        self.assertEqual(summary['all_failure_n'],1);self.assertEqual(summary['not_all_failure_n'],191)

    def test_exact_tie_is_not_reclassified(self):
        row=detail(side(20.,8.),side(20.,8.))
        self.assertTrue(row['equal_both']);self.assertFalse(row['worse_either']);self.assertFalse(row['improved_either'])

    def test_ulp_worsening_is_strict_worsening(self):
        row=detail(side(math.nextafter(20.,math.inf),8.),side(20.,8.))
        self.assertTrue(row['T_worse']);self.assertTrue(row['worse_either'])

    def test_unavailable_pose_is_retained_excluded(self):
        row=detail(side(None,None,False),side(10.,8.))
        self.assertFalse(row['both_axes_comparable']);self.assertIsNone(row['worse_either'])
        self.assertEqual(row['T_selected_exclusion_reasons'],'actual_pose_unavailable;metric_pose_unavailable;null_error')
        summary=A.summarize([row],'CAL46','h1','N0',46)
        self.assertEqual(summary['all_failure_n'],1);self.assertEqual(summary['T_excluded_n'],1)
        self.assertEqual(summary['T_delta']['n'],0);self.assertIsNone(summary['T_delta']['sum'])

    def test_nonfinite_error_and_overflow_excluded(self):
        row=detail(side(math.inf,8.),side(10.,8.));self.assertFalse(row['T_comparable'])
        self.assertIn('nonfinite_error',row['T_selected_exclusion_reasons'])
        row=detail(side(1e308,8.),side(-1e308,8.))
        self.assertFalse(row['T_comparable']);self.assertEqual(row['T_delta_exclusion_reason'],'nonfinite_subtraction')

    def test_empty_population_has_null_moments(self):
        row=A.summarize([],'TEST49','h1','N0',49)
        self.assertEqual(row['all_failure_n'],0);self.assertEqual(row['not_all_failure_n'],49)
        self.assertEqual(row['T_delta'],dict(n=0,sum=None,mean=None,min=None,max=None,aggregate_nonfinite=False))

    def test_moments_nonfinite_sum_is_explicit(self):
        row=A.moments([1e308,1e308]);self.assertTrue(row['aggregate_nonfinite']);self.assertIsNone(row['sum'])

    def test_own_pose_metric_and_strict_metric(self):
        reference=dict(R=pose()['R_physical'],t=[0.,0.,0.],order=2)
        selected,stable=A.metric_side(pose(.06),reference)
        self.assertFalse(stable['success']);self.assertEqual(selected['translation_cm'],6.)
        candidate=pose(.03);self.assertTrue(A.I.pose_metric(candidate,reference)['success'])

    def test_missing_bool_field_fails_exact_check(self):
        problems=[];A.compare(False,None,'x',dict(aggregate_abs=1e-7,relative=1e-10),problems)
        self.assertEqual(problems[0]['kind'],'EXACT')


if __name__=='__main__':unittest.main()

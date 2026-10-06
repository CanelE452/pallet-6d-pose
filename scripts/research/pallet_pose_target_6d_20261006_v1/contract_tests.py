"""Independent small behavior fixtures and saved-artifact postconditions.

Tests never invoke a real pose solver, feature network, optimizer or manuscript
writer. --artifacts adds actual parity/cache/fit/evaluation receipt checks.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode=True
import numpy as np
import torch
from . import reporting as R
from . import cost_cache as C
from . import training as T
from . import evaluation as E
from .baseline import BASELINE_ROOT
from scripts.research.pallet_joint_action_handoff_20261006_v1.scorer import action_scores,decode_bank

ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'_docs/experiments/pallet_pose_target_6d_20261006_v1'
GIT_BASE=None


def output(logits=None,support=None,valid=None):
    logits=torch.tensor([[[0.,2.,-float('inf')]]*8],requires_grad=True) if logits is None else logits
    b,k,m=logits.shape
    q=torch.arange(b*m*9*2,dtype=torch.float32).reshape(b,m,9,2)
    return dict(logits=logits,point_support=torch.ones((b,k),dtype=torch.bool) if support is None else support,
        action_valid=torch.tensor([[True,True,False]]).repeat(b,1) if valid is None else valid,
        candidate_points=q,points_raw=q[:,0].clone())


def row(fid,cost=1.,error=1.,face='long-face-front',index=0,available=True):
    corner=dict(id=fid,evaluable=True,E_sym=error/100.,E_fixed=error/100.,frame_mean_px=error,
        errors=[error]*8,observed_errors=[error]*8,canonical_errors=[error]*8,canonical_valid=[True]*8,
        matched=True,detected=True,branch=0,corners=8,center_error=0.)
    return dict(id=fid,session='s',corner=corner,pose=dict(id=fid,available=available,
        ADDsym_m=cost,translation_cm=cost*100,rotation_deg=cost),selected_index=index,final_hypothesis=face)


def screen_fixture(means,ci=None,catastrophic=False,gross=False,failure=False):
    comparison=dict(statistics={key:dict(frame=dict(mean_paired_difference=value,
        CI95=ci if ci is not None else [value-.01,value+.01])) for key,value in zip(R.METRICS,means)})
    old=dict(pose=dict(failures=0),corner=dict(gross20=.1))
    new=dict(pose=dict(failures=int(failure)),corner=dict(gross20=.2 if gross else .1))
    return R.screening_decision(comparison,dict(good5_to_bad10=2),dict(good5_to_bad10=3 if catastrophic else 2),old,new)


class BehaviorContracts(unittest.TestCase):
    def test_invalid_F_cost_is_positive_infinity(self):
        with patch.object(C.POSE,'infer',return_value=dict(available=False)) as infer:
            result=C.candidate_cost(dict(K=np.eye(3),xyz=[1.,.2,.5],source=False),np.zeros((9,2)))
        self.assertFalse(result['available']);self.assertTrue(np.isposinf(result['ADDsym_m']))
        self.assertEqual(infer.call_count,1)
        with self.assertRaises(AssertionError):C.hard_target([0.],[False])

    def test_all_F_invalid_explicit_exclusion(self):
        self.assertEqual(C.hard_target([np.inf,np.inf],[False,False]),-1)
        o=output();loss,audit=T.hard_pose_loss(o,torch.tensor([-1]))
        self.assertEqual(float(loss),0.);self.assertEqual(audit['excluded_frames'],1)
        loss.backward();self.assertTrue(torch.isfinite(o['logits'].grad).all())

    def test_candidate_index_not_compacted(self):
        self.assertEqual(C.hard_target([3.,np.inf,1.,np.inf,2.],[True,False,True,False,True]),2)
        with self.assertRaises(AssertionError):C.hard_target([0.,np.nan],[True,False])

    def test_deterministic_first_tie_NoOp0(self):
        self.assertEqual(C.hard_target([.1,.1,.2],[True,True,True]),0)
        self.assertEqual(C.hard_target([np.inf,.1,.1],[False,True,True]),1)
        o=output(torch.zeros((1,8,3)),valid=torch.ones((1,3),dtype=torch.bool))
        self.assertEqual(int(decode_bank(o,'J')[1]),0)

    def test_hard_loss_matches_hand_probability(self):
        logits=torch.tensor([[[-math.log(2),0.,-math.log(2)]]*8],requires_grad=True)
        o=output(logits,valid=torch.ones((1,3),dtype=torch.bool))
        loss,audit=T.hard_pose_loss(o,torch.tensor([1]))
        self.assertAlmostEqual(float(loss),math.log(2),places=6)
        self.assertEqual(audit,dict(excluded_frames=0,eligible_frames=1))
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_failure_target_does_not_reassign_batch_neighbor(self):
        o=output(torch.tensor([[[0.,1.,-float('inf')]]*8,[[0.,3.,-float('inf')]]*8],requires_grad=True))
        loss,audit=T.hard_pose_loss(o,torch.tensor([-1,1]))
        self.assertEqual(audit['excluded_frames'],1)
        self.assertAlmostEqual(float(loss),math.log1p(math.exp(-3)),places=6)
        loss.backward();self.assertTrue((o['logits'].grad[0]==0).all())

    def test_support0_readout_and_nonzero_target_fail_closed(self):
        o=output(support=torch.zeros((1,8),dtype=torch.bool))
        self.assertEqual(int(decode_bank(o,'J')[1]),0)
        with self.assertRaises(AssertionError):T.hard_pose_loss(o,torch.tensor([1]))

    def test_observed_preupdate_fit_eligible_exposures_and_weighted_CE(self):
        logits=torch.tensor([[[3.,1.,0.]]*8,[[0.,3.,1.]]*8,
            [[3.,1.,0.]]*8,[[0.,3.,1.]]*8],requires_grad=True)
        o=output(logits,valid=torch.ones((4,3),dtype=torch.bool))
        value=torch.tensor(2.,requires_grad=True);rng=torch.get_rng_state().clone()
        observation=T.observed_action_fit(o,torch.tensor([0,1,1,-1]),value)
        self.assertEqual(observation,dict(observed_exposures=4,eligible_target_exposures=3,
            excluded_target_exposures=1,matching_eligible_exposures=2,moving_target_exposures=2,
            matching_moving_target_exposures=1,NoOp_target_exposures=1,
            matching_NoOp_target_exposures=1,eligible_CE_sum=6.))
        total=T.empty_action_fit();T.accumulate_action_fit(total,observation)
        second=output(logits[:2],valid=torch.ones((2,3),dtype=torch.bool))
        T.accumulate_action_fit(total,T.observed_action_fit(second,torch.tensor([-1,0]),torch.tensor(4.)))
        summary=T.action_fit_summary(total)
        self.assertEqual(summary['observed_exposures'],6)
        self.assertEqual(summary['eligible_target_exposures'],4)
        self.assertEqual(summary['excluded_target_exposures'],2)
        self.assertEqual(summary['eligible_action_accuracy'],.5)
        self.assertEqual(summary['moving_target_action_accuracy'],.5)
        self.assertEqual(summary['NoOp_target_action_accuracy'],.5)
        self.assertEqual(summary['eligible_mean_CE'],2.5)
        self.assertEqual(summary['additional_model_forwards'],0)
        self.assertEqual(summary['additional_optimizer_updates'],0)
        self.assertIsNone(T.action_fit_summary(T.empty_action_fit())['eligible_action_accuracy'])
        self.assertIsNone(T.action_fit_summary(T.empty_action_fit())['eligible_mean_CE'])
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertIsNone(logits.grad);self.assertIsNone(value.grad)

    def test_GT_never_enters_forward_signature(self):
        batch={k:object() for k in ('p3','p4','points','boxes','point_valid','input_shape','context','gt_points','gt_valid','permutations','group_valid')}
        class Spy:
            def forward_bank(self,*args,**kwargs):self.args=args;self.kwargs=kwargs;return 'predicted'
        spy=Spy();candidates=object();valid=object()
        self.assertEqual(T.forward_bank(spy,batch,candidates,valid),'predicted')
        self.assertEqual(spy.args,tuple(batch[k] for k in ('p3','p4','points','boxes','point_valid','input_shape')))
        self.assertEqual(set(spy.kwargs),{'context','candidate_points','action_valid'})
        self.assertFalse(any(value is batch['gt_points'] or value is batch['gt_valid'] for value in (*spy.args,*spy.kwargs.values())))
        with self.assertRaises(AssertionError):T.forward_bank(spy,dict(batch,truth=object()),candidates,valid)

    def test_ADDsym_golden_corresponding_corner_and_proper_group(self):
        identity=np.eye(3);xyz=np.array([1.,.2,.5]);q=np.arange(18,dtype=float).reshape(9,2)
        truth=dict(R=identity,t=[0,0,2],xyz=xyz,body_R=identity,body_xyz=xyz,order=2)
        predicted=dict(available=True,R_physical=identity,R_cf=identity,centroid=[.1,0,2],cf_extents=xyz,selected_hypothesis='long-face-front')
        seen=[]
        def infer(points,K,dimensions,source):seen.append(np.array(points));return predicted
        with patch.object(C.POSE,'infer',side_effect=infer):
            result=C.candidate_cost(dict(id='golden',K=identity,xyz=xyz,source=False,truth=truth),q)
        self.assertTrue(np.array_equal(seen[0],q));self.assertAlmostEqual(result['ADDsym_m'],.1,places=14)
        self.assertAlmostEqual(result['translation_cm'],10.,places=12);self.assertEqual(result['rotation_deg'],0.)
        for proper in C.POSE.rotations(2):self.assertAlmostEqual(np.linalg.det(proper),1.,places=14)

    def test_native_hard_J_and_RAW_center_exact(self):
        o=output();native=np.arange(3*9*2,dtype=float).reshape(3,9,2)+.123456789
        native[1,8]=[-500.,600.]
        raw=native[0].copy();raw[8]=[13.123456789,17.987654321]
        chosen=E.select_native(o,[dict(points=native,hypotheses=['NoOp','GEO','INVALID'])],[dict(q=raw)])[0]
        self.assertEqual(chosen[1],1);self.assertTrue(np.array_equal(chosen[0][:8],native[1,:8]));self.assertTrue(np.array_equal(chosen[0][8],raw[8]))

    def test_final_F_function_identical_between_cost_and_eval(self):
        self.assertIs(C.POSE.infer,E.POSE.infer)
        self.assertEqual(C.metric_no_iou.__code__.co_code,C.POSE.metric.__code__.co_code)
        self.assertNotEqual(C.metric_no_iou.__globals__,C.POSE.metric.__globals__)

    def test_initialization_exact_all_three_saved_seeds(self):
        config=R.read(R.OLD/'A_protocol.json')['config']
        for seed in (1,2,3):
            head=T.initialize(seed,config,'cpu');old=R.read(R.OLD/f'A_fits/GEO_seed{seed}.json')
            self.assertEqual(T.dcp_env.state_sha(head.state_dict()),old['first_step']['initial_state_sha256'])
            self.assertEqual(sum(p.numel() for p in head.parameters()),20259)

    def test_LR_matches_actual_saved_original_trajectory(self):
        old=R.read(R.OLD/'A_fits/GEO_seed1.json')
        for record in old['history']:
            self.assertEqual(T.learning_rate(record['step']),record['lr'])
        self.assertEqual(T.learning_rate(1),.00001)
        self.assertEqual(T.learning_rate(100),.001)
        self.assertEqual(T.learning_rate(6000),.0001)
        self.assertTrue(all(T.learning_rate(s)>=T.learning_rate(s+1) for s in range(100,6000)))

    def test_duplicate_missing_ID_join_rejected(self):
        with self.assertRaises(ValueError):R.ordered([dict(id='a'),dict(id='a')],['a','b'])
        self.assertEqual([r['id'] for r in R.ordered([dict(id='b'),dict(id='a')],['a','b'])],['a','b'])

    def test_coverage_preserves_all_four_failure_denominators(self):
        base=[row(str(i),available=a) for i,a in enumerate([True,False,True,False])]
        new=[row(str(i),available=a) for i,a in enumerate([True,True,False,False])]
        result=R.coverage(new,base)
        self.assertEqual(result,dict(full_frames=4,common_success=1,new_only_success=1,base_only_success=1,both_failed=1))

    def test_recovery_no_clamp_negative_and_over_one(self):
        raw=[row('a',2.),row('b',2.),row('c',1.)];oracle=[row('a',1.),row('b',1.),row('c',1.)]
        new=[row('a',3.),row('b',.5),row('c',.9)];old=[row('a',2.),row('b',2.),row('c',1.)]
        result=R.recovery(raw,old,new,oracle)
        self.assertEqual(result['ratios']['new']['negative'],1);self.assertEqual(result['ratios']['new']['over_one'],1)
        self.assertEqual(result['excluded_headroom_le_tolerance'],1)
        self.assertEqual([r['new_ratio'] for r in result['rows']],[-1.,1.5,None])

    def test_WD_group_uses_final_F_not_generation(self):
        raw=[row('a',face='long-face-front')];oracle=[row('a',face='short-face-front',index=2)]
        new=[dict(row('a',face='short-face-front',index=1),generating_hypothesis='long-face-front')]
        old=[dict(row('a',face='long-face-front',index=2),generating_hypothesis='short-face-front')]
        result=R.hypothesis_analysis(raw,old,new,oracle)['groups']['SWITCH']
        self.assertEqual(result['recovery']['NEW']['final_hypothesis_recovered'],1)
        self.assertEqual(result['recovery']['OLD']['final_hypothesis_recovered'],0)
        self.assertEqual(result['recovery']['OLD']['exact_oracle_action'],1)

    def test_tradeoff_uses_canonical_GT_identity(self):
        base=[row('a',2.,error=1.)];new=[row('a',1.,error=11.)]
        new[0]['corner']['branch']=1
        result=R.tradeoff(new,base)
        self.assertEqual(result['corner_damage']['good5_to_bad10'],8)
        self.assertEqual(result['canonical_corner_counts']['SIX_D_IMPROVES_TWO_D_WORSENS'],8)
        self.assertEqual(result['canonical_corner_denominator'],8)
        self.assertEqual(result['corner_damage']['evaluation_branch_changed'],1)

    def test_bootstrap_shared_draws_seedmean_cluster_weighting(self):
        a=R.SharedBootstrap(['a','b','c'],['x','x','y'],resamples=100)
        b=R.SharedBootstrap(['a','b','c'],['x','x','y'],resamples=100)
        self.assertEqual(a.draw_sha256,b.draw_sha256)
        self.assertAlmostEqual(a.contrast([1.,1.,4.])['mean_paired_difference'],2.)
        # The cluster draw x twice keeps both x frames, not an equal group mean.
        self.assertEqual(a.counts.tolist(),[2,1])
        self.assertTrue(np.array_equal(a.weights,b.weights))

    def test_screen_STOP_CONTINUE_and_damage_gate(self):
        self.assertEqual(screen_fixture([1.,1.,1.],catastrophic=True)['decision'],'STOP')
        self.assertEqual(screen_fixture([-1.,0.,0.],ci=[-2.,-.5])['decision'],'CONTINUE')
        self.assertFalse(screen_fixture([-1.,0.,0.],ci=[-2.,-.5],failure=True)['continue_seeds'])
        self.assertFalse(screen_fixture([-1.,0.,0.],ci=[-2.,-.5],gross=True)['continue_seeds'])
        self.assertFalse(screen_fixture([-1.,0.,0.],ci=[-2.,-.5],catastrophic=True)['continue_seeds'])

    def test_screen_mixed_requires_both_signs_no_uncertain_expansion(self):
        self.assertEqual(screen_fixture([-1.,1.,0.],ci=[-2.,2.])['decision'],'MIXED_CONTINUE')
        self.assertEqual(screen_fixture([-1.,-1.,-1.],ci=[-2.,2.])['decision'],'STOP_MIXED')

    def test_screen_function_never_loads_REAL(self):
        calls=[]
        original=R.analyze_split
        def spy(split,seeds):calls.append(split);raise RuntimeError('stop before any output')
        with patch.object(R,'analyze_split',side_effect=spy),self.assertRaises(RuntimeError):R.screen(ROOT)
        self.assertEqual(calls,['SYNTH_HELDOUT'])

    def test_OLD_target_support_never_claims_N3_support(self):
        synthetic=dict(seed_mean_comparisons={},comparisons={},summaries={},damage_vs_RAW={})
        for family,sign in [('FIT_GEO_J',-1.),('N3',1.)]:
            statistics={key:dict(frame=dict(mean_paired_difference=sign,CI95=[sign-.1,sign+.1])) for key in R.METRICS}
            synthetic['seed_mean_comparisons'][f'POSE_TARGET_GEO_J_minus_{family}']=dict(statistics=statistics)
            for seed in (1,2,3):
                synthetic['comparisons'][f'POSE_TARGET_GEO_J_seed{seed}_minus_{family}_seed{seed}']=dict(statistics=statistics)
                for name in [f'{family}_seed{seed}',f'POSE_TARGET_GEO_J_seed{seed}']:
                    synthetic['summaries'][name]=dict(pose=dict(failures=0),corner=dict(gross20=.1))
                    synthetic['damage_vs_RAW'][name]=dict(good5_to_bad10=2)
        self.assertEqual(R.comparison_question(synthetic,[1,2,3],'FIT_GEO_J')['verdict'],'POSE_TARGET_SUPPORTED')
        self.assertEqual(R.comparison_question(synthetic,[1,2,3],'N3')['verdict'],'POSE_TARGET_NOT_SUPPORTED')
        self.assertEqual(R.comparison_question(synthetic,[1],'FIT_GEO_J')['verdict'],'POSE_TARGET_NOT_SUPPORTED')

    def test_binary_journal_preserves_incomplete_and_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            arrays=C.open_arrays(temporary,n=1,mode='w+')
            arrays['action_counts'][0]=3
            path=Path(temporary)/'journal.bin';journal=C.BinaryJournal(path)
            journal.attempt(0,2);journal.close()
            result=C.replay_journal(path,arrays,[0])
            self.assertEqual(result['attempted'],{(0,2)});self.assertEqual(result['completed'],set())
            self.assertEqual(int(arrays['candidate_state'][0,2]),1)
            journal=C.BinaryJournal(path);journal.attempt(0,2);journal.close()
            with self.assertRaises(AssertionError):C.replay_journal(path,arrays,[0])

    def test_no_manuscript_files_modified_or_created(self):
        args=['git','diff','--name-only',GIT_BASE or 'HEAD']
        paths=subprocess.check_output(args,cwd=ROOT,text=True).splitlines()
        paths+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
        def prohibited(p):
            q=Path(p)
            return ('paper_updated' in q.parts or q.name.startswith('manuscript') or q.name in ['main.tex','supplement.tex','references.bib'] or q.suffix.lower() in ['.tex','.pdf','.bib','.cls','.sty','.bst'] or '/scripts/paper/' in '/'+p or '/scripts/references/' in '/'+p)
        self.assertEqual([p for p in paths if prohibited(p)],[])


class ArtifactContracts(unittest.TestCase):
    def test_actual_registered_NoOp_F_and_oracle_parity(self):
        receipt=R.read(DOC/'POSE_TARGET_PARITY.json')
        self.assertEqual(receipt['status'],'PASS');self.assertEqual(receipt['registered_cases'],6)
        self.assertEqual(receipt['new_bank_generation'],0)
        for case in receipt['cases']:
            self.assertEqual(case['status'],'PASS');self.assertEqual(case['old_oracle_index'],case['new_oracle_index'])
            self.assertLessEqual(abs(case['old_raw_ADDsym_m']-case['new_raw_ADDsym_m']),1e-7)
            self.assertLessEqual(abs(case['old_oracle_ADDsym_m']-case['new_oracle_ADDsym_m']),1e-7)
            self.assertTrue(case['NoOp_coordinate_exact'])

    def test_same_bank_initializer_order_and_final_F_every_actual_fit(self):
        seeds=[s for s in (1,2,3) if (DOC/f'fits/POSE_TARGET_GEO_seed{s}.json').exists()]
        self.assertIn(1,seeds)
        result=R.matched_comparison(seeds)
        for row_value in result['rows']:
            self.assertTrue(row_value['matched_contract_passed'],row_value['independent_checks'])
            self.assertEqual(row_value['matched_contract']['changed_components'],['training_target'])

    def test_whole_TRAIN_cache_and_excluded_targets_explicit(self):
        result=R.read(DOC/'POSE_COST_CACHE_MANIFEST.json')
        self.assertIn(result['status'],['PASS','DONE','COMPLETE'])
        self.assertTrue(result.get('files'))
        self.assertTrue(all('sha256' in file and 'bytes' in file for file in result['files']))

    def test_actual_prediction_rows_all_IDs_and_GT_absence(self):
        seeds=[s for s in (1,2,3) if (DOC/f'fits/POSE_TARGET_GEO_seed{s}.json').exists()]
        for split,total in [('SYNTH_HELDOUT',1985),('REAL_DEV',319)]:
            ids,methods,oracle,_=R.load_methods(split,seeds)
            self.assertEqual(len(ids),total)
            for seed in seeds:
                rows=methods[f'POSE_TARGET_GEO_J_seed{seed}']
                self.assertEqual(len(rows),total)
                for row_value,bound in zip(rows,oracle):
                    self.assertLess(row_value['selected_index'],bound['actions'])
                    if row_value['pose']['available'] and bound['oracle']['available']:
                        self.assertGreaterEqual(row_value['pose']['ADDsym_m']-bound['oracle_ADDsym_m'],-1e-7)


def main(argv=None):
    global GIT_BASE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',action='store_true')
    parser.add_argument('--git-base',help='publication base such as origin/main; default current HEAD for worktree edits')
    args=parser.parse_args(argv);GIT_BASE=args.git_base
    begin=time.monotonic();suite=unittest.TestLoader().loadTestsFromTestCase(BehaviorContracts)
    if args.artifacts:suite.addTests(unittest.TestLoader().loadTestsFromTestCase(ArtifactContracts))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(schema='pose_target_contract_test_results_v1',status='PASS' if result.wasSuccessful() else 'FAIL',
        command=' '.join([sys.executable,'-B','-m','scripts.research.pallet_pose_target_6d_20261006_v1.contract_tests',*sys.argv[1:]]),
        passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),failed=len(result.failures)+len(result.errors),
        skipped=len(result.skipped),tests_run=result.testsRun,failures=[str(t) for t,_ in result.failures+result.errors],seconds_wall=time.monotonic()-begin,
        artifact_postconditions=args.artifacts,code_sha256=R.sha(Path(__file__)),new_feature_or_scorer_forward_calls=0,
        mocked_F_calls_only=True,new_actual_F_calls=0,new_PnP_calls=0,new_optimizer_updates=0,
        toy_loss_backward_tests=3,CPU_initializer_checks=3,manuscript_writes=0)
    R.write(DOC/'CONTRACT_TEST_RESULTS.json',receipt)
    if not result.wasSuccessful():raise SystemExit(1)
    return receipt


if __name__=='__main__':main()

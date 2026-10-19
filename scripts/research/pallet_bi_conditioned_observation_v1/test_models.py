"""Independent constructed model/training checks; AdamW is always mocked.

No real image, scientific label ledger, or reference file is read. Mock steps
leave parameters unchanged and their attempts live in a separate artifact root.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import resource
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import models as M
from . import training as T
from .supervision import targets_for_frame
from .test_supervision import fixture

ARTIFACT_ROOT = None
MOCK_CONSTRUCTIONS = 0
MOCK_STEPS = 0


class MockAdamW:
    """Log API calls without making any optimizer or changing any tensor."""
    def __init__(self, parameters, lr, weight_decay):
        global MOCK_CONSTRUCTIONS
        MOCK_CONSTRUCTIONS += 1
        self.parameters = list(parameters)
        self.groups = dict(lr=lr,weight_decay=weight_decay)
        self.steps = 0

    def state_dict(self):
        return dict(state={},param_groups=[dict(self.groups)],mock_steps=self.steps,
                    CONSTRUCTED_MOCK_ONLY=True)

    def load_state_dict(self,state):
        if not state['CONSTRUCTED_MOCK_ONLY']:
            raise AssertionError('Scientific optimizer checkpoint reached constructed test')
        self.steps = state['mock_steps']

    def zero_grad(self,set_to_none=True):
        for parameter in self.parameters:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)

    def step(self):
        global MOCK_STEPS
        self.steps += 1
        MOCK_STEPS += 1


def feature_corner(k=0, candidates=2, state='B',partial=False):
    values = np.zeros((candidates,M.FEATURE_DIMENSION))
    valid = np.zeros_like(values)
    values[:,0] = np.arange(candidates)
    valid[:,0] = 1
    return dict(native_corner=k,state=state,candidate_ids=[f'{k}:keep']+[f'{k}:alternative{j}' for j in range(1,candidates)],
        applied_xy=[[float(j),0.] for j in range(candidates)],x_values=values.tolist(),x_validity=valid.tolist(),
        partial_replaces_full_likelihood=partial)


def feature_frame(ident='constructed',split='train',corners=None):
    return dict(id=ident,split=split,corner_rows=corners or [feature_corner()])


def label_frame(frame):
    return dict(id=frame['id'],split=frame['split'],corner_rows=[dict(native_corner=r['native_corner'],
        candidate_ids=r['candidate_ids'],valid_target=[not r['partial_replaces_full_likelihood']]*len(r['candidate_ids']),
        target_log1p_error=[None if r['partial_replaces_full_likelihood'] else 0.]*len(r['candidate_ids']))
        for r in frame['corner_rows']])


def prepared_training():
    frames = [feature_frame('constructed_train_'+str(i)) for i in range(4)]
    val = feature_frame('constructed_validation','validation')
    norm = M.normalizer(frames)
    items = dict(train=[M.prepare_frame(r,label_frame(r),norm) for r in frames],
                 validation=[M.prepare_frame(val,label_frame(val),norm)])
    return items,norm


def artifact_folder(label):
    if ARTIFACT_ROOT is not None:
        ARTIFACT_ROOT.mkdir(parents=True,exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=label+'_',dir=ARTIFACT_ROOT))


def evaluation_schedule(items, values, interrupt_epoch=None):
    index = 0
    def evaluate(model,current):
        nonlocal index
        if current is items['train']:
            if interrupt_epoch is not None and index == interrupt_epoch-1:
                raise RuntimeError('CONSTRUCTED_INTERRUPT_AFTER_TWO_EPOCHS')
            loss = .25
        else:
            loss = values[min(index,len(values)-1)]
            index += 1
        return dict(loss=loss,supervised_frames=len(current),full_frame_denominator=len(current),
                    supervised_corners=len(current),supervised_candidates=2*len(current))
    return evaluate


class VectorScorer(nn.Module):
    def __init__(self,values):
        super().__init__();self.values=nn.Parameter(torch.tensor(values,dtype=torch.float64))
    def forward(self,x,c):
        return self.values


class ConstantScorer(nn.Module):
    def __init__(self):
        super().__init__();self.calls=0
    def forward(self,x,c):
        self.calls += 1
        return torch.ones(len(x))


class ModelContractTest(unittest.TestCase):
    def test_27585_paired_initialization_for_all_fixed_seeds(self):
        for seed in M.SEEDS:
            torch.manual_seed(seed);common=M.Scorer('COMMON')
            torch.manual_seed(seed);bi=M.Scorer('BI')
            self.assertEqual(sum(p.numel() for p in common.parameters()),27585)
            self.assertEqual(M.state_hash(common),M.state_hash(bi))
            for a,b in zip(common.parameters(),bi.parameters()):
                self.assertTrue(torch.equal(a,b))

    def test_common_is_condition_invariant_bi_uses_only_shared_condition_slots(self):
        common,bi=M.Scorer('COMMON'),M.Scorer('BI')
        with torch.no_grad():
            for p in common.parameters():p.zero_()
            common.layers[0].weight[0,-2:]=torch.tensor([1.,2.])
            common.layers[2].weight[0,0]=1.
            common.layers[4].weight[0,0]=1.
        bi.load_state_dict(common.state_dict())
        x=torch.zeros((2,M.MODEL_INPUT_DIMENSION))
        c=torch.tensor([[1.,0.],[0.,1.]])
        torch.testing.assert_close(common(x,c),common(x,c.flip(1)),rtol=0,atol=0)
        torch.testing.assert_close(common(x,c),torch.nn.functional.softplus(torch.tensor([1.5,1.5])))
        torch.testing.assert_close(bi(x,c),torch.nn.functional.softplus(torch.tensor([1.,2.])))
        self.assertTrue(torch.equal(x,torch.zeros_like(x)))

    def test_train_only_valid_normalization_distinguishes_observed_zero(self):
        corner=feature_corner()
        corner['x_values'][1][0]=4.
        corner['x_validity'][0][1]=1.
        frame=feature_frame(corners=[corner]);norm=M.normalizer([frame])
        self.assertEqual(norm['counts'][0:2],[2,1])
        self.assertEqual(norm['mean'][0:2],[2.,0.])
        self.assertEqual(norm['standard_deviation'][0:2],[2.,1.])
        x=M.normalized(corner,norm)
        self.assertEqual(x.shape,(2,362))
        np.testing.assert_array_equal(x[:,0],[-1.,1.])
        self.assertEqual(x[0,M.FEATURE_DIMENSION+1],1.)
        self.assertEqual(x[1,M.FEATURE_DIMENSION+1],0.)
        with self.assertRaisesRegex(ValueError,'TRAIN only'):
            M.normalizer([frame,feature_frame('forbidden_validation','validation')])
        broken=deepcopy(corner);broken['x_values'][0][2]=5.
        with self.assertRaisesRegex(ValueError,'Missing numeric'):
            M.normalized(broken,norm)

    def test_candidate_corner_frame_loss_and_gradient_by_hand(self):
        def corner(k,n):
            return dict(native_corner=k,candidate_ids=[str(i) for i in range(n)],
                x=np.zeros((n,362),np.float32),c=np.zeros((n,2),np.float32),
                y=np.zeros(n,np.float64),valid=np.ones(n,bool),partial=False)
        items=[dict(corners=[corner(0,3),corner(1,1)]),dict(corners=[corner(2,2)]),dict(corners=[])]
        batch=M.collate(items)
        scorer=VectorScorer([0.,1.,2.,3.,.5,.5])
        loss=M.loss_value(scorer,batch)
        self.assertAlmostEqual(float(loss),41./48.,places=14)
        loss.backward()
        np.testing.assert_allclose(scorer.values.grad.numpy(),[0.,1/12,1/12,1/4,1/8,1/8],rtol=0,atol=1e-15)
        self.assertEqual(batch['frame_count'],2)
        self.assertIsNone(M.collate([dict(corners=[])]))

    def test_fixed_partial_excluded_and_never_scored(self):
        frame=feature_frame(corners=[feature_corner(0,2,partial=True),feature_corner(1,1)])
        norm=M.normalizer([frame]);labels=label_frame(frame)
        prepared=M.prepare_frame(frame,labels,norm)
        self.assertEqual(M.collate([prepared])['candidate_count'],1)
        scorer=ConstantScorer();selected=M.choose(scorer,frame,norm)
        self.assertEqual(scorer.calls,1)
        self.assertEqual(selected[0]['selection_reason'],'FIXED_SHARED_PARTIAL_PATH')
        self.assertIsNone(selected[0]['scores'])
        labels['corner_rows'][0]['valid_target'][0]=True
        labels['corner_rows'][0]['target_log1p_error'][0]=0.
        with self.assertRaisesRegex(ValueError,'Fixed partial'):
            M.prepare_frame(frame,labels,norm)

    def test_actual_applied_target_integration_and_missing_target_guard(self):
        up,frame,gt,_=fixture(order=1)
        frame['corner_rows']=frame['corner_rows'][:1]
        f=feature_corner(0,1)
        frame['corner_rows'][0].update({k:v for k,v in f.items() if k not in ('applied_xy','candidate_ids')})
        frame['corner_rows'][0]['applied_xy'][0][0]+=6.
        frame['corner_rows'][0]['raw_intersection_xy']=[[9999.,9999.]]
        labels=targets_for_frame(up,frame,gt);norm=M.normalizer([frame])
        prepared=M.prepare_frame(frame,labels,norm)
        self.assertAlmostEqual(float(prepared['corners'][0]['y'][0]),math.log1p(6/2),places=7)
        labels['corner_rows'][0]['target_log1p_error'][0]=None
        with self.assertRaisesRegex(ValueError,'Missing reference'):
            M.prepare_frame(frame,labels,norm)

    def test_ties_keep_q0_and_ignore_candidate_names(self):
        frame=feature_frame();norm=M.normalizer([frame]);scorer=ConstantScorer()
        first=M.choose(scorer,frame,norm)
        self.assertEqual(first[0]['selected_candidate_id'],'0:keep')
        renamed=deepcopy(frame);renamed['corner_rows'][0]['candidate_ids']=['zzzz','aaaa']
        selected=M.choose(scorer,renamed,norm)
        self.assertEqual(selected[0]['selected_candidate_id'],'zzzz')
        self.assertEqual(first[0]['scores'],selected[0]['scores'])


class MockTrainingTest(unittest.TestCase):
    def test_mock_fit_first_strict_minimum_patience_logging_and_reload(self):
        items,norm=prepared_training();root=artifact_folder('strict_minimum')
        bindings=dict(scientific=dict(constructed=True),execution_source=dict(constructed=True))
        constructions_before=MOCK_CONSTRUCTIONS
        with patch.object(T.torch.optim,'AdamW',MockAdamW),patch.object(M,'evaluate_loss',evaluation_schedule(items,[3.,2.,2.,2.,2.,2.,2.])):
            receipt=T.fit(items,norm,root,'COMMON',M.SEEDS[0],bindings)
            cached=T.fit(items,norm,root,'COMMON',M.SEEDS[0],bindings)
        self.assertEqual(MOCK_CONSTRUCTIONS-constructions_before,1)
        self.assertEqual(receipt,cached)
        self.assertEqual(receipt['best_epoch'],2)
        self.assertEqual(receipt['epochs'],7)
        self.assertEqual(receipt['optimizer_steps_this_attempt'],7)
        folder=root/'training'/f'OBS_COMMON_s{M.SEEDS[0]}'
        best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        self.assertEqual(best['runtime']['next_epoch'],2)
        self.assertEqual(last['runtime']['wait'],5)
        restored=M.Scorer('COMMON');restored.load_state_dict(best['model_state_dict'])
        self.assertEqual(M.state_hash(restored),receipt['initial_state_sha256'])
        events=[json.loads(s)['event'] for s in (root/'RUN_LOG.jsonl').read_text().splitlines()]
        self.assertEqual(events.count('TRAIN_EPOCH'),7)
        self.assertEqual(events.count('FIT_COMPLETE'),1)
        self.assertEqual(len((root/'GLOBAL_FIT_ATTEMPTS.jsonl').read_text().splitlines()),1)

    def test_mock_failure_and_same_contract_resume_preserve_epoch_order(self):
        items,norm=prepared_training();root=artifact_folder('technical_resume')
        bindings=dict(scientific=dict(constructed=True),execution_source=dict(constructed=True))
        with patch.object(T.torch.optim,'AdamW',MockAdamW),patch.object(M,'evaluate_loss',evaluation_schedule(items,[3.,2.],interrupt_epoch=3)):
            with self.assertRaisesRegex(RuntimeError,'CONSTRUCTED_INTERRUPT'):
                T.fit(items,norm,root,'BI',M.SEEDS[0],bindings)
        folder=root/'training'/f'OBS_BI_s{M.SEEDS[0]}'
        failures=list((folder/'attempts').glob('*/FAILED_TECHNICAL.json'))
        self.assertEqual(len(failures),1)
        failure=json.loads(failures[0].read_text())
        self.assertEqual(failure['finished_history_epochs'],2)
        self.assertEqual(failure['optimizer_steps_this_attempt'],3)
        self.assertTrue((failures[0].parent/'last.pt').is_file())
        with patch.object(T.torch.optim,'AdamW',MockAdamW),patch.object(M,'evaluate_loss',evaluation_schedule(items,[2.5]*5)):
            receipt=T.fit(items,norm,root,'BI',M.SEEDS[0],bindings,resume=True)
        self.assertTrue(receipt['resumed'])
        self.assertEqual(receipt['epochs'],7)
        self.assertEqual(receipt['best_epoch'],2)
        self.assertEqual(receipt['optimizer_steps_this_attempt'],5)
        history=json.loads((folder/'HISTORY.json').read_text())
        rng=np.random.default_rng(M.SEEDS[0])
        expected=[hashlib.sha256(rng.permutation(4).astype('<i8').tobytes()).hexdigest() for _ in range(7)]
        self.assertEqual([h['training_order_sha256'] for h in history],expected)
        self.assertEqual(len((root/'GLOBAL_FIT_ATTEMPTS.jsonl').read_text().splitlines()),2)


def main():
    global ARTIFACT_ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root',required=True,type=Path)
    args=parser.parse_args();ARTIFACT_ROOT=args.artifact_root
    ARTIFACT_ROOT.mkdir(parents=True,exist_ok=True)
    started,cpu=time.monotonic(),time.process_time()
    torch.set_num_threads(1)
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__,fromlist=['*']))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(schema='BI_CONSTRUCTED_MODEL_TRAINING_TESTS_V1',status='PASS' if result.wasSuccessful() else 'FAIL',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        mock_optimizer_constructions=MOCK_CONSTRUCTIONS,mock_step_calls=MOCK_STEPS,
        real_optimizer_constructions=0,scientific_optimizer_updates=0,scientific_reference_reads=0,
        parameters_changed_by_mock_steps=False,
        cost=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(M.__file__),Path(T.__file__)]})
    path=ARTIFACT_ROOT/'CONSTRUCTED_TEST_RECEIPT.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()

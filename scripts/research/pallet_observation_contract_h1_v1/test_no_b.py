"""Constructed feature-closure and unchanged H1-contract tests, optimizer0."""
from __future__ import annotations
from copy import deepcopy
import unittest
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from scripts.research.pallet_boundary_action_repair_v1 import action_data as D
from scripts.research.pallet_boundary_action_repair_v1 import action_models as M
from . import no_b as B
from . import no_b_train as T


def frame():
    pose=lambda t:dict(available=True,R_physical=np.eye(3).tolist(),centroid=[float(t),0.,3.])
    candidates=[]
    for index,t in enumerate((0.,.2,.4)):
        candidates.append(dict(candidate_id='N0'if index==0 else'c'+str(index),actual_pose=pose(t),eligible=True,
            source_aliases=['N0']if index==0 else[],features=[float(j+index+1)for j in range(31)]+[1.]*31))
    return dict(id='TOY',object_scale_m=1.,feature_names=list(B.FEATURE_NAMES),fallback=pose(0.),candidates=candidates)


def normalizer():return dict(mean=[.25]*31,scale=[2.]*31,valid_train_counts=[3]*31)


class NoBTests(unittest.TestCase):
    def test_explicit_dependency_exact(self):
        self.assertEqual(B.REMOVED_INDICES,(2,3,4,10,13,14,15,23,25,28,29,30))
        self.assertEqual(len(set(B.FINAL_MASK)),72);self.assertEqual(len(B.KEPT_FINAL),116)
        self.assertNotIn(7,B.REMOVED_INDICES);self.assertEqual(B.FEATURE_NAMES[7],'outline_energy')

    def test_closed_before_all_pooling(self):
        f=frame();original,_=D.inputs(f,normalizer(),'H1');x,_=B.inputs(f,normalizer())
        self.assertTrue(np.any(original[:,B.FINAL_MASK]!=0));self.assertTrue(np.all(x[:,B.FINAL_MASK]==0))
        self.assertTrue(np.array_equal(x[:,62:124],np.repeat(x[:,:62].mean(0)[None],len(x),0)))
        self.assertTrue(np.array_equal(x[:,124:186],np.repeat(x[:,:62].max(0)[None],len(x),0)))

    def test_every_retained_input_bitwise_exact(self):
        f=frame();a,old=D.inputs(f,normalizer(),'H1');b,new=B.inputs(f,normalizer())
        self.assertEqual(old,new);self.assertEqual(a[:,B.KEPT_FINAL].tobytes(),b[:,B.KEPT_FINAL].tobytes())
        self.assertEqual(b.shape,(3,188));self.assertEqual(b.dtype,np.float32)

    def test_removed_values_and_validity_cannot_change_input_or_logit(self):
        first=frame();second=deepcopy(first)
        for i,c in enumerate(second['candidates']):
            for j in B.REMOVED_INDICES:c['features'][j]=(-1.)**i*1e12;c['features'][31+j]=float((i+j)%2)
        a,_=B.inputs(first,normalizer());b,_=B.inputs(second,normalizer());self.assertEqual(a.tobytes(),b.tobytes())
        torch.manual_seed(123);model=M.make_model('MLP')
        self.assertTrue(torch.equal(model(torch.from_numpy(a)),model(torch.from_numpy(b))))

    def test_retained_information_can_affect_nonzero_test_weights(self):
        f=frame();g=deepcopy(f);g['candidates'][1]['features'][0]+=10
        a,_=B.inputs(f,normalizer());b,_=B.inputs(g,normalizer());model=M.make_model('MLP')
        with torch.no_grad():
            for value in model.parameters():value.zero_()
            model[0].weight[0,0]=1.;model[2].weight[0,0]=1.;model[4].weight[0,0]=1.
        self.assertNotEqual(float(model(torch.from_numpy(a))[1]),float(model(torch.from_numpy(b))[1]))

    def test_original_candidate_journal_not_mutated(self):
        f=frame();before=deepcopy(f);B.inputs(f,normalizer());self.assertEqual(f,before)

    def test_physical_order_dedup_aliases_and_eligibility_unchanged(self):
        f=frame();f['candidates'][2]['eligible']=False;duplicate=deepcopy(f['candidates'][0]);duplicate['candidate_id']='duplicate';duplicate['source_aliases']=['E0'];f['candidates'].append(duplicate)
        self.assertEqual(B.inputs(f,normalizer())[1],D.inputs(f,normalizer(),'H1')[1])
        self.assertEqual(len(B.inputs(f,normalizer())[1]),2)

    def test_candidate_id_rename_and_input_row_permutation(self):
        f=frame();g=deepcopy(f);g['candidates'].reverse()
        for i,c in enumerate(g['candidates']):c['candidate_id']='arbitrary_'+str(i)
        a,aa=B.inputs(f,normalizer());b,bb=B.inputs(g,normalizer())
        self.assertTrue(np.array_equal(a,b));self.assertEqual([D.pose_key(c['actual_pose'])for c in aa],[D.pose_key(c['actual_pose'])for c in bb])

    def test_exact_ties_keep_N0_original_readout(self):
        f=frame();model=M.make_model('MLP')
        with torch.no_grad():
            for value in model.parameters():value.zero_()
        r=B.infer(f,model,normalizer());self.assertEqual(r['action_role'],'N0');self.assertFalse(r['fallback']);self.assertEqual(r['actual_pose'],f['fallback'])

    def test_no_pose_is_preserved(self):
        f=frame();f['candidates']=[];f['fallback']={'available':False};x,entries=B.inputs(f,normalizer());self.assertEqual(x.shape,(0,188));self.assertEqual(entries,[])
        self.assertEqual(B.infer(f,M.make_model('MLP'),normalizer())['prediction_status'],'NO_POSE')

    def test_wrong_feature_meaning_is_rejected(self):
        f=frame();f['feature_names'][2]='not_B_energy'
        with self.assertRaises(ValueError):B.inputs(f,normalizer())

    def test_unseen_retained_values_and_validity_match_original(self):
        n=normalizer();n['valid_train_counts'][0]=0;f=frame();f['candidates'][1]['features'][31+1]=0
        a,_=D.inputs(f,n,'H1');b,_=B.inputs(f,n);self.assertTrue(np.array_equal(a[:,B.KEPT_FINAL],b[:,B.KEPT_FINAL]))

    def test_same_original_BCE_and_padding(self):
        model=nn.Linear(188,1);model.weight.data.zero_();model.bias.data.zero_()
        records=[dict(x=np.ones((2,188),np.float32),target=np.array([1.,0.],np.float32)),dict(x=np.ones((1,188),np.float32),target=np.array([0.],np.float32))]
        loss=M.loss_value(model,M.collate(records),'H1');self.assertAlmostEqual(float(loss),np.log(2),places=6);loss.backward();self.assertTrue(torch.isfinite(model.weight.grad).all())

    def test_source_exact_independent_initialization_all_three_seeds(self):
        for seed in D.SEEDS:
            torch.manual_seed(seed);original=M.make_model('MLP')
            torch.manual_seed(seed);independent=nn.Sequential(nn.Linear(188,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
            self.assertEqual(T.model_hash(original),T.model_hash(independent));self.assertEqual(sum(p.numel()for p in original.parameters()),16321)

    def test_fixed_original_frame_schedule_all50epochs(self):
        for seed in D.SEEDS:
            old=np.random.default_rng(seed);new=np.random.default_rng(seed)
            for epoch in range(50):self.assertTrue(np.array_equal(old.permutation(768),new.permutation(768)))

    def test_GO_guard_runs_before_plan_or_input_read(self):
        with self.assertRaises(PermissionError):T.run(SimpleNamespace(training_GO=False))

    def test_actual_common_logger_owns_timestamp_for_all_fit_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for stage in ('OPTIMIZER_CREATED','FIRST_OPTIMIZER_STEP','FIT_COMPLETE','FIT_FAILED'):
                T.log_fit(root,stage,arm=T.ARM,seed=20261018,attempt=1)
            records=[json.loads(s) for s in (root/'RUN_LOG.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),4)
            self.assertTrue(all(isinstance(r['epoch'],float) for r in records))
            self.assertEqual(records[-1]['stage'],'FIT_FAILED')

    def test_reserved_logger_timestamp_is_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(ValueError):T.log_fit(root,'TOY',epoch=1)
            self.assertFalse((root/'RUN_LOG.jsonl').exists())

    def test_verified_retry_is_new_counted_attempt_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old={'arm':T.ARM,'seed':20261018,'attempt':1}
            (root/'GLOBAL_FIT_ATTEMPTS.jsonl').write_text(json.dumps(old)+'\n')
            with self.assertRaises(RuntimeError):T.reserve(root,20261018,{})
            receipt={'retry_seed':20261018,'failed_attempt':1}
            second=T.reserve(root,20261018,{},receipt)
            self.assertEqual(second['attempt'],2);self.assertEqual(second['retry_of_attempt'],1)
            with self.assertRaises(RuntimeError):T.reserve(root,20261018,{},receipt)

    def test_global_nine_cap_applies_to_technical_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'GLOBAL_FIT_ATTEMPTS.jsonl').write_text(''.join(json.dumps({'arm':'OTHER','seed':s,'attempt':s+1})+'\n' for s in range(9)))
            with self.assertRaises(RuntimeError):T.reserve(root,20261018,{}, {'retry_seed':20261018,'failed_attempt':1})

    def test_two_preserved_failures_allow_exactly_five_no_B_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);receipt={'retry_seed':20261018,'failed_attempt':2,'prior_failed_attempts':[1,2]}
            (root/'GLOBAL_FIT_ATTEMPTS.jsonl').write_text(''.join(json.dumps(dict(arm=T.ARM,seed=20261018,attempt=i))+'\n'for i in (1,2)))
            third=T.reserve(root,20261018,{},receipt);fourth=T.reserve(root,20261019,{},receipt);fifth=T.reserve(root,20261020,{},receipt)
            self.assertEqual([third['attempt'],fourth['attempt'],fifth['attempt']],[3,4,5]);self.assertEqual(third['retry_of_attempt'],2)
            with self.assertRaises(RuntimeError):T.reserve(root,20261018,{},receipt)

    def test_full_run_with_mock_optimizer_exercises_checkpoint_history_manifest_logging(self):
        """Real common I/O and old loss/eval; no scientific AdamW construction."""
        created=[]
        class MockOptimizer:
            def __init__(self,parameters,**kwargs):
                self.parameters=list(parameters);self.steps=0;created.append(self)
                if kwargs!={'lr':.001,'weight_decay':.0001}:raise AssertionError(kwargs)
            def zero_grad(self,set_to_none=True):
                for parameter in self.parameters:parameter.grad=None
            def step(self):self.steps+=1
            def state_dict(self):return dict(MOCK_OPTIMIZER_NOT_SCIENTIFIC=True,steps=self.steps)
        x,entries=B.inputs(frame(),normalizer())
        item=dict(id='TOY',x=x,entries=entries,target=np.array([1.,0.,0.],np.float32),
            metrics=[dict(success_5cm5deg=i==0,translation_cm=float(i),rotation_deg=float(i))for i in range(3)])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan=root/'PLAN.json';plan.write_text(json.dumps(dict(status='FROZEN_TOY_ONLY')))
            args=SimpleNamespace(training_GO=True,plan=plan,plan_sha256=D.sha(plan),run_dir=root,
                features=None,feature_seal=None,labels_dir=None,split_lock=None,old_training_manifest=None,technical_resume_receipt=None)
            with patch.object(T,'prepare_items',return_value=({'train':[item],'validation':[item]},normalizer(),{'toy_only':True})),patch.object(torch.optim,'AdamW',MockOptimizer):
                actual=T.run(args)
            self.assertEqual(len(created),3);self.assertEqual(len(actual),3)
            manifest=json.loads((root/'NO_B_TRAINING_MANIFEST.json').read_text())
            self.assertEqual(manifest['status'],'COMPLETE');self.assertEqual(manifest['actual_new_optimizer_attempts'],3)
            self.assertEqual(manifest['preserved_failed_optimizer_attempts'],0)
            events=[json.loads(s)for s in (root/'RUN_LOG.jsonl').read_text().splitlines()]
            self.assertEqual(len(events),9);self.assertEqual(sum(r['stage']=='FIT_COMPLETE'for r in events),3)
            for row in actual:
                self.assertEqual(row['epochs'],6);self.assertEqual(row['best_epoch'],1);self.assertEqual(row['optimizer_steps'],6)
                self.assertTrue(row['first_epoch_cost']['wall_seconds']>=0)
                last=torch.load(row['last_checkpoint'],map_location='cpu',weights_only=False)
                self.assertTrue(last['optimizer_state_dict']['MOCK_OPTIMIZER_NOT_SCIENTIFIC'])
                self.assertEqual(last['optimizer_state_dict']['steps'],6)
                self.assertEqual(last['training_runtime_state']['patience_wait'],5)
                self.assertEqual(len(last['training_runtime_state']['history']),6)
                self.assertEqual(len(json.loads(Path(row['history']).read_text())),6)
                self.assertEqual(last['numpy_rng_state']['bit_generator'],'PCG64')
                self.assertTrue(torch.is_tensor(last['torch_rng_state']))
                self.assertEqual(D.sha(row['checkpoint']),row['checkpoint_sha256'])
                self.assertEqual(D.sha(row['last_checkpoint']),row['last_checkpoint_sha256'])

    def test_terminal_failure_counts_are_derived_from_preserved_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for i in (1,2):
                folder=root/'failed_attempts'/str(i);folder.mkdir(parents=True)
                (folder/'FAILED_TECHNICAL.json').write_text(json.dumps(dict(arm=T.ARM,attempt={'attempt':i},optimizer_created=True,optimizer_steps=0 if i==1 else 24)))
            self.assertEqual(set(T.terminal_failures(root)),{1,2})


if __name__=='__main__':unittest.main()

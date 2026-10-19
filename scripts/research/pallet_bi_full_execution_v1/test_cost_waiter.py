"""Temporary metadata fixtures and mocked subprocesses only."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import cost_waiter as C


def put(path,value):C.write(path,value);return C.sha(path)


def service(unit=C.UNIT,substate='exited',status='0',restarts='0'):
    return dict(unit=unit,Id=unit,returncode=0,SubState=substate,Result='success',ExecMainStatus=status,NRestarts=restarts,
        CPUUsageNSec='2500000000',MemoryPeak='1234',ExecMainStartTimestampMonotonic='1000000',ExecMainExitTimestampMonotonic='11000000')


class WaiterTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.r=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def change(self,p,**values):
        d=C.read(p);d.update(values);p.write_text(json.dumps(d));return d

    def controller(self,name,stages,source):
        root=self.r/name;plan_path=root/'PLAN.json';go_path=self.r/(name+'_GO.json')
        plan=dict(output=str(root),controller_sha256=C.sha(source),sources={str(source):C.sha(source)},fixed_inputs={},stage_order=stages)
        put(plan_path,plan);put(go_path,dict(approved=True,plan_sha256=C.sha(plan_path),controller_sha256=C.sha(source)))
        (root/'ACTUAL_COMMAND_LEDGER.jsonl').write_text('')
        cps={}
        for stage in stages:
            artifact=root/(stage+'_artifact.json');put(artifact,dict(complete=True))
            cp=root/'checkpoints'/(stage+'.json')
            put(cp,dict(complete=True,identity=dict(input_bindings={'PLAN.json':C.sha(plan_path),'ROOT_GO':C.sha(go_path)}),
                completed_files={str(artifact):C.sha(artifact)}));cps[cp.name]=C.sha(cp)
        seal=dict(complete=True,plan_sha256=C.sha(plan_path),ROOT_GO_sha256=C.sha(go_path),source_sha256=C.sha(source),
            stage_order=stages,actual_command_ledger_sha256=C.sha(root/'ACTUAL_COMMAND_LEDGER.jsonl'),checkpoints=cps)
        return plan_path,go_path,seal

    def derived_fixture(self):
        ns=self.r/'ns';ns.mkdir()
        for name in ['master_costs.py','derived_completion.py','final_sequence.py']:(ns/name).write_text('fixture source '+name)
        sp,sg,ss=self.controller('SEQUENCE',C.SEQUENCE_STAGES,ns/'final_sequence.py')
        put(sp.parent/'FINAL_SEQUENCE_SEAL.json',ss)
        dp,dg,ds=self.controller('DERIVED',C.DERIVED_STAGES,ns/'derived_completion.py')
        p=C.read(dp);p.update(sequence_plan=str(sp),sequence_GO=str(sg),sequence_output=str(sp.parent),scatter_pairs=[{},{}]);dp.write_text(json.dumps(p))
        self.change(dg,plan_sha256=C.sha(dp));ds.update(plan_sha256=C.sha(dp),ROOT_GO_sha256=C.sha(dg))
        for stage in C.DERIVED_STAGES:
            cp=dp.parent/'checkpoints'/(stage+'.json');v=C.read(cp)
            v['identity']['input_bindings'].update({'PLAN.json':C.sha(dp),'ROOT_GO':C.sha(dg)})
            cp.write_text(json.dumps(v));ds['checkpoints'][cp.name]=C.sha(cp)
        put(dp.parent/'ARTIFACT_SIZES.json',{'fixture':True})
        ds.update(schema='FULL_CAMPAIGN_DERIVED_COMPLETION_SEAL_V1',upstream_sequence_seal_sha256=C.sha(sp.parent/'FINAL_SEQUENCE_SEAL.json'),
            contrasts=30,scalar_arms=40,scalar_population_rows=160,scalar_frame_arm_rows=9520,gallery_groups=5,gallery_seed_count=3,
            scatter_datasets=6,new_H1_fits=0,new_pose_solves=0,new_pose_metric_evaluations=0,raw_RGB_copied=False,
            public_poses_or_GT=False,all_outliers_and_failures_retained=True,generated_inputs_sha256={},public_contrasts_files={},
            artifact_sizes_sha256=C.sha(dp.parent/'ARTIFACT_SIZES.json'))
        put(dp.parent/'DERIVED_COMPLETION_SEAL.json',ds)
        return dict(master_source=str(ns/'master_costs.py'),derived_plan=str(dp),derived_GO=str(dg))

    def lifecycle_fixture(self):
        source=self.r/'ROOT_CAPTURE_SERVICE_LIFECYCLE.py';source.write_text('fixture capture source')
        root=self.r/'lifecycle';put(root/'LOCK.json',dict(source_sha256=C.sha(source),units=sorted(C.FOUR_UNITS)))
        snapshot=dict(units=[service(unit=u)for u in sorted(C.FOUR_UNITS)])
        (root/'SNAPSHOTS.jsonl').write_text(json.dumps(snapshot)+'\n')
        put(root/'COMPLETION.json',dict(complete=True,scientific_completion_claim=False,source_sha256=C.sha(source),
            snapshot_sha256=C.sha(root/'SNAPSHOTS.jsonl'),last_snapshot=snapshot))
        return dict(lifecycle_root=str(root),run=str(self.r),sources={str(source):C.sha(source)})

    def test_V4_is_copy_adds_only_derived_ledger_metadata(self):
        original=dict(run=str(self.r),ledgers=[],metadata=[],schema='cost')
        value=C.version4(original,self.r)
        self.assertEqual(original['ledgers'],[]);self.assertEqual(len(value['ledgers']),1)
        self.assertEqual(value['metadata'][0]['name'],'EXTERNAL_DERIVED_SERVICE')

    def test_V4_duplicate_derived_ledger_rejected(self):
        value=C.version4(dict(run=str(self.r),ledgers=[],metadata=[]),self.r)
        with self.assertRaises(ValueError):C.version4(value,self.r)

    def test_full_derived_and_upstream_checkpoint_bindings(self):
        p=self.derived_fixture();self.assertEqual(C.verify_derived(p)['stage_count'],8)

    def test_derived_partial_stage_rejected(self):
        p=self.derived_fixture();t=Path(C.read(p['derived_plan'])['output'])/'DERIVED_COMPLETION_SEAL.json';s=C.read(t)
        s['checkpoints'].pop('scatter.json');t.write_text(json.dumps(s))
        with self.assertRaises(ValueError):C.verify_derived(p)

    def test_derived_population_change_rejected(self):
        p=self.derived_fixture();t=Path(C.read(p['derived_plan'])['output'])/'DERIVED_COMPLETION_SEAL.json';self.change(t,scalar_frame_arm_rows=9519)
        with self.assertRaises(ValueError):C.verify_derived(p)

    def test_derived_artifact_change_rejected(self):
        p=self.derived_fixture();root=Path(C.read(p['derived_plan'])['output']);(root/'scatter_artifact.json').write_text('{}')
        with self.assertRaises(ValueError):C.verify_derived(p)

    def test_upstream_sequence_missing_checkpoint_rejected(self):
        p=self.derived_fixture();dp=C.read(p['derived_plan']);seq=C.read(dp['sequence_plan']);t=Path(seq['output'])/'checkpoints/F_finish.json';t.unlink()
        with self.assertRaises(FileNotFoundError):C.verify_derived(p)

    def test_full_four_service_capture(self):self.assertIn('completion_sha256',C.verify_lifecycle(self.lifecycle_fixture()))

    def test_service_lifecycle_changed_snapshot_rejected(self):
        p=self.lifecycle_fixture();(Path(p['lifecycle_root'])/'SNAPSHOTS.jsonl').write_text('{}\n')
        with self.assertRaises(ValueError):C.verify_lifecycle(p)

    def test_service_CPU_and_elapsed_units(self):
        r=C.service_record(service(),C.UNIT);self.assertEqual(r['CPU_seconds'],2.5);self.assertEqual(r['elapsed_wall_seconds'],10.)

    def test_running_fifth_unit_cannot_be_finalized(self):
        with self.assertRaises(ValueError):C.service_record(service(substate='running'),C.UNIT)

    def test_restart_rejected(self):
        with self.assertRaises(ValueError):C.service_record(service(restarts='1'),C.UNIT)

    def test_failed_service_cost_retained_but_success_gate_rejects(self):
        r=C.service_record(service(status='1'),C.UNIT,successful=False);self.assertEqual(r['ExecMainStatus'],'1')
        with self.assertRaises(ValueError):C.service_record(service(status='1'),C.UNIT)

    def test_query_whitelist_and_actualargv(self):
        seen=[]
        def mocked(argv,**kwargs):
            seen.append(argv);return SimpleNamespace(stdout='Id='+C.UNIT+'\n',stderr='',returncode=0)
        value,argv=C.query_service(C.UNIT,mocked)
        self.assertEqual(seen,[argv]);self.assertEqual(argv[:4],['systemctl','--user','show',C.UNIT])
        self.assertIn('query_child_CPU_seconds',value)
        with self.assertRaises(ValueError):C.query_service('other.service',mocked)

    def test_source_guard_preserves_frozen_master(self):
        source=self.r/'master';source.write_text('modified')
        with self.assertRaises(ValueError):C.source_guard(dict(sources={str(source):C.sha(source)},fixed_inputs={},master_source=str(source)))

    def test_repeated_source_guard_rejects_catalog_tamper(self):
        source=Path(C.__file__).with_name('master_costs.py')
        catalog=self.r/'catalog.json';put(catalog,dict(reviewed=True))
        plan=dict(sources={str(source):C.MASTER_SHA},fixed_inputs={},master_source=str(source),
            catalog=str(catalog),catalog_sha256=C.sha(catalog))
        C.source_guard(plan)
        self.change(catalog,reviewed=False)
        with self.assertRaises(ValueError):C.source_guard(plan)

    def test_actual_fifth_launch_env_and_tail_bound(self):
        fixture=self.derived_fixture();fixture.update(python='/fixture/python',worktree='/fixture/worktree')
        dp=Path(fixture['derived_plan']);dg=Path(fixture['derived_GO'])
        args=['systemd-run','--unit='+C.UNIT.removesuffix('.service'),'--property=Restart=no','--property=CPUAccounting=yes',
            '--setenv=PYTHONPATH=/fixture/worktree']+['--setenv='+n+'=1'for n in ('MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')]
        args += [fixture['python'],'-B','-m','scripts.research.pallet_bi_full_execution_v1.derived_completion','run',
            '--plan',str(dp),'--root-GO',str(dg)]
        path=self.r/'launch.json';put(path,dict(actual_service_argv=args,service_returncode=0,plan_sha256=C.sha(dp),
            GO_sha256=C.sha(dg),source_sha256=C.read(dp)['controller_sha256']));fixture['derived_launch']=str(path)
        self.assertEqual(C.validate_derived_launch(fixture),C.sha(path))
        args.remove('--setenv=PYTHONPATH=/fixture/worktree');self.change(path,actual_service_argv=args)
        with self.assertRaises(ValueError):C.validate_derived_launch(fixture)

    def test_unbound_ROOT_GO_cannot_launch(self):
        plan=self.r/'plan';go=self.r/'go';put(plan,{});put(go,dict(approved=False,metadata_only=True))
        with self.assertRaises(ValueError):C.guard_GO(plan,go)

    def test_prepare_refuses_existing_output_before_any_campaign_read(self):
        output=self.r/'existing';output.mkdir()
        with self.assertRaises(ValueError):C.prepare(self.r,self.r/'missing',output,self.r/'masterout')

    def test_partial_run_not_automatically_repeated(self):
        output=self.r/'wait';output.mkdir();put(output/'RUN_STARTED.json',{})
        plan=dict(output=str(output),run=str(self.r),master_output=str(self.r/'master'))
        with patch.object(C,'guard_GO',return_value=plan),patch.object(C,'completed_return',return_value=None),patch.object(C.subprocess,'call')as call:
            with self.assertRaises(ValueError):C.run('unused','unused')
            call.assert_not_called()

    def test_mocked_whole_metadata_execution_one_child_includes_output_write(self):
        output=self.r/'wait';output.mkdir();put(output/'PLAN.json',{})
        derived=self.r/'derived';derived.mkdir();put(derived/'DERIVED_COMPLETION_SEAL.json',{})
        dp=self.r/'derivedplan';put(dp,dict(output=str(derived)))
        lifecycle=self.r/'lifecycle';lifecycle.mkdir();put(lifecycle/'COMPLETION.json',{})
        catalog=self.r/'catalog';put(catalog,{})
        go=self.r/'GO';put(go,{})
        master=self.r/'masterout'
        plan=dict(output=str(output),run=str(self.r),master_output=str(master),derived_plan=str(dp),
            lifecycle_root=str(lifecycle),systemctl_unit=C.UNIT,python='/fixture/python',catalog=str(catalog),worktree='/fixture/worktree')
        calls=[]
        def child(argv,**kwargs):
            calls.append((argv,kwargs));master.mkdir()
            put(master/'MASTER_COSTS.json',dict(auditor_source_sha256=C.MASTER_SHA,H1=dict(all_expected_fits_complete=True,
                completed_new_fits=33,all_declared_prediction_seals_complete=True,tiny_head_real_prediction_forwards=7854),
                campaign_elapsed_wall_seconds='UNKNOWN_NOT_ZERO',campaign_elapsed=dict(measurement_cutoff_scope=C.CUTOFF_SCOPE)))
            put(master/'PUBLIC_COST_SUMMARY.json',dict(private_master_receipt_sha256=C.sha(master/'MASTER_COSTS.json')))
            (master/'MASTER_COSTS.md').write_text('fixture')
            return 0
        with patch.object(C,'guard_GO',return_value=plan),patch.object(C,'source_guard'),\
             patch.object(C,'verify_derived',return_value={'full':True}),patch.object(C,'verify_lifecycle',return_value={'full':True}),\
             patch.object(C,'validate_derived_launch',return_value='launch_hash'),\
             patch.object(C,'query_service',return_value=(service(),['systemctl','fixture'])),\
             patch.object(C.subprocess,'call',side_effect=child):
            result=C.run(output/'PLAN.json',go)
        self.assertEqual(len(calls),1);self.assertTrue(result['complete']);self.assertFalse(result['campaign_complete'])
        self.assertEqual(calls[0][1]['env']['PYTHONPATH'],'/fixture/worktree')
        self.assertEqual(calls[0][1]['env']['OPENBLAS_NUM_THREADS'],'1')
        self.assertEqual(C.read(self.r/'ROOT_COST_ELAPSED_CUTOFF.json')['scope'],C.CUTOFF_SCOPE)
        self.assertEqual(C.read(output/'MASTER_CHILD_ROOT_GO.json')['auditor_source_sha256'],C.MASTER_SHA)
        self.assertEqual(result['actual_master_child']['return_code'],0)


if __name__=='__main__':unittest.main()

"""Small temp-file lifecycle checks; no real jobs, models or reference data."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from . import campaign as C
from . import extend_batch as X
from .common import rows,save_rows,sha,write


class CampaignPackagingTests(unittest.TestCase):
    def test_controller_proof_reuse_and_missing_hook_are_not_fabricated_done(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';run.mkdir();source=root/'scripts/research/pallet_statewise_end_to_end_recovery_v1';source.mkdir(parents=True)
            (source/'fixture.py').write_text('fixture')
            write(run/'CONFIG_FROZEN.json',dict(fixed_scientific_choice=1));write(run/'ARM_REGISTRY.json',dict(arms=[]))
            plan=dict(config_sha256=sha(run/'CONFIG_FROZEN.json'),arm_registry_sha256=sha(run/'ARM_REGISTRY.json'),
                core_source_sha256={'fixture.py':sha(source/'fixture.py')},frozen_epoch=time.time(),runtime_budget=dict(wall_hours=1,workers=1))
            write(run/'PLAN_LOCK.json',plan);proof=run/'proof.json';write(proof,dict(complete=True))
            config=dict(run_dir=str(run),source_root=str(root),scientific_config_sha256=sha(run/'CONFIG_FROZEN.json'),
                plan_lock_sha256=sha(run/'PLAN_LOCK.json'),stages=[dict(id='done',proofs=[dict(path=str(proof),fields={'complete':True})]),
                    dict(id='blocked',depends_on=['done'],proofs=[],blocked_reason='input not supplied')])
            path=run/'runtime.json';write(path,config)
            first=C.run(path,wait=False)
            self.assertEqual(first['stages']['done']['status'],'DONE')
            self.assertEqual(first['stages']['blocked']['status'],'BLOCKED_INPUT')
            self.assertEqual(C.run(path,wait=False)['stages']['done']['proof_sha256'],first['stages']['done']['proof_sha256'])
            write(proof,dict(complete=False),replace=True)
            changed=C.run(path,wait=False)
            self.assertEqual(changed['stages']['done']['status'],'FAILED_TECHNICAL')

    def test_live_handle_requires_current_PID_start_and_argv(self):
        current=C._proc(os.getpid());self.assertIsNotNone(C._same_live(current))
        self.assertIsNone(C._same_live({**current,'start_ticks':'0'}))

    def test_exact_input_rows_reuse_hardlinks_at_expanded_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pilot=root/'pilot';pilot.mkdir();config=root/'config.json';write(config,{'fixed':1})
            old=[dict(id='a',payload=[1]),dict(id='b',payload=[2])]
            full=[old[1],dict(id='c',payload=[3]),old[0]]
            pinput=root/'pilot.jsonl.gz';einput=root/'full.jsonl.gz';save_rows(pinput,old);save_rows(einput,full)
            values=[]
            for i,frame in enumerate(old):
                folder=pilot/'frames'/f'{i:04d}';folder.mkdir(parents=True)
                value=dict(id=frame['id'],reference_inputs=False,boundary={});values.append(value)
                save_rows(folder/'RESULT.jsonl.gz',[value]);write(folder/'FRAME_SEAL.json',dict(id=frame['id'],
                    input_row_sha256=X._row_sha(frame),config_sha256=sha(config),result_sha256=sha(folder/'RESULT.jsonl.gz'),
                    complete=True,reference_inputs=False,resource={}))
            save_rows(pilot/'PREDICTIONS.jsonl.gz',values);save_rows(pilot/'HEAD_FEATURES.jsonl.gz',values)
            core={name:sha(Path(X.__file__).with_name(name)) for name in X.CORE}
            lock=dict(ordered_ids=['a','b'],config_sha256=sha(config),source_sha256=core)
            write(pilot/'BATCH_LOCK.json',lock)
            write(pilot/'BATCH_SEAL.json',dict(**lock,complete=True,reference_inputs=False,failures=[],input_sha256=sha(pinput),
                predictions_sha256=sha(pilot/'PREDICTIONS.jsonl.gz'),head_features_sha256=sha(pilot/'HEAD_FEATURES.jsonl.gz')))
            output=root/'full';result=X.prepare(pinput,einput,config,pilot,output)
            self.assertEqual((result['reused_frames'],result['new_pending_frames']),(2,1))
            self.assertEqual(next(rows(output/'frames/0000/RESULT.jsonl.gz'))['id'],'b')
            self.assertEqual(os.stat(output/'frames/0000/RESULT.jsonl.gz').st_ino,os.stat(pilot/'frames/0001/RESULT.jsonl.gz').st_ino)
            self.assertEqual(X.prepare(pinput,einput,config,pilot,output)['frame_bindings'],result['frame_bindings'])
            modified=root/'modified.jsonl.gz';save_rows(modified,[{**old[1],'payload':[999]},old[0]])
            with self.assertRaises(ValueError):X.prepare(pinput,modified,config,pilot,root/'bad')


if __name__=='__main__':unittest.main()

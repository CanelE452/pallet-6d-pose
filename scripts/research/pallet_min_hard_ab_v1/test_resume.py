"""No-GPU resume regression checks; temporary artifacts only."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import common as C
from . import run_ab


class ResumeTests(unittest.TestCase):
    def test_missing_stage_not_done(self):
        with tempfile.TemporaryDirectory() as d,patch.object(C,'DOC',Path(d)):
            self.assertFalse(run_ab.stage_done('teacher','missing.json'))

    def test_completed_fit_hash_checked_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);checkpoint=root/'last.pt';checkpoint.write_bytes(b'test fixture, not a checkpoint')
            with patch.object(C,'ROOT',root),patch.object(C,'DOC',root):
                C.save(root/'FIT.json',dict(complete=True,steps=320,epochs=list(range(5)),checkpoint=C.bind(checkpoint)))
                self.assertTrue(run_ab.stage_done('train:H_MANUAL','FIT.json'))
                checkpoint.write_bytes(b'corrupt fixture')
                with self.assertRaises(AssertionError):run_ab.stage_done('train:H_MANUAL','FIT.json')

    def test_incomplete_fit_marker_rejected(self):
        with tempfile.TemporaryDirectory() as d,patch.object(C,'DOC',Path(d)):
            C.save(Path(d)/'FIT.json',dict(complete=False,steps=64,epochs=[0]))
            with self.assertRaises(AssertionError):run_ab.stage_done('train:H_MANUAL','FIT.json')

    def test_missing_upstream_not_regenerated_under_completed_fit(self):
        with tempfile.TemporaryDirectory() as d,patch.object(C,'DOC',Path(d)):
            C.save(Path(d)/'FIT.json',dict(complete=True))
            with patch.object(C,'read',return_value={}),patch.object(run_ab,'STAGES',[('teacher','TEACHER.json'),('train:H_MANUAL','FIT.json')]),patch.object(run_ab,'stage_done',return_value=False),patch.object(run_ab.subprocess,'run') as launch:
                with self.assertRaisesRegex(RuntimeError,'Missing upstream marker'):run_ab.run(True)
                launch.assert_not_called()

    def test_finalization_missing_compute_stage_fails_without_launch(self):
        with patch.object(C,'read',return_value={}),patch.object(run_ab,'stage_done',return_value=False),patch.object(run_ab.subprocess,'run') as launch:
            with self.assertRaisesRegex(RuntimeError,'Finalization forbids'):run_ab.run(False)
            launch.assert_not_called()

    def test_completed_pipeline_only_runs_presentation_audits(self):
        with patch.object(C,'read',return_value={}),patch.object(C,'save'),patch.object(run_ab,'stage_done',side_effect=lambda stage,artifact:artifact is not None),patch.object(run_ab.subprocess,'run') as launch:
            events=run_ab.run(False)
            self.assertEqual([e['stage'] for e in events if e['action']=='EXECUTED'],['final_report','completion_audit','directive_audit'])
            self.assertEqual(launch.call_count,3)
            self.assertTrue(all(e['action']=='VERIFIED_SKIP' for e in events if e['stage'].startswith('train:')))

    def test_interrupted_fit_not_restarted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'runs/H_MANUAL').mkdir(parents=True)
            with patch.object(C,'RAW',root),patch.object(C,'read',return_value={}),patch.object(run_ab,'STAGES',[('train:H_MANUAL','FIT_H_MANUAL.json')]),patch.object(run_ab,'stage_done',return_value=False),patch.object(run_ab.subprocess,'run') as launch:
                with self.assertRaisesRegex(RuntimeError,'Interrupted fit'):run_ab.run(True)
                launch.assert_not_called()

    def test_partial_evaluation_marker_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as d,patch.object(C,'DOC',Path(d)):
            C.save(Path(d)/'DECISION.json',dict(primary='fixture'))
            with self.assertRaisesRegex(AssertionError,'Incomplete scoring'):run_ab.stage_done('evaluate','DECISION.json')

    def test_cli_dispatches_every_post_label_state(self):
        from .cli import resume
        states=['HARD_LABELS_LOCKED_TRAINING_PENDING','TRAINING','FIT_COMPLETE_EVALUATION_PENDING','PREDICTIONS_FROZEN_SCORING_PENDING','EVALUATED_REPORT_PENDING','COMPLETE']
        for state in states:
            with self.subTest(state=state),patch.object(C,'exclusive',side_effect=lambda _:nullcontext()),patch.object(C,'state',return_value=dict(status=state)),patch.object(run_ab,'main') as main:
                resume();main.assert_called_once_with(allow_compute=state!='COMPLETE')

    def test_full_status_has_directive_sections(self):
        from .status_output import collect
        data=collect()
        for key in ('CANDIDATE_POOL','DIFFICULTY_TAGGING','HARD_SELECTION','ANNOTATION','TEACHER_SUPPORT','TRAINING','PAIR_INTEGRITY',
                    'BASE_CLEAN','HMANUAL_CLEAN','BASE_MODERATE','HMANUAL_MODERATE','BASE_SEVERE','HMANUAL_SEVERE','HPSEUDO_MODERATE','HPSEUDO_SEVERE',
                    'VERIFIED_HARD36','SOURCE256','PRIMARY_DECISION','STOP_MORE_HARD_LABELING','INTERPRETATION','NEXT_ONE_STEP','COMMITS','PUSH','LATEST_REMOTE_HEAD','GIT_STATUS'):
            self.assertIn(key,data)
        self.assertEqual(data['TRAINING']['NEW_UPDATES_DURING_FINALIZATION'],0)

    def test_cli_continues_after_approved_label_lock(self):
        from .cli import resume
        from . import lock_assisted_labels
        with tempfile.TemporaryDirectory() as d,patch.object(C,'DOC',Path(d)):
            C.save(Path(d)/'PNP_BOX_USER_APPROVAL.json',dict(approved=True))
            with patch.object(C,'exclusive',side_effect=lambda _:nullcontext()),patch.object(C,'state',side_effect=[dict(status='WAITING_FOR_HUMAN_HARD_METADATA'),dict(status='HARD_LABELS_LOCKED_TRAINING_PENDING')]),patch.object(lock_assisted_labels,'main') as lock,patch.object(run_ab,'main') as run:
                resume();lock.assert_called_once();run.assert_called_once_with()

    def test_finish_missing_decision_from_frozen_results_idempotently(self):
        from .evaluate import finish_decision
        results=C.read(C.DOC/'RESULTS.json')['groups'];fit=C.read(C.DOC/'TRAIN_FIT.json')['groups'];expected=C.read(C.DOC/'DECISION.json')
        with tempfile.TemporaryDirectory() as d,patch.object(C,'DOC',Path(d)),patch.object(C,'set_state'):
            finish_decision(results,fit);p=Path(d)/'DECISION.json';before=C.sha(p)
            self.assertEqual(C.read(p),expected)
            finish_decision(results,fit);self.assertEqual(C.sha(p),before)


def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ResumeTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    C.save(C.DOC/'RESUME_TESTS.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
           cases=unittest.defaultTestLoader.getTestCaseNames(ResumeTests),new_training=0,new_inference=0,
           method='temporary marker/checkpoint fixtures + mocked stage execution + real read-only status output'))
    assert result.wasSuccessful()


if __name__=='__main__':main()

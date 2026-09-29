"""Lock the scope of additional controls, not a retroactive preregistration."""
from __future__ import annotations
import subprocess
from pathlib import Path
from . import common as C


def run():
    path = C.DOC / 'START.json'
    if path.exists():
        for binding in C.read(path)['immutable_inputs']:
            C.verify(binding)
        print('EXISTING_START_REUSED')
        return
    old = C.OLD.DOC
    ledger = C.read(old / 'RESOURCE_LEDGER.json')
    assert ledger['totals']['student_fits'] == 6
    assert ledger['caps'] == dict(student_fits=10, selector_fits=1, GPU_training_seconds=21600)
    inputs = [old / n for n in ('RESOURCE_LEDGER.json', 'FINAL_DECISION.json',
        'CLEAN_LOCK.json', 'PRIMARY_PROTOCOL.json', 'SELECTOR_SUPERVISION_PROVENANCE.json')]
    user_file = C.ROOT / 'data/evaluation/pallet_eval_v1/reports/ANNOTATION_PROGRESS.md'
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=C.ROOT, text=True).strip()
    start = dict(created_at=C.now(), HEAD=git('rev-parse','HEAD'),
        branch=git('branch','--show-current'), remote_HEAD=git('rev-parse','origin/main'),
        recent_commits=git('log','-8','--oneline').splitlines(),
        tracked_status=git('status','--short','--branch','--untracked-files=no').splitlines(),
        user_changes_preserved=C.bind(user_file), immutable_inputs=[C.bind(p) for p in inputs],
        posthoc_adaptive_DEV_followup=True, retrospective_preregistration=False,
        primary_population='Ordinary Plastic NATURAL99=MODERATE21+SEVERE78; CLEAN29 and FULL128 retained',
        contract='Same pallet-center cm / C2 full rotation degrees / geometry / failed denominator / reference',
        prior_goal_turn='Interrupted before tool work: no progress. New directive read and HEAD verified here.',
        new_training_allowed_only_after_control_results_and_frozen_decision=True,
        new_manual=0,new_training_RGB=0,old_FINAL_preserved=True)
    C.save(path,start,True)
    C.save(C.DOC/'RESOURCE_LEDGER.json',dict(caps=ledger['caps'],
        inherited_ledger=C.bind(old/'RESOURCE_LEDGER.json'), inherited_totals=ledger['totals'],
        new_events=[],totals=ledger['totals']),True)
    C.save(C.DOC/'STATE.json',dict(stage='FROZEN_CONTROL_COMPLETION',new_fits=0,
        method_development_stopped=False,automatic_background_resume=False),True)
    print('START_LOCKED',start['HEAD'],ledger['totals'])


if __name__ == '__main__':
    run()

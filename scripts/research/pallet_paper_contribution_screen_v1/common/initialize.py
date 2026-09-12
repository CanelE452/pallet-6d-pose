"""Bind the existing evidence without altering or re-evaluating it."""
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.contracts import *

def main():
    old = ROOT/'data/pallet/results/pallet_line_pose_v1'
    read = lambda name: json.loads((old/name).read_text())
    s, v, r, p = [read(x) for x in ['SUMMARY.json','VERDICT.json','RUNTIME.json','TRAIN_PROTOCOL.json']]
    ck = old/'runs/image_line_only_seed1/last.pt'
    assert sha(ck) == '1fc71445c041eb3f14db34d03decfa605b0d7fe0b1f6a6458be8bac0ee436192'
    start = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    write(DOC/'REPO_STATE_START.json', dict(main_sha=start, origin_main_sha=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip(),
        initial_worktree_clean=True, branch='main', required_ancestor='15ff8e1ab26f79d92e77d697a8bc88accf3fae64',
        note='Initial fetch/clean check performed before source creation; wiring probe ran before writing this receipt, with no performance evaluation or student training.'))
    write(DOC/'MASTER_PROTOCOL_LOCK.json', dict(schema='paper_contribution_screen_v1',time_utc=datetime.now(timezone.utc).isoformat(),
        user_instruction_sha=sha('/home/minjae/.codex/attachments/f05f9bb0-7e09-4f0e-9936-ab5e3348805f/pasted-text.txt'),
        sequence=['A','C','D','B','E'],seeds=[1,2,3],student_steps=900,trust_steps=1500,
        gate_failure_stops_only_own_track=True,no_performance_retuning=True,independent_confirmation_required=True,
        paper_final_write=False,probe_order_note='C wiring-only smoke completed while auditing existing A evidence; A frozen before any C student fit.'))
    freeze = dict(candidate='image_line_only',seed=1,checkpoint=str(ck.relative_to(ROOT)),sha256=sha(ck),
        temperature=1.,lam=.25,cap=None,new_training_updates=0,evidence_level='DEVELOPMENT',
        reason='Existing image_joint vs image_line_only confidence intervals all include zero. Identical deployed architecture; line-only removes the auxiliary corner training objective. First numeric seed; no new DEV selection. Runtime measurements do not establish line-only speed superiority.',
        parameter_architecture=p['model_config'],runtime=r['by_run']['image_line_only_seed1'],
        joint_line_comparison=s['comparisons']['image_joint_vs_image_line_only'],
        original_verdict=v,confirmation_status='NOT_RUN',verdict='A_NEEDS_NEW_CONFIRMATION_DATA')
    write(DOC/'A_architecture_confirmation/CANDIDATE_FREEZE.json',freeze)
    write(DOC/'A_architecture_confirmation/EXISTING_EVIDENCE.json', dict(baseline=s['baseline'],arms=s['arms'],runs=s['runs'],
        runtime_by_run=r['by_run'],identity=s.get('identity'),sources={n:sha(old/n) for n in ['SUMMARY.json','VERDICT.json','TRAIN_PROTOCOL.json','SELECTION.json','RUNTIME.json']}))
    prior = ROOT/'_docs/experiments/pallet_dht_pseudoline_selftrain_v1/DATA_PROVENANCE.json'
    write(DOC/'DATA_INVENTORY.json',dict(r0=dict(path=str(R0.relative_to(ROOT)),sha256=sha(R0)),
        local_line_source_manifest=dict(path=str((old/'SOURCE_MANIFEST.json').relative_to(ROOT)),sha256=sha(old/'SOURCE_MANIFEST.json'),partitions=p['partitions']),
        prior_provenance=json.loads(prior.read_text()),
        confirmation='No population with documented never-consulted selection history established. FINAL path names alone do not prove untouched status.',
        gpu=dict(device='RTX 3080',temperature_c=51,compute_processes=[],process_library_path='/tmp/nvidia-580.173.02-userspace',system_nvml_mismatch=True,reboot=False)))
    preserved = {str(R0.relative_to(ROOT)):sha(R0)}
    for directory in ['_docs/paper/final','_docs/experiments/pallet_line_pose_v1','_docs/experiments/pallet_dht_pseudoline_selftrain_v1']:
        preserved.update({str(f.relative_to(ROOT)):sha(f) for f in (ROOT/directory).rglob('*') if f.is_file()})
    for f in old.glob('*.json'):
        preserved[str(f.relative_to(ROOT))]=sha(f)
    preserved[str(ck.relative_to(ROOT))]=sha(ck)
    write(DOC/'PRESERVED_SOURCE_SHA.json',preserved)
    print('A freeze and master bindings written; no new confirmation claim.')

if __name__ == '__main__': main()

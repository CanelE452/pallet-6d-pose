import subprocess,time
from . import common as C

def main():
    if (C.DOC/'INPUT_BINDINGS.json').exists():
        for b in C.read(C.DOC/'INPUT_BINDINGS.json')['files']:C.verify(b)
        print('FOLLOWUP_BASELINE_ALREADY_LOCKED');return
    paths=[C.P.DOC/x for x in ('CORE_RESULTS.json','CORE_COMPARABILITY_AUDIT.json','MANUAL_SUPERVISION_BUDGET.json','PREDICTIONS_LOCK.json')]
    paths += [C.M.DOC/x for x in ('WOOD_RESULTS.json','WOOD_PAIR_PREFLIGHT.json','WOOD_PROVENANCE_AUDIT.json','WOOD_PREDICTIONS_LOCK.json','METHOD_LOCK.json')]
    paths += [C.ROOT/'_docs/experiments/pallet_visible_transfer_closure_v1'/x for x in ('REPORT_KO.md','TRAIN_TARGET_TRANSFER.json','RESULTS.json')]
    paths += [C.P.FINAL,C.P.TRUTH,C.P.SPLIT,C.P.RAW/'PREDICTIONS.json',C.P.RAW/'POSE_PREDICTIONS.json',C.P.RAW/'ANCHOR_POINTS.json']
    paths += [C.ROOT/'_docs/paper/selftraining_submission_v1'/x for x in ('manuscript.tex','manuscript.pdf','material_extension.tex')]
    paths += [C.ROOT/C.checkpoint(mat,arm)['path'] for mat in ('PLASTIC','WOOD') for arm in (*C.ARMS,'SYN_LR5','TEACHER')]
    paths += [C.ROOT/'scripts/research/pallet_type_selftrain_v1'/x for x in ('recovery_pose.py','recovery_pose_trainer.py','train.py')]
    status=subprocess.check_output(['git','status','--short','--branch'],text=True)
    C.save(C.RAW/'START_GIT_PRIVATE.json',dict(status=status,log=subprocess.check_output(['git','log','-8','--oneline'],text=True)))
    C.save(C.DOC/'INPUT_BINDINGS.json',dict(head_start=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        remote_start=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip(),branch='main',
        files=[C.bind(p) for p in sorted(set(paths))],existing_tracked_changes=[s for s in status.splitlines()[1:] if not s.startswith('??')],
        existing_untracked_path_entries=sum(s.startswith('??') for s in status.splitlines()),
        baseline_contract='Original320-update material comparison; not previous640 or S1/H_MANUAL',utc=C.now()),True)
    C.save(C.DOC/'RESOURCE_LEDGER.json',dict(start_unix=1790559722,start_utc='2026-09-28T01:42:02Z',
        caps=dict(cycles=3,fits=12,optimizer_updates=7680,per_fit_updates=640,gpu_seconds=21600,wall_seconds=36000),
        reserve='Prefer final third of training budget for matched replication/control',events=[],totals={}),True)
    C.resource('prepare_and_cuda_readonly_preflight',details='Host RTX3080 torch2.1.1+cu118 works; sandbox device access not GPU failure; no updates')
    C.save(C.DOC/'DATA_ROLE_LEDGER.json',dict(
        TRAIN_FIT=dict(plastic_unique=217,wood_unique=361,real_slots_per_material=512,synthetic_slots=512,
          teacher_manual_images=9,teacher_manual_corners=38,teacher_materials=['PLASTIC','WOOD'],new_manual=0),
        TRAIN_CALIBRATION='Only preselected existing TRAIN rows; no DEV-derived weights/labels',
        REUSED_DEV=dict(PLASTIC=dict(images=128,corners=985,visible_verified=66,visible_images=16),WOOD=dict(images=45,corners=346,verified_visible_provenance_points=0)),
        SYNTH_CHECK='Existing exact synthetic replay/heldout roles retained, not new real-generalization evidence',
        UNTOUCHED_CONFIRMATION='None opened; historical audit reused',
        roles='Oracle/reference arrays private diagnostic-only; never production training/selection input',
        aliases='Reuse existing Wood/Plastic recording SHA role audits; do not equate session filenames with recordings'),True)
    C.state('BASELINE_LOCKED',['start_git','baseline_bindings','data_roles','host_cuda'],
            'Parallel prior/literature/pose-oracle; native Wood TRAIN following and gradient diagnostics; then lock first cycle')
    print('FOLLOWUP_BASELINE_LOCKED',flush=True)

if __name__=='__main__':main()

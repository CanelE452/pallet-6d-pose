import subprocess
from . import common as C

SOURCES=[
 'scripts/annotate/annotate_pnp.py',
 'scripts/annotate/convert_to_camera_facing_v4.py',
 '_docs/archive/paper_support_20260830/real_gt_v2/FRAME_CONVENTION.md',
 'scripts/research/pallet_dht_structured_v2/SEMANTICS.md',
 '_docs/experiments/pallet_type_selftrain_v1/selftrain_recovery_v1/renderer_front_visibility_audit/RESULTS.json',
 '_docs/experiments/pallet_type_selftrain_v1/selftrain_recovery_v1/renderer_front_visibility_audit/COMPLETION_AUDIT.json',
 '_docs/experiments/pallet_sensors_submission_v1/R0_PRETRAINING_PROVENANCE.json',
 'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json',
 'challenge/yolo_pose_one_model/spatial_concat_scratch/build_dataset.py',
 'data/pallet/results/pallet_existing_data_transfer_v1/TARGETS.json',
 'data/pallet/results/pallet_existing_data_transfer_v1/DIAGNOSTICS.json',
 '_docs/experiments/pallet_existing_data_transfer_v1/THREE_CORNER_AUDIT.json',
 '_docs/experiments/pallet_existing_data_transfer_v1/THREE_CORNER_LOSS_PROBE.json']

def main():
    if (C.DOC/'INPUT_BINDINGS.json').exists():
        for b in C.read(C.DOC/'INPUT_BINDINGS.json')['files']:assert C.sha(C.ROOT/b['path'])==b['sha256']
        print('PREFLIGHT bindings reverified');return
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.ROOT,text=True).strip()
    status=subprocess.check_output(['git','status','--short','--branch'],cwd=C.ROOT,text=True)
    C.put(C.RAW/'GIT_STATUS_BEFORE.txt',status)
    C.put(C.DOC/'INPUT_BINDINGS.json',dict(HEAD_BEFORE=head,frame=C.FRAME,
        files=[C.bind(p) for p in [C.ANN,C.RGB,*[C.ROOT/s for s in SOURCES]]],
        no_training=True,no_inference=True,geometry_phase_model_payload_opened=False,
        prior_exposure='Prior conversation and THREE_CORNER audit already exposed model outputs. This is not a retrospectively blind investigator study. New geometry scripts do not load model payloads; UI hides models.'))
    C.put(C.DOC/'PREFLIGHT_AUDIT.md',f'# Preflight\n\nHEAD_BEFORE: `{head}`\n\nFrame: `{C.FRAME}`. No training/optimizer/inference/GT edits.\n\nTracked files were clean at entry; many unrelated untracked files retained. Full git status stored privately.\n\nEarlier model exposure is disclosed; only the new procedural geometry-before-model-read ordering can be tested.\n')
    print('PREFLIGHT',head)

if __name__=='__main__':main()

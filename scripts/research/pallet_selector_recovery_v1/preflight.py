import subprocess
from pathlib import Path
from . import common as C
from .feature_contract import main as lock_features

def main():
    assert not C.DOC.exists(),'Do not overwrite experiment'
    for p in (C.DOC,C.RAW,C.OUT):p.mkdir(parents=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    C.freeze(C.RAW/'WORKTREE_START.json',dict(head=head,branch=branch,status=subprocess.check_output(['git','status','--short','--branch'],text=True),created_at=C.now()))
    lock_features()  # Before loading any reference or derived outcome from the previous run.
    files=[]
    for b in C.read(C.PREV_DOC/'INPUT_BINDINGS.json')['files']:C.verify(b);files.append(b)
    for root in (C.PREV_DOC,C.PREV_RAW):files.extend(C.bind(p) for p in root.glob('*.json'))
    geometry=C.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
    source=C.ROOT/'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json'
    files.extend(C.bind(p) for p in (geometry,source,C.ROOT/'challenge/evaluation_v2/pnp_selector.py',C.ROOT/'scripts/annotate/pallet_geometry.py'))
    fits={a:C.read(C.STRUCT/f'FIT_PLASTIC_{a}.json')['checkpoint'] for a in C.ARMS}
    for b in fits.values():C.verify(b)
    C.freeze(C.DOC/'INPUT_BINDINGS.json',dict(head_start=head,branch=branch,files=list({b['path']:b for b in files}.values()),base_checkpoints=fits,created_at=C.now()))
    C.freeze(C.RAW/'BASE_INPUTS.json',dict(checkpoints=fits))
    C.save(C.DOC/'PREFLIGHT_AUDIT.md',f'# Selector recovery preflight\n\nHEAD `{head}` / {branch}. Existing inputs hashed; feature contract locked before Stage1 reference access.\n\nFrozen S0/S1 only; no base optimizer. Synthetic training runs in a separate process with runtime read guard. Real DEV is previously viewed; process isolation is not analyst blindness.\n')
    print('PREFLIGHT_LOCKED',len(files),flush=True)

if __name__=='__main__':main()

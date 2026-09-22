"""Entry point. Never loads model weights; never creates an optimizer."""
import subprocess
from pathlib import Path
import traceback
import sys
import torch
from . import diagnose as D

def main():
    resume='--resume-comparison-type-fix' in sys.argv
    if resume:
        assert 'ufunc' in D.read(D.DOC/'STOP.json')['error']
        assert not (D.DOC/'PARITY.json').exists()
        for b in D.read(D.DOC/'INPUT_BINDINGS.json')['files']:D.C.verify(b)
    else:
        assert not D.DOC.exists() and not D.RAW.exists(),'Namespace already exists: do not overwrite'
        D.DOC.mkdir(parents=True);D.RAW.mkdir(parents=True)
    def forbidden(*args,**kwargs):raise AssertionError('NEW OPTIMIZER FORBIDDEN')
    torch.optim.Optimizer.__init__=forbidden
    D.cv2.setNumThreads(1);torch.set_num_threads(1)
    try:
        data=D.load()
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
        status=subprocess.check_output(['git','status','--short'],text=True)
        bindings=[D.C.bind(p) for p in data['paths']]
        if not resume:
            D.save(D.DOC/'INPUT_BINDINGS.json',dict(head=head,branch=branch,files=bindings))
            D.save(D.RAW/'PREFLIGHT_PRIVATE.json',dict(status=status,records=data['records']))
            D.save(D.DOC/'PREFLIGHT_AUDIT.md',f'# Preflight\n\nHEAD: `{head}`\n\nBranch: `{branch}`\n\nFrozen population: 300; unchanged human severity. No training, optimizer, checkpoint loading. Exact input bindings saved. Unrelated worktree status recorded privately; untouched.\n\nImportant: production selector requires finite (9,2); true LOO cannot be executed without extending that contract. Attempt rejection is recorded, not counted as pose harm.\n')
        else:D.save(D.DOC/'COMPARISON_TYPE_CORRECTION.json',dict(reason='Test helper attempted np.allclose on lists containing JSON null; recursive comparison fixed. Not a numeric parity mismatch.',original_stop_preserved=True))
        frames,checks=D.parity_and_frames(data)
        D.save(D.DOC/'PARITY.json',dict(passed=True,checks=checks,tolerance=1e-7))
        D.save(D.DOC/'FRAME_DIAGNOSTICS.json',frames)
        D.save(D.DOC/'OFFICIAL_RESULTS_SNAPSHOT.json',data['results'])
        tt=D.transitions(frames,data);D.save(D.DOC/'TRANSITIONS.json',tt)
        oracle=D.oracles(frames);D.save(D.DOC/'ORACLE_WD_AUDIT.json',oracle)
        D.save(D.DOC/'HYPOTHESIS_AUDIT.json',D.distributions(frames))
        loo,rep=D.perturbations(frames,data,tt)
        D.save(D.DOC/'LOO_CORNER_INFLUENCE.json',loo);D.save(D.DOC/'GT_REPLACEMENT_ORACLE.json',rep)
        pair=D.pair_geometry(frames,data);D.save(D.DOC/'PAIR_GEOMETRY_AUDIT.json',pair)
        D.save(D.DOC/'AUGMENTATION_FREQUENCY_SNAPSHOT.json',D.read(D.C.DOC/'AUGMENTATION_FREQUENCY.json'))
        for b in bindings:D.C.verify(b)
        D.save(D.DOC/'ANALYSIS_COMPLETE.json',dict(complete_except_LOO_contract=True,new_training=0,optimizer_steps=0,bindings_verified=True,head_before=head))
        print('ANALYSIS_COMPLETE_WITH_LOO_CONTRACT_LIMIT',flush=True)
    except Exception as exc:
        D.save(D.DOC/('STOP_AFTER_RESUME.json' if resume else 'STOP.json'),dict(error=str(exc),traceback=traceback.format_exc(),training=0))
        raise

if __name__=='__main__':main()

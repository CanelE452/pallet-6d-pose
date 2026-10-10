"""Fresh Stage-0/1 reproduction with bounded-memory reference loading.

Use a new PALLET_WD_OUTPUT directory. Original frozen selection code is unchanged;
the explicitly pinned wrapper replaces only SYNTH reference preparation after
that population's selection seal exists.
"""
from pathlib import Path
from . import common as C
from . import inputs as I
from . import preflight
from . import stage1
from . import stage1_resume


def run():
    assert not (C.DOC/'INPUT_AUDIT.json').exists(), 'Fresh outputs only'
    preflight.run()
    C.write(C.DOC/'FRESH_WRAPPER_LOCK.json',dict(
        status='LOCKED_BEFORE_ANY_SOLVER_CALL',
        bindings=[C.binding(Path(__file__),C.ROOT),
                  C.binding(Path(stage1_resume.__file__),C.ROOT)],
        original_core_changed=False,
        correction='SYNTH reference-only NPZ members loaded once after population selection seal; identical arithmetic.',
        source_lock_sha256=C.sha(C.source_lock_path())))
    original=I.truth
    def bounded_reference(population):
        if population in ('SYNTH','SYNTH_HELDOUT'):
            assert (C.DOC/'STAGE1_SELECTION_SEAL_SYNTH.json').exists()
            return stage1_resume.minimal_truth('SYNTH',with_margin=True)
        return original(population)
    I.truth=bounded_reference
    try:
        stage1.run()
    finally:
        I.truth=original


if __name__=='__main__':
    run()

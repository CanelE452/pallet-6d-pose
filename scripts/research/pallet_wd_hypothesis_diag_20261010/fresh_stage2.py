"""Reproduce Stage2 from authenticated, already published Stage1 in a new folder.

This reuses the first experiment's actual publication receipt, not a claimed
publication of newly rerun Stage1. Fresh Stage1 is a separate entry point.
"""
from pathlib import Path
import shutil
from . import common as C


def run():
    archive=C.ROOT/'_docs/experiments/pallet_wd_hypothesis_diag_20261010'
    assert C.DOC != archive and C.DOC.is_relative_to(C.ROOT)
    assert not C.DOC.exists(), 'Choose a new nonexistent output namespace'
    publication=C.read(archive/'STAGE1_PUBLICATION.json')
    assert publication['status']=='PASS'
    names={p.name for p in archive.glob('STAGE1*') if p.is_file()}
    snapshot=C.read(archive/'STAGE1_PUBLISHED_SNAPSHOT.json')
    names.update(entry['preserved'] for entry in snapshot['files'])
    names.update(('INPUT_AUDIT.json','STAGE0_PARITY.json','METHOD_KO.md',
        'METHOD_LOCK.json','SOURCE_LOCK_STAGE2.json','DEPTH_PREPARATION.json'))
    source=C.read(archive/'SOURCE_LOCK_STAGE2.json')
    for group in ('new_core','original_core'):
        for b in source[group]:
            owner=C.SOURCE if b['owner']=='historical_source' else C.ROOT
            assert C.sha(owner/b['path'])==b['sha256']
    C.DOC.mkdir(parents=True)
    copied={}
    for name in sorted(names):
        src,dst=archive/name,C.DOC/name
        assert src.is_file()
        expected=C.sha(src);shutil.copyfile(src,dst)
        assert C.sha(dst)==expected
        copied[name]=expected
    C.write(C.DOC/'REPRODUCTION_INPUT_LOCK.json',dict(
        status='ARCHIVED_PUBLISHED_STAGE1_COPIED_BEFORE_NEW_STAGE2',
        archived_namespace=str(archive.relative_to(C.ROOT)),
        publication_receipt_sha256=C.sha(archive/'STAGE1_PUBLICATION.json'),
        files_sha256=copied,fresh_Stage1_or_publication_claimed=False,
        new_PnP_calls_before_lock=0,model_forwards=0,training_updates=0,
        reproduction_wrapper_binding=C.binding(Path(__file__),C.ROOT)))
    from . import run_stage2
    return run_stage2.run()


if __name__=='__main__':run()

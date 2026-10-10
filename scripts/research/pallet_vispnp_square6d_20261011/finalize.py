"""Freeze verified scientific artifacts after actual visual inspection."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
from PIL import Image
from . import common as C

def main():
    p=argparse.ArgumentParser();p.add_argument('--visual-inspected',action='store_true');a=p.parse_args()
    assert a.visual_inspected,'Open and inspect every generated PNG before freezing'
    assert not (C.DOC/'SHA256_MANIFEST.json').exists()
    verification=C.read(C.DOC/'VERIFICATION.json');assert verification['status']=='PASS'
    assert verification['original_preservation_and_private_inputs']['status']=='PASS'
    figures=C.read(C.DOC/'FIGURE_INDEX.json')
    for b in figures['figures']:
        assert C.sha(C.DOC/b['path'])==b['sha256']
        with Image.open(C.DOC/b['path']) as im:assert im.format=='PNG';im.verify()
    figures['visual_inspection']='PASS_ROOT_OPENED_ALL_PNGS'
    C.write(C.DOC/'FIGURE_INDEX.json',figures)
    synth=C.read(C.DOC/'SYNTH_EXECUTION.json')
    real=C.read(C.DOC/'REAL_EXECUTION.json') if (C.DOC/'REAL_EXECUTION.json').exists() else None
    verdict=C.read(C.DOC/'VERDICT.json') if real else C.read(C.DOC/'SYNTH_VERDICT.json')
    summary=dict(status='COMPLETE_WITH_DECLARED_SQUARE_BLOCKER',A_phase='A2' if real else 'STOPPED_AT_A1',
        A_verdict=verdict['verdict'],B_status=C.read(C.DOC/'SQUARE_REFERENCE_POSES.json')['status'],
        new_manual_annotations=0,new_training_updates=0,new_models=0,new_synthetic_images=0,
        original_RGB_published=False,paper_latex_pdf_changes=0,
        actual_experiment_F_calls=1276+synth['actual_F_calls']+(real['actual_F_calls'] if real else 0),
        A0_actual_F=1276,A1_actual_F=synth['actual_F_calls'],A2_actual_F=real['actual_F_calls'] if real else 0,
        original_checkout_preservation='VERIFICATION.json',independent_numeric_verification='VERIFICATION.json',
        scientific_source_lock='SOURCE_LOCK.json',figures=len(figures['figures']),visual_inspection=True,
        finalized_utc=datetime.now(timezone.utc).isoformat(),
        publication=dict(repository='CanelE452/pallet-6d-pose',target='main',authorized_by='explicit user request',
            normal_push_only=True,force_push=False,remote_receipt='REMOTE_PUBLICATION_VERIFICATION.json added after first normal push'))
    C.write(C.DOC/'EXECUTION_AND_PUBLICATION.json',summary)
    files=[p for directory in [C.DOC,Path(__file__).parent] for p in sorted(directory.rglob('*')) if p.is_file()]
    C.write(C.DOC/'SHA256_MANIFEST.json',dict(status='SEALED',files=[dict(path=str(p.relative_to(C.ROOT)),sha256=C.sha(p),bytes=p.stat().st_size) for p in files],
        excludes_self=True,scope='Scientific artifacts and code frozen before publication; later remote receipt is additional evidence.'))
    print('FINALIZED',summary['A_phase'],summary['A_verdict'],summary['actual_experiment_F_calls'],flush=True)

if __name__=='__main__':main()

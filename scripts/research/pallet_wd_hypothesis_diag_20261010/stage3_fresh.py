"""Fresh fixed official Small depth execution after completed Stage2 report.

Use a fresh private parent with its fetched metric3d subdirectory. Unlike the
historical recovery entry point, this performs no failed-key/diagnostic attempt.
"""
import argparse
from pathlib import Path
from . import common as C
from . import depth as D
from . import stage3 as E
from .stage3_official_runtime import OfficialSmallRunner


def run(private_dir):
    private=Path(private_dir).resolve()
    assert C.read(C.DOC/'STAGE2_REPORT_RECEIPT.json')['status']=='COMPLETE'
    assert not any((C.DOC/n).exists() for n in
        ('SOURCE_LOCK_STAGE3.json','STAGE3_RUNTIME_AMENDMENT.json','DEPTH_GATE.json'))
    assert not (private/'depth_accuracy_cache').exists()
    C.verify_core()
    C.write(C.DOC/'STAGE3_RUNTIME_AMENDMENT.json',dict(
        status='LOCKED_BEFORE_FIRST_DEPTH_FORWARD',kind='FRESH_REPRODUCTION_OFFICIAL_LOADER',
        reason='Pinned upstream strict=False; only absent unused zero mask_token allowed',
        sole_missing_key='depth_model.encoder.mask_token',unexpected_keys=[],
        actual_prior_model_constructions=0,actual_prior_model_forwards=0,
        checkpoint_or_model_change=False,configuration_or_threshold_change=False,
        GT_accuracy_read_before_amendment=False,
        runtime_binding=C.binding(Path(__file__).with_name('stage3_official_runtime.py'),C.ROOT),
        fresh_entry_binding=C.binding(Path(__file__),C.ROOT)))
    original=D.MetricDepthRunner
    D.MetricDepthRunner=OfficialSmallRunner
    try:result=E.run(private)
    finally:D.MetricDepthRunner=original
    execution=C.read(C.DOC/'STAGE3_EXECUTION.json')
    C.write(C.DOC/'STAGE3_RECOVERY_EXECUTION.json',dict(status='COMPLETE',
        kind='FRESH_REPRODUCTION_OFFICIAL_LOADER',
        actual_model_constructions_total=execution['actual_model_constructions'],
        actual_depth_forwards_total=execution['actual_depth_forwards'],
        failed_guard_model_constructions=0,diagnostic_model_constructions=0,
        prior_attempt_forwards=0,training_updates=0,
        scientific_execution_path='STAGE3_EXECUTION.json',
        runtime_amendment_path='STAGE3_RUNTIME_AMENDMENT.json',
        runtime_source_sha256=C.sha(Path(__file__).with_name('stage3_official_runtime.py'))))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-dir',type=Path,required=True)
    args=parser.parse_args();run(args.private_dir)

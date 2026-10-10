"""Replay into an explicitly fresh directory, respecting every stop gate."""
import argparse
from pathlib import Path
import unittest
from . import common as C

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    C.DOC=a.output.resolve();assert not C.DOC.exists(),'Never replace published or interrupted results'
    suite=unittest.defaultTestLoader.loadTestsFromNames([
        'scripts.research.pallet_vispnp_square6d_20261011.test_adapter',
        'scripts.research.pallet_vispnp_square6d_20261011.test_statistics'])
    checks=unittest.TextTestRunner(verbosity=1).run(suite)
    assert checks.wasSuccessful(),'Contract unit checks failed before replay'
    from .preflight import main as preflight
    preflight();C.write(C.DOC/'UNIT_CHECKS.json',dict(status='PASS',tests_run=checks.testsRun,
        actual_Frozen_F_unit_checks=True,threshold_and_seed_aggregation_checks=True))
    C.lock_sources()
    from .synth import run
    verdict=run()
    if verdict['continue_to_A2']:
        from .real import main as real
        from .summarize_real import main as summarize
        real();summarize()
    print('REPLAY_COMPLETE',verdict['verdict'],'REAL_run',verdict['continue_to_A2'],flush=True)

if __name__=='__main__':main()

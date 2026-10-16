"""Run original learned observation/pose stages in a new isolated output directory.

This adapter changes checkpoint/output paths only. Frozen Base inputs/initial
poses/controls are read through immutable dependency paths; decoder and pose
implementations remain the original code. It performs no work when imported.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def run(args):
    from . import retrain as R
    # Check completed, authorized exact replay before loading models or fitting.
    protocol = R.read(args.protocol)
    completed = R.read(Path(args.fits) / 'TRAINING_COMPLETION.json')
    assert completed['complete'] and completed['formal_updates'] == 9000
    assert completed['protocol']['sha256'] == R.sha(args.protocol)
    assert completed['input_hashes_unchanged']
    for key in ['original_model','original_decoder','original_evaluation','original_solver',
                'original_point_line','original_common','original_pose_inference','original_pose_scoring',
                'frozen_real_inputs','frozen_real_initial_observations','frozen_real_controls','downstream_adapter']:
        b = protocol['fixed_inputs'][key]
        path = R.REPO / b['path']; assert path.is_file() and R.sha(path) == b['sha256']
    output = Path(args.output).resolve()
    R.reject_protected_output(output,args.source_root,allow_repair_doc=True)
    if output != R.DOC.resolve() and output.exists():
        if args.stage == 'infer':assert not any(output.iterdir()),'External observation output must be empty'
        else:
            previous=R.read(output/'REPAIR_INFER_ADAPTER_RECEIPT.json')
            assert previous['schema']=='original_path_supervision_repair_adapter_v1'
            assert previous['protocol']['sha256']==R.sha(args.protocol),'External evaluation must follow this exact repair inference receipt'
    output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('PALLET_SOURCE_ROOT', str(Path(args.source_root).resolve()))
    from scripts.research.pallet_observation_refiner_20261009_v1 import common as C
    original_read, original_iter_rows = C.read, C.iter_rows
    readonly = {name: R.OLD_DOC / name for name in ['INPUTS.json','OBSERVATIONS.jsonl.gz','FIXED_CONTROLS.jsonl.gz']}
    def redirect(path):
        path = Path(path)
        if path.parent.resolve() == output:
            if path.name in readonly: return readonly[path.name]
            if path.name == 'TRAINING_COMPLETION.json': return Path(args.fits) / path.name
        return path
    C.DOC = output
    C.read = lambda path: original_read(redirect(path))
    C.iter_rows = lambda path: original_iter_rows(redirect(path))
    # Original modules import the shared C object, so outputs use only this DOC.
    if args.stage == 'infer':
        assert not (output / 'LEARNED_OBSERVATIONS.jsonl.gz').exists()
        assert not (output / 'OBSERVATION_SEAL.json').exists()
        from scripts.research.pallet_observation_refiner_20261009_v1 import learned_infer as L
        L.FITS = Path(args.fits).resolve()
        sys.argv = [sys.argv[0], '--output', 'LEARNED_OBSERVATIONS.jsonl.gz']
        L.main()
    else:
        assert not (output / 'LEARNED_PREDICTIONS.jsonl.gz').exists()
        assert not (output / 'LEARNED_GEOMETRY_SEALED.jsonl.gz').exists()
        from scripts.research.pallet_observation_refiner_20261009_v1 import learned_evaluate as E
        E.run()
    R.write_new(output / ('REPAIR_' + args.stage.upper() + '_ADAPTER_RECEIPT.json'),
                dict(schema='original_path_supervision_repair_adapter_v1', stage=args.stage,
                     protocol=R.binding(args.protocol), completed_training=R.binding(Path(args.fits)/'TRAINING_COMPLETION.json'),
                     implementation='original model/decoder/evaluation/solver unchanged; checkpoint/output path adapter only',
                     readonly_inputs={key:R.binding(path) for key,path in readonly.items()}))


def main():
    from . import retrain as R
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['infer','evaluate'])
    p.add_argument('--protocol', default=str(R.DOC/'RETRAINING_PROTOCOL.json'))
    p.add_argument('--fits', default='/dev/shm/pallet-kp-supervision-repair-private-20261010/learned_fits')
    p.add_argument('--output', default=str(R.DOC))
    p.add_argument('--source-root', default=os.environ.get('PALLET_SOURCE_ROOT'))
    run(p.parse_args())


if __name__ == '__main__': main()

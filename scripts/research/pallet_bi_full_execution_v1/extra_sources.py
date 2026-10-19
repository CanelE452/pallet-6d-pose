"""Read-only byte/receipt checks for sealed shared complete-action sources."""
from pathlib import Path

from . import s1_union as U
from scripts.research.pallet_boundary_action_repair_v1 import action_data as D


def verify(extra_path, seal_path=None):
    extra_path = Path(extra_path)
    if seal_path is None:
        possible = [extra_path.parent/name for name in ('FULL_RECEIPT.json', 'PREDICTION_SEAL.json')]
        found = [path for path in possible if path.exists()]
        if len(found) != 1:
            raise ValueError('An unambiguous completed extra producer seal is required')
        seal_path = found[0]
    seal_path = Path(seal_path); seal = U.read(seal_path); digest = U.sha(extra_path)
    if not seal.get('complete'):
        raise ValueError('Extra producer is not complete')
    binding = dict(extra_path=str(extra_path.resolve()), extra_sha256=digest,
        producer_seal_path=str(seal_path.resolve()), producer_seal_sha256=U.sha(seal_path))
    if 'output_artifacts' in seal:
        if seal.get('rows') != 1241 or seal.get('all1241_completed') is not True:
            raise ValueError('S3/S4 extras require the full 1241-frame producer receipt')
        if seal['output_artifacts'].get(extra_path.name) != digest:
            raise ValueError('Extra output bytes differ from completed S3/S4 receipt')
        lock_path = seal_path.parent/'METHOD_LOCK_RESOURCE_LP4.json'
        if seal.get('method_lock_sha256') != U.sha(lock_path):
            raise ValueError('Extra producer method lock differs')
        binding['producer_method_lock_path'] = str(lock_path.resolve())
        binding['producer_method_lock_sha256'] = U.sha(lock_path)
    else:
        D.bound(extra_path, seal_path)
    return binding

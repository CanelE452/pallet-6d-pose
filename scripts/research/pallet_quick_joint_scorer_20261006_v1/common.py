"""Read-only reused inputs; all new outputs have separate fixed destinations."""
from pathlib import Path
from scripts.research.pallet_quick_pose_loss_20261006_v1.common import (
    ROOT,BANKS,COST,OLD_DOC,HARD_DOC,BASELINE_ROOT,Data,Banks,read,write,sha,hash_value,
    dcp_env,numeric_contract,learning_rate,forward_bank,action_scores)
from scripts.research.pallet_quick_pose_loss_20261006_v1.losses import pose_loss
QUICK_DOC=ROOT/'_docs/experiments/pallet_quick_pose_loss_20261006_v1'
LOSS_DOC=QUICK_DOC
LOSS_SETUP=QUICK_DOC/'LOSS_SETUP.json'
DOC=ROOT/'_docs/experiments/pallet_quick_joint_scorer_20261006_v1'
OUTPUT=Path('/tmp/pallet-quick-joint-scorer-20261006-cache')


def load_setup():
    setup=read(LOSS_SETUP)
    assert setup['status']=='PASS'
    return setup


def code_bindings():
    return {name:sha(Path(__file__).parent/name) for name in ('common.py','model.py','training.py')}


def fit_receipts():
    path=DOC/'RUN_RECEIPTS.json'
    return read(path) if path.exists() else {'methods':{}}


def record_method(method,receipt):
    result=fit_receipts();result.setdefault('methods',{})[method]=receipt
    write(DOC/'RUN_RECEIPTS.json',result)


def verify_derived(setup):
    """Reuse initial fresh target hashes and check unchanged current state."""
    ready=fit_receipts()['inputs']
    assert ready['status']=='PASS' and ready['loss_setup_sha256']==sha(LOSS_SETUP)
    for entry in ready['states']:
        st=Path(entry['path']).stat()
        assert st.st_size==entry['bytes'] and st.st_mtime_ns==entry['mtime_ns'],entry['path']

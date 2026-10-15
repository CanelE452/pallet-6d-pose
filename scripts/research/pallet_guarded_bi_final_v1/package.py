"""Whitelist small non-reference evidence; retain all full private artifacts."""
from pathlib import Path
from copy import deepcopy
import argparse
import gzip
import shutil
from scripts.research.pallet_minimal_bi_ablation_v1 import batch as B
from scripts.research.pallet_minimal_bi_ablation_v1 import runtime as RT

COPY={
    'score/ROW_METRICS.csv':'metrics/ROW_METRICS.csv',
    'score/RESULTS.json':'metrics/RESULTS.json',
    'score/SCORE_SEAL.json':'evidence/SCORE_SEAL.json',
    'PREDICTION_SEAL.json':'evidence/PREDICTION_SEAL.json',
    'SELECTION_SEAL.json':'evidence/SELECTION_SEAL.json',
    'PREDICTIONS.jsonl.gz':'evidence/PREDICTIONS.jsonl.gz',
    'GUARD_AUDITS.jsonl.gz':'evidence/GUARD_AUDITS.jsonl.gz',
    'e_inputs/CONTRACT.json':'evidence/E_INPUT_CONTRACT.json',
    'e_inputs/INPUT_SEAL.json':'evidence/E_INPUT_SEAL.json',
    'e_run/OBSERVATION_SEAL.json':'evidence/E_OBSERVATION_SEAL.json',
    'cal/RESULTS.json':'metrics/CAL_RESULTS.json',
    'timing/TIMING.json':'evidence/TIMING.json',
    'independent_v1/RECEIPT.json':'evidence/INDEPENDENT_RECEIPT.json',
    'diagnostics/DIAGNOSTICS.json':'diagnostics/DIAGNOSTICS.json',
    'RESOURCE_LEDGER.jsonl':'evidence/RESOURCE_LEDGER.jsonl',
}

def package(folder,prior,output):
    folder,prior,output=Path(folder),Path(prior),Path(output);output.mkdir(parents=True,exist_ok=True)
    receipt=B.read(folder/'independent_v1/RECEIPT.json')
    if receipt['failure_count'] or receipt['rows']!=238 or receipt['arm_rows']!=2856 or receipt['guard_decisions']!=1428:
        raise ValueError('Complete independent pass required')
    bindings={}
    for source,target in COPY.items():
        src,dst=folder/source,output/target
        raw=src.read_bytes()
        plain=gzip.decompress(raw) if source.endswith('.gz') else raw
        if b'/home/' in plain or b'/tmp/' in plain:raise ValueError('Private absolute path in '+source)
        if dst.exists():raise FileExistsError(dst)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        bindings[target]=dict(sha256=RT.sha256(dst),exact_private_file=source)
    for source,target in (('e_inputs/INPUTS.json','evidence/E_INPUTS.json.gz'),
                          ('score/SCORED_ROWS.json','metrics/SCORED_ROWS.json.gz'),
                          ('cal/SCORED_ROWS.json','metrics/CAL_SCORED_ROWS.json.gz')):
        src,dst=folder/source,output/target
        plain=src.read_bytes()
        if b'/home/' in plain or b'/tmp/' in plain:raise ValueError('Private path in '+source)
        if dst.exists():raise FileExistsError(dst)
        dst.write_bytes(gzip.compress(plain,mtime=0))
        if gzip.decompress(dst.read_bytes())!=plain:raise ValueError('Compression changed bytes')
        bindings[target]=dict(sha256=RT.sha256(dst),decompressed_sha256=RT.sha256(src),exact_private_file=source)
    protocol=deepcopy(B.read(folder/'METHOD_PROTOCOL.json'))
    protocol['publication_view_of_private_protocol_sha256']=RT.sha256(folder/'METHOD_PROTOCOL.json')
    local=protocol['immutable_inputs_sha256'];portable={}
    for name,h in local.items():
        p=Path(name)
        prefix='prior/' if p.is_relative_to(prior) else 'this/'
        base=prior if prefix=='prior/' else folder
        portable[prefix+str(p.relative_to(base))]=h
    protocol['immutable_inputs_sha256']=portable
    protocol['publication_view_only_change']='Replace absolute local artifact roots with prior/ and this/; private original remains byte-sealed.'
    B.write(output/'evidence/METHOD_PROTOCOL_PUBLIC_VIEW.json',protocol)
    B.write(output/'evidence/PACKAGE_BINDINGS.json',dict(files=bindings,
        private_protocol_sha256=RT.sha256(folder/'METHOD_PROTOCOL.json'),
        no_reference_coordinates_or_full_masks_or_weights_included=True,
        all_full_private_artifacts_preserved=True,public_RGB_permission='same existing V11 hash inventory, checked by gallery'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();package(a.folder,a.prior,a.output)

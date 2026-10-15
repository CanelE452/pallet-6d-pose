"""Publish small fixed-data evidence without weights, masks or references."""
from pathlib import Path
from copy import deepcopy
import argparse
import gzip
import shutil
from scripts.research.pallet_minimal_bi_ablation_v1 import batch as B
from scripts.research.pallet_minimal_bi_ablation_v1 import runtime as RT

COPIES={
    'inputs_v1/INPUT_SEAL.json':'evidence/INPUT_SEAL.json',
    'inputs_v1/CONTRACT.json':'evidence/INPUT_CONTRACT.json',
    'inputs_v1/F_PARITY.json':'evidence/F_PARITY.json',
    'PREDICTION_SEAL.json':'evidence/PREDICTION_SEAL.json',
    'PREDICTIONS.jsonl.gz':'evidence/PREDICTIONS.jsonl.gz',
    'run/OBSERVATION_SEAL.json':'evidence/OBSERVATION_SEAL.json',
    'score/SCORE_SEAL.json':'evidence/SCORE_SEAL.json',
    'score/RESULTS.json':'metrics/RESULTS.json',
    'score/ROW_METRICS.csv':'metrics/ROW_METRICS.csv',
    'score/CHANGED_CASES.csv':'metrics/CHANGED_CASES.csv',
    'independent_v1/RECEIPT.json':'evidence/INDEPENDENT_RECEIPT.json',
    'diagnostics/CAL_STATE_DIAGNOSTICS.json':'diagnostics/CAL_STATE_DIAGNOSTICS.json',
    'diagnostics/DIAGNOSTIC_SEAL.json':'evidence/DIAGNOSTIC_SEAL.json',
    'RESOURCE_LEDGER.jsonl':'evidence/RESOURCE_LEDGER.jsonl',
}

def package(folder,prior,output):
    folder,prior,output=map(Path,(folder,prior,output));output.mkdir(parents=True,exist_ok=True)
    receipt=B.read(folder/'independent_v1/RECEIPT.json')
    if receipt['failure_count'] or receipt['rows']!=238 or receipt['arm_rows']!=1904:raise ValueError('Complete independent pass required')
    bindings={}
    def portable(data,name):
        if b'/home/' in data or b'/tmp/' in data or b'/root/' in data:raise ValueError('Private path in '+name)
    for name,target in COPIES.items():
        src,dst=folder/name,output/target;raw=src.read_bytes()
        portable(gzip.decompress(raw) if name.endswith('.gz') else raw,name)
        if dst.exists():raise FileExistsError(dst)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        bindings[target]=dict(sha256=RT.sha256(dst),exact_private_file=name)
    for name,target in (('inputs_v1/INPUTS.json','evidence/INPUTS.json.gz'),
            ('score/SCORED_ROWS.json','metrics/SCORED_ROWS.json.gz'),
            ('diagnostics/STAGE_DIAGNOSTICS.json','diagnostics/STAGE_DIAGNOSTICS.json.gz')):
        src,dst=folder/name,output/target;raw=src.read_bytes();portable(raw,name)
        if dst.exists():raise FileExistsError(dst)
        dst.write_bytes(gzip.compress(raw,mtime=0))
        if gzip.decompress(dst.read_bytes())!=raw:raise ValueError('Lossless publication required')
        bindings[target]=dict(sha256=RT.sha256(dst),decompressed_sha256=RT.sha256(src),exact_private_file=name)
    p=deepcopy(B.read(folder/'PROTOCOL.json'));p['publication_view_of_private_protocol_sha256']=RT.sha256(folder/'PROTOCOL.json')
    mapped={}
    for name,h in p['immutable_inputs_sha256'].items():
        path=Path(name);old=path.is_relative_to(prior);base=prior if old else folder
        mapped[('prior/' if old else 'this/')+str(path.relative_to(base))]=h
    p['immutable_inputs_sha256']=mapped
    p['publication_view_only_change']='Replace private absolute artifact roots with prior/ and this/; original private protocol remains byte-sealed.'
    B.write(output/'evidence/PROTOCOL_PUBLIC_VIEW.json',p)
    B.write(output/'evidence/PACKAGE_BINDINGS.json',dict(files=bindings,
        public_data_contains_no_reference_pose_or_manual_coordinate_arrays=True,
        original_private_full_masks_observations_inputs_and_failures_preserved=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();package(a.folder,a.prior,a.output)

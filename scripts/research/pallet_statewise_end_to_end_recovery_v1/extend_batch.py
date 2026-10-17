"""Reuse sealed pilot frame bytes in an expanded prediction-only batch.

No solver, image model, evaluation target or metric is imported. Source pilot
directories are immutable. Canonical full input rows must match, not just IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import time

from .common import canonical,read,rows,sha,write

CORE=('batch.py','solver.py','owner_model.py','boundary_model.py','common.py')


def _row_sha(frame):
    return hashlib.sha256(canonical(frame).encode()).hexdigest()


def _link(source,destination):
    if destination.exists():
        if sha(source)!=sha(destination):
            raise ValueError('Existing destination differs; preserve it: '+str(destination))
        return 'verified_existing'
    os.link(source,destination)
    return 'hardlinked'


def prepare(pilot_inputs,expanded_inputs,config_path,pilot_batch,output,*,expanded_input_seal=None):
    start=time.monotonic()
    pilot_inputs,expanded_inputs,config_path,pilot_batch,output=map(Path,
        (pilot_inputs,expanded_inputs,config_path,pilot_batch,output))
    seal=read(pilot_batch/'BATCH_SEAL.json');lock=read(pilot_batch/'BATCH_LOCK.json')
    config_digest=sha(config_path)
    source_hashes={name:sha(Path(__file__).with_name(name)) for name in CORE}
    if (seal.get('complete') is not True or seal.get('reference_inputs') is not False
            or seal.get('failures') or seal['input_sha256']!=sha(pilot_inputs)
            or seal['config_sha256']!=config_digest or lock['config_sha256']!=config_digest
            or seal['source_sha256']!=source_hashes or lock['source_sha256']!=source_hashes
            or seal['predictions_sha256']!=sha(pilot_batch/'PREDICTIONS.jsonl.gz')
            or seal['head_features_sha256']!=sha(pilot_batch/'HEAD_FEATURES.jsonl.gz')):
        raise ValueError('Completed unchanged reference-free pilot and identical frozen core/config required')
    if expanded_input_seal is not None:
        input_seal=read(expanded_input_seal)
        if sha(expanded_inputs) not in (input_seal.get('predictions_sha256'),input_seal.get('output_sha256')):
            raise ValueError('Expanded predictor-only inputs changed')
        if input_seal.get('GT_masks_or_labels_read') not in (None,False):
            raise ValueError('GT mask/label inputs are forbidden')
    old=list(rows(pilot_inputs));full=list(rows(expanded_inputs))
    old_ids=[f['id'] for f in old];full_ids=[f['id'] for f in full]
    if (len(set(old_ids))!=len(old) or len(set(full_ids))!=len(full)
            or old_ids!=seal['ordered_ids'] or old_ids!=lock['ordered_ids']
            or not set(old_ids).issubset(full_ids)):
        raise ValueError('Exact unique pilot subset of expanded IDs required')
    oldmap={f['id']:(i,f) for i,f in enumerate(old)}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'BATCH_SEAL.json').exists():
        raise ValueError('Expanded batch already completed; use batch.run seal reuse instead')
    linked=[];pending=[];operations={}
    for index,frame in enumerate(full):
        fid=frame['id']
        if fid not in oldmap:
            pending.append(fid);continue
        old_index,previous=oldmap[fid]
        digest=_row_sha(frame)
        if digest!=_row_sha(previous):
            raise ValueError('Same-ID pilot/expansion full input rows differ: '+fid)
        source=pilot_batch/'frames'/f'{old_index:04d}'
        destination=output/'frames'/f'{index:04d}'
        fs=read(source/'FRAME_SEAL.json')
        if (fs['id']!=fid or fs['input_row_sha256']!=digest or fs['config_sha256']!=config_digest
                or fs.get('complete') is not True or fs.get('reference_inputs') is not False
                or fs['result_sha256']!=sha(source/'RESULT.jsonl.gz')):
            raise ValueError('Pilot frame proof changed: '+fid)
        result=next(rows(source/'RESULT.jsonl.gz'))
        if result['id']!=fid or result.get('reference_inputs') is not False:
            raise ValueError('Pilot result identity/reference-free contract changed')
        bundle_sha=result.get('boundary',{}).get('bundle_sha256')
        if bundle_sha is not None and sha(source/'BOUNDARY_BUNDLE.jsonl.gz')!=bundle_sha:
            raise ValueError('Original actual RGB bundle bytes changed')
        destination.mkdir(parents=True,exist_ok=True)
        names=['RESULT.jsonl.gz']+(['BOUNDARY_BUNDLE.jsonl.gz'] if bundle_sha is not None else [])
        # A valid frame seal is published last, so a partial packaging attempt
        # cannot masquerade as completed solver execution.
        names+=['FRAME_SEAL.json']
        for name in names:
            status=_link(source/name,destination/name)
            operations[status]=operations.get(status,0)+1
        linked.append(dict(id=fid,pilot_index=old_index,expanded_index=index,
                           canonical_input_row_sha256=digest,result_sha256=fs['result_sha256'],
                           source_frame_seal_sha256=sha(source/'FRAME_SEAL.json')))
    receipt=dict(schema='EXACT_PILOT_TO_EXPANDED_BATCH_REUSE_V1',complete=True,
        pilot_frames=len(old),expanded_frames=len(full),reused_frames=len(linked),new_pending_frames=len(pending),
        reused_ids=[r['id'] for r in linked],pending_ids=pending,frame_bindings=linked,
        pilot_input_sha256=sha(pilot_inputs),expanded_input_sha256=sha(expanded_inputs),
        expanded_input_seal_sha256=sha(expanded_input_seal) if expanded_input_seal else None,
        pilot_batch_seal_sha256=sha(pilot_batch/'BATCH_SEAL.json'),config_sha256=config_digest,
        batch_core_source_sha256=source_hashes,link_operations=operations,
        original_pilot_unchanged=True,canonical_full_rows_not_just_IDs_verified=True,
        new_solves=0,new_model_forwards=0,reference_reads=0,elapsed_seconds=time.monotonic()-start)
    target=output/'EXTENSION_REUSE_RECEIPT.json'
    if target.exists():
        prior=read(target)
        keys=('pilot_input_sha256','expanded_input_sha256','pilot_batch_seal_sha256','config_sha256',
              'batch_core_source_sha256','reused_ids','pending_ids','frame_bindings')
        if any(prior[k]!=receipt[k] for k in keys):raise ValueError('Existing extension receipt differs')
        return prior
    write(target,receipt)
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot-inputs',required=True);p.add_argument('--expanded-inputs',required=True)
    p.add_argument('--config',required=True);p.add_argument('--pilot-batch',required=True);p.add_argument('--output',required=True)
    p.add_argument('--expanded-input-seal');a=p.parse_args()
    result=prepare(a.pilot_inputs,a.expanded_inputs,a.config,a.pilot_batch,a.output,
                   expanded_input_seal=a.expanded_input_seal)
    print(canonical({k:result[k] for k in ('complete','pilot_frames','expanded_frames','reused_frames','new_pending_frames','new_solves')}))


if __name__=='__main__':main()

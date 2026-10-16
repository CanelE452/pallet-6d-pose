"""Lossless float32 storage for sealed logits; retain the original private artifact."""
import base64
from pathlib import Path
import shutil
import numpy as np
from . import common as C

def decode(row):
    if 'candidate_logits_f32_base64' not in row:return row
    row=dict(row);blob=row.pop('candidate_logits_f32_base64');shape=row.pop('candidate_logits_shape')
    row.pop('unencoded_semantic_sha256',None)
    logits=np.frombuffer(base64.b64decode(blob),dtype='<f4').reshape(shape)
    row['queries']=[dict(q,candidate_logits=logits[i].tolist()) for i,q in enumerate(row['queries'])]
    return row

def run():
    path=C.DOC/'LEARNED_OBSERVATIONS.jsonl.gz';seal_path=C.DOC/'OBSERVATION_SEAL.json'
    seal=C.read(seal_path);assert seal['records']['sha256']==C.sha(path)
    original=C.SCRATCH/'UNCOMPACTED_LEARNED_OBSERVATIONS.jsonl.gz'
    assert not original.exists()
    C.write(C.SCRATCH/'OBSERVATION_SEAL_BEFORE_COMPACTION.json',seal)
    shutil.move(path,original);count=0
    def encoded_rows():
        nonlocal count
        for row in C.iter_rows(original):
            if row.get('queries'):
                arrays=np.asarray([q['candidate_logits'] for q in row['queries']],dtype='<f4')
                compact=dict(row,candidate_logits_f32_base64=base64.b64encode(arrays.tobytes()).decode(),
                    candidate_logits_shape=list(arrays.shape),unencoded_semantic_sha256=C.digest(row))
                compact['queries']=[{k:v for k,v in q.items() if k!='candidate_logits'} for q in row['queries']]
                assert C.digest(decode(compact))==compact['unencoded_semantic_sha256'], 'Storage must be numerically lossless'
                yield compact
            else:yield row
            count+=1
    C.save_rows(path,encoded_rows())
    seal.update(records=C.binding(path),logits_encoding='little-endian float32 base64, per row shape84x66; lossless original tensor values',
        source_numeric_semantics_unchanged=True,private_original_sha256=C.sha(original),private_original_bytes=original.stat().st_size,
        lossless_rows_verified=count,observations_not_regenerated=True)
    C.write(seal_path,seal)
    C.write(C.DOC/'OBSERVATION_STORAGE_AUDIT.json',dict(complete=True,rows=count,
        original_sha256=C.sha(original),original_bytes=original.stat().st_size,public=C.binding(path),
        float32_logits_exact_roundtrip=True,all_original_fields_reconstructable=True,
        decoder='compact_observations.decode',new_detector_calls=0,new_head_calls=0,new_solver_calls=0,
        original_preserved_in_private_cache=True))
    print('OBSERVATION LOSSLESS COMPACTION',count,original.stat().st_size,path.stat().st_size,flush=True)

if __name__=='__main__':run()

"""Check complete and interrupted full-witness serialization before real runs."""
from __future__ import annotations
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from . import common as C
from .run import RowWriter,FrameStreams


def read_rows(path):
    with gzip.open(path,'rt') as stream:return [json.loads(line) for line in stream]


def byte_exact(root):
    original=root/'original';streamed=root/'streamed';original.mkdir();streamed.mkdir()
    rows=[dict(id='fixture0',nested={'all_candidate_solutions':[dict(generator_ids=[1,2,4,6],R_cf=[[1,0,0],[0,1,0],[0,0,1]])]},xy=[float('nan'),2.],label='경계'),dict(id='fixture1',empty=[],value=None)]
    first=original/'records.jsonl.gz';second=streamed/'records.jsonl.gz'
    C.save_rows(first,rows);writer=RowWriter(second);[writer.write(row) for row in rows]
    assert writer.count==2 and not second.exists();writer.promote()
    assert writer.closed and writer.published and not writer.close_errors
    assert not writer.pending_path.exists() and first.read_bytes()==second.read_bytes()
    assert read_rows(second)==C.finite(rows)
    return dict(same_original_gzip_bytes=True,full_candidate_witness_retained=True,rows=2)


def interrupted_and_exclusive(root):
    target=root/'unfinished.jsonl.gz';retained=root/'INTERRUPTED_unfinished.jsonl.gz'
    writer=RowWriter(target,interrupted_path=retained)
    writer.write(dict(id='prefix0',all_candidate_solutions=[{'state':'recorded'}]));writer.preserve_interrupted()
    assert not target.exists() and writer.pending_path.exists() and retained.exists()
    assert read_rows(retained)==[dict(id='prefix0',all_candidate_solutions=[{'state':'recorded'}])]
    assert retained.stat().st_ino==writer.pending_path.stat().st_ino and not writer.preservation_errors
    replacement=root/'collision.jsonl.gz';protected=b'user-owned-file';w=RowWriter(replacement,interrupted_path=root/'INTERRUPTED_collision.jsonl.gz');w.write(dict(id='prefix1'));replacement.write_bytes(protected)
    try:w.promote()
    except FileExistsError:pass
    else:raise AssertionError('existing final file was overwritten')
    w.preserve_interrupted();assert replacement.read_bytes()==protected and w.pending_path.exists()
    return dict(interrupted_prefix_readable=True,existing_final_never_overwritten=True)


def fsync_failure(root):
    target=root/'fsync.jsonl.gz';retained=root/'INTERRUPTED_fsync.jsonl.gz';w=RowWriter(target,interrupted_path=retained);w.write(dict(id='prefix2'))
    saved=os.fsync
    def fail(_):raise OSError('injected fixture fsync failure')
    try:
        os.fsync=fail;w.close()
    finally:os.fsync=saved
    assert any(e['phase']=='raw_fsync' for e in w.close_errors)
    try:w.promote()
    except RuntimeError:pass
    else:raise AssertionError('unfsynced output was published')
    w.preserve_interrupted();assert not target.exists() and read_rows(retained)==[dict(id='prefix2')]
    return dict(fsync_failure_blocks_final_publication=True,prefix_retained=True)


def population_and_duplicate(root):
    output=root/'population';args=SimpleNamespace(output=str(output),source_root='/home/minjae/Documents/github/pallet-pose',baseline_root='/home/minjae/Documents/github/pallet-pose-handoff-20261006',fits='/dev/shm/pallet-kp-supervision-repair-private-20261010/learned_fits')
    streams=FrameStreams(args)
    rows=dict(geometry=[dict(id='same',method=m,witness={'all_candidate_solutions':[{'k':i}]}) for i,m in enumerate(C.METHODS)],fixed=[dict(id='same',method=m) for m in ('BASE','N3_SUBPIX')],observations=[dict(id='same',head_arm=a) for a in C.ARMS])
    for kind,value in rows.items():streams.append(kind,value)
    before=streams.counts()
    try:streams.append('geometry',[rows['geometry'][0]])
    except RuntimeError:pass
    else:raise AssertionError('duplicate was serialized')
    assert streams.counts()==before==dict(geometry=8,fixed=2,observations=3)
    streams.publish()
    for kind,value in rows.items():assert read_rows(output/streams.NAMES[kind])==value
    assert streams.method_counts==Counter(C.METHODS) and streams.head_counts==Counter(C.ARMS)
    return dict(eight_plus_two_plus_three_population=True,duplicate_rejected_before_write=True,full_witnesses_unchanged=True)


def main():
    target=C.DOC/'STREAMING_CHECKS.json';C.require(not target.exists(),'preserve stream checks')
    C.PRIVATE.mkdir(parents=True,exist_ok=True)
    checks=[]
    with tempfile.TemporaryDirectory(prefix='stream-fixtures-',dir=C.PRIVATE) as temp:
        root=Path(temp)
        for fn in (byte_exact,interrupted_and_exclusive,fsync_failure,population_and_duplicate):
            try:checks.append(dict(name=fn.__name__,passed=True,details=fn(root)))
            except BaseException as e:checks.append(dict(name=fn.__name__,passed=False,error=dict(type=type(e).__name__,message=str(e))))
    result=dict(passed=all(r['passed'] for r in checks),complete=True,checks=checks,check_groups=4,code={k:C.binding(p) for k,p in dict(tests=__file__,writer=Path(__file__).with_name('run.py'),runtime=Path(__file__).with_name('runtime.py'),evaluate=Path(__file__).with_name('evaluate.py'),renderer=Path(__file__).with_name('render.py')).items()},actual_models_PnP_GT_optimization_training_RGB_calls=0,limits='Owned synthetic serialized fixtures only; does not certify real inference or accuracy')
    C.write_new(target,result);print('STREAMING_CHECKS',result['passed'],4,flush=True)
    raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()

"""Same frozen pose costs, with durable row reservations and buffered WAL.

Reservation is not an actual F call. Missing completions in an interrupted
reservation block replay: they can never be quietly retried. Completed old
chunks are read-only. No candidate geometry, pose arithmetic or target changes.
"""
from pathlib import Path
import argparse, fcntl, json, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from . import cost_cache as old

DOC=old.DOC
PNP=old.PNP
WORKER={}


def durable_write(path,value):
    """Sync both replacement content and its directory entry before F."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    with temporary.open('w') as stream:
        stream.write(json.dumps(old.clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def reserve_row(path,binding,writer_sha,row,remaining_indices,prefix_bytes):
    packet=dict(schema='durable_row_reservation_v1',status='ROW_RESERVED',binding=binding,
                writer_sha256=writer_sha,row=int(row),remaining_indices=list(map(int,remaining_indices)),
                prefix_bytes=int(prefix_bytes),reserved_count=len(remaining_indices),
                actual_F_started=None,meaning='reserved is not actual; only durable COMPLETE proves executed F',time_unix=time.time())
    assert len(packet['remaining_indices'])==len(set(packet['remaining_indices']))
    durable_write(path,packet);return packet


def guard_recovery(guard,completed_keys,tail_bytes=0):
    row=int(guard['row']);reserved={(row,int(i)) for i in guard['remaining_indices']}
    complete=reserved&set(completed_keys);missing=sorted(i for r,i in reserved-complete)
    return dict(status='COMPLETE_PROVEN' if not missing and not tail_bytes else 'BLOCKED',
                row=row,reserved_count=len(reserved),durable_complete=len(complete),
                ambiguous_remaining=missing,tail_bytes=int(tail_bytes),actual_F_started_exact=len(reserved) if not missing and not tail_bytes else None,
                actual_F_started_bounds=[len(complete),len(reserved)],new_F_calls=0)


def require_quiescence(transition):
    assert transition['status']=='OLD_QUIESCED_READY_FAST'
    assert transition['incomplete_attempted_F']==0
    processes=transition['old_producer_processes'];assert processes
    assert len({int(p['pid']) for p in processes})==len(processes)
    for process in processes:
        path=Path('/proc')/str(int(process['pid']))/'stat'
        try:fields=path.read_text().rsplit(') ',1)[1].split()
        except FileNotFoundError:continue
        if int(fields[19])==int(process['start_ticks']):
            assert fields[0]=='Z',('Old producer is still alive',process['pid'],fields[0])


class BufferedJournal:
    """Original byte layout; row guard is durable before buffered records."""
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        existed=self.path.exists()
        self.fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        if not existed:
            # The first durable row must retain the WAL filename after crash.
            os.fdatasync(self.fd)
            directory=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
    def append(self,packet):
        count=os.write(self.fd,packet)
        if count!=len(packet):raise IOError('Partial WAL write: block the reserved row')
    def attempt(self,row,index):self.append(old.ATTEMPT.pack(row,index,1))
    def complete(self,row,index,result,pnp):
        self.append(old.COMPLETE.pack(row,index,2,result['ADDsym_m'],result['translation_cm'],result['rotation_deg'],result['WD'],*(pnp[k] for k in PNP)))
    def sync(self):os.fdatasync(self.fd)
    def close(self):os.close(self.fd)


def readonly_reuse(cache,chunk,rows,binding,arrays):
    """Do not rewrite a successful legacy receipt or its arrays/journal."""
    path=Path(cache)/'receipts'/f'chunk_{chunk:04d}.json'
    if not path.exists():return None
    receipt=old.read(path)
    if receipt['status']!='PASS':return None
    expected=sum(int(arrays['action_counts'][r]) for r in rows)
    assert receipt['binding']==binding and receipt['rows']==list(map(int,rows))
    assert receipt['attempted']==receipt['completed']==expected and receipt['incomplete_rows']==0
    assert arrays['done'][rows].all() and (arrays['row_state'][rows]==2).all()
    journal=Path(cache)/'journals'/f'chunk_{chunk:04d}.bin'
    assert journal.stat().st_size==receipt['journal_bytes']==37*expected and old.sha(journal)==receipt['journal_sha256']
    return dict(receipt,new_F_calls=0,fast_invocation_new_F_calls=0,readonly_reuse=True,
                original_receipt_sha256=old.sha(path),original_receipt_new_F_calls=receipt['new_F_calls'])


def _worker(task):
    source_root,bank_cache,cache,chunk,rows,binding,writer_sha,io_binding=task
    old.cv2.setNumThreads(1);old.torch.set_num_threads(1)
    assert old.sha(Path(__file__))==writer_sha,'Fast writer changed after seal'
    key=(source_root,bank_cache,cache,binding,writer_sha)
    if WORKER.get('key')!=key:
        data=old.Data(source_root);bank=old.ReadOnlyBanks(data,bank_cache);arrays=old.open_arrays(cache,len(data.indices))
        WORKER.update(key=key,data=data,bank=bank,arrays=arrays)
    data,bank,arrays=(WORKER[k] for k in ('data','bank','arrays'))
    reused=readonly_reuse(cache,chunk,rows,binding,arrays)
    if reused is not None:return reused
    journal_path=Path(cache)/'journals'/f'chunk_{chunk:04d}.bin';meta=journal_path.with_suffix('.binding.json')
    if meta.exists():assert old.read(meta)==dict(binding=binding,chunk=chunk)
    else:
        assert not journal_path.exists(),'Unbound old WAL must not be reused'
        durable_write(meta,dict(binding=binding,chunk=chunk))
    guard_path=Path(cache)/'fast_guards'/f'chunk_{chunk:04d}.json'
    # Dirty pages surviving a process crash are made durable before any reuse.
    journal=BufferedJournal(journal_path);journal.sync()
    prior=old.replay_journal(journal_path,arrays,rows)
    guard=old.read(guard_path) if guard_path.exists() else None
    if guard is not None:
        assert guard['binding']==binding and guard['writer_sha256']==writer_sha
        recovered=guard_recovery(guard,prior['completed'],prior['tail_bytes'])
        if recovered['status']=='BLOCKED':
            journal.close();return dict(chunk=chunk,status='BLOCKED_INTERRUPTED_RESERVATION',rows=rows,
                new_F_calls=0,fast_invocation_new_F_calls=0,guard=recovered,
                WAL_attempt_records=len(prior['attempted']),durable_completed_F=len(prior['completed']))
        guard.update(status='ROW_COMPLETE',actual_F_started=len(guard['remaining_indices']),
                     completed_F=len(guard['remaining_indices']),wal_bytes=journal_path.stat().st_size,
                     recovered_from_full_WAL=True,completion_observed_unix=time.time())
        durable_write(guard_path,guard)
    if prior['tail_bytes'] or prior['attempted']!=prior['completed']:
        journal.close();return dict(chunk=chunk,status='BLOCKED_OLD_INCOMPLETE_ATTEMPT',rows=rows,
            new_F_calls=0,fast_invocation_new_F_calls=0,WAL_attempt_records=len(prior['attempted']),
            durable_completed_F=len(prior['completed']),tail_bytes=prior['tail_bytes'])
    # Root's quiescence makes every old chunk PASS or unstarted. Partial data
    # therefore must be protected by this writer's durable guard.
    assert guard is not None or not prior['attempted'],'Unreceipted old partial chunk needs independent review'
    started=time.monotonic();new_calls=0;completed_now=0;reserved_now=0
    active_guard=None
    try:
        with old.pnp_counter() as counter:
            for row in rows:
                frame=data.source_frame(row);b=bank.get(row,frame);n=len(b['points'])
                remaining=np.flatnonzero(arrays['candidate_state'][row,:n]!=2).tolist()
                if not remaining:
                    arrays['oracle_index'][row]=old.hard_target(arrays['cost_ADDsym_m'][row,:n],arrays['F_available'][row,:n])
                    arrays['done'][row]=True;arrays['row_state'][row]=2;continue
                assert (arrays['candidate_state'][row,remaining]==0).all()
                active_guard=reserve_row(guard_path,binding,writer_sha,row,remaining,journal_path.stat().st_size)
                reserved_now+=len(remaining)
                for index in remaining:
                    journal.attempt(row,index);arrays['candidate_state'][row,index]=1;before=counter.copy();new_calls+=1
                    result=old.candidate_cost(frame,b['points'][index]);pnp={k:counter[k]-before[k] for k in PNP}
                    journal.complete(row,index,result,pnp);completed_now+=1
                    arrays['cost_ADDsym_m'][row,index]=result['ADDsym_m'];arrays['translation_cm'][row,index]=result['translation_cm'];arrays['rotation_deg'][row,index]=result['rotation_deg'];arrays['WD_hypothesis'][row,index]=result['WD'];arrays['F_available'][row,index]=result['available'];arrays['candidate_state'][row,index]=2
                journal.sync() # Every row completion is durable before next reservation.
                arrays['oracle_index'][row]=old.hard_target(arrays['cost_ADDsym_m'][row,:n],arrays['F_available'][row,:n]);arrays['done'][row]=True;arrays['row_state'][row]=2
            if active_guard is not None:
                active_guard.update(status='ROW_COMPLETE',actual_F_started=len(active_guard['remaining_indices']),
                                    completed_F=len(active_guard['remaining_indices']),wal_bytes=journal_path.stat().st_size,
                                    completion_observed_unix=time.time())
                durable_write(guard_path,active_guard)
            for array in arrays.values():array.flush()
            cumulative=old.replay_journal(journal_path,arrays,rows)
            expected=sum(int(arrays['action_counts'][r]) for r in rows)
            assert cumulative['tail_bytes']==0 and len(cumulative['completed'])==len(cumulative['attempted'])==expected
            assert arrays['done'][rows].all()
            elapsed=time.monotonic()-started
            phase_path=Path(cache)/'fast_phases'/f'chunk_{chunk:04d}.json'
            previous=old.read(phase_path) if phase_path.exists() else {'seconds':0,'actual_F_started':0,'actual_F_completed':0,'invocations':[]}
            invocation=dict(actual_F_started=new_calls,actual_F_completed=completed_now,reserved_candidates=reserved_now,
                            seconds=elapsed,PnP_counts=counter,time_unix=time.time())
            previous['seconds']+=elapsed;previous['actual_F_started']+=new_calls;previous['actual_F_completed']+=completed_now;previous['invocations'].append(invocation)
            durable_write(phase_path,previous)
            receipt=dict(chunk=chunk,status='PASS',rows=rows,binding=binding,new_F_calls=expected,
                attempted=expected,completed=expected,incomplete_rows=0,
                all_F_invalid_rows=int((arrays['oracle_index'][rows]<0).sum()),PnP_counts=cumulative['PnP_counts'],
                seconds=previous['seconds'],this_invocation_seconds=elapsed,
                journal_sha256=old.sha(journal_path),journal_bytes=journal_path.stat().st_size,
                IO_binding=io_binding,fast_writer_sha256=writer_sha,
                fast_invocation_new_F_calls=new_calls,reserved_candidates_this_invocation=reserved_now,
                actual_F_started_this_invocation=new_calls,actual_F_completed_this_invocation=completed_now,
                actual_F_completed_cumulative=expected,guard_policy='row reservation durable before any F; fdatasync WAL before next guard')
            durable_write(Path(cache)/'receipts'/f'chunk_{chunk:04d}.json',receipt)
            return receipt
    except Exception as error:
        journal.sync()
        if active_guard is not None:
            active_guard.update(status='FAILED_RESERVED_ROW',exception=repr(error),
                                actual_F_started_this_invocation=new_calls,actual_F_completed_this_invocation=completed_now)
            durable_write(guard_path,active_guard)
        return dict(chunk=chunk,status='BLOCKED_RESERVED_ROW_EXCEPTION',rows=rows,new_F_calls=new_calls,
                    fast_invocation_new_F_calls=new_calls,actual_F_completed_this_invocation=completed_now,reason=repr(error))
    finally:journal.close()


def _generate_locked(source_root,bank_cache,cache,workers,bindings,preflight,transition):
    started=time.monotonic();writer_sha=old.sha(Path(__file__))
    cache=Path(cache).resolve();transition=Path(transition).resolve();phase=old.read(transition)
    require_quiescence(phase)
    data,bank,arrays,header=old.prepare(source_root,bank_cache,cache,bindings,preflight)
    slow_seconds=float(phase['slow_full_elapsed_seconds']);assert slow_seconds>=0
    if 'binding' in phase:assert phase['binding']==header['binding']
    io_contract=dict(science_binding=header['binding'],cost_cache_code_sha256=header['cost_cache_code_sha256'],
        fast_writer_sha256=writer_sha,transition_sha256=old.sha(transition),transition_path=str(transition),
        native_auditor_sha256=old.sha(Path(__file__).with_name('native_cost_audit.py')),
        unchanged='frozen prepare/candidate_cost/hard_target/nativebanks; same FP64 ADD, proper symmetry, native indices, original F',
        policy='durable row-before reservation; buffered original 37byte WAL; fdatasync perrow; interrupted unknown reservation BLOCKED without retry')
    io_binding=old.hash_value(io_contract);io_path=cache/'FAST_IO_BINDING.json'
    if io_path.exists():assert old.read(io_path)==dict(binding=io_binding,contract=io_contract)
    else:durable_write(io_path,dict(binding=io_binding,contract=io_contract))
    rows=list(map(int,data.train_rows));tasks=[];reports=[]
    for start in range(0,len(rows),64):
        group=rows[start:start+64];chunk=start//64
        reused=readonly_reuse(cache,chunk,group,header['binding'],arrays)
        if reused is None:tasks.append((str(source_root),str(bank_cache),str(cache),chunk,group,header['binding'],writer_sha,io_binding))
        else:reports.append(reused)
    if workers==1:
        for task in tasks:
            reports.append(_worker(task));print('FAST_COST_CHUNK',task[3],reports[-1]['status'],round(time.monotonic()-started,1),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for report in pool.map(_worker,tasks,chunksize=1):
                reports.append(report)
                if len(reports)%10==0:print('FAST_COST_PROGRESS',len(reports),874,round(time.monotonic()-started,1),flush=True)
    success=len(reports)==874 and all(r['status']=='PASS' for r in reports)
    result=old.summarize(cache,hash_arrays=success)
    # Never expose PASS before every scheduled task, row and guard is complete.
    if not success:result['status']='BLOCKED_INTERRUPTED_RESERVATION'
    disk_receipts=[old.read(p) for p in sorted((cache/'receipts').glob('*.json'))]
    fast_elapsed=time.monotonic()-started
    result.update(invocation_new_F_calls=sum(r['new_F_calls'] for r in disk_receipts),
                  invocation_new_F_calls_scope='logical whole generation after initial first16 timing; slow+fast disk receipt union, not current fast invocation',
                  fast_invocation_new_F_calls=sum(r.get('fast_invocation_new_F_calls',0) for r in reports if not r.get('readonly_reuse')),
                  seconds_wall=slow_seconds+fast_elapsed,slow_full_elapsed_seconds=slow_seconds,
                  fast_elapsed_seconds=fast_elapsed,workers=workers,cache_path=str(cache),
                  header_sha256=old.sha(cache/'cache_header.json'),IO_binding=io_binding,
                  fast_writer_sha256=writer_sha,transition_sha256=old.sha(transition),
                  reservation_scope='Reserved candidates are not actual F calls; missing durable COMPLETE after interruption is unknown and BLOCKED',
                  blocked_chunks=[r for r in reports if r['status']!='PASS'])
    if success:
        from .native_cost_audit import verify_native_cost
        assert old.sha(Path(__file__).with_name('native_cost_audit.py'))==io_contract['native_auditor_sha256']
        audit_started=time.monotonic()
        durable_write(cache/'PRETRAIN_COST_MEMORY_MANIFEST.json',result)
        audit=verify_native_cost(cache,result)
        assert audit['status']=='PASS'
        result['pretrain_cost_verification_sha256']=old.sha(DOC/'PRETRAIN_COST_VERIFICATION.json')
        result['native_audit_wall_seconds']=time.monotonic()-audit_started
    else:result['native_audit_wall_seconds']=0
    fast_elapsed=time.monotonic()-started
    result.update(seconds_wall=slow_seconds+fast_elapsed,fast_elapsed_seconds=fast_elapsed)
    durable_write(DOC/'POSE_COST_CACHE_MANIFEST.json',result)
    durable_write(DOC/'FAST_IO_EXECUTION.json',dict(status=result['status'],IO_binding=io_binding,
        fast_writer_sha256=writer_sha,original_cost_code_sha256=header['cost_cache_code_sha256'],
        readonly_reused_chunks=sum(bool(r.get('readonly_reuse')) for r in reports),new_chunks=len(tasks),
        fast_actual_F_started_this_invocation=result['fast_invocation_new_F_calls'],
        slow_full_elapsed_seconds=slow_seconds,fast_elapsed_seconds=fast_elapsed,
        original_header_bytes_unchanged=True,original_source_bank_math_unchanged=True,
        blocked_chunks=result['blocked_chunks']))
    return result


def generate(source_root,bank_cache,cache,workers,bindings,preflight,transition):
    """Exclusive fast-producer lock; old PID quiescence is checked separately."""
    cache=Path(cache).resolve();assert cache.is_dir() and workers>0
    fd=os.open(cache/'PRODUCER.lock',os.O_RDWR|os.O_CREAT,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _generate_locked(source_root,bank_cache,cache,workers,bindings,preflight,transition)
    finally:os.close(fd)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','bank-cache','cache-dir','bindings','preflight','transition'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args(argv)
    result=generate(args.source_root,args.bank_cache,args.cache_dir,args.workers,args.bindings,args.preflight,args.transition)
    print(json.dumps({k:v for k,v in old.clean(result).items() if k not in ('files','blocked_chunks')},ensure_ascii=False),flush=True)
    if result['status']!='PASS':raise SystemExit(2)


if __name__=='__main__':main()

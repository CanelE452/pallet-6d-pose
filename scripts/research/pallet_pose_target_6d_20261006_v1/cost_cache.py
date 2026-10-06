"""Read immutable native GEO actions, solve final F once/action, cache ADDsym.

TRAIN source-row indexing is retained. Durable binary write-ahead journals keep
completed actions reusable and prohibit repeating an incomplete attempted F.
No training, geometry generation, inference scoring, paper or legacy entrypoint.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse,hashlib,json,os,struct,sys,time,types
sys.dont_write_bytecode=True
import numpy as np
import cv2
import torch
from concurrent.futures import ProcessPoolExecutor
from .baseline import resolve
BASELINE_ROOT=resolve()
from scripts.research.pallet_joint_action_handoff_20261006_v1.a_data import Data
from scripts.research.pallet_joint_action_handoff_20261006_v1.a_common import POSE

ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'_docs/experiments/pallet_pose_target_6d_20261006_v1'
OLD=BASELINE_ROOT/'_docs/experiments/pallet_joint_action_handoff_20261006_v1'
MAX_ACTIONS=201
WD={'UNAVAILABLE':0,'SQUARE_IDENTICAL_WD':1,'long-face-front':2,'short-face-front':3}
PNP=('solvePnP','solvePnPGeneric','solvePnPRefineLM')
# Same function bytecode, same ADD/T/R operations. Only unused IoU computation
# is disabled in a private globals copy; original POSE module is never patched.
metric_no_iou=types.FunctionType(POSE.metric.__code__,dict(POSE.metric.__globals__,oriented_iou_3d=lambda *a,**k:float('nan')),'inherited_metric_without_unused_iou')
ATTEMPT=struct.Struct('<IHB') # source row, unchanged action index, record tag1
COMPLETE=struct.Struct('<IHBdffBHHH') # tag2, ADD64,T32,R32,WD8,internalPnP16x3
SPECS={'cost_ADDsym_m':('float64',(MAX_ACTIONS,),float('inf')),'translation_cm':('float32',(MAX_ACTIONS,),float('inf')),'rotation_deg':('float32',(MAX_ACTIONS,),float('inf')),'F_available':('bool',(MAX_ACTIONS,),False),'WD_hypothesis':('uint8',(MAX_ACTIONS,),0),'candidate_state':('uint8',(MAX_ACTIONS,),0),'action_counts':('uint16',(),0),'oracle_index':('int16',(),-1),'row_state':('uint8',(),0),'done':('bool',(),False),'usable_train':('bool',(),False)}


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def hash_value(x):return hashlib.sha256(json.dumps(clean(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.pending')
    with tmp.open('w') as f:f.write(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def array_hash(x):
    x=np.ascontiguousarray(x);h=hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode());h.update(x.tobytes());return h.hexdigest()


@contextmanager
def pnp_counter():
    counts={name:0 for name in PNP};original={name:getattr(cv2,name) for name in PNP}
    def wrap(name):
        def call(*a,**k):counts[name]+=1;return original[name](*a,**k)
        return call
    for name in PNP:setattr(cv2,name,wrap(name))
    try:yield counts
    finally:
        for name in PNP:setattr(cv2,name,original[name])


def candidate_cost(frame,q):
    """One original final F, then the original approved-group metric."""
    pose=POSE.infer(q,frame['K'],frame['xyz'],frame['source'])
    if not pose['available']:return dict(available=False,ADDsym_m=float('inf'),translation_cm=float('inf'),rotation_deg=float('inf'),WD=0,final_hypothesis=None,pose=pose)
    m=metric_no_iou((frame['id'],pose,frame['truth']))
    assert np.isfinite(m['ADDsym_m']) and np.isfinite(m['translation_cm']) and np.isfinite(m['rotation_deg'])
    name=pose['selected_hypothesis'];assert name in WD,('Unknown final hypothesis',name)
    return dict(available=True,ADDsym_m=m['ADDsym_m'],translation_cm=m['translation_cm'],rotation_deg=m['rotation_deg'],WD=WD[name],final_hypothesis=name,pose=pose)


def hard_target(cost,available):
    """First native candidate index; every failed action is +inf."""
    cost=np.asarray(cost,dtype=np.float64);available=np.asarray(available,bool)
    assert cost.ndim==1 and available.shape==cost.shape and (~available|np.isfinite(cost)).all()
    assert np.isposinf(cost[~available]).all()
    return int(np.argmin(cost)) if available.any() else -1


class ReadOnlyBanks:
    def __init__(self,data,directory):
        self.data=data;self.directory=Path(directory)
        self.points=np.load(self.directory/'source_banks.npy',mmap_mode='r');self.counts=np.load(self.directory/'source_counts.npy',mmap_mode='r');self.hyp=np.load(self.directory/'source_hypothesis.npy',mmap_mode='r')
        self.names=read(self.directory/'BANK_NAMES.json');self.binding=read(self.directory/'BANK_BINDING.json')['binding']
        assert self.points.dtype==np.float64 and self.points.shape==(len(data.indices),201,9,2) and self.counts.shape==(len(data.indices),)
        assert not self.points.flags.writeable and not self.counts.flags.writeable
        assert (self.counts[data.train_rows]>0).all(),'Missing original TRAIN bank; regeneration prohibited'
        for p,h in read(self.directory/'GENERATION_CODE_BINDINGS.json')['code_bindings'].items():assert sha(BASELINE_ROOT/p)==h,('Original generation/F code changed',p)
    def get(self,row,frame=None):
        n=int(self.counts[row]);assert 0<n<=201
        points=np.array(self.points[row,:n],copy=True)
        if frame is not None:
            assert np.array_equal(points[0],frame['q'],equal_nan=True),'Original NoOp changed'
            c=points[:,8:];q=frame['q'][None,8:];assert np.all((c==q)|(np.isnan(c)&np.isnan(q))),'Original center changed'
        return dict(points=points,hypotheses=[self.names[int(h)] for h in self.hyp[row,:n]])


class BinaryJournal:
    """Each small record is synchronously durable before/after its F call."""
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_APPEND|os.O_DSYNC,0o600)
    def append(self,packet):
        n=os.write(self.fd,packet)
        if n!=len(packet):raise IOError('Incomplete durable journal write; stop before another F')
    def attempt(self,row,index):self.append(ATTEMPT.pack(row,index,1))
    def complete(self,row,index,result,pnp):
        self.append(COMPLETE.pack(row,index,2,result['ADDsym_m'],result['translation_cm'],result['rotation_deg'],result['WD'],*(pnp[k] for k in PNP)))
    def close(self):os.close(self.fd)


def replay_journal(path,arrays,allowed_rows):
    """Recover complete records; pending attempted actions can never be retried."""
    p=Path(path);attempted=set();completed=set();pnp={k:0 for k in PNP};tail_bytes=0
    if not p.exists():return dict(attempted=attempted,completed=completed,PnP_counts=pnp,tail_bytes=0)
    contents=p.read_bytes();offset=0;allowed=set(map(int,allowed_rows))
    while offset<len(contents):
        if len(contents)-offset<ATTEMPT.size:tail_bytes=len(contents)-offset;break
        row,index,tag=ATTEMPT.unpack_from(contents,offset);assert row in allowed and 0<=index<int(arrays['action_counts'][row])
        key=(row,index)
        if tag==1:
            assert key not in attempted,'Duplicate F attempt in journal';attempted.add(key);arrays['candidate_state'][row,index]=1;offset+=ATTEMPT.size
        elif tag==2:
            if len(contents)-offset<COMPLETE.size:tail_bytes=len(contents)-offset;break
            assert key in attempted and key not in completed,'Completion without unique attempt'
            values=COMPLETE.unpack_from(contents,offset);_,_,_,cost,T,R,WDcode,*counts=values
            assert WDcode in WD.values() and (np.isfinite(cost) or np.isposinf(cost))
            arrays['cost_ADDsym_m'][row,index]=cost;arrays['translation_cm'][row,index]=T;arrays['rotation_deg'][row,index]=R;arrays['WD_hypothesis'][row,index]=WDcode;arrays['F_available'][row,index]=np.isfinite(cost);arrays['candidate_state'][row,index]=2
            for name,num in zip(PNP,counts):pnp[name]+=num
            completed.add(key);offset+=COMPLETE.size
        else:raise ValueError('Invalid binary journal tag; never reset attempted budget')
    return dict(attempted=attempted,completed=completed,PnP_counts=pnp,tail_bytes=tail_bytes)


def open_arrays(directory,n,mode='r+'):
    return {k:np.lib.format.open_memmap(Path(directory)/(k+'.npy'),mode=mode,dtype=dtype,shape=(n,*shape)) for k,(dtype,shape,_) in SPECS.items()}


def require_ready(bindings_path,preflight_path):
    p=read(preflight_path);assert p['status']=='PASS'
    assert p['input_bindings_sha256']==sha(bindings_path) and p['protocol_sha256']==sha(DOC/'PROTOCOL.json')
    assert p['cost_cache_code_sha256']==sha(Path(__file__)),'Cost code changed after preflight'
    return p


def prepare(source_root,bank_cache,cache_dir,bindings_path,preflight_path):
    require_ready(bindings_path,preflight_path)
    source_root=Path(source_root).resolve();bank_cache=Path(bank_cache).resolve();cache_dir=Path(cache_dir).resolve()
    assert source_root not in cache_dir.parents and bank_cache not in cache_dir.parents and cache_dir not in (source_root,bank_cache)
    data=Data(source_root);bank=ReadOnlyBanks(data,bank_cache);rows=np.array(data.train_rows,np.int64)
    assert len(rows)==55915 and len(data.indices)==60000
    identities=[data.source['records'][data.indices[r]]['id'] for r in rows]
    manifest=read(OLD/'A_manifest.json');assert manifest['bank_binding']==bank.binding
    banksha=next(x['sha256'] for x in manifest['cache_files'] if x['name']=='source_banks.npy')
    binding=hash_value(dict(inputs_sha256=sha(bindings_path),protocol_sha256=sha(DOC/'PROTOCOL.json'),cost_code_sha256=sha(Path(__file__)),bank_sha256=banksha,bank_binding=bank.binding,train_rows_sha256=array_hash(rows),train_ids_sha256=hash_value(identities)))
    header_path=cache_dir/'cache_header.json'
    if header_path.exists():
        header=read(header_path);assert header['binding']==binding
        arrays=open_arrays(cache_dir,len(data.indices));assert np.array_equal(np.flatnonzero(arrays['usable_train']),rows)
    else:
        cache_dir.mkdir(parents=True,exist_ok=True)
        arrays={}
        for key,(dtype,shape,initial) in SPECS.items():
            a=np.lib.format.open_memmap(cache_dir/(key+'.npy'),mode='w+',dtype=dtype,shape=(len(data.indices),*shape));a[:]=initial;a.flush();arrays[key]=a
        arrays['usable_train'][rows]=True;arrays['row_state'][rows]=1;arrays['action_counts'][rows]=bank.counts[rows]
        for a in arrays.values():a.flush()
        np.save(cache_dir/'train_rows.npy',rows)
        header=dict(schema='final_F_ADDsym_TRAIN_cache_v1',binding=binding,source_rows=60000,usable_TRAIN_rows=55915,bank_binding=bank.binding,bank_sha256=banksha,source_root=str(source_root),bank_cache=str(bank_cache),cache_dir=str(cache_dir),input_bindings_sha256=sha(bindings_path),protocol_sha256=sha(DOC/'PROTOCOL.json'),cost_cache_code_sha256=sha(Path(__file__)),train_rows_sha256=array_hash(rows),train_ids_sha256=hash_value(identities),train_rows_file='train_rows.npy',candidate_index='exact original native bank index;NoOp0',cost='unchanged corresponding8 approved proper-rotation group ADDsym from actual final F;float64 argmin first tie',WD_codes=WD,row_state_codes={'OUT_OF_SCOPE':0,'PENDING':1,'COMPLETE':2,'INCOMPLETE_ATTEMPT':3},candidate_state_codes={'UNATTEMPTED':0,'ATTEMPTED_NO_DURABLE_COMPLETION':1,'COMPLETE':2},files={k:dict(file=k+'.npy',dtype=dtype,shape=[60000,*shape]) for k,(dtype,shape,_) in SPECS.items()},journal_record_bytes=dict(attempt=ATTEMPT.size,complete=COMPLETE.size),excluded_all_F_invalid_target=-1,unused_IoU='disabled only in private copy of original metric globals; ADD/T/R original bytecode remains identical',created_unix=time.time())
        write(header_path,header)
    return data,bank,arrays,header


_WORKER={}

def _worker(task):
    source_root,bank_cache,cache_dir,chunk,rows,binding=task
    cv2.setNumThreads(1);torch.set_num_threads(1)
    key=(source_root,bank_cache,cache_dir,binding)
    if _WORKER.get('key')!=key:
        data=Data(source_root);bank=ReadOnlyBanks(data,bank_cache);arrays=open_arrays(cache_dir,len(data.indices));_WORKER.update(key=key,data=data,bank=bank,arrays=arrays)
    data,bank,arrays=(_WORKER[k] for k in ('data','bank','arrays'));header=read(Path(cache_dir)/'cache_header.json');assert header['binding']==binding
    journal_path=Path(cache_dir)/'journals'/f'chunk_{chunk:04d}.bin';journal_meta=journal_path.with_suffix('.binding.json')
    if journal_meta.exists():assert read(journal_meta)==dict(binding=binding,chunk=chunk)
    else:
        assert not journal_path.exists(),'Unbound journal must never be reused'
        write(journal_meta,dict(binding=binding,chunk=chunk))
    existing=replay_journal(journal_path,arrays,rows)
    if existing['tail_bytes']:
        # Never append after a torn tail or guess whether its call happened.
        return dict(chunk=chunk,status='BLOCKED_JOURNAL_TAIL',rows=rows,attempted=len(existing['attempted']),completed=len(existing['completed']),tail_bytes=existing['tail_bytes'],new_F_calls=0,PnP_counts=existing['PnP_counts'])
    journal=BinaryJournal(journal_path);start=time.monotonic();new_calls=0;incomplete=0;allfail=0
    try:
        with pnp_counter() as counter:
            for row in rows:
                frame=data.source_frame(row);b=bank.get(row,frame);n=len(b['points'])
                for index,q in enumerate(b['points']):
                    state=int(arrays['candidate_state'][row,index])
                    if state==2:continue
                    if state==1:continue # Charged attempt; no second call ever.
                    journal.attempt(row,index);arrays['candidate_state'][row,index]=1;before=counter.copy();new_calls+=1
                    result=candidate_cost(frame,q);counts={k:counter[k]-before[k] for k in PNP};journal.complete(row,index,result,counts)
                    arrays['cost_ADDsym_m'][row,index]=result['ADDsym_m'];arrays['translation_cm'][row,index]=result['translation_cm'];arrays['rotation_deg'][row,index]=result['rotation_deg'];arrays['WD_hypothesis'][row,index]=result['WD'];arrays['F_available'][row,index]=result['available'];arrays['candidate_state'][row,index]=2
                if (arrays['candidate_state'][row,:n]==2).all():
                    idx=hard_target(arrays['cost_ADDsym_m'][row,:n],arrays['F_available'][row,:n]);arrays['oracle_index'][row]=idx;arrays['row_state'][row]=2;arrays['done'][row]=True;allfail+=idx<0
                else:arrays['row_state'][row]=3;arrays['done'][row]=False;arrays['oracle_index'][row]=-1;incomplete+=1
        for a in arrays.values():a.flush()
        receipt_path=Path(cache_dir)/'receipts'/f'chunk_{chunk:04d}.json';prior=read(receipt_path) if receipt_path.exists() else {}
        elapsed=time.monotonic()-start
        receipt=dict(chunk=chunk,status='PASS' if not incomplete else 'INCOMPLETE_ATTEMPTS',rows=rows,binding=binding,new_F_calls=new_calls,attempted=len(existing['attempted'])+new_calls,completed=int(sum((arrays['candidate_state'][r,:int(arrays['action_counts'][r])]==2).sum() for r in rows)),incomplete_rows=incomplete,all_F_invalid_rows=allfail,PnP_counts={k:existing['PnP_counts'][k]+counter[k] for k in PNP},seconds=prior.get('seconds',0)+elapsed,this_invocation_seconds=elapsed,journal_sha256=sha(journal_path),journal_bytes=journal_path.stat().st_size)
        write(receipt_path,receipt)
        return receipt
    finally:journal.close()


def summarize(cache_dir,hash_arrays=False):
    directory=Path(cache_dir);h=read(directory/'cache_header.json');a=open_arrays(directory,h['source_rows'],mode='r');rows=np.load(directory/'train_rows.npy');done=a['done'][rows];targets=a['oracle_index'][rows];counts=a['action_counts'][rows]
    receipts=[read(p) for p in sorted((directory/'receipts').glob('*.json'))]
    complete_candidates=int((a['candidate_state'][rows]==2).sum());pending_attempts=int((a['candidate_state'][rows]==1).sum());failures=int(((a['candidate_state'][rows]==2)&~a['F_available'][rows]).sum());valid=int(a['F_available'][rows].sum())
    indices=np.flatnonzero(done&(targets>=0));raw=a['cost_ADDsym_m'][rows[indices],0];oracle=a['cost_ADDsym_m'][rows[indices],targets[indices]];both=np.isfinite(raw)&np.isfinite(oracle);headroom=raw[both]-oracle[both]
    result=dict(schema=h['schema'],status='PASS' if done.all() and pending_attempts==0 else 'INCOMPLETE',binding=h['binding'],full_TRAIN_rows=55915,completed_rows=int(done.sum()),pending_rows=int((a['row_state'][rows]==1).sum()),incomplete_attempt_rows=int((a['row_state'][rows]==3).sum()),available_targets=int((done&(targets>=0)).sum()),excluded_all_F_invalid=int((done&(targets<0)).sum()),candidate_F_attempted=complete_candidates+pending_attempts,candidate_F_completed=complete_candidates,incomplete_attempted_F=pending_attempts,candidate_F_available=valid,candidate_F_failures=failures,registered_F_calls_total=int(counts.sum()),PnP_counts={k:sum(r.get('PnP_counts',{}).get(k,0) for r in receipts) for k in PNP},chunk_seconds_sum=sum(r.get('seconds',0) for r in receipts),oracle_NoOp=int((done&(targets==0)).sum()),headroom=dict(common_success=int(both.sum()),positive_gt_1e_7=int((headroom>1e-7).sum()),mean_m=float(headroom.mean()) if len(headroom) else None,median_m=float(np.median(headroom)) if len(headroom) else None,P90_m=float(np.quantile(headroom,.9)) if len(headroom) else None),WD_codes=WD,files=[])
    for key in [*SPECS,'train_rows']:
        p=directory/(key+'.npy');entry=dict(file=p.name,bytes=p.stat().st_size)
        if hash_arrays:entry['sha256']=sha(p)
        result['files'].append(entry)
    return result


def generate(source_root,bank_cache,cache_dir,workers,bindings_path,preflight_path,max_rows=None):
    begin=time.monotonic();data,bank,arrays,header=prepare(source_root,bank_cache,cache_dir,bindings_path,preflight_path)
    rows=[int(r) for r in data.train_rows];selected=rows if max_rows is None else rows[:max_rows]
    # Fixed chunking remains unchanged between first16 timing and full resume.
    tasks=[];selected_set=set(selected)
    for start in range(0,len(rows),64):
        subset=[r for r in rows[start:start+64] if r in selected_set]
        if subset:tasks.append((str(source_root),str(bank_cache),str(cache_dir),start//64,subset,header['binding']))
    reports=[]
    if workers==1:
        for task in tasks:reports.append(_worker(task));print('POSE_COST_CHUNK',task[3],reports[-1]['completed'],round(time.monotonic()-begin,1),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for report in pool.map(_worker,tasks,chunksize=1):
                reports.append(report)
                if len(reports)%10==0:print('POSE_COST_PROGRESS',len(reports),len(tasks),sum(r['completed'] for r in reports),round(time.monotonic()-begin,1),flush=True)
    result=summarize(cache_dir,hash_arrays=len(selected)==55915 and all(r['status']=='PASS' for r in reports));result.update(invocation_new_F_calls=sum(r['new_F_calls'] for r in reports),seconds_wall=time.monotonic()-begin,workers=workers,cache_path=str(Path(cache_dir).resolve()),header_sha256=sha(Path(cache_dir)/'cache_header.json'))
    write(DOC/'POSE_COST_CACHE_MANIFEST.json',result)
    if max_rows is not None:
        result['timing_registered_rows']=len(selected);result['estimated_remaining_serial_seconds']=result['seconds_wall']*(55915-len(selected))/len(selected);write(DOC/'COST_TIMING.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--bank-cache',type=Path,required=True);p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--bindings',type=Path,required=True);p.add_argument('--preflight',type=Path,required=True);p.add_argument('--workers',type=int,default=1);p.add_argument('--max-rows',type=int)
    a=p.parse_args();generate(a.source_root,a.bank_cache,a.cache_dir,a.workers,a.bindings,a.preflight,a.max_rows)
if __name__=='__main__':main()

"""Reference-free streaming availability/variation audit of saved F1 scalars.

No feature recomputation, fitted normalization, scores, targets, model or solver
imports. Validity zero is unavailable OR invalid; physical absence is not inferred.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

from .prediction_guard import PredictionGuard


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb')as stream:
        for value in iter(lambda:stream.read(1048576),b''):h.update(value)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(path):return json.loads(Path(path).read_text())
def records(path):
    with gzip.open(path,'rt')as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def analysis_guard(files,repo,output):
    return PredictionGuard(allowed_files=files,
        allowed_read_roots=[repo/'scripts',Path(sys.prefix),Path('/usr/lib'),Path('/usr/local/lib'),output],output_root=output)


class Moments:
    def __init__(self,m):
        self.n=np.zeros(m,dtype=np.int64);self.mean=np.zeros(m);self.m2=np.zeros(m)
        self.minimum=np.full(m,np.inf);self.maximum=np.full(m,-np.inf)

    def add(self,x,v):
        n=v.sum(axis=0);total=self.n+n
        mean=np.divide(np.where(v,x,0.).sum(axis=0),n,out=np.zeros_like(self.mean),where=n>0)
        m2=np.where(v,(x-mean)**2,0.).sum(axis=0);delta=mean-self.mean
        self.m2+=m2+np.divide(delta**2*self.n*n,total,out=np.zeros_like(self.mean),where=total>0)
        self.mean+=np.divide(delta*n,total,out=np.zeros_like(self.mean),where=total>0)
        self.n=total
        self.minimum=np.minimum(self.minimum,np.where(v,x,np.inf).min(axis=0))
        self.maximum=np.maximum(self.maximum,np.where(v,x,-np.inf).max(axis=0))

    def scalar(self,j):
        if not self.n[j]:return dict(n=0,minimum=None,maximum=None,mean=None,population_SD=None)
        return dict(n=int(self.n[j]),minimum=float(self.minimum[j]),maximum=float(self.maximum[j]),
            mean=float(self.mean[j]),population_SD=float(np.sqrt(max(self.m2[j]/self.n[j],0.))))


class Population:
    def __init__(self,m,normalizer):
        self.frames=0;self.actions=0;self.ids=[];self.frame_present=np.zeros(m,dtype=np.int64)
        self.frame_varied=np.zeros(m,dtype=np.int64);self.valid_zero=np.zeros(m,dtype=np.int64)
        self.raw=Moments(m);self.signed=Moments(m);self.normalized=Moments(m)
        self.train_mean=np.asarray(normalizer['mean']);self.train_scale=np.asarray(normalizer['scale'])
        self.train_seen=np.asarray(normalizer['valid_train_counts'])>0
        if any(a.shape!=(m,)for a in (self.train_mean,self.train_scale,self.train_seen))or not np.all(self.train_scale>0):
            raise ValueError('Fixed normalizer shape/positive scale mismatch')

    def add(self,row,m):
        if not row['actions']:raise ValueError('F1 action universe cannot be empty')
        x=np.asarray([a['extra_values']for a in row['actions']],dtype=np.float64)
        raw_v=np.asarray([a['extra_validity']for a in row['actions']])
        if x.shape!=(len(row['actions']),m)or raw_v.shape!=x.shape or not np.isfinite(x).all()or not np.isin(raw_v,[0,1]).all():
            raise ValueError('F1 scalar/validity contract violated')
        if len({a['candidate_id']for a in row['actions']})!=len(row['actions']):raise ValueError('Duplicate canonical action ID')
        v=raw_v==1;z=np.sign(x)*np.log1p(np.abs(x))
        model=np.where(v&self.train_seen,(z-self.train_mean)/self.train_scale,0.)
        self.frames+=1;self.actions+=len(x);self.ids.append(row['id'])
        self.frame_present+=v.any(axis=0);self.valid_zero+=((x==0)&v).sum(axis=0)
        lo=np.where(v,x,np.inf).min(axis=0);hi=np.where(v,x,-np.inf).max(axis=0)
        self.frame_varied+=v.any(axis=0)&(lo!=hi)
        self.raw.add(x,v);self.signed.add(z,v);self.normalized.add(model,v)

    def export(self,spec,normalizer):
        rows=[]
        for j,name in enumerate(spec['scalar_order']):
            valid=int(self.raw.n[j]);channel=spec['channels'][j]
            rows.append(dict(index=j,name=name,family=channel['family'],unit=channel['unit'],
                train_valid_actions=int(normalizer['valid_train_counts'][j]),
                train_numeric_support=bool(self.train_seen[j]),model_numeric_neutralized_without_train_support=not bool(self.train_seen[j]),
                valid_actions=valid,invalid_or_unavailable_actions=self.actions-valid,
                valid_zero_actions=int(self.valid_zero[j]),
                frames_with_valid_action=int(self.frame_present[j]),frames_without_valid_action=self.frames-int(self.frame_present[j]),
                frames_with_multiple_distinct_valid_values=int(self.frame_varied[j]),
                validity_varies_across_actions=0<valid<self.actions,
                numeric_varies_over_valid_actions=bool(valid and self.raw.minimum[j]!=self.raw.maximum[j]),
                raw=self.raw.scalar(j),signed_log1p=self.signed.scalar(j),
                fixed_normalized_float64_over_valid_actions=self.normalized.scalar(j)))
        return dict(frames=self.frames,actions=self.actions,ordered_IDs_sha256=hashlib.sha256(canonical(self.ids)).hexdigest(),scalars=rows)


def run(args):
    if not args.analysis_GO:raise PermissionError('Explicit root reference-free --analysis-GO required')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    files=[Path(getattr(args,name)).resolve()for name in ('synthetic','real','spec','normalizer','execution_summary')]
    repo=Path(__file__).resolve().parents[3]
    guard=analysis_guard(files,repo,output)
    start,cpu=time.monotonic(),time.process_time()
    with guard:
        synth,real,specpath,normpath,executionpath=files
        spec=read(specpath);norm=read(normpath);execution=read(executionpath)
        bindings={name:sha(p)for name,p in zip(('synthetic','real','spec','normalizer','execution_summary'),files)}
        for name,path in [('synthetic',synth),('real',real)]:
            if sha(path)!=execution['branches'][name]['features_sha256']:raise ValueError('Original sealed F1 features changed: '+name)
        if sha(specpath)!=execution['branches']['synthetic']['spec_sha256']or sha(specpath)!=execution['branches']['real']['spec_sha256']:
            raise ValueError('Original frozen feature spec changed')
        if sha(normpath)!=execution['normalizer']['sha256']or norm['features_sha256']!=sha(synth):raise ValueError('Original train-only normalizer changed')
        m=spec['scalar_count']
        if m!=144 or len(spec['scalar_order'])!=m or len(spec['channels'])!=m:raise ValueError('Frozen144 specification required')
        pops={name:Population(m,norm)for name in ('SYNTH_TRAIN768','SYNTH_VAL235','PRIMARY192','CAL46')}
        all_ids=[]
        for path in (synth,real):
            for row in records(path):
                if path==synth:
                    if row['split']not in ('train','validation'):raise ValueError('Unexpected synthetic split')
                    pop='SYNTH_TRAIN768'if row['split']=='train'else'SYNTH_VAL235'
                else:
                    if row['partition']not in ('TEST','REMAINDER','CAL'):raise ValueError('Unexpected real partition')
                    pop='CAL46'if row['partition']=='CAL'else'PRIMARY192'
                pops[pop].add(row,m);all_ids.append(row['id'])
        if len(all_ids)!=1241 or len(set(all_ids))!=1241:raise ValueError('Exact full1241 unique frame denominator required')
        expected=dict(SYNTH_TRAIN768=768,SYNTH_VAL235=235,PRIMARY192=192,CAL46=46)
        if any(pops[k].frames!=n for k,n in expected.items()):raise ValueError('Original partition denominators changed')
        if set(pops['SYNTH_TRAIN768'].ids)!=set(norm['training_frame_ids']):raise ValueError('Original normalizer train ID membership changed')
        if pops['SYNTH_TRAIN768'].raw.n.tolist()!=norm['valid_train_counts']:raise ValueError('Train scalar support counts disagree')
        data={name:p.export(spec,norm)for name,p in pops.items()}
        zero=[j for j,n in enumerate(norm['valid_train_counts'])if not n]
        zero_summary=[dict(index=j,name=spec['scalar_order'][j],populations={name:{key:p['scalars'][j][key]for key in
            ('valid_actions','invalid_or_unavailable_actions','frames_with_valid_action','frames_without_valid_action','numeric_varies_over_valid_actions')}
            for name,p in data.items()},model_numeric_values='always0 under frozen normalizer; original validity bit is retained')for j in zero]
        value=dict(schema='F1_REFERENCE_FREE_INFORMATION_AVAILABILITY_V1',status='COMPLETE',
            source_sha256=sha(__file__),input_sha256=bindings,scalar_count=m,populations=data,
            zero_train_support_scalars=zero_summary,
            scope='Saved scalar/validity availability and action-wise variation only; no accuracy or causal efficacy inference',
            missingness='validity0 means invalid OR unavailable under frozen spec; physical observation absence cannot be fully distinguished from these arrays alone',
            variation='Exact numeric inequality; no new thresholds. Descriptive action-weighted population SD, not evaluation sample error variance.',
            normalization='Existing signed-log1p train768 normalizer only, descriptive float64 expression before original float32 cast; no refit, clipping, new feature choice, or weights',
            train_support_count_exact=True,train_ID_membership_exact=True,
            train_feature_order_equals_normalizer_ID_order=pops['SYNTH_TRAIN768'].ids==norm['training_frame_ids'],new_features=0,new_model_calls=0,
            new_solve_calls=0,new_GT_or_reference_files_read=0)
        (output/'F1_INFORMATION_AVAILABILITY.json').write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        flat=[]
        for name,p in data.items():
            for row in p['scalars']:
                item=dict(population=name,frames=p['frames'],actions=p['actions'])
                for k,v in row.items():
                    if isinstance(v,dict):item.update({k+'/'+kk:vv for kk,vv in v.items()})
                    else:item[k]=v
                flat.append(item)
        with open(output/'F1_INFORMATION_AVAILABILITY.csv','w',newline='')as stream:
            writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
        result=dict(schema='F1_INFORMATION_ANALYSIS_SEAL_V1',status='COMPLETE',input_sha256=bindings,source_sha256=sha(__file__),
            artifacts={name:sha(output/name)for name in ('F1_INFORMATION_AVAILABILITY.json','F1_INFORMATION_AVAILABILITY.csv')},
            wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,guard=guard.receipt(),
            no_GT_or_reference_read=True,no_model_or_solver_import=True)
        (output/'ANALYSIS_SEAL.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('synthetic','real','spec','normalizer','execution-summary','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--analysis-GO',action='store_true');a=p.parse_args();result=run(a)
    print(json.dumps({k:result[k]for k in ('status','wall_seconds','CPU_seconds','peak_rss_kib')}))


if __name__=='__main__':main()

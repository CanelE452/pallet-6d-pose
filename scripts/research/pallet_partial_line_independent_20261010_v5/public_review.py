"""Standard-library public arithmetic review; no private models/data required."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
DOC=REPO/'_docs/experiments/pallet_partial_line_independent_20261010_v5'

def read(path):return json.loads(Path(path).read_text())
def rows(path):
    with gzip.open(path,'rt') as stream:return [json.loads(line) for line in stream if line.strip()]
def check_binding(path,b):
    assert path.stat().st_size==b['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==b['sha256'],str(path)
def quantile(a,p):
    if not a:return None
    a=sorted(a);i=(len(a)-1)*p;lo=math.floor(i);hi=math.ceil(i)
    return a[lo]+(a[hi]-a[lo])*(i-lo)
def close(a,b):
    assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<=1e-9+abs(b)*1e-12),(a,b)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=DOC)
    p.add_argument('--output',type=Path,required=True,help='new review receipt, e.g. /tmp/cornerwise_public_review.json')
    p.add_argument('--export-csv',type=Path,help='optional new all1470 per-frame CSV')
    args=p.parse_args();folder=args.input
    assert not args.output.exists() and not args.output.is_symlink(),'preserve review receipt'
    if args.export_csv:assert not args.export_csv.exists(),'preserve per-frame CSV'
    protocol=read(folder/'PROTOCOL.json');seal=read(folder/'GEOMETRY_SEAL.json');scoring=read(folder/'SCORING_RECEIPT.json');metrics=read(folder/'METRICS.json')
    for key,name in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('fixed_geometry','FIXED_GEOMETRY_SEALED.jsonl.gz'),('observations','OBSERVATIONS.jsonl.gz'),('parity','BASE_N3_PARITY.json')):
        check_binding(folder/name,seal[key])
    for key,name in (('predictions','PREDICTIONS.jsonl.gz'),('fixed_predictions','FIXED_PREDICTIONS.jsonl.gz')):check_binding(folder/name,scoring[key])
    public_bound=0;external_unchecked=[]
    for name,b in protocol['inputs'].items():
        if b['origin']=='public_repository':check_binding(REPO/b['path'],b);public_bound+=1
        else:external_unchecked.append(name)
    raw=rows(folder/'PREDICTIONS.jsonl.gz')+rows(folder/'FIXED_PREDICTIONS.jsonl.gz')
    methods=('BASE','N3_SUBPIX',*protocol['methods'])
    assert len(raw)==1470 and len({(r['method'],r['id']) for r in raw})==1470
    assert Counter(r['method'] for r in raw)=={m:245 for m in methods}
    cohort=read(REPO/protocol['inputs']['cohort']['path']);label={r['id']:r['label'] for r in cohort['frames']}
    assert set(label)=={r['id'] for r in raw}
    checks=0
    for stratum,labels in [('combined',{'clean','moderate'}),('easy',{'clean'}),('medium',{'moderate'})]:
        for method in methods:
            data=[r for r in raw if r['method']==method and label[r['id']] in labels]
            block=metrics['strata'][stratum]['methods'][method]
            assert block['denominator']==len(data)
            assert block['statuses']==dict(Counter(r['output_status'] for r in data))
            for scope,selected in [('operational',[r for r in data if r['pose']['available']]),('new_pose',[r for r in data if r['pose']['available'] and r['new_pose_estimated']]),('fallback',[r for r in data if r['pose']['available'] and r['fallback_used']])]:
                for key in ('translation_cm','rotation_deg','ADDsym_cm'):
                    values=[float(r['pose']['ADDsym_m'])*100 if key=='ADDsym_cm' else float(r['pose'][key]) for r in selected]
                    n=len(values);mean=math.fsum(values)/n if n else None
                    var=math.fsum((v-mean)**2 for v in values)/(n-1) if n>1 else None
                    actual=block['metrics'][scope][key];assert actual['n']==n
                    for k,v in dict(mean=mean,sample_variance=var,sample_std=math.sqrt(var) if var is not None else None,median=quantile(values,.5),P90=quantile(values,.9),max=max(values) if values else None).items():close(actual[k],v);checks+=1
    if args.export_csv:
        with args.export_csv.open('x',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['id','session','difficulty','method','status','new_pose','fallback','T_cm','R_deg','ADDsym_cm','excluded_H','fit_ids','inlier_ids','selected_boundary_ids'])
            for r in raw:
                solver=r.get('solver') or {};pose=r['pose'];contract=r.get('observation_contract') or {}
                writer.writerow([r['id'],r['session'],label[r['id']],r['method'],r['output_status'],r['new_pose_estimated'],r['fallback_used'],pose.get('translation_cm'),pose.get('rotation_deg'),float(pose['ADDsym_m'])*100 if pose.get('ADDsym_m') is not None else None,json.dumps(r['hidden_initial']),json.dumps(solver.get('fit_input_ids',[])),json.dumps(solver.get('final_inliers',[])),json.dumps(contract.get('hybrid_boundary_corner_ids',[]) if r['method'] in (protocol['primary'],'N3_INDEPENDENT_CORNERWISE_ROLE') else [])])
    result=dict(passed=True,rows=1470,frames=245,methods=list(methods),moment_checks=checks,
        public_frozen_input_bindings_checked=public_bound,external_frozen_dependencies_not_opened=external_unchecked,
        private_GT_model_PnP_training_RGB_calls=0,independent_physical_truth_or_GPU_accuracy_rerun=False,
        bootstrap_CI_checked_here=False,full_independent_CI_audit='VERIFICATION.json / verify.py requires original dependency bindings')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()

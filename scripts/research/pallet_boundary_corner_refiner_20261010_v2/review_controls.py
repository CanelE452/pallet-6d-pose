"""Run public-only positive, semantic negative and output-guard review controls.

Only an exclusive isolated fixture is changed. The scientific verifier and all
actual result files remain byte unchanged. No experiment module is imported.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode=True
NAME='pallet_boundary_corner_refiner_20261010_v2'
VERIFIER_SHA='6f6d4710fe2a1fed7735085e8a18bf50bdd81adde1f9b166830a2eb0aeebed59'


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):
    with Path(path).open() as f:return json.load(f)


def rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(s) for s in f if s.strip()]


def save_json(path,value):
    with Path(path).open('w') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def save_rows(path,values):
    with Path(path).open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:
        for value in values:f.write((json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n').encode())


def rebind(item,path):
    out=dict(item);out.update(sha256=sha(path),bytes=Path(path).stat().st_size);return out


def copy_new(source,target):
    with Path(source).open('rb') as f,Path(target).open('xb') as out:shutil.copyfileobj(f,out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    p.add_argument('--workspace',type=Path,default=Path('/dev/shm/pallet-boundary-review-controls-20261010-v2'))
    args=p.parse_args();root=args.root.resolve();doc=root/'_docs/experiments'/NAME
    workspace=args.workspace.absolute();require(not workspace.exists() and not workspace.is_symlink(),'preserve previous control workspace')
    require(not workspace.resolve().is_relative_to(root),'fixture must be external to publication repository')
    code=root/'scripts/research'/NAME/'review_verify.py';require(sha(code)==VERIFIER_SHA,'verifier differs from first PASS')
    source_receipt=read(doc/'REVIEW_CHECKS_ATTEMPT_A.json');require(source_receipt['status']=='PASS' and source_receipt['complete'],'actual baseline review must already pass')
    outputs=['REVIEW_PUBLIC_POSITIVE.json','REVIEW_CONTROL_MEAN_FAILURE.json','REVIEW_CONTROL_HIDDEN_FAILURE.json','REVIEW_CONTROL_CHECKS.json']
    require(all(not (doc/n).exists() and not (doc/n).is_symlink() for n in outputs),'preserve existing control receipts')
    exports=read(doc/'PUBLIC_EXPORT.json')['copies']
    public_aliases={r['public']['sha256']:root/r['public']['path'] for r in exports}
    sources={code.relative_to(root):code}
    for item in source_receipt['verified_inputs']:
        path=root/item['path']
        if not path.is_file() or sha(path)!=item['sha256']:path=doc/Path(item['path']).name
        if not path.is_file() or sha(path)!=item['sha256']:path=public_aliases.get(item['sha256'],path)
        require(path.is_file() and sha(path)==item['sha256'] and path.stat().st_size==item['bytes'],'missing public baseline input '+Path(item['path']).name)
        sources[path.relative_to(root)]=path
    for item in exports:
        b=item['public'];path=root/b['path'];require(sha(path)==b['sha256'],'lossless public export differs');sources[path.relative_to(root)]=path
    before={str(rel):dict(sha256=sha(path),bytes=path.stat().st_size) for rel,path in sources.items()}
    workspace.mkdir();fixture=workspace/'public_fixture';fixture.mkdir()
    for rel,path in sources.items():
        dst=fixture/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
    fd=fixture/'_docs/experiments'/NAME;fc=fixture/'scripts/research'/NAME/'review_verify.py'
    require(all(sha(fixture/rel)==v['sha256'] for rel,v in before.items()),'fixture is not byte-exact public copy')
    controls=[];started=time.monotonic()

    def invoke(name,expected,expected_group=None):
        output=fd/('REVIEW_FIXTURE_'+name.upper()+'.json')
        cmd=[sys.executable,'-I','-S',str(fc),'--require-runtime','--output',str(output)]
        tick=time.monotonic();run=subprocess.run(cmd,cwd=fixture,text=True,capture_output=True)
        require(output.is_file(),'semantic control did not produce its fresh review receipt')
        receipt=read(output);status=receipt['status'];require(status==expected and run.returncode==(0 if expected=='PASS' else 1),'unexpected control result '+name)
        failure=next((c for c in receipt['checks'] if c['status']=='FAIL'),None)
        if expected_group:require(failure and failure['name']==expected_group and 'SHA binding' not in failure['error'],'control failed on binding rather than independent arithmetic')
        out=dict(name=name,expected_status=expected,observed_status=status,returncode=run.returncode,
                 exact_cli=cmd,wall_seconds=time.monotonic()-tick,failed_group=failure,
                 receipt=dict(path=str(output.relative_to(workspace)),sha256=sha(output),bytes=output.stat().st_size),
                 stdout=run.stdout,stderr=run.stderr)
        controls.append(out);return output,receipt

    positive,_=invoke('positive','PASS');copy_new(positive,doc/'REVIEW_PUBLIC_POSITIVE.json')
    metric_path=fd/'METRICS.json';original_metric=metric_path.read_bytes();metric=read(metric_path)
    affected={'N3_SUBPIX','N3_VALIDATED_ROLE'}
    for value in list(metric['strata'].values())+[dict(methods=metric['methods'],contrasts=metric['contrasts'])]:
        for method in affected:
            for group in value['methods'][method]['metrics'].values():
                if group['translation_cm']['mean'] is not None:group['translation_cm']['mean']+=1.
        for contrast,pairs in value['contrasts'].items():
            a,b=contrast.split('_minus_')
            for pair in pairs.values():
                for method,key in ((a,'new_marginals'),(b,'comparator_marginals')):
                    if method in affected and pair[key]['translation_cm']['mean'] is not None:pair[key]['translation_cm']['mean']+=1.
    require(metric['methods']==metric['strata']['combined']['methods'] and metric['contrasts']==metric['strata']['combined']['contrasts'],'mean mutation broke combined aliases')
    save_json(metric_path,metric);case_mean=workspace/'mean_mutation';case_mean.mkdir();shutil.copyfile(metric_path,case_mean/'METRICS.json')
    bad,_=invoke('mean','FAIL','metrics');copy_new(bad,doc/'REVIEW_CONTROL_MEAN_FAILURE.json');metric_path.write_bytes(original_metric)

    changed={}
    def preserve(name):
        path=fd/name;changed.setdefault(name,path.read_bytes());return path
    geometry_path=preserve('GEOMETRY_SEALED.jsonl.gz');geometry=rows(geometry_path)
    panel={v['id'] for v in read(fd/'RUNTIME.json')['panel']}
    target=min((r for r in geometry if r['method']=='N3_VALIDATED_ROLE' and r['id'] in panel and r['new_pose_estimated'] and r['hidden_initial']),key=lambda r:r['id'])
    fid=target['id'];corner=target['hidden_initial'][0];target['native_points'][corner][0]+=1.
    save_rows(geometry_path,geometry)
    predicted_path=preserve('PREDICTIONS.jsonl.gz');predicted=rows(predicted_path)
    for r in predicted:
        if r['method']=='N3_VALIDATED_ROLE' and r['id']==fid:r['native_points'][corner][0]+=1.
    save_rows(predicted_path,predicted)
    seal_path=preserve('GEOMETRY_SEAL.json');seal=read(seal_path);seal['geometry']=rebind(seal['geometry'],geometry_path);save_json(seal_path,seal)
    score_path=preserve('SCORING_RECEIPT.json');score=read(score_path)
    for k,path in [('geometry_seal',seal_path),('geometry',geometry_path),('predictions',predicted_path)]:score[k]=rebind(score[k],path)
    save_json(score_path,score)
    scored_start_path=preserve('SCORING_STARTED.json');value=read(scored_start_path);value['geometry_seal']=rebind(value['geometry_seal'],seal_path);save_json(scored_start_path,value)
    runtime_rows_path=preserve('RUNTIME_ROWS.jsonl.gz');runtime_rows=rows(runtime_rows_path)
    for r in runtime_rows:
        if r['arm']=='N3_VALIDATED_ROLE' and r['id']==fid:r['final_points'][corner][0]+=1.
    save_rows(runtime_rows_path,runtime_rows)
    runtime_path=preserve('RUNTIME.json');runtime=read(runtime_path);runtime['accuracy_seal']=rebind(runtime['accuracy_seal'],seal_path);runtime['raw_rows']=rebind(runtime['raw_rows'],runtime_rows_path);save_json(runtime_path,runtime)
    resume_protocol_path=preserve('RUNTIME_RESUME_PROTOCOL.json');rp=read(resume_protocol_path);rp['fixed_inputs']['accuracy_seal']=rebind(rp['fixed_inputs']['accuracy_seal'],seal_path);save_json(resume_protocol_path,rp)
    resume_receipt_path=preserve('RUNTIME_RESUME_RECEIPT.json');rr=read(resume_receipt_path)
    for k,path in [('resume_protocol',resume_protocol_path),('result',runtime_path),('rows',runtime_rows_path)]:rr[k]=rebind(rr[k],path)
    save_json(resume_receipt_path,rr)
    mp=preserve('METRICS.json');metric=read(mp);metric['bindings']['new_scored_rows']=rebind(metric['bindings']['new_scored_rows'],predicted_path);save_json(mp,metric)
    hidden_case=workspace/'hidden_mutation';hidden_case.mkdir()
    for name in changed:shutil.copyfile(fd/name,hidden_case/name)
    bad,_=invoke('hidden','FAIL','geometry');copy_new(bad,doc/'REVIEW_CONTROL_HIDDEN_FAILURE.json')
    controls[-1]['mutation']=dict(id=fid,corner_id=corner,delta_x_px=1.,coherent_seal_score_runtime_hashes_updated=True,pose_matrices_unchanged=True)
    for name,value in changed.items():(fd/name).write_bytes(value)

    def guard(name,output,protected):
        snapshot=protected.read_bytes();cmd=[sys.executable,'-I','-S',str(fc),'--require-runtime','--output',str(output)]
        tick=time.monotonic();run=subprocess.run(cmd,cwd=fixture,text=True,capture_output=True)
        require(run.returncode!=0 and protected.read_bytes()==snapshot,'output guard failed '+name)
        message='preserve existing receipt/output' if name=='overwrite' else 'receipt output is a symlink'
        require(message in run.stderr and 'Traceback' in run.stderr,'unexpected output guard rejection '+name)
        controls.append(dict(name=name,expected_rejection=message,observed_rejection=message,returncode=run.returncode,
                             exact_cli=cmd,wall_seconds=time.monotonic()-tick,protected_sha256=sha(protected),stdout=run.stdout,stderr=run.stderr,
                             arithmetic_started=False,protected_bytes_unchanged=True))
    overwrite=fd/'REVIEW_PROTECTED_OUTPUT.json';overwrite.write_text('Preserved overwrite control.\n');guard('overwrite',overwrite,overwrite)
    marker=workspace/'symlink_marker.txt';marker.write_text('Preserved symlink target.\n');link=fd/'REVIEW_SYMLINK_OUTPUT.json';link.symlink_to(marker);guard('symlink',link,marker)
    after={str(rel):dict(sha256=sha(path),bytes=path.stat().st_size) for rel,path in sources.items()}
    require(before==after and sha(code)==VERIFIER_SHA,'actual publication/source results changed during controls')
    result=dict(schema='boundary_refiner_public_review_controls_v2',complete=True,passed=True,controls=controls,
                verifier_sha256=VERIFIER_SHA,control_code=dict(path=str(Path(__file__).relative_to(root)),sha256=sha(__file__),bytes=Path(__file__).stat().st_size),
                source_materialized_file_count=len(before),source_before_after_same=True,source_inputs=before,
                public_only_fixture=True,private_data_or_weights_copied=False,
                actual_calls=dict(reviewer_subprocesses=5,positive_saved_arithmetic=1,semantic_negative_saved_arithmetic=2,
                                  pre_arithmetic_output_guards=2,detector=0,head=0,PnP=0,rays=0,optimizer_updates=0,private_GT=0,timing=0),
                fixture_retained='isolated external control workspace',wall_seconds=time.monotonic()-started,
                limits=['Controls validate arithmetic detection and output preservation; they do not certify physical GT, model execution, GPU timing, or bootstrap CI endpoints.'])
    with (doc/'REVIEW_CONTROL_CHECKS.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,controls=len(controls),actual_calls=result['actual_calls'],output=str(doc/'REVIEW_CONTROL_CHECKS.json'))))


if __name__=='__main__':main()

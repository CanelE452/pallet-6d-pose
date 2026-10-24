"""Execute only the two pre-registered source fits after independent GT audit."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--independent-audit',required=True);args=ap.parse_args()
    audit_path=Path(args.independent_audit).resolve();audit=json.loads(audit_path.read_text())
    assert audit.get('PASS')is True,'independent source target audit must PASS'
    lock=ROOT/'source_cache/SOURCE_INPUT_LOCK.json';spec=json.loads(lock.read_text())
    for rel,expected in spec['bindings'].items():assert sha(ROOT/rel)==expected,rel
    out=ROOT/'TWO_SOURCE_FITS_EXECUTION.json';assert not out.exists()
    info=dict(schema='V65_TWO_LOCKED_SOURCE_FITS',source_lock_sha256=sha(lock),independent_audit=dict(path=str(audit_path),sha256=sha(audit_path)),
              code_sha256=sha(__file__),python=sys.executable,private_data_read=False,PnP_calls=0,runs=[],started_unix=time.time())
    out.write_text(json.dumps(info,indent=2)+'\n')
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for candidate in ['SYNTH','MIXED']:
        command=[sys.executable,'model/fit_global_amodal_head.py','--train',f'source_cache/{candidate}_TRAIN_GLOBAL.npz',
             '--val',f'source_cache/{candidate}_VALIDATION_GLOBAL.npz','--out',f'runs/{candidate}_GLOBAL_UNWEIGHTED',
             '--eval-denylist','../query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json',
             '--provenance','source_cache/SOURCE_INPUT_LOCK.json','--protocol','PROTOCOL.json',
             '--max-updates','2000','--val-every-updates','100','--batch-size','256','--learning-rate','.001',
             '--weight-decay','.0001','--dense-loss-weight','.25','--global-loss-weight','.25','--seed','20261022','--device','cuda']
        log=ROOT/f'{candidate}_FIT.log'
        with log.open('x')as f:result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        info['runs'].append(dict(candidate=candidate,command=command,exit_code=result.returncode,log_sha256=sha(log)))
        out.write_text(json.dumps(info,indent=2)+'\n')
        if result.returncode:raise SystemExit(result.returncode)
    info['completed_unix']=time.time();info['optimizer_updates_total']=4000
    out.write_text(json.dumps(info,indent=2)+'\n')

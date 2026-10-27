"""Observe an explicit ROOT command; append actual postseal execution evidence only."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
D=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    spec_path=Path(sys.argv[1]);spec=json.loads(spec_path.read_text());before=sha(spec_path)
    assert spec['authorized'] is True
    label=spec['id'];assert label and all(c.isalnum() or c in '_-' for c in label)
    assert (D/'preflight/FINALIZATION_JOURNAL_CLOSED.json').exists()
    command=spec['command'];assert isinstance(command,list) and all(isinstance(x,str) for x in command)
    for p,h in spec.get('input_authority_sha256',{}).items():assert sha(p)==h,p
    for p,h in spec.get('source_sha256',{}).items():assert sha(p)==h,p
    log=D/'logs/finalization'/(label+'.stdout.log');log.parent.mkdir(exist_ok=True)
    started=utc();start=time.monotonic()
    with log.open('xb') as stream:result=subprocess.run(command,cwd=str(D),stdout=stream,stderr=subprocess.STDOUT)
    elapsed=time.monotonic()-start;ended=utc();assert sha(spec_path)==before
    receipts={str(p):sha(p) for p in map(Path,spec.get('output_receipts',[])) if p.is_file()}
    item=dict(schema='finalization_observed_command_v1',id=label,command=command,cwd=str(D),
        started_utc=started,ended_utc=ended,elapsed_seconds=elapsed,returncode=result.returncode,
        stdout_log=str(log),stdout_sha256=sha(log),ROOT_command_spec=str(spec_path),ROOT_command_spec_sha256=before,
        observer_source_sha256=sha(__file__),executed_source_sha256=spec.get('source_sha256',{}),
        input_authority_sha256=spec.get('input_authority_sha256',{}),output_receipt_sha256=receipts)
    assert (D/'preflight/FINALIZATION_JOURNAL_CLOSED.json').exists()
    data=(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n').encode()
    fd=os.open(str(D/'logs/FINALIZATION_POSTSEAL_COMMANDS.jsonl'),os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o644)
    try:assert os.write(fd,data)==len(data)
    finally:os.close(fd)
    print(json.dumps(item,ensure_ascii=False),flush=True)
    raise SystemExit(result.returncode)
if __name__=='__main__':main()

"""Run training through reporting; preserve a failure record on interruption."""
from pathlib import Path
import subprocess,sys,json,time
KIT=Path(__file__).resolve().parent
ROOT=KIT.parents[2]
RUN=ROOT/'data/pallet/results/pallet_observation_v25'
STAGES=[('intervention25.py',[],'INTERVENTION_DIAGNOSIS.json'),('train25.py',[], 'TRAINING_COMPLETE.json'),('eval25.py',['--phase','inference'],'INFERENCE_COMPLETE.json'),('eval25.py',['--phase','score'],'SUMMARY.json'),('audit25.py',[],'AUDIT.json'),('provenance25.py',[],'PROVENANCE_CHECK.json'),('report25.py',[],'REPORT_COMPLETE.json'),('delivery25.py',[],'DELIVERY_CHECK.json'),('complete25.py',[],'COMPLETION.json')]
def main():
    for script,args,marker in STAGES:
        if (RUN/marker).exists():continue
        status=dict(stage=script,args=args,started=time.strftime('%Y-%m-%d %H:%M:%S'),state='RUNNING')
        (RUN/'PROGRESS.json').write_text(json.dumps(status,indent=2)+'\n')
        try:
            subprocess.run([sys.executable,str(KIT/script),*args],cwd=ROOT,check=True)
            assert (RUN/marker).exists(),('Missing completion artifact',marker)
            doc=json.loads((RUN/marker).read_text());assert doc.get('complete') or doc.get('status')=='PASS'
        except BaseException as e:
            status.update(state='FAILED',error=str(e));(RUN/'PROGRESS.json').write_text(json.dumps(status,indent=2)+'\n');raise
    (RUN/'PROGRESS.json').write_text(json.dumps(dict(state='PIPELINE_COMPLETE',report=str(RUN/'index.html')),indent=2)+'\n')
if __name__=='__main__':main()

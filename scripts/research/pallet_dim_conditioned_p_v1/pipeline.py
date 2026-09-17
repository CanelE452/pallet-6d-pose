"""Sequential resumable stage runner. Errors stop, never silently change protocol."""
import argparse,os,subprocess,sys,time
import dcp_env as E

STAGES=[
 ('paper_logits',['paper_evaluate.py','cache'],'PAPER_LOGITS_COMPLETE.json'),
 ('paper_calibration',['paper_evaluate.py','calibrate'],'CALIBRATION_AND_SELECTION.json'),
 ('source_detection_audit',['paper_evaluate.py','baseline'],'SYNTH_DETECTION_AUDIT.json'),
 ('synthetic_heldout',['paper_evaluate.py','evaluate'],'SYNTH_HELDOUT_RESULTS.json'),
 ('DEV_observations',['dev_evaluate.py','cache'],'DEV_CACHE_COMPLETE.json'),
 ('DEV_inference',['dev_evaluate.py','infer'],'DEV_INFERENCE_COMPLETE.json'),
 ('DEV_metrics',['dev_evaluate.py','evaluate'],'REAL_DEV_RESULTS.json'),
 ('paired_statistics',['paired_statistics.py'],'PAIRED_STATISTICS.json'),
 ('paper_pose',['pose.py'],'PAPER_POSE_RESULTS.json'),
 ('metadata_sensitivity',['metadata_sensitivity.py'],'METADATA_SENSITIVITY.json'),
 ('runtime',['runtime.py'],'RUNTIME_AND_MEMORY.json'),
 ('square_cache',['square_data.py'],'SQUARE_CACHE_COMPLETE.json'),
 ('square_training',['train.py','square'],'SQUARE_TRAINING_COMPLETE.json'),
 ('square_evaluation',['secondary_evaluate.py','square'],'SQUARE_RESULTS.json'),
 ('square_pose',['pose.py','--track','square'],'SQUARE_POSE_RESULTS.json'),
 ('mixed_training',['train.py','mixed'],'MIXED_TRAINING_COMPLETE.json'),
 ('mixed_evaluation',['secondary_evaluate.py','mixed'],'MIXED_SQUARE_RESULTS.json'),
 ('mixed_pose',['pose.py','--track','mixed'],'MIXED_POSE_RESULTS.json'),
 ('closeout',['closeout.py'],'FINAL_AUDIT.json')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);args=p.parse_args();logroot=E.RAW/'logs';logroot.mkdir(parents=True,exist_ok=True)
    if args.wait_pid:
        while True:
            try:os.kill(args.wait_pid,0)
            except ProcessLookupError:break
            E.write(E.DOC/'PIPELINE_STATE.json',dict(status='WAITING_FOR_EXISTING_PAPER_TRAINER',pid=args.wait_pid,time=E.now()))
            time.sleep(20)
    assert E.read(E.DOC/'PAPER_TRAINING_COMPLETE.json')['complete'],'Paper training not completed; inspect exact failure, do not continue'
    for stage,argv,marker in STAGES:
        if (E.DOC/marker).exists() and E.read(E.DOC/marker).get('complete'):continue
        E.write(E.DOC/'PIPELINE_STATE.json',dict(status='RUNNING',stage=stage,argv=argv,time=E.now()))
        print('STAGE_START',stage,flush=True)
        path=logroot/(stage+'.log')
        with path.open('a') as f:
            result=subprocess.run([sys.executable,'-B',str(E.HERE/argv[0]),*argv[1:]],cwd=E.ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            E.write(E.DOC/'PIPELINE_STATE.json',dict(status='FAILED',stage=stage,returncode=result.returncode,log=str(path.relative_to(E.ROOT)),time=E.now()))
            print('STAGE_FAILED',stage,path,flush=True);raise SystemExit(result.returncode)
        assert E.read(E.DOC/marker).get('complete'),('Missing completion',stage)
        print('STAGE_COMPLETE',stage,flush=True)
    E.write(E.DOC/'PIPELINE_STATE.json',dict(status='COMPLETE_PENDING_GIT',time=E.now()));print('ALL_STAGES_COMPLETE_PENDING_GIT',flush=True)
if __name__=='__main__':main()

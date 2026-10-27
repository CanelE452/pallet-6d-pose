"""ROOT-authorized CPU queue: completed PRIMARY372 -> six-pair comparisons.

No IO on import; no NN, GPU lock, checkpoint, RGB, annotation or reference XY.
Requires DELIVERY/preflight/MAIN_MODEL_COMPARISONS_QUEUE_ROOT_REVIEW_V2.json:
{passed:true, source_sha256: exact self+FROZEN map}. Separate RUN GO at
MAIN_MODEL_COMPARISONS_QUEUE_GO_V2.json must contain schema main_comparisons_queue_go_v2,
passed:true, queue_authorized:true, combined_tag:final_v1, output_tag:main_v1,
source_sha256: same map, ROOT_review_sha256, producer:{pid:ROOT_RUNTIME_PID,
create_time:ROOT_RUNTIME_CREATE_TIME,source_path:.../finish_merged_experiments_v2.py,
source_sha256:ROOT_FROZEN_V2_SHA},resume_authority_sha256:ROOT_AUTHORITY_SHA.
V2 admits the preserved failures only through the explicit immutable ROOT resume map.
Only exact four completed inputs are sealed; existing artifacts never overwrite.
"""
from pathlib import Path
import csv
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parents[1];BASE=OUT.parent
WOOD=BASE/'dope-wood640-eval80-20261027-v1';COMBINED=WOOD/'combined372'
MAIN=BASE/'dope-filter-ablation-per-type-20261027-v1';SOURCE=Path(__file__).resolve()
CHILD=WOOD/'code/main_model_comparisons.py';PARENT=OUT/'code/finish_merged_experiments_v2.py'
FROZEN={str(CHILD):'96fcb6c233bc4f768f558bc1bc8d57cbb9f813f308fde63fe0ba19dcfc73a46e',
    str(WOOD/'code/combined_metrics.py'):'ce51e0bd3637d47eaae666e3ab1eb50e0a8ae5432aee35e8f0bac659ef861088',
    str(MAIN/'code/evaluate_models.py'):'a58463788e470470b0bf55326305393ddf1953cbfecd0c65274421755fead615'}
FOLDER=COMBINED/'evaluation/final_v1/final';TARGET=COMBINED/'comparisons/main_v1'
GO=WOOD/'preflight/MAIN_MODEL_COMPARISONS_GO.json'
REVIEW=OUT/'preflight/MAIN_MODEL_COMPARISONS_QUEUE_ROOT_REVIEW_V2.json'
RUN_GO=OUT/'preflight/MAIN_MODEL_COMPARISONS_QUEUE_GO_V2.json'
QA=OUT/'preflight/MAIN_MODEL_COMPARISONS_QUEUE_QA.json'


def require(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(path):
    p=Path(path);require(p.is_file() and not p.is_symlink() and p.resolve()==p,'Unsafe/missing exact input')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


# Immutable ROOT authority records the old technical failures; it is not a waiver
# of scientific/core checks. Producer SHA placeholders live in ROOT metadata only.
RESUME_AUTHORITY=OUT/'preflight/OUTER_WORKFLOW_RESUME_AUTHORITY_V2.json'
OLD_FAILURES={OUT/'FINISH_EXPERIMENTS_FAILURE.json',OUT/'FILTER_DIAGNOSTIC_WORKFLOW_FAILURE.json',
    OUT/'preflight/FILTER_STATISTICS_QUEUE_FAILURE.json',OUT/'preflight/MAIN_MODEL_COMPARISONS_QUEUE_FAILURE.json'}


def resume_binding(go,producer_source,allowed_failures):
    before=sha(RESUME_AUTHORITY);authority=json.loads(RESUME_AUTHORITY.read_text())
    require(sha(RESUME_AUTHORITY)==before and go['resume_authority_sha256']==before,
        'ROOT V2 resume authority absent/stale')
    require(authority['schema']=='outer_workflow_resume_authority_v2' and authority['passed'] is True
        and authority['technical_repair_only'] is True and authority['scientific_core_or_target_tags_changed'] is False,
        'Technical-only ROOT resume authority required')
    old=authority['old_failures'];require(set(old)=={str(p) for p in OLD_FAILURES},'Exact four old failure records required')
    for p in OLD_FAILURES:require(sha(p)==old[str(p)],'Preserved old failure SHA changed: '+str(p))
    item=authority['approved_rebindings'][str(SOURCE)];producer=go['producer']
    require(item['source_sha256']==sha(SOURCE)
        and set(item['allowed_old_failures'])=={str(p) for p in allowed_failures},'This V2 replacement is not authorized')
    require(set(producer)=={'pid','create_time','source_path','source_sha256'}
        and type(producer['pid']) is int and producer['pid']>0
        and isinstance(producer['create_time'],(int,float)) and 0<producer['create_time']<1e12
        and producer['source_path']==str(producer_source)
        and isinstance(producer['source_sha256'],str) and len(producer['source_sha256'])==64,
        'ROOT runtime producer identity incomplete')
    static={k:producer[k] for k in ('source_path','source_sha256')}
    require(item['producer']==static and sha(producer_source)==producer['source_sha256'],
        'Approved static V2 producer source differs')
    return before

def gates():
    expected={**FROZEN,str(SOURCE):sha(SOURCE)};review=read(REVIEW);go=read(RUN_GO)
    require(review['passed'] is True and review['source_sha256']==expected,'ROOT queue source review absent/stale')
    require(go['schema']=='main_comparisons_queue_go_v2' and go['passed'] is True and go['queue_authorized'] is True
        and go['combined_tag']=='final_v1' and go['output_tag']=='main_v1' and go['source_sha256']==expected
        and go['ROOT_review_sha256']==sha(REVIEW),'Explicit ROOT queue GO absent/stale')
    resume_binding(go,PARENT,{OUT/'FINISH_EXPERIMENTS_FAILURE.json',OUT/'preflight/MAIN_MODEL_COMPARISONS_QUEUE_FAILURE.json'})
    for path,digest in FROZEN.items():require(sha(path)==digest,'Frozen numerical/source implementation changed')
    return go


def ready():
    gates()  # Old failures are admitted only through the exact ROOT resume map.
    for name in ('FINISH_EXPERIMENTS_FAILURE_V2.json','MERGED_WORKFLOW_FAILURE.json','ABLATION_FINALIZE_FAILURE.json'):
        require(not (OUT/name).exists(),'Upstream queue failure requires ROOT repair')
    p=FOLDER/'COMPLETE.json'
    if not p.is_file():return False
    try:value=read(p)
    except (json.JSONDecodeError,UnicodeError,OSError):return False
    require(value['complete'] is True and value['passed'] is True and value['main_groups']==51 and value['frame_rows']==6324
        and value['primary_frames']==372 and value['primary_reference_points']==2891,'Not a completed PRIMARY372 score result')
    require(value['source_sha256'].get(str(WOOD/'code/combined_metrics.py'))==FROZEN[str(WOOD/'code/combined_metrics.py')],
        'Completed score producer SHA differs')
    return True


def wait():
    while not ready():
        p=gates()['producer']
        import psutil
        try:
            process=psutil.Process(p['pid'])
            live=(abs(process.create_time()-p['create_time'])<.01 and p['source_path'] in process.cmdline()
                and process.status()!=psutil.STATUS_ZOMBIE)
            if not live:
                if ready():return
                raise RuntimeError('Producer identity/state changed before combined scores')
        except psutil.NoSuchProcess:
            if ready():return
            raise RuntimeError('Producer exited before completed combined scores')
        time.sleep(30)


def seal():
    require(ready() and not GO.exists(),'Completed source required; preserve existing comparison GO')
    paths=[FOLDER/n for n in ('SCORES.json','BOOTSTRAP_RECEIPT.json','COMPLETE.json')]+[COMBINED/'MERGED_MANIFEST.json']
    bindings={str(p):sha(p) for p in paths};done=read(FOLDER/'COMPLETE.json')
    for p in paths[:2]:require(done['output_sha256'].get(str(p))==bindings[str(p)],'Completed output SHA differs')
    require(done['source_sha256'].get(str(paths[3]))==bindings[str(paths[3])],'Recording metadata producer SHA differs')
    require(sha(paths[2])==bindings[str(paths[2])],'Completion changed during seal')
    value=dict(schema='main_model_comparisons_go_v1',passed=True,utc=utc(),combined_tag='final_v1',
        cached_scalar_score_analysis_authorized=True,recording_metadata_only_authorized=True,
        source_sha256=FROZEN,input_sha256=bindings,queue_source_sha256=sha(SOURCE),ROOT_queue_GO_sha256=sha(RUN_GO))
    with GO.open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    for p,digest in bindings.items():require(sha(p)==digest,'Completed input changed during seal')
    return bindings


def verify_result(bindings):
    done=read(TARGET/'COMPLETE.json');boot=read(TARGET/'BOOTSTRAP_RECEIPT.json')
    require(done['complete'] is True and done['passed'] is True and done['combined_tag']=='final_v1' and done['tag']=='main_v1'
        and done['source_groups_checked']==51 and done['source_frame_rows_checked']==6324
        and done['primary_CI_rows']==90 and done['GREEN_manual_CI_rows']==done['WOOD_manual_CI_rows']==30
        and done['WOOD_recording_CI_rows']==60 and done['reference_or_prediction_XY_read'] is False,'Completed210-CI comparison gate differs')
    require(boot['input_weight_hashes_exactly_reproduced'] is True and boot['original_GREEN_RNG_consumes_WOOD45'] is True
        and boot['seed']==20261026 and boot['replicates']==10000 and boot['no_undefined_draws_discarded'] is True,'Bootstrap identity/policy gate differs')
    for p,digest in {**bindings,**FROZEN,str(GO):sha(GO)}.items():require(done['input_sha256'].get(p)==digest and sha(p)==digest,'Comparison input/source binding differs')
    for p,digest in done['output_sha256'].items():require(Path(p).parent==TARGET and sha(p)==digest,'Comparison output SHA differs')
    counts={}
    for name,n in (('PAIRED_CI.csv',90),('GREEN_MANUAL_PAIRED_CI.csv',30),('WOOD_MANUAL_PAIRED_CI.csv',30),('WOOD_RECORDING_PAIRED_CI.csv',60)):
        with (TARGET/name).open(newline='') as f:rows=list(csv.DictReader(f))
        require(len(rows)==n and len({(r['kind'],r['comparison'],r['panel'],r['metric']) for r in rows})==n,'Missing/duplicate comparison catalogue')
        require({r['metric'] for r in rows}=={'PCK5','PCK10','PCK20','coverage','conditional_precision10'},'Five-metric catalogue differs')
        counts[name]=n
    require(sum(counts.values())==210,'Expected210 total main/manual/recording intervals')
    return counts,sha(TARGET/'COMPLETE.json')


def main():
    require(sys.argv[1:]==['--run'],'Explicit --run required');gates()
    require(not QA.exists() and not TARGET.exists() and not GO.exists()
        and not (OUT/'preflight/MAIN_MODEL_COMPARISONS_QUEUE_FAILURE_V2.json').exists(),
        'Preserve completed/partial comparison artifacts or V2 failure')
    with (OUT/'logs/MAIN_MODEL_COMPARISONS_QUEUE_V2.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            wait();gates();bindings=seal();log=OUT/'logs/main_model_comparisons_main_v1.stdout.log'
            require(not log.exists(),'Preserve prior child log')
            command=[sys.executable,'-B',str(CHILD),'--combined-tag','final_v1','--tag','main_v1']
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            begin=time.monotonic()
            with log.open('xb',buffering=0) as f:result=subprocess.run(command,cwd=WOOD,env=env,stdout=f,stderr=subprocess.STDOUT)
            require(result.returncode==0,'CPU comparison child failed');gates();counts,digest=verify_result(bindings)
            with QA.open('x') as f:f.write(json.dumps(dict(passed=True,complete=True,utc=utc(),CI_rows=counts,total_CI_rows=210,
                comparison_COMPLETE_sha256=digest,GO_sha256=sha(GO),queue_source_sha256=sha(SOURCE),command=command,
                stdout_path=str(log),stdout_sha256=sha(log),elapsed_seconds=time.monotonic()-begin,input_sha256=bindings,
                resume_authority_sha256=sha(RESUME_AUTHORITY),NN_forwards=0,PnP_fits=0,outside_GLOBAL_lock_acquired=False,published=False),indent=2)+'\n')
        except Exception as error:
            p=OUT/'preflight/MAIN_MODEL_COMPARISONS_QUEUE_FAILURE_V2.json'
            with p.open('x') as f:f.write(json.dumps(dict(utc=utc(),error=repr(error),automatic_retry=False,published=False),indent=2)+'\n')
            raise


if __name__=='__main__':main()

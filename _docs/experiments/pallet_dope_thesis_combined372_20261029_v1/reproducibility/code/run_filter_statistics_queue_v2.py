"""CPU-only filter-statistics verification, followed by ratio exposure tables.

Import performs no IO. ROOT must write preflight/FILTER_STATISTICS_QUEUE_ROOT_REVIEW_V2.json
{passed:true,source_sha256:exact SOURCES plus dynamic producer SHA and self SHA} and
preflight/FILTER_STATISTICS_QUEUE_GO_V2.json {schema:filter_statistics_queue_go_v2,
passed:true,queue_authorized:true,tag:filter372_v1,source_sha256:same map,
ROOT_review_sha256,actions:[verify_merged_filter_statistics,
summarize_ratio_per_frame_exposures],producer:{pid:ROOT_RUNTIME_PID,create_time:ROOT_RUNTIME_CREATE_TIME,
source_path:.../run_merged_filter_diagnostics_v2.py,source_sha256:ROOT_FROZEN_V2_SHA},
resume_authority_sha256:ROOT_FROZEN_AUTHORITY_SHA}.
V2 rebinds only with the preserved four-failure ROOT resume authority.
Both child GOs are generated only after these exact source authorizations.
No upstream private XY cache, raw image/annotation, checkpoint or SUBSETS is
opened or rehashed. All output and partial logs are immutable; no auto retry.
"""
from pathlib import Path
import csv
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parents[1];BASE=OUT.parent;SOURCE=Path(__file__).resolve()
WOOD=BASE/'dope-wood640-eval80-20261027-v1';RATIO=BASE/'dope-r2-ratio-per-type-20261027-v1'
TAG='filter372_v1';FOLDER=WOOD/'combined372/filter_diagnostics'/TAG
PRODUCER=OUT/'code/run_merged_filter_diagnostics_v2.py';CORE=WOOD/'code/merged_filter_diagnostics.py'
TEST=WOOD/'code/test_merged_filter_diagnostics.py';CHILD=OUT/'code/verify_merged_filter_statistics.py'
EXPOSURE=OUT/'code/summarize_ratio_exposures.py';EXPOSURE_OUT=OUT/'results/ratio_exposures'
NORMALIZER=OUT/'code/prepare_ratio_evaluation_v2.py';ADAPTER=OUT/'code/run_filter_diagnostics_with_metadata_v2.py'
SOURCES={str(NORMALIZER):'11278e1b623c251c98276305390de32cb3cd030cd740c25c52aeeb19da959322',
    str(ADAPTER):'ca4df1f250ae8db1fae7dedfa50c77d4cae1cfa9939fda4fc7c3438ff14ab837',str(CORE):'e83f23bfe3f1e1429654ed7e2036993abd68849498ff0cb0b30a1f9693f6832a',
    str(TEST):'a159c64b3ab0371fe85b6ffbc794fc03d71311350000016b6412823bbe18e2b0',
    str(CHILD):'ad79ec5bf0f327a8464269ffe96df4427a299e67a4857ef99d269b06963a477e',
    str(EXPOSURE):'09671da49f1e5a9b1f94cd79b4466ce5fc200186130efbd19a164f35130b5572'}
REVIEW=OUT/'preflight/FILTER_STATISTICS_QUEUE_ROOT_REVIEW_V2.json';RUN_GO=OUT/'preflight/FILTER_STATISTICS_QUEUE_GO_V2.json'
MANIFEST=OUT/'preflight/MERGED_FILTER_STATISTICS_INPUTS.json';STATS_GO=OUT/'preflight/MERGED_FILTER_STATISTICS_GO.json'
STATS_QA=OUT/'preflight/MERGED_FILTER_STATISTICS_QA.json';EXPOSURE_GO=OUT/'preflight/ROOT_RATIO_EXPOSURE_GO.json'
WORKFLOW=OUT/'FILTER_DIAGNOSTIC_WORKFLOW_COMPLETE.json';INFER=FOLDER/'infer/COMPLETE.json'
QA=OUT/'preflight/FILTER_STATISTICS_QUEUE_QA.json';FAILURE=OUT/'preflight/FILTER_STATISTICS_QUEUE_FAILURE_V2.json'
FILES={'component_summary':FOLDER/'score/public/COMPONENT_QUALITY.csv','stage_summary':FOLDER/'score/public/T3_FILTER_STAGES.csv',
    'component_frames':FOLDER/'score/private/COMPONENT_FRAME_COUNTERS.csv','stage_frames':FOLDER/'score/private/T3_FRAME_COUNTERS.csv',
    'selection_equivalence':FOLDER/'select/EQUIVALENCE.json','diagnostic_complete':FOLDER/'score/COMPLETE.json',
    'merged_manifest':WOOD/'combined372/MERGED_MANIFEST.json'}
RECEIPTS=[RATIO/'trials'/k/c/'TRAIN_COMPLETE.json' for k in ('WOOD','GREEN','PLASTIC') for c in 'ABCDE']
EXPOSURE_OUTPUTS=[EXPOSURE_OUT/n for n in ('PER_FRAME_EXPOSURES.csv','SUMMARY.csv','SUMMARY.md','INPUT_SOURCE_SHA256.json')]
STAGE_LOGS={s:OUT/'logs'/('filter_statistics_queue_'+s+'.stdout.log') for s in ('statistics','exposures')}
ADAPTER_RECEIPTS={FOLDER/('METADATA_ADAPTER_'+stage+'_COMPLETE.json') for stage in ('infer','select','score')}
# Immutable ROOT authority records the old technical failures; it is not a waiver
# of scientific/core checks. Producer SHA placeholders live in ROOT metadata only.
RESUME_AUTHORITY=OUT/'preflight/OUTER_WORKFLOW_RESUME_AUTHORITY_V2.json'
OLD_FAILURES={OUT/'FINISH_EXPERIMENTS_FAILURE.json',OUT/'FILTER_DIAGNOSTIC_WORKFLOW_FAILURE.json',
    OUT/'preflight/FILTER_STATISTICS_QUEUE_FAILURE.json',OUT/'preflight/MAIN_MODEL_COMPARISONS_QUEUE_FAILURE.json'}

HASHABLE={SOURCE,PRODUCER,RESUME_AUTHORITY,*OLD_FAILURES,*ADAPTER_RECEIPTS,*map(Path,SOURCES),REVIEW,RUN_GO,MANIFEST,STATS_GO,STATS_QA,EXPOSURE_GO,WORKFLOW,INFER,
          *FILES.values(),*RECEIPTS,*EXPOSURE_OUTPUTS,EXPOSURE_OUT/'QA.json',*STAGE_LOGS.values()}
JSON_READS={REVIEW,RUN_GO,WORKFLOW,INFER,FILES['selection_equivalence'],FILES['diagnostic_complete'],STATS_QA,EXPOSURE_OUT/'QA.json'}


def require(ok,msg):
    if not ok:raise RuntimeError(msg)


class MetadataInFlux(RuntimeError):
    pass


def sha(path):
    p=Path(path);require(p in HASHABLE and p.resolve()==p and p.is_file() and not p.is_symlink(),'Unlisted/unsafe hash path')
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    require(path in JSON_READS,'Unlisted metadata body read')
    before=sha(path);value=json.loads(path.read_text())
    if sha(path)!=before:raise MetadataInFlux('Metadata changed during read')
    return value


def write(path,value):
    require(path in (MANIFEST,STATS_GO,EXPOSURE_GO,QA,FAILURE) and path.resolve()==path and not path.exists(),'Preserve existing/unsafe output')
    with path.open('x') as stream:stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')



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
    review=read(REVIEW);go=read(RUN_GO)
    resume_binding(go,PRODUCER,{OUT/'FILTER_DIAGNOSTIC_WORKFLOW_FAILURE.json',OUT/'preflight/FILTER_STATISTICS_QUEUE_FAILURE.json'})
    expected={**SOURCES,str(PRODUCER):go['producer']['source_sha256'],str(SOURCE):sha(SOURCE)}
    require(review['passed'] is True and review['source_sha256']==expected,'ROOT source review missing/stale')
    require(go['schema']=='filter_statistics_queue_go_v2' and go['passed'] is True and go['queue_authorized'] is True
        and go['tag']==TAG and go['source_sha256']==expected and go['ROOT_review_sha256']==sha(REVIEW)
        and go['actions']==['verify_merged_filter_statistics','summarize_ratio_per_frame_exposures'],'Explicit ROOT CPU actions GO required')
    for path,digest in SOURCES.items():require(sha(path)==digest,'Frozen child/producer source changed')
    return go,sha(RUN_GO)


def ready():
    gate=gates()[0]
    require(not (OUT/'FILTER_DIAGNOSTIC_WORKFLOW_FAILURE_V2.json').exists(),'V2 diagnostic producer failed; ROOT repair required')
    if not WORKFLOW.is_file():return False
    try:value=read(WORKFLOW)
    except (json.JSONDecodeError,UnicodeError,MetadataInFlux):return False
    expected={str(PRODUCER):gate['producer']['source_sha256'],
        **{str(p):SOURCES[str(p)] for p in (CORE,TEST,NORMALIZER,ADAPTER)}}
    require(value['complete'] is True and value['passed'] is True and value['tag']==TAG and value['new_forwards']==320
        and value['source_sha256']==expected and value['score_complete_path']==str(FILES['diagnostic_complete'])
        and value['score_complete_sha256']==sha(FILES['diagnostic_complete'])
        and [r['stage'] for r in value['stages']]==['infer','select','score']
        and all(r['returncode']==0 for r in value['stages']),'Diagnostic workflow completion failed')
    require(set(value['metadata_adapter_receipt_sha256'])=={str(p) for p in ADAPTER_RECEIPTS}
        and all(sha(p)==value['metadata_adapter_receipt_sha256'][str(p)] for p in ADAPTER_RECEIPTS)
        and all(r['command'][2]==str(ADAPTER) for r in value['stages']),'Explicit V2 metadata adapter execution evidence differs')
    return True


def wait():
    while not ready():
        p=gates()[0]['producer']
        import psutil
        try:
            process=psutil.Process(p['pid']);live=(abs(process.create_time()-p['create_time'])<.01
                and p['source_path'] in process.cmdline() and process.status()!=psutil.STATUS_ZOMBIE)
            if not live:
                if ready():return
                raise RuntimeError('Producer identity/state changed before diagnostic completion')
        except psutil.NoSuchProcess:
            if ready():return
            raise RuntimeError('Producer exited without diagnostic completion')
        time.sleep(30)


def seal_statistics(root_go_sha):
    require(not any(p.exists() for p in (MANIFEST,STATS_GO,STATS_QA)),'Preserve prior/partial statistics artifacts')
    infer=read(INFER);proof=read(FILES['selection_equivalence']);done=read(FILES['diagnostic_complete'])
    require(infer['complete'] is True and infer['passed'] is True and infer['new_forwards']==infer['attempts']==infer['completed']==320
        and infer['R0_original_canonical_parity_frames']==80 and proof['source_input_sha256'].get(str(INFER))==sha(INFER),'Actual320/canonical parity evidence differs')
    for k,v in dict(complete=True,passed=True,combined_teacher_frame_pairs=1116,new_chosen_comparisons=480,
        mask9_pass_inliers_exact=True,original_final_decomposition_exact=True,raw_targets_unchanged=True,
        centroid_mask_false=True,reference_coordinates_read=False).items():require(proof.get(k)==v,'Equivalence gate failed: '+k)
    for k,v in dict(complete=True,passed=True,frames=372,WOOD_frames=125,component_rows=35,T3_stage_rows=165,
        component_frame_rows=4312,T3_frame_rows=20328,original292_component28_exact=True,
        original292_T3_stage132_exact=True,selection_frozen_before_reference_read=True).items():require(done.get(k)==v,'Score gate failed: '+k)
    bindings={str(p):sha(p) for p in FILES.values()};upstream={}
    for mapping in (proof['source_input_sha256'],done['input_sha256']):
        for path,digest in mapping.items():
            require(Path(path).is_absolute() and isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest)
                and (path not in upstream or upstream[path]==digest),'Conflicting/malformed upstream metadata SHA')
            upstream[path]=digest  # These paths are declarations only: NEVER opened/hashed.
    for role in ('component_summary','stage_summary','component_frames','stage_frames'):
        require(done['output_sha256'].get(str(FILES[role]))==bindings[str(FILES[role])],'Counter output binding differs')
    for role in ('selection_equivalence','merged_manifest'):
        require(done['input_sha256'].get(str(FILES[role]))==bindings[str(FILES[role])],'Frozen score input differs')
    write(MANIFEST,dict(schema='primary372_filter_statistics_inputs_v1',verifier_source_sha256=SOURCES[str(CHILD)],
        producer_source_sha256=SOURCES[str(CORE)],input_files={str(p):dict(role=r,sha256=bindings[str(p)]) for r,p in FILES.items()},
        upstream_sha256=upstream,ROOT_queue_GO_sha256=root_go_sha,queue_source_sha256=sha(SOURCE)))
    write(STATS_GO,dict(schema='primary372_filter_statistics_go_v1',passed=True,statistics_verification_authorized=True,
        input_manifest_sha256=sha(MANIFEST),verifier_source_sha256=SOURCES[str(CHILD)],ROOT_queue_GO_sha256=root_go_sha))
    for p,digest in bindings.items():require(sha(p)==digest,'Diagnostic input changed during seal')
    return bindings


def call(stage,path):
    log=STAGE_LOGS[stage];require(not log.exists(),'Preserve prior/partial CPU stage log')
    command=[sys.executable,'-B',str(path)];begin=time.monotonic()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with log.open('xb',buffering=0) as stream:result=subprocess.run(command,cwd=OUT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    require(result.returncode==0,'CPU child failed: '+stage)
    return dict(stage=stage,command=command,returncode=result.returncode,elapsed_seconds=time.monotonic()-begin,stdout_path=str(log),stdout_sha256=sha(log))


def validate_statistics(bindings):
    qa=read(STATS_QA)
    require(qa['passed'] is True and qa['complete'] is True and qa['mismatch_count']==0 and qa['frames']==372
        and qa['verifier_source_sha256']==SOURCES[str(CHILD)] and qa['producer_source_sha256']==SOURCES[str(CORE)]
        and qa['checks']['components']['summary_rows']==35 and qa['checks']['components']['frame_rows']==4312
        and qa['checks']['T3_stages']['summary_rows']==165 and qa['checks']['T3_stages']['frame_rows']==20328
        and qa['checks']['T3_stages']['stages_per_teacher']==11,'Independent statistics QA failed')
    for p,digest in {**bindings,str(MANIFEST):sha(MANIFEST),str(STATS_GO):sha(STATS_GO),str(CORE):SOURCES[str(CORE)]}.items():
        require(qa['source_sha256'].get(p)==digest and sha(p)==digest,'Independent statistics source binding differs')


def exposures(root_go_sha):
    require(not EXPOSURE_GO.exists() and not EXPOSURE_OUT.exists(),'Preserve prior/partial exposure artifacts')
    receipts={str(p):sha(p) for p in RECEIPTS}
    write(EXPOSURE_GO,dict(authorized=True,action='summarize_ratio_per_frame_exposures',source_sha256=SOURCES[str(EXPOSURE)],
        subsets_sha256='0fa15df85bf89826c972d90316b8fdabc72a924bcdfd55cefe732fbcb7d6fded',input_sha256=receipts,
        ROOT_queue_GO_sha256=root_go_sha,independent_filter_statistics_QA_sha256=sha(STATS_QA)))
    stage=call('exposures',EXPOSURE);qa=read(EXPOSURE_OUT/'QA.json')
    require(qa['passed'] is True and qa['complete'] is True and qa['receipts']==15 and qa['summary_rows']==30
        and qa['per_frame_rows']==14224 and qa['committed_draws_per_branch_condition']==6624 and qa['SD_ddof']==0
        and qa['R2_only'] is True and set(qa['output_sha256'])=={str(p) for p in EXPOSURE_OUTPUTS},'Exposure output catalogue failed')
    for p,digest in {**receipts,str(EXPOSURE):SOURCES[str(EXPOSURE)],str(EXPOSURE_GO):sha(EXPOSURE_GO)}.items():
        require(qa['input_and_source_sha256'].get(p)==digest and sha(p)==digest,'Exposure input/source binding differs')
    for p,digest in qa['output_sha256'].items():require(sha(p)==digest,'Exposure output changed')
    for filename,n in (('SUMMARY.csv',30),('PER_FRAME_EXPOSURES.csv',14224)):
        with (EXPOSURE_OUT/filename).open(newline='') as stream:require(sum(1 for _ in csv.DictReader(stream))==n,'Exposure CSV row count differs')
    return stage


def main():
    require(sys.argv[1:]==['--run'],'Explicit --run required');_,root_go_sha=gates()
    require(not QA.exists() and not FAILURE.exists(),'Preserve prior queue outcome')
    with (OUT/'logs/FILTER_STATISTICS_QUEUE_V2.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            wait();require(gates()[1]==root_go_sha,'ROOT run GO changed while waiting');bindings=seal_statistics(root_go_sha)
            stages=[call('statistics',CHILD)];validate_statistics(bindings)
            require(gates()[1]==root_go_sha,'ROOT run GO changed');stages.append(exposures(root_go_sha))
            validate_statistics(bindings);require(gates()[1]==root_go_sha,'ROOT run GO/source changed')
            write(QA,dict(passed=True,complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,
                source_sha256={**SOURCES,str(PRODUCER):gates()[0]['producer']['source_sha256'],str(SOURCE):sha(SOURCE)},
                resume_authority_sha256=sha(RESUME_AUTHORITY),ROOT_run_GO_sha256=root_go_sha,stages=stages,
                statistics_QA_sha256=sha(STATS_QA),exposure_QA_sha256=sha(EXPOSURE_OUT/'QA.json'),statistics_mismatches=0,
                statistics_frame_rows=[4312,20328],exposure_summary_rows=30,exposure_per_frame_rows=14224,
                raw_RGB_reads=0,raw_annotation_reads=0,checkpoint_reads=0,private_XY_cache_bytes_read=0,
                neural_forwards=0,PnP_fits=0,outside_GLOBAL_lock_acquired=False,published=False))
        except Exception as error:
            write(FAILURE,dict(passed=False,error=repr(error),automatic_retry=False,published=False));raise


if __name__=='__main__':main()

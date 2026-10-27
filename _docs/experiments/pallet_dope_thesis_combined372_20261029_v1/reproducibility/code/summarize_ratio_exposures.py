"""Summarize committed R2 exposure IDs from exactly15 completed receipts.

Source-only until a separate ROOT GO. Never opens SUBSETS, checkpoints, raw
images, annotations or prediction arrays. Import performs no IO and imports no
training/NN modules. Exposure means committed training draws, not unique images.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import datetime
import hashlib
import json
import math
import re
import sys

OUT=Path(__file__).resolve().parents[1]
RATIO=OUT.parent/'dope-r2-ratio-per-type-20261027-v1'
GO=OUT/'preflight/ROOT_RATIO_EXPOSURE_GO.json'
TARGET=OUT/'results/ratio_exposures'
KINDS=('WOOD','GREEN','PLASTIC')
LEVELS={'A':20,'B':50,'C':100,'D':200,'E':400}
REAL={'WOOD':550,'GREEN':550,'PLASTIC':20}
SUBSETS_SHA='0fa15df85bf89826c972d90316b8fdabc72a924bcdfd55cefe732fbcb7d6fded'
UPDATES,EPOCHS,STEPS,BATCH,DRAWS=828,3,276,8,6624


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha_bytes(value):return hashlib.sha256(value).hexdigest()


def digest_ok(value):return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def read_exact(path,expected):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve()==path,'Missing/unsafe exact metadata file: '+str(path))
    raw=path.read_bytes();require(sha_bytes(raw)==expected,'ROOT-approved metadata bytes differ: '+str(path))
    return json.loads(raw)


def schedule_digest(ids):return sha_bytes(json.dumps(ids,separators=(',',':')).encode())


def branch_counts(history,schedules,branch,count):
    field=branch+'_ids';total=Counter();epoch_reports=[];known=None
    for epoch in range(1,EPOCHS+1):
        rows=history[(epoch-1)*STEPS:epoch*STEPS];batches=[r[field] for r in rows]
        require(all(isinstance(batch,list) and len(batch)==BATCH
            and all(isinstance(i,str) and i for i in batch) for batch in batches),'Eight nonempty string IDs per branch/update required')
        counts=Counter(i for batch in batches for i in batch)
        require(len(counts)==count and sum(counts.values())==STEPS*BATCH,'Epoch unique-ID/draw budget differs')
        require(min(counts.values())==STEPS*BATCH//count and max(counts.values())==math.ceil(STEPS*BATCH/count),
            'Independent shuffle-cycle exposure balance differs')
        if known is None:known=set(counts)
        require(set(counts)==known,'Committed unique record IDs changed between epochs')
        require(schedules[epoch-1]['epoch']==epoch,'Schedule epoch order differs')
        saved=schedules[epoch-1][branch]
        seed=42+2000000+(epoch-1)*1000+(333 if branch=='real' else 0)
        duplicates=sum(BATCH-len(set(batch)) for batch in batches)
        expected={'unique_pool':count,'draws':STEPS*BATCH,'unique_exposed':count,
            'repeated_draws':STEPS*BATCH-count,'within_batch_duplicate_draws':duplicates,
            'seed':seed,'policy':'independent_shuffle_cycle_fill','id_schedule_sha256':schedule_digest(batches)}
        require(all(saved.get(k)==v for k,v in expected.items()),'Receipt schedule/history identity differs: '+branch)
        total.update(counts);epoch_reports.append({'epoch':epoch,'unique_records':count,'draws':sum(counts.values()),
            'min_exposures':min(counts.values()),'max_exposures':max(counts.values()),'within_batch_duplicate_draws':duplicates,
            'id_schedule_sha256':expected['id_schedule_sha256']})
    require(len(total)==count and sum(total.values())==DRAWS,'Committed branch total differs')
    return total,epoch_reports


def verify_receipt(receipt,kind,condition):
    real=REAL[kind];synthetic=real*LEVELS[condition]//100
    require(receipt.get('complete') is True and receipt['kind']==kind and receipt['condition']==condition
        and receipt['round']==2 and receipt['synthetic_percent']==LEVELS[condition],'Completed trial identity differs')
    require(receipt['subsets_sha256']==SUBSETS_SHA and receipt['optimizer_steps']==[1656]
        and receipt['real_unique_count']==real and receipt['synthetic_unique_count']==synthetic,'Frozen subset/count/update contract differs')
    require(receipt['checkpoint_path']==str(RATIO/'trials'/kind/condition/'R2.pth')
        and digest_ok(receipt['checkpoint_sha256']) and digest_ok(receipt['teacher_checkpoint_sha256']),
        'Checkpoint declaration differs; no checkpoint bytes are opened')
    require(receipt['neural_teacher_forwards']==receipt['PnP_fits']==0 and receipt['evaluation_performed'] is False
        and receipt['raw_teacher_targets_only'] is True and receipt['centroid_supervision'] is False,'Training receipt scope differs')
    stats=receipt['stats']
    expected={'complete':True,'round_idx':2,'updates_completed':UPDATES,'updates_required':UPDATES,'persistent_Adam_steps':1656,
        'epochs':EPOCHS,'effective_synthetic_batch':BATCH,'effective_real_batch':BATCH,'microbatch':BATCH,'amp':True,
        'loss_dtype':'float32','parameter_dtype':'float32','loss_mode':'original_branch_means_sum','lambda_real':1.0,
        'pseudo_samples':real,'synthetic_unique_pool':synthetic,'real_unique_pool':real,
        'committed_synthetic_draws':DRAWS,'committed_real_draws':DRAWS}
    require(all(stats.get(k)==v for k,v in expected.items()),'Completed committed training statistics differ')
    history=stats['history'];schedules=stats['schedules']
    require(isinstance(history,list) and len(history)==UPDATES and isinstance(schedules,list) and len(schedules)==EPOCHS,
        'Complete828-update/3-epoch history required')
    for update,row in enumerate(history,1):
        epoch,local=divmod(update-1,STEPS)
        require(row['update']==update and row['epoch']==epoch+1 and row['epoch_step']==local+1,
            'History update/epoch/epoch-step sequence differs')
    result={}
    for branch,count in (('real',real),('synthetic',synthetic)):
        result[branch]=branch_counts(history,schedules,branch,count)
    return result


def summary(kind,condition,branch,counts,checkpoint_sha):
    values=list(counts.values());n=len(values);mean=math.fsum(values)/n
    variance=math.fsum((v-mean)**2 for v in values)/n
    return dict(kind=kind,condition=condition,synthetic_percent=LEVELS[condition],branch=branch,unique_records=n,
        total_draws=sum(values),mean_exposures=mean,population_variance_exposures=variance,population_SD_exposures=math.sqrt(variance),
        min_exposures=min(values),max_exposures=max(values),n_updates=UPDATES,draws_per_update=BATCH,epochs=EPOCHS,
        checkpoint_sha256=checkpoint_sha,subsets_sha256=SUBSETS_SHA)


def csv_new(path,rows):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def json_new(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def contract():
    return {'ROOT_GO_path':str(GO),'GO':{'authorized':True,'action':'summarize_ratio_per_frame_exposures',
        'source_sha256':'CURRENT_SUMMARIZER_SHA256','subsets_sha256':SUBSETS_SHA,
        'input_sha256':{str(RATIO/'trials'/kind/condition/'TRAIN_COMPLETE.json'):'ACTUAL_COMPLETED_RECEIPT_SHA256'
            for kind in KINDS for condition in LEVELS}},'output_directory':str(TARGET),
        'outputs':['PER_FRAME_EXPOSURES.csv','SUMMARY.csv','SUMMARY.md','INPUT_SOURCE_SHA256.json','QA.json'],
        'expected_receipts':15,'expected_summary_rows':30,'expected_per_frame_rows':14224,
        'branch_draws_per_condition':DRAWS,'checkpoint_and_SUBSETS_bytes_read':0,
        'training_source_evidence':{'count_trainer.py':'cycle_schedule/id_schedule/train_r2.status/history',
            'experiment_driver.py':'trial TRAIN_COMPLETE schema; original R1 teacher SHA declaration only'}}


def run():
    source=Path(__file__).resolve();source_sha=sha_bytes(source.read_bytes())
    require(not TARGET.exists(),'Preserve any existing exposure output directory')
    require(GO.is_file() and not GO.is_symlink() and GO.resolve()==GO,'Explicit ROOT exposure GO required')
    go_raw=GO.read_bytes();go_sha=sha_bytes(go_raw);go=json.loads(go_raw)
    expected_paths={str(RATIO/'trials'/kind/condition/'TRAIN_COMPLETE.json') for kind in KINDS for condition in LEVELS}
    require(go.get('authorized') is True and go.get('action')=='summarize_ratio_per_frame_exposures'
        and go['source_sha256']==source_sha and go['subsets_sha256']==SUBSETS_SHA,'ROOT exposure authorization/source binding differs')
    require(set(go['input_sha256'])==expected_paths and all(digest_ok(v) for v in go['input_sha256'].values()),
        'ROOT must SHA-bind exactly15 completed trial receipts before any receipt is opened')
    details=[];summaries=[];epoch_checks=[];receipt_reports=[];sets={}
    for kind in KINDS:
        for condition in LEVELS:
            path=RATIO/'trials'/kind/condition/'TRAIN_COMPLETE.json';receipt=read_exact(path,go['input_sha256'][str(path)])
            verified=verify_receipt(receipt,kind,condition)
            receipt_reports.append(dict(kind=kind,condition=condition,receipt=str(path),sha256=go['input_sha256'][str(path)],
                checkpoint_sha256_declared=receipt['checkpoint_sha256'],teacher_checkpoint_sha256_declared=receipt['teacher_checkpoint_sha256'],
                checkpoint_bytes_opened=False,subsets_sha256_declared=receipt['subsets_sha256']))
            for branch,(counts,epochs) in verified.items():
                sets[kind,condition,branch]=set(counts)
                details.extend(dict(kind=kind,condition=condition,branch=branch,record_id=i,n_exposures=counts[i]) for i in sorted(counts))
                summaries.append(summary(kind,condition,branch,counts,receipt['checkpoint_sha256']))
                epoch_checks.append(dict(kind=kind,condition=condition,branch=branch,epochs=epochs))
    for kind in KINDS:
        require(all(sets[kind,c,'real']==sets[kind,'A','real'] for c in LEVELS),'Same-kind real unique IDs differ across ratio conditions')
        for a,b in zip(tuple(LEVELS),tuple(LEVELS)[1:]):
            require(sets[kind,a,'synthetic']<sets[kind,b,'synthetic'],'Shared synthetic-prefix unique ID nesting differs')
    require(len(summaries)==30 and len(details)==14224 and sum(r['total_draws'] for r in summaries)==30*DRAWS,
        'Exposure output catalogue or global committed draw count differs')
    def unchanged():
        require(sha_bytes(GO.read_bytes())==go_sha and sha_bytes(source.read_bytes())==source_sha,'ROOT GO/source changed')
        for path,digest in go['input_sha256'].items():require(sha_bytes(Path(path).read_bytes())==digest,'Input receipt changed')
    unchanged();TARGET.mkdir(parents=True,exist_ok=False)
    csv_new(TARGET/'PER_FRAME_EXPOSURES.csv',details);csv_new(TARGET/'SUMMARY.csv',summaries)
    lines=['# 비율 실험의 영상별 학습 노출 횟수','',
        '완료된15개 R2 학습 영수증의828개 업데이트 기록에서 실제 사용 ID를 집계하였다. 각 조건의 실제·합성 분기는 업데이트마다8번, 합계6624번의 committed draw를 사용하였다. R1의 노출은 포함하지 않는다.',
        '','| 종류 | 조건 | 분기 | 고유 ID 수 | draw 수 | 영상당 평균 ± 모집단SD | 최소 | 최대 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append('| '+' | '.join(map(str,[r['kind'],r['condition'],r['branch'],r['unique_records'],r['total_draws'],
        f"{r['mean_exposures']:.6f} ± {r['population_SD_exposures']:.6f}",r['min_exposures'],r['max_exposures']]))+' |')
    lines+=['','고유 수는 기록된 record ID 기준이며 새 RGB 해시 검사는 수행하지 않았다. 같은 영상의 반복 draw는 새 영상으로 세지 않는다. SD는 영상별 노출 횟수의 모집단 SD(ddof=0)이며 학습 seed 간 변동이나 성능 SD가 아니다.',
        '','독립 shuffle-cycle-fill 정책 때문에 데이터 수가 적을수록 한 영상의 반복 노출이 많다. 이는 Lsyn+Lreal의 분기 평균 가중치가 바뀌었다는 뜻이 아니다. 증강 이후의 픽셀 동일성이나 실제 성능 효과를 이 표만으로 판단하지 않는다.',
        '','SUBSETS는 사전 고정 SHA의 영수증 선언만 대조하였다. 해당 파일 본문과 체크포인트·RGB·GT는 열지 않았다.15개 비율 본 실험의 완료 영수증을 사용했으며 기술재현 모델은 집계 범위에 포함하지 않았다.']
    with (TARGET/'SUMMARY.md').open('x') as stream:stream.write('\n'.join(lines)+'\n')
    bindings={str(GO):go_sha,str(source):source_sha,**go['input_sha256']}
    json_new(TARGET/'INPUT_SOURCE_SHA256.json',dict(command=sys.argv,source_sha256=source_sha,ROOT_GO_sha256=go_sha,input_and_source_sha256=bindings,
        subsets_sha256_declaration=SUBSETS_SHA,subsets_file_opened=False,checkpoint_files_opened=False))
    unchanged();outputs={str(p):sha_bytes(p.read_bytes()) for p in TARGET.iterdir() if p.is_file()}
    json_new(TARGET/'QA.json',dict(passed=True,complete=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        receipts=15,summary_rows=30,per_frame_rows=len(details),committed_draws_per_branch_condition=DRAWS,
        all_828_update_sequences_exact=True,all_epoch_schedules_and_digests_exact=True,all_unique_counts_exact=True,
        same_kind_real_ID_sets_exact=True,synthetic_ID_prefix_nesting_exact=True,SD_ddof=0,R2_only=True,
        input_and_source_sha256=bindings,output_sha256=outputs,receipt_declarations=receipt_reports,epoch_checks=epoch_checks,
        command=sys.argv,raw_RGB_reads=0,raw_annotation_reads=0,checkpoint_reads=0,SUBSETS_body_reads=0,
        neural_forwards=0,PnP_fits=0,bootstrap_draws=0))
    unchanged();print('RATIO_EXPOSURES_COMPLETE',15,30,len(details),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--describe-contract',action='store_true');args=parser.parse_args()
    if args.describe_contract:print(json.dumps(contract(),ensure_ascii=False,indent=2))
    else:run()


if __name__=='__main__':main()

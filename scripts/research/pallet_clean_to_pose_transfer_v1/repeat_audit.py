"""반복43의 독립 CPU 감사. 새 fit/신경망 추론/원본수정 없이 캐시를검산한다."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from . import common as C
from .followup_pair import StageContext,evaluation_paths,PAIR_ARMS


def require(condition,message):
    if not condition:raise AssertionError(message)


def close(a,b):
    if isinstance(a,dict):
        require(isinstance(b,dict) and set(a)==set(b),'dict fields mismatch')
        for key in a:close(a[key],b[key])
    elif isinstance(a,list):
        require(isinstance(b,list) and len(a)==len(b),'list mismatch')
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        require(np.isclose(a,b,atol=1e-8,rtol=1e-7),'numeric mismatch')
    else:require(a==b,'value mismatch')


class Verifier:
    def __init__(self):self.seen=set()
    def binding(self,row):
        key=row['path'],row['sha256']
        if key in self.seen:return
        C.verify(row)
        if 'bytes' in row:require((C.ROOT/row['path']).stat().st_size==row['bytes'],'binding size')
        self.seen.add(key)
    def nested(self,value):
        if isinstance(value,dict):
            if {'path','sha256'}<=set(value):self.binding(value)
            for child in value.values():self.nested(child)
        elif isinstance(value,list):
            for child in value:self.nested(child)


def direction_counts(before,after,ids):
    output=Counter()
    sign=lambda x:'IMPROVE' if x<0 else 'WORSEN' if x>0 else 'TIE'
    for fid in ids:
        if not (before[fid]['available'] and after[fid]['available']):continue
        dt=after[fid]['translation_cm']-before[fid]['translation_cm']
        dr=after[fid]['rotation_deg']-before[fid]['rotation_deg']
        output[f'T_{sign(dt)}__R_{sign(dr)}']+=1
    return {f'T_{t}__R_{r}':output[f'T_{t}__R_{r}'] for t in ('IMPROVE','TIE','WORSEN') for r in ('IMPROVE','TIE','WORSEN')}


def trace_audit(traces,preflight,original42):
    raw,ref=[traces[arm] for arm in PAIR_ARMS]
    require(len(raw)==len(ref)==320,'320 actual batches required')
    stats={}
    for arm,actual in traces.items():
        require([r['batch'] for r in actual]==list(range(320)),'batch index sequence')
        require(Counter(r['epoch'] for r in actual)=={e:64 for e in range(5)},'5x64 epoch trace')
        for batch,expected in zip(actual[:8],preflight[arm]):
            # preflight RGB is uint8, actual trainer RGB is /255 float; input SHA per image is identical.
            for key in ('names','before_images','after_images','boxes','support','coordinates','batch_idx','roles','transfer'):
                require(batch[key]==expected[key],'actual/preflight '+arm+'/'+key)
        old=original42[arm]
        differences={key:sum(a[key]!=b[key] for a,b in zip(old,actual)) for key in ('names','before_images','images')}
        differences['plan_batches']=sum([x['plan'] for x in a['transfer']]!=[x['plan'] for x in b['transfer']] for a,b in zip(old,actual))
        require(all(n>0 for n in differences.values()),'old ineffective seed repetition')
        role=Counter(i['role'] for b in actual for i in b['transfer'])
        require(role=={'REAL':2560,'SOURCE':2560},'role exposure changed')
        first={}
        for seed,trace in [(42,old),(43,actual)]:
            first[seed]={}
            for b in trace:
                for info in b['transfer']:
                    if info['role']=='REAL':first[seed].setdefault(info['name'],info)
        common=set(first[42]) & set(first[43])
        require(len(common)==78,'real member mismatch')
        rgb_changed=sum(first[42][n]['before_image']!=first[43][n]['before_image'] for n in common)
        plan_seed_changed=sum((first[42][n]['plan'] or {}).get('seed')!=(first[43][n]['plan'] or {}).get('seed') for n in common)
        require(rgb_changed>0 and plan_seed_changed>0,'same-ID seed stream ineffective')
        stats[arm]=dict(batches=320,role_occurrences=dict(role),different_from42=differences,
            matched_first_real_identities=len(common),same_identity_base_RGB_changed=rgb_changed,
            same_identity_plan_seed_changed=plan_seed_changed,actual_first8_match_preflight=True)
    coordinate_different=0
    for a,b in zip(raw,ref):
        for key in ('epoch','batch','names','images','before_images','after_images','boxes','support','batch_idx','roles'):
            require(a[key]==b[key],'RAW/REF pair '+key)
        coordinate_different+=a['coordinates']!=b['coordinates']
        for x,y in zip(a['transfer'],b['transfer']):
            for key in ('name','role','recording','plan','scheduled','applied','geometric_demotions'):
                require(x[key]==y[key],'RAW/REF occurrence '+key)
            if x['role']=='SOURCE':
                require(not x['applied'] and x['before_image']==x['after_image'],'source modified')
    require(coordinate_different>0,'coordinate contrast missing')
    return dict(passed=True,full320_pair_independently_checked=True,coordinate_different_batches=coordinate_different,arms=stats)


def state_audit(context,protocol,verifier):
    import torch
    from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import pose_parameter
    model=torch.load(C.ROOT/protocol['initialization']['path'],map_location='cpu',weights_only=False)['model'].float()
    base=model.state_dict();allowed={name for name,_ in model.named_parameters() if pose_parameter(name)}
    protected=set(base)-allowed
    require((len(base),len(allowed),len(protected))==(879,132,747),'state inventory')
    fits={};traces={};preflight={};original42={};states={}
    for arm in PAIR_ARMS:
        fit_path=context.DOC/f'FIT_{arm}_S43.json';fit=context.read(fit_path);verifier.nested(fit)
        require(fit['complete'] and fit['seed']==43 and fit['optimizer_steps']==320,'fit contract')
        require(fit['initialization']==protocol['initialization'],'same R0 initialization')
        require(fit['exact_R0_initialization'] and fit['protected_state_exact'],'runtime initial/protected assertion missing')
        require(fit['protected_tensors']==747 and fit['trainable_tensors']==132,'declared inventory')
        final=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)['model'].float().state_dict()
        require(set(final)==set(base),'checkpoint state keys')
        require(all(torch.equal(base[k],final[k]) for k in protected),'protected tensor changed')
        changed={k for k in base if not torch.equal(base[k],final[k])}
        require(changed and changed<=allowed and changed==set(fit['changed_tensors']),'changed state outside scope')
        previous=C.read(C.DOC/f'FIT_{arm}_S42.json')
        previous_state=torch.load(C.ROOT/previous['checkpoint']['path'],map_location='cpu',weights_only=False)['model'].float().state_dict()
        differ42=sum(not torch.equal(previous_state[k],final[k]) for k in base)
        require(differ42>0,'bit-identical old ineffective seed repetition')
        require([h['steps'] for h in fit['history']]==[64,128,192,256,320],'optimizer history')
        require(all(h['protected_tensors']==747 for h in fit['history']),'epoch frozen check')
        with (C.ROOT/fit['results_csv']['path']).open() as stream:
            require(len(list(csv.DictReader(stream)))==5,'5 epoch CSV')
        start=context.read(context.RAW/f'START_{arm}_S43.json')
        require(datetime.fromisoformat(protocol['created_utc'])<datetime.fromisoformat(start['utc']),'protocol later than fit')
        fits[arm]=fit;traces[arm]=C.read(C.ROOT/fit['trace']['path'])
        preflight[arm]=context.read(context.RAW/f'PREFLIGHT_TRACE_S43_{fit["target"]}_OCC_PRIVATE.json')
        original42[arm]=C.read(C.RAW/f'TRACE_{arm}_S42.json')
        states[arm]=dict(state879_inventory_verified=True,protected747_CPU_bit_exact=True,
            allowed132_parameters_verified=True,changed_tensors=len(changed),state_tensors_different_from_seed42=differ42,
            initial_step0_evidence='runtime879 torch.equal assertion + boundR0; 별도step0tensor snapshot은없음')
    return fits,dict(states=states,trace=trace_audit(traces,preflight,original42))


def resource_audit(context,protocol,fits,verifier):
    snapshot_path=context.DOC/'RESOURCE_LEDGER_PREFIT_S43.json'
    snapshot=context.read(snapshot_path);actual=C.read(C.DOC/'RESOURCE_LEDGER.json')
    require(C.sha(snapshot_path)==protocol['global_resource_ledger']['sha256'],'prefit ledger snapshot hash')
    require(snapshot['totals']['student_fits']==4 and snapshot['totals']['optimizer_updates']==1280,'prefit cost')
    require(actual['events'][:len(snapshot['events'])]==snapshot['events'],'historical ledger overwritten')
    totals={key:sum(e[key] for e in actual['events']) for key in ('GPU_training_seconds','student_fits','optimizer_updates','selector_fits')}
    close(totals,actual['totals'])
    require(len({e['event'] for e in actual['events']})==len(actual['events']),'duplicate ledger event')
    require(totals['student_fits']==6 and totals['optimizer_updates']==1920 and totals['selector_fits']==0,'unplanned fit count')
    require(totals['GPU_training_seconds']<=21600,'GPU cap')
    for arm,fit in fits.items():
        event=next(e for e in actual['events'] if e['event']==context.stage+'::FIT_'+arm+'_S43')
        require(event['student_fits']==1 and event['optimizer_updates']==320,'repeat event cost')
        close(event['GPU_training_seconds'],fit['seconds'])
    require(not (context.DOC/'RESOURCE_LEDGER.json').exists(),'separate resetting budget ledger exists')
    return dict(passed=True,prefit_snapshot=C.bind(snapshot_path),current_ledger=C.bind(C.DOC/'RESOURCE_LEDGER.json'),
        totals=totals,global_mutable_field_is_historical_snapshot=True,
        original_protocol_unchanged=True,one_global_ledger_only=True)


def evaluation_audit(context,verifier):
    from . import eval_student as E
    from scripts.research.pallet_pose_objective_followup_v2 import metric_baseline as M
    paths=evaluation_paths(context,43)
    lock=context.read(paths['lock']);result=context.read(paths['result'])
    verifier.nested(lock);verifier.nested(result)
    require(lock['all_configured_arms_locked'] and lock['configured_new_arms']==2 and lock['no_evaluation_reference_coordinates_read'],'prediction lock')
    require(lock['original_D9'],'selector changed in D9 table')
    rows=context.read(paths['metadata']);ids=E.validate_membership(rows);groups=E.group_ids(rows)
    primary_rows=C.read(C.RAW/'evaluation/S42/METADATA.json')
    require(rows==primary_rows,'repeat population or inference metadata changed')
    predictions=context.read(paths['predictions']);poses=context.read(paths['poses'])
    frames=context.read(paths['raw']/'FRAME_METRICS.json');metrics=context.read(paths['raw']/'POSE_METRICS.json')
    for arm in predictions:
        require(list(predictions[arm])==ids and set(poses[arm])==set(ids),'inference order/population')
        if arm in PAIR_ARMS:E.assert_native_parity(rows,predictions['R0'],predictions[arm])
    checked=0
    for group,members in groups.items():
        for arm,cache in metrics.items():
            values=[cache[fid] for fid in members];valid=[v for v in values if v['available']]
            current=result['groups'][group][arm]
            require(current['frames']==len(members) and current['valid_pose']==len(valid),'pose denominator')
            for metric in ('translation_cm','rotation_deg','yaw_deg'):
                for stat,q in [('median',.5),('P90',.9)]:
                    actual=float(np.quantile([v[metric] for v in valid],q)) if valid else None
                    close(actual,current['conditional'][metric][stat])
            errors=[error for fid in members for error in frames[arm][fid]['errors']]
            require(len(errors)==current['twoD']['corners'],'2D denominator')
            for threshold in (5,10,20):
                close(sum(error<=threshold for error in errors)/len(errors),current['twoD']['PCK'][str(threshold)])
            close(M.summarize(values)['ADDsym_AUC'],current['ADDsym_AUC'])
            checked+=1
        for comparison,paired in result['contrasts'][group].items():
            after,before=comparison.split('-minus-')
            require(direction_counts(metrics[before],metrics[after],members)==paired['paired_direction_counts'],'paired counts')
            for metric in ('translation_cm','rotation_deg','yaw_deg'):
                a=[metrics[before][fid][metric] for fid in members if metrics[before][fid]['available']]
                b=[metrics[after][fid][metric] for fid in members if metrics[after][fid]['available']]
                close(float(np.median(b)-np.median(a)),paired['difference_of_conditional_medians'][metric])
    require(len(groups['NATURAL99'])==99 and len(groups['FULL128'])==128,'primary denominator')
    for arm in metrics:
        require(result['groups']['FULL128'][arm]['twoD']['corners']==985,'full corners')
    require(paths['lock'].stat().st_mtime<=paths['result'].stat().st_mtime,'result predates lock')
    candidate_lock=context.read(paths['candidate_lock']);verifier.nested(candidate_lock)
    require(candidate_lock['no_reference_coordinates_read'],'candidate lock')
    candidate_metrics=context.read(paths['raw']/'ORACLE_METRICS_SELECTIONS_PRIVATE.json')['candidate_metrics']
    candidate_records=context.read(paths['candidates'])
    for arm in PAIR_ARMS:
        require(set(candidate_metrics[arm])==set(ids),'candidate denominator')
        for fid in ids:
            selected=next((h['metric'] for h in candidate_metrics[arm][fid] if h['name']==candidate_records[arm][fid]['selected_name']),None)
            if selected is not None:close(selected,metrics[arm][fid])
    require(paths['candidate_lock'].stat().st_mtime<=(paths['raw']/'ORACLE_METRICS_SELECTIONS_PRIVATE.json').stat().st_mtime,'candidate score before lock')
    return dict(passed=True,aggregate_arm_groups_recomputed=checked,all_paired_counts_recomputed=True,
        same128_metadata_and99_primary=True,full_2D_denominator985=True,new_pair_detector_parity=True,
        candidate_selected_metric_parity=256,
        lock_order_evidence='별도infer/candidate-freeze산출물hash+고정score코드상선행검증+mtime; 별도SCORING_START이벤트나파일접근syscall로그는없음',
        reference_limitation='실제저장per-frame metric재집계; 원영상/GT다시PnP하거나IoU를독립재계산한감사는아님')


def main(stage='REPEAT_PRIMARY_S43'):
    context=StageContext(stage);paths=evaluation_paths(context,43)
    required=[context.DOC/'PRIMARY_PROTOCOL.json',context.DOC/'PREFLIGHT.json',context.DOC/'PAIR_INTEGRITY_S43.json',
        context.DOC/'CODE_LOCK.json',paths['result'],paths['candidate_lock'],paths['raw']/'ORACLE_METRICS_SELECTIONS_PRIVATE.json']
    required.extend(context.DOC/f'FIT_{arm}_S43.json' for arm in PAIR_ARMS)
    missing=[str(p.relative_to(C.ROOT)) for p in required if not p.exists()]
    if missing:
        print(json.dumps(dict(status='NOT_READY',missing=missing)),flush=True)
        return
    verifier=Verifier();protocol=context.read(context.DOC/'PRIMARY_PROTOCOL.json')
    # mutable globalledger는관측시점사본으로검증한다. 다른모든hash는현재파일검증이다.
    copy_protocol=dict(protocol);copy_protocol.pop('global_resource_ledger')
    verifier.nested(copy_protocol)
    for name in ('CODE_LOCK.json','PREFLIGHT.json','PAIR_INTEGRITY_S43.json'):
        value=context.read(context.DOC/name);verifier.nested(value)
    require(protocol['followup_kind']=='SEED_REPEAT' and protocol['seeds']==[43],'not exact repeat')
    primary=C.read(C.DOC/'PRIMARY_PROTOCOL.json')
    for key in ('args','masking','initialization','teacher','support_contract','target_contract','evaluation'):
        require(protocol[key]==primary[key],'recipe changed '+key)
    fits,state=state_audit(context,protocol,verifier)
    resources=resource_audit(context,protocol,fits,verifier)
    evaluation=evaluation_audit(context,verifier)
    failure=context.read(context.RAW/'PREFLIGHT_SANDBOX_IPC_FAILURE_PRIVATE.json')
    require(failure['new_fits']==failure['optimizer_updates']==failure['GPU_seconds']==0,'IPC retry counted as training')
    result=dict(status='PASS',passed=True,stage=stage,seed=43,checkpoint_and_training=state,resource_audit=resources,evaluation=evaluation,
        unique_input_bindings_verified=len(verifier.seen),new_audit_fits=0,new_audit_optimizer_updates=0,GPU_seconds=0,
        technical_retry='sandboxworkers2 IPC실패흔적보존후같은코드hostCPUretry; 알고리즘/성능재시도아님',
        inputs=[C.bind(Path(__file__)),C.bind(context.DOC/'PRIMARY_PROTOCOL.json'),C.bind(context.DOC/'PREFLIGHT.json'),
            C.bind(context.DOC/'PAIR_INTEGRITY_S43.json'),C.bind(paths['result']),C.bind(paths['lock']),
            C.bind(context.DOC/'RESOURCE_LEDGER_PREFIT_S43.json'),C.bind(context.RAW/'PREFLIGHT_SANDBOX_IPC_FAILURE_PRIVATE.json')])
    destination=context.DOC/'REPEAT_AUDIT_S43.json';context.save(destination,result,True)
    lines=['# 같은OCC recipe seed43 반복 독립감사','',f'상태: **PASS**. 고유입력해시 {len(verifier.seen)}개검증. 새학습/optimizer/GPU감사0회.','',
        '- RAW/REF 각각320update·5epoch, 같은R0·78실사·512source·augmentation·mask·box·실제입력계약을확인했다.',
        '- 각checkpoint879개state중보호된747개를R0와CPU bit-exact비교했다. 나머지변경은허용132parameter범위안이다.',
        '- 실제320batch RAW/REF 순서/RGB/support/box/가림계획을독립비교했다. 좌표값은달라야한다.',
        '- seed42와43의실제순서·RGB·가림계획·최종가중치가달랐다. 같은이미지첫occurrence도기본증강/RNG변경을확인했다.',
        '- 첫8개실제학습입력이CPU사전검사와일치했다. 사전검사의uint8 RGB batch와실제/255 float batch hash를같다고주장하지않고image별before/after hash를비교했다.',
        '- 전역원장6fit·1920update·selector fit0을검산했다. 사전원장4fit의불변사본과현재변하는전역원장을구분했다.',
        '- 동일128/99및985코너분모,그룹별T/R/yaw median/P90·PCK/AUC·paired9분류를저장per-frame오차에서다시계산했다.',
        '- 예측잠금선행은고정코드의읽기순서·hash·산출물mtime에근거한다. 별도SCORING_START이벤트나파일접근로그가있는것으로꾸미지않았다.',
        '- 초기879개tensor동일성은runtime torch.equalassertion과R0binding에근거한다. 저장되지않은step0snapshot을사후복원했다고쓰지않는다.','',
        '[상세 JSON](REPEAT_AUDIT_S43.json)','']
    context.save(destination.with_suffix('.md'),'\n'.join(lines),True)
    C.save(C.DOC/'REPEAT_AUDIT_S43.json',result,True)
    C.save(C.DOC/'REPEAT_AUDIT_S43.md','\n'.join(lines),True)
    print('REPEAT_AUDIT_PASS',len(verifier.seen),resources['totals'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',default='REPEAT_PRIMARY_S43')
    main(parser.parse_args().stage)

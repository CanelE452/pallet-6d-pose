"""조건부2팔 실행 도구. 이파일 작성은 개입승인/프로토콜잠금/fit 실행이 아니다.

PRIMARY의train/augmentation파일은 수정하지 않는다. 별도stage공간에서 동일
train engine을 호출하며 자원 회계는 최상위RESOURCE_LEDGER 하나로만 한다.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import re
import time

import numpy as np

from . import common as ROOT_C

PAIR_ARMS=('CLEAN_RAW_OCC','CLEAN_REF_OCC')


class StageContext:
    def __init__(self,stage):
        assert re.fullmatch(r'[A-Z][A-Z0-9_]+',stage) and stage!='PRIMARY'
        self.stage=stage
        self.DOC=ROOT_C.DOC/'followups'/stage
        self.RAW=ROOT_C.RAW/'followups'/stage
        self.OUT=ROOT_C.OUT/'followups'/stage
        self.ROOT=ROOT_C.ROOT

    def __getattr__(self,name):
        return getattr(ROOT_C,name)

    def read(self,path):
        path=Path(path)
        if path==self.DOC/'RESOURCE_LEDGER.json':
            path=ROOT_C.DOC/'RESOURCE_LEDGER.json'
        return ROOT_C.read(path)

    def save(self,path,value,freeze=False):
        path=Path(path)
        assert path!=self.DOC/'RESOURCE_LEDGER.json', 'stage별0예산 ledger생성금지'
        assert any(path.resolve().is_relative_to(root.resolve()) for root in (self.DOC,self.RAW,self.OUT))
        ROOT_C.save(path,value,freeze)

    def resource(self,event,*args,**kwargs):
        return ROOT_C.resource(self.stage+'::'+event,*args,**kwargs)


def base_context(name):
    return ROOT_C if name=='PRIMARY' else StageContext(name)


@contextmanager
def runtime(stage,modules=()):
    """별도CLI process의한번실행에만engine C/policy를교체하고복구한다."""
    context=StageContext(stage)
    protocol=context.read(context.DOC/'PRIMARY_PROTOCOL.json')
    previous=[(module,module.C) for module in modules]
    try:
        for module,_ in previous:
            module.C=context
        assert protocol['followup_kind']=='SEED_REPEAT' and protocol['masking']['schedule']==.5
        yield context,protocol
    finally:
        for module,old in previous:
            module.C=old


def freeze(stage):
    """root의명시적사전decision 파일이있을때만새stage계약을생성한다."""
    import yaml
    from .prepare import local_link,label_path
    context=StageContext(stage)
    destination=context.DOC/'PRIMARY_PROTOCOL.json'
    if destination.exists():
        for binding in context.read(destination)['inputs']+context.read(destination)['sources']:
            ROOT_C.verify(binding)
        return
    decision_path=ROOT_C.DOC/f'FOLLOWUP_DECISION_{stage}.json'
    decision=ROOT_C.read(decision_path)
    assert decision['approved'] and decision['locked_before_fit'] and decision['why']
    assert decision['kind']=='SEED_REPEAT' and decision['base_stage']=='PRIMARY'
    base=base_context(decision['base_stage'])
    original=base.read(base.DOC/'PRIMARY_PROTOCOL.json')
    p=copy.deepcopy(original)
    seed=int(decision['seed'])
    assert seed==43 and original['masking']['schedule']==.5
    p.update(created_utc=ROOT_C.now(),arms={key:original['arms'][key] for key in PAIR_ARMS},seeds=[seed],
        primary_fits=2,followup_kind=decision['kind'],base_stage=decision['base_stage'],
        base_seed=42,decision=ROOT_C.bind(decision_path),locked_before_fit=True,
        deviation='현재고정clean78/OCC계약에서실제로다른seed43으로한쌍반복;새recipe없음',
        global_resource_ledger=ROOT_C.bind(ROOT_C.DOC/'RESOURCE_LEDGER.json'))
    bindings=[]
    for target in ('RAW','REF'):
        src=original['datasets'][target]
        directory=context.RAW/'dataset'/target
        lists={}
        for role in ('train','val'):
            oldlist=ROOT_C.ROOT/src[role+'_list']['path']
            ROOT_C.verify(src[role+'_list'])
            aliases=[]
            for text in oldlist.read_text().splitlines():
                image=Path(text)
                alias=directory/'images'/image.name
                local_link(image,alias)
                local_link(label_path(image),directory/'labels'/image.with_suffix('.txt').name)
                aliases.append(str(alias))
            listing=directory/f'{role}.txt'
            context.save(listing,'\n'.join(aliases)+'\n',True)
            lists[role]=listing
            bindings.append(ROOT_C.bind(listing))
        data=yaml.safe_load((ROOT_C.ROOT/src['data']['path']).read_text())
        data.update(path=str(directory),train=str(lists['train']),val=str(lists['val']))
        datapath=directory/'data.yaml'
        context.save(datapath,yaml.safe_dump(data,sort_keys=False),True)
        p['datasets'][target]=dict(src,data=ROOT_C.bind(datapath),train_list=ROOT_C.bind(lists['train']),val_list=ROOT_C.bind(lists['val']))
        bindings.append(ROOT_C.bind(datapath))
    for name in ('PRIMARY_INPUT_BINDINGS_PRIVATE.json','CLEAN_LOCKED_PRIVATE.json'):
        context.save(context.RAW/name,ROOT_C.read(ROOT_C.RAW/name),True)
    p['inputs']=original['inputs']+bindings
    p['sources']=original['sources']+[ROOT_C.bind(base.DOC/'PRIMARY_PROTOCOL.json'),ROOT_C.bind(decision_path),
        ROOT_C.bind(ROOT_C.DOC/'REPLICATION_DECISION.json'),ROOT_C.bind(Path(__file__))]
    context.save(destination,p,True)
    print('FOLLOWUP_PROTOCOL_LOCKED',stage,decision['kind'],seed,flush=True)


def pair_trace_checks(raw,ref,full=False):
    assert len(raw)==len(ref)==(320 if full else 8)
    changed=0
    for a,b in zip(raw,ref):
        for key in ('names','images','before_images','after_images','boxes','support','batch_idx','roles'):
            assert a[key]==b[key],key
        changed+=a['coordinates']!=b['coordinates']
        for x,y in zip(a['transfer'],b['transfer']):
            for key in ('name','role','recording','plan','scheduled','applied','geometric_demotions'):
                assert x[key]==y[key],key
    assert changed>0
    exposure={}
    for arm,trace in [('RAW',raw),('REF',ref)]:
        real=[info for batch in trace for info in batch['transfer'] if info['role']=='REAL']
        source=[info for batch in trace for info in batch['transfer'] if info['role']=='SOURCE']
        for info in source:
            assert not info['applied'] and info['before_image']==info['after_image']
        if full:
            assert len(real)==len(source)==2560
            assert len({i['name'] for i in real})==78 and len({i['name'] for i in source})==512
        exposure[arm]=dict(real_occurrences=len(real),source_occurrences=len(source),
            scheduled=sum(i['scheduled'] for i in real),applied=sum(i['applied'] for i in real),
            covered=sum(i['actual_covered'] for i in real),remaining=sum(i['actual_remaining'] for i in real),
            failed_placements=sum(bool(i['plan']) and i['plan']['scheduled'] and not i['plan']['applied'] for i in real),
            placement_reasons=dict(Counter(i['reason'] for i in real)),
            bbox_fraction_sum=sum(i['plan']['bbox_fraction'] for i in real if i['applied']),
            recording={rec:dict(images=sum(i['recording']==rec for i in real),
                applied=sum(i['recording']==rec and i['applied'] for i in real),
                covered=sum(i['actual_covered'] for i in real if i['recording']==rec)) for rec in sorted({i['recording'] for i in real})})
    return dict(passed=True,batches=len(raw),coordinate_different_batches=changed,exposures=exposure,
        RGB_order_support_boxes_plan_exact=True,source_RGB_unmasked=True)


def preflight(stage):
    from . import preflight as P
    from .augmentation import load_paired_labels
    with runtime(stage,(P,)) as (context,p):
        if (context.DOC/'PREFLIGHT.json').exists():
            assert context.read(context.DOC/'PREFLIGHT.json')['passed']
            return
        parent_check=ROOT_C.read(ROOT_C.DOC/'PREFLIGHT.json')
        assert parent_check['passed']
        for binding in parent_check['bindings']:
            ROOT_C.verify(binding)
        paired=load_paired_labels(*[ROOT_C.ROOT/p['datasets'][t]['train_list']['path'] for t in ('RAW','REF')])
        recordings={r['train_id']+'.png':r['recording'] for r in context.read(context.RAW/'CLEAN_LOCKED_PRIVATE.json')['rows']}
        streams={};bindings=[]
        for target in ('RAW','REF'):
            stream,binding,_=P.collect(p,paired,recordings,43,target,'OCC')
            streams[43,target,'OCC']=stream
            bindings.append(binding)
            prior_path=ROOT_C.RAW/f'PREFLIGHT_TRACE_S43_{target}_OCC_PRIVATE.json'
            assert stream==ROOT_C.read(prior_path),'변경없어야하는seed43입력계약이달라짐'
            bindings.append(ROOT_C.bind(prior_path))
        checks={'43':pair_trace_checks(streams[43,'RAW','OCC'],streams[43,'REF','OCC'])}
        for target in ('RAW','REF'):
            streams[42,target,'OCC']=ROOT_C.read(ROOT_C.RAW/f'PREFLIGHT_TRACE_S42_{target}_OCC_PRIVATE.json')
        seeds=P.seed_checks(streams,ROOT_C.read(ROOT_C.RAW/'PREFLIGHT_TRACE_S43_REF_OCC_REPEAT_PRIVATE.json'))
        context.save(context.DOC/'PREFLIGHT.json',dict(passed=True,checks=checks,seed_checks=seeds,
            seed43_actual_stream_matches_prior_preflight=True,bindings=bindings+[ROOT_C.bind(context.DOC/'PRIMARY_PROTOCOL.json'),ROOT_C.bind(Path(__file__))],
            true_ignore_loss_test_reused=ROOT_C.bind(ROOT_C.DOC/'PREFLIGHT.json'),
            fits=0,optimizer_updates=0,GPU_seconds=0),True)


def train(stage,arm):
    from . import train as T
    with runtime(stage,(T,)) as (context,p):
        assert arm in PAIR_ARMS
        for binding in context.read(context.DOC/'PREFLIGHT.json')['bindings']:
            ROOT_C.verify(binding)
        T.train(arm,p['seeds'][0])


def parity(stage):
    context=StageContext(stage)
    p=context.read(context.DOC/'PRIMARY_PROTOCOL.json')
    seed=p['seeds'][0]
    fits={arm:context.read(context.DOC/f'FIT_{arm}_S{seed}.json') for arm in PAIR_ARMS}
    for fit in fits.values():
        assert fit['complete'] and fit['optimizer_steps']==320 and fit['protected_state_exact'] and fit['exact_R0_initialization']
        assert fit['initialization']==p['initialization']
        for key in ('trace','checkpoint','protocol'):
            ROOT_C.verify(fit[key])
    traces=[ROOT_C.read(ROOT_C.ROOT/fits[arm]['trace']['path']) for arm in PAIR_ARMS]
    result=pair_trace_checks(*traces,full=True)
    differences={}
    for arm,trace in zip(PAIR_ARMS,traces):
        previous=ROOT_C.read(ROOT_C.RAW/f'TRACE_{arm}_S42.json')
        counts={key:sum(a[key]!=b[key] for a,b in zip(previous,trace)) for key in ('names','before_images','images')}
        counts['plan_batches']=sum([v['plan'] for v in a['transfer']]!=[v['plan'] for v in b['transfer']] for a,b in zip(previous,trace))
        assert all(v>0 for v in counts.values()),'실제다른seed가아님'
        differences[arm]=counts
    result.update(same_initialization=True,protected_state_exact=True,updates_per_arm=320,
        sources=[ROOT_C.bind(context.DOC/'PRIMARY_PROTOCOL.json'),ROOT_C.bind(Path(__file__))],
        fit_bindings=[ROOT_C.bind(context.DOC/f'FIT_{a}_S{seed}.json') for a in PAIR_ARMS],
        trace_bindings=[fit['trace'] for fit in fits.values()],actual_stream_different_from_seed42=differences)
    context.save(context.DOC/f'PAIR_INTEGRITY_S{seed}.json',result,True)
    print('FOLLOWUP_PAIR_PARITY_PASS',stage,flush=True)


def evaluation_paths(context,seed):
    raw=context.RAW/'evaluation'/f'S{seed}'
    return dict(raw=raw,metadata=raw/'METADATA.json',predictions=raw/'PREDICTIONS.json',poses=raw/'POSES.json',
        lock=raw/'PREDICTIONS_LOCK.json',result=context.DOC/f'EVAL_RESULTS_S{seed}.json',
        candidates=raw/'CANDIDATES.json',candidate_lock=raw/'CANDIDATES_LOCK.json')


def infer(stage):
    import cv2
    import torch
    from ultralytics import YOLO
    from . import eval_student as E
    from scripts.research.pallet_visible_transfer_closure_v1.infer_train import predict
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import thermal_guard
    context=StageContext(stage);p=context.read(context.DOC/'PRIMARY_PROTOCOL.json');seed=p['seeds'][0]
    paths=evaluation_paths(context,seed)
    if paths['lock'].exists():
        for binding in context.read(paths['lock'])['files']+context.read(paths['lock'])['sources']:
            ROOT_C.verify(binding)
        return
    assert context.read(context.DOC/f'PAIR_INTEGRITY_S{seed}.json')['passed']
    base=base_context(p['base_stage']);bp=evaluation_paths(base,p['base_seed'])
    for binding in ROOT_C.read(bp['lock'])['files']+ROOT_C.read(bp['lock'])['sources']:
        ROOT_C.verify(binding)
    rows=ROOT_C.read(bp['metadata']);E.validate_membership(rows)
    oldpred,oldpose=ROOT_C.read(bp['predictions']),ROOT_C.read(bp['poses'])
    aliases={'R0':'R0','OLD_REF':'OLD_REF','BASE_RAW_OCC':'CLEAN_RAW_OCC','BASE_REF_OCC':'CLEAN_REF_OCC'}
    predictions={a:oldpred[b] for a,b in aliases.items()};poses={a:oldpose[b] for a,b in aliases.items()}
    assert torch.cuda.is_available()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    files=[];checkpoints={}
    for arm in PAIR_ARMS:
        fit_path=context.DOC/f'FIT_{arm}_S{seed}.json';fit=context.read(fit_path)
        ROOT_C.verify(fit['checkpoint']);checkpoints[arm]=fit['checkpoint']
        cache=paths['raw']/f'EVAL_{arm}.json'
        if cache.exists():
            stored=context.read(cache);assert stored['checkpoint']==fit['checkpoint'] and stored['reference_coordinates_read'] is False
            ROOT_C.verify(context.read(cache.with_name(cache.stem+'_LOCK.json'))['file'])
            values=stored['predictions']
        else:
            thermal_guard();model=YOLO(str(ROOT_C.ROOT/fit['checkpoint']['path']),task='pose');values={}
            for index,row in enumerate(rows):
                if index%32==0:thermal_guard()
                ROOT_C.verify(row['image']);image=cv2.imread(str(ROOT_C.ROOT/row['image']['path']))
                assert image is not None and list(image.shape[:2])==row['hw']
                values[row['id']]=predict(model,image)
            E.assert_native_parity(rows,predictions['R0'],values)
            context.save(cache,dict(predictions=values,checkpoint=fit['checkpoint'],reference_coordinates_read=False),True)
            context.save(cache.with_name(cache.stem+'_LOCK.json'),dict(file=ROOT_C.bind(cache)),True)
            del model;torch.cuda.empty_cache()
        E.assert_native_parity(rows,predictions['R0'],values)
        predictions[arm]=values
        poses[arm]={r['id']:O.D.Pose.infer(O.D.points(values[r['id']]),np.asarray(r['K']),np.asarray(r['xyz']),False) for r in rows}
        files.extend([cache,cache.with_name(cache.stem+'_LOCK.json')])
    for key,value in [('metadata',rows),('predictions',predictions),('poses',poses)]:
        context.save(paths[key],value,True);files.append(paths[key])
    sources=[context.DOC/'PRIMARY_PROTOCOL.json',context.DOC/f'PAIR_INTEGRITY_S{seed}.json',bp['lock'],Path(__file__),Path(O.D.Pose.__file__)]
    sources.extend(context.DOC/f'FIT_{a}_S{seed}.json' for a in PAIR_ARMS)
    context.save(paths['lock'],dict(created_at=ROOT_C.now(),all_configured_arms_locked=True,configured_new_arms=2,
        files=[ROOT_C.bind(f) for f in files],sources=[ROOT_C.bind(f) for f in sources],checkpoints=checkpoints,
        no_evaluation_reference_coordinates_read=True,original_D9=True),True)


def score(stage):
    from concurrent.futures import ProcessPoolExecutor
    from . import eval_student as E
    from scripts.research.pallet_pose_objective_followup_v2 import metric_baseline as M
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O,cycle_affine_eval as A
    from scripts.research.pallet_material_selftrain_closure_v1.score_eval import score_frame
    context=StageContext(stage);p=context.read(context.DOC/'PRIMARY_PROTOCOL.json');seed=p['seeds'][0]
    paths=evaluation_paths(context,seed);lock=context.read(paths['lock'])
    assert lock['all_configured_arms_locked'] and lock['configured_new_arms']==2 and lock['no_evaluation_reference_coordinates_read']
    for binding in lock['files']+lock['sources']+list(lock['checkpoints'].values()):ROOT_C.verify(binding)
    if paths['result'].exists():return
    # 두학생native와D9 hash lock 뒤에만참조좌표를연다.
    truth=ROOT_C.read(O.P.TRUTH);_,pose_truth=O.D.Pose.metadata('REAL_DEV')
    rows,predictions,poses=[context.read(paths[k]) for k in ('metadata','predictions','poses')]
    groups=E.group_ids(rows);ids=groups['FULL128']
    base=base_context(p['base_stage']);bp=evaluation_paths(base,p['base_seed'])
    aliases={'R0':'R0','OLD_REF':'OLD_REF','BASE_RAW_OCC':'CLEAN_RAW_OCC','BASE_REF_OCC':'CLEAN_REF_OCC'}
    collections=[ROOT_C.read(bp['raw']/f'{name}.json') for name in ('FRAME_METRICS','FIXED_ID_METRICS','POSE_METRICS')]
    frames,fixed,metrics=[{a:data[b] for a,b in aliases.items()} for data in collections]
    for arm in PAIR_ARMS:
        frames[arm]={fid:score_frame(fid,predictions[arm][fid],truth[fid]) for fid in ids}
        fixed[arm]={fid:score_frame(fid,predictions[arm][fid],truth[fid],fixed=True) for fid in ids}
        with ProcessPoolExecutor(max_workers=4) as pool:
            metrics[arm]=dict(pool.map(E.pose_job,[(fid,poses[arm][fid],pose_truth[fid]) for fid in ids],chunksize=8))
    groups,summaries=E.aggregate(rows,frames,fixed,metrics)
    for value in summaries['FULL128'].values():
        assert value['frames']==128 and value['twoD']['corners']==985 and value['twoD']['matched']==120
    pairs=[('BASE_RAW_OCC','CLEAN_RAW_OCC'),('BASE_REF_OCC','CLEAN_REF_OCC'),('CLEAN_RAW_OCC','CLEAN_REF_OCC'),('R0','CLEAN_REF_OCC'),('OLD_REF','CLEAN_REF_OCC')]
    contrasts={g:{b+'-minus-'+a:dict(**M.paired(metrics[a],metrics[b],ii),auxiliary_2D_and_ADD=A.contrast(frames[a],frames[b],metrics[a],metrics[b],ii)) for a,b in pairs} for g,ii in groups.items()}
    files=[]
    for name,value in [('FRAME_METRICS',frames),('FIXED_ID_METRICS',fixed),('POSE_METRICS',metrics)]:
        path=paths['raw']/f'{name}.json';context.save(path,M.clean(value),True);files.append(path)
    result=dict(stage=stage,seed=seed,groups=summaries,contrasts=contrasts,
        classification={base:M.classify_candidate(summaries['NATURAL99']['CLEAN_REF_OCC'],summaries['NATURAL99'][base]) for base in ('R0','OLD_REF','BASE_REF_OCC','CLEAN_RAW_OCC')},
        reference='동일legacy128/자연99, 기존D9, geometry-derived6D, reusedDEV',
        prediction_lock=ROOT_C.bind(paths['lock']),private_artifacts=[ROOT_C.bind(f) for f in files],
        sources=[ROOT_C.bind(paths['lock']),ROOT_C.bind(Path(__file__)),ROOT_C.bind(O.P.TRUTH),ROOT_C.bind(bp['result']),
            ROOT_C.bind(O.D.Pose.E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json')],new_scoring_fits=0,optimizer_updates=0)
    context.save(paths['result'],M.clean(result),True)
    print('FOLLOWUP_PAIR_SCORED',stage,result['classification'],flush=True)


def candidate_freeze(stage):
    from . import eval_student as E
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    context=StageContext(stage);protocol=context.read(context.DOC/'PRIMARY_PROTOCOL.json')
    paths=evaluation_paths(context,protocol['seeds'][0]);lock=context.read(paths['lock'])
    assert lock['no_evaluation_reference_coordinates_read']
    for binding in lock['files']+lock['sources']:ROOT_C.verify(binding)
    if paths['candidate_lock'].exists():
        for binding in context.read(paths['candidate_lock'])['files']+context.read(paths['candidate_lock'])['sources']:ROOT_C.verify(binding)
        return
    rows,predictions,poses=[context.read(paths[k]) for k in ('metadata','predictions','poses')]
    E.validate_membership(rows)
    values={a:{} for a in PAIR_ARMS}
    for arm in PAIR_ARMS:
        for row in rows:
            fid=row['id'];record=O.candidate_record(predictions[arm][fid],row)
            O.D.close(record['current'],poses[arm][fid]);values[arm][fid]=record
    context.save(paths['candidates'],values,True)
    context.save(paths['candidate_lock'],dict(no_reference_coordinates_read=True,
        files=[ROOT_C.bind(paths['candidates'])],sources=[ROOT_C.bind(paths['lock']),ROOT_C.bind(Path(__file__)),ROOT_C.bind(Path(O.__file__))],
        existing_D9_current_parity=256,new_selector=False,new_fits=0,optimizer_updates=0,GPU_seconds=0),True)


def candidate_score(stage):
    from concurrent.futures import ProcessPoolExecutor
    from . import eval_student as E
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    context=StageContext(stage);protocol=context.read(context.DOC/'PRIMARY_PROTOCOL.json')
    paths=evaluation_paths(context,protocol['seeds'][0]);lock=context.read(paths['candidate_lock'])
    assert lock['no_reference_coordinates_read']
    for binding in lock['files']+lock['sources']:ROOT_C.verify(binding)
    destination=paths['raw']/'ORACLE_METRICS_SELECTIONS_PRIVATE.json'
    if destination.exists():return
    assert paths['result'].exists(),'실제D9 평가를먼저완료'
    _,truth=O.D.Pose.metadata('REAL_DEV')
    candidates=context.read(paths['candidates']);current=context.read(paths['raw']/'POSE_METRICS.json');choices={}
    for arm,records in candidates.items():
        with ProcessPoolExecutor(max_workers=4) as pool:
            choices[arm]=dict(pool.map(E.candidate_job,[(fid,record,truth[fid]) for fid,record in records.items()],chunksize=8))
        for fid,record in records.items():
            selected=next((v['metric'] for v in choices[arm][fid] if v['name']==record['selected_name']),None)
            if selected is not None:O.D.close(selected,current[arm][fid])
    context.save(destination,dict(candidate_metrics=choices,posthoc_selections={},no_oracle_deployment=True),True)
    context.save(paths['raw']/'CANDIDATE_METRICS_LOCK.json',dict(file=ROOT_C.bind(destination),
        candidate_lock=ROOT_C.bind(paths['candidate_lock']),implementation=ROOT_C.bind(Path(__file__)),new_fits=0),True)
    print('REPEAT_CANDIDATE_METRICS_COMPLETE',stage,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['freeze','preflight','train','parity','infer','score','candidate-freeze','candidate-score'])
    parser.add_argument('--stage',required=True)
    parser.add_argument('--arm',choices=PAIR_ARMS)
    args=parser.parse_args()
    if args.phase=='train':train(args.stage,args.arm)
    else:globals()[args.phase.replace('-','_')](args.stage)

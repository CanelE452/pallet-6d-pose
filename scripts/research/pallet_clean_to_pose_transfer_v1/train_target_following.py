"""실제 clean78 저장타깃의 native 추종 진단; GT 정확도나 증강TRAIN-fit이 아님.

원래217 감사의 native RGB/좌표 역변환을 재사용하되 새512 real slot 반복수를
복원한다. 실제 새label bytes와 원래label이 같은지 확인한다. 최고confidence
검출만사용하며 타깃에가까운후보매칭을하지않는다. fit/optimizer step은0회다.
"""
import argparse
from collections import Counter
import math
from pathlib import Path
import time

import numpy as np

from . import common as C
from . import eval_student as E

PARENT=C.ROOT/'data/pallet/results/pallet_visible_transfer_closure_v1'
RAW=C.RAW/'train_target_following'


def selected(prediction):
    index=prediction.get('selected_index')
    return prediction['candidates'][index] if index is not None else None


def restore_native(label, hw):
    values=np.asarray(label,float)
    assert values.shape==(32,)
    q=values[5:].reshape(9,3)
    h,w=hw
    return q[:,:2]*[w+200,h+200]-100,q[:,2]==2


def prepare():
    destination=RAW/'TRAIN_TARGETS_PRIVATE.json'
    lock_path=RAW/'TARGETS_LOCK.json'
    if lock_path.exists():
        lock=C.read(lock_path)
        for b in lock['files']+lock['sources']:
            C.verify(b)
        return C.read(destination)
    protocol=C.read(C.DOC/'PRIMARY_PROTOCOL.json')
    assert protocol['locked_before_fit'] and protocol['train_unique_images']==78
    source_rows={r['id']:r for r in C.read(PARENT/'TRAIN_TARGETS_PRIVATE.json')}
    lists={}
    for target in ('RAW','REF'):
        b=protocol['datasets'][target]['train_list']
        C.verify(b)
        lists[target]=[Path(s) for s in (C.ROOT/b['path']).read_text().splitlines() if s.strip()]
    assert [p.stem for p in lists['RAW']]==[p.stem for p in lists['REF']]
    counts=Counter(p.stem for p in lists['REF'] if p.stem.startswith('PLASTIC__'))
    assert len(counts)==78 and sum(counts.values())==512
    output=[]
    labels=[]
    for fid,count in sorted(counts.items()):
        old=source_rows[fid]
        C.verify(old['image'])
        actual={}
        for target in ('RAW','REF'):
            image=next(p for p in lists[target] if p.stem==fid)
            label=image.parent.parent/'labels'/f'{fid}.txt'
            C.verify(old['padded_label_bindings'][target])
            assert C.sha(label)==old['padded_label_bindings'][target]['sha256']
            xy,mask=restore_native(label.read_text().split(),old['hw'])
            np.testing.assert_array_equal(mask,old['common_support'])
            np.testing.assert_allclose(xy,old[target.lower()+'_target'],atol=1e-7,rtol=0)
            actual[target]=C.bind(label)
            labels.append(label)
        output.append(dict(old,occurrences_per_epoch=count,exposures_5epochs=count*protocol['epochs'],actual_label_bindings=actual))
    E.save(destination,output)
    sources=[C.DOC/'PRIMARY_PROTOCOL.json',PARENT/'TRAIN_TARGETS_PRIVATE.json',Path(__file__)]
    sources.extend(C.ROOT/protocol['datasets'][t]['train_list']['path'] for t in ('RAW','REF'))
    sources.extend(labels)
    E.save(lock_path,dict(files=[C.bind(destination)],sources=[C.bind(p) for p in sources],
        unique=78,real_slots_per_epoch=512,new_coordinate_conversion=False,
        actual_label_parity=True,native_unaugmented_only=True,
        post_affine_common_support_not_replayed=True))
    return output


def infer(seed=42):
    rows=prepare()
    _,fits=E.fits_for(seed)
    destination=RAW/f'S{seed}'
    lock_path=destination/'PREDICTIONS_LOCK.json'
    if lock_path.exists():
        for b in C.read(lock_path)['files']+C.read(lock_path)['sources']:
            C.verify(b)
        return
    import cv2
    import torch
    from ultralytics import YOLO
    from scripts.research.pallet_visible_transfer_closure_v1.infer_train import predict
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import thermal_guard,assert_detector_parity
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    r0_old=C.read(PARENT/'TRAIN_PREDICTIONS_R0.json')
    protocol=C.read(C.DOC/'PRIMARY_PROTOCOL.json')
    assert r0_old['checkpoint']==protocol['initialization']
    r0={r['id']:r0_old['predictions'][r['id']] for r in rows}
    E.save(destination/'R0.json',dict(predictions=r0,checkpoint=protocol['initialization'],reused_frozen=True))
    files=[destination/'R0.json']
    start=time.monotonic()
    for arm in E.ARMS:
        path=destination/f'{arm}.json'
        seal=destination/f'{arm}_LOCK.json'
        if path.exists():
            assert seal.exists()
            C.verify(C.read(seal)['file'])
            assert C.read(path)['checkpoint']==fits[arm]['checkpoint']
            files.extend([path,seal])
            continue
        thermal_guard()
        model=YOLO(str(C.ROOT/fits[arm]['checkpoint']['path']),task='pose')
        predictions={}
        for index,row in enumerate(rows):
            if index%24==0:
                thermal_guard()
                print('CLEAN_TRAIN_TARGET_INFER',arm,index,len(rows),flush=True)
            C.verify(row['image'])
            image=cv2.imread(str(C.ROOT/row['image']['path']))
            assert image is not None and list(image.shape[:2])==row['hw']
            pred=predict(model,image)
            assert_detector_parity(r0[row['id']],pred)
            predictions[row['id']]=pred
        E.save(path,dict(predictions=predictions,checkpoint=fits[arm]['checkpoint'],native_unaugmented=True,
            highest_confidence_only=True,target_or_GT_matching=False))
        E.save(seal,dict(file=C.bind(path)))
        files.extend([path,seal])
        del model
        torch.cuda.empty_cache()
    sources=[RAW/'TARGETS_LOCK.json',PARENT/'TRAIN_PREDICTIONS_R0.json',Path(__file__),
        C.ROOT/'scripts/research/pallet_visible_transfer_closure_v1/infer_train.py']
    sources.extend(C.DOC/f'FIT_{arm}_S{seed}.json' for arm in E.ARMS)
    E.save(lock_path,dict(created_at=C.now(),files=[C.bind(p) for p in files],sources=[C.bind(p) for p in sources],
        no_eval_reference_read=True,TRAIN_pseudo_targets_not_physical_truth=True,
        new_fits=0,optimizer_updates=0,seconds=time.monotonic()-start))


def summarize_errors(points, weighted=False):
    if not points:
        return dict(points=0,mean_px=None,median_px=None,P90_px=None)
    errors=[]
    for p in points:
        errors.extend([p['error_px']]*(p['occurrences_per_epoch'] if weighted else 1))
    return dict(points=len(points),effective_point_occurrences=len(errors),mean_px=float(np.mean(errors)),
        median_px=float(np.median(errors)),P90_px=float(np.quantile(errors,.9)),
        missing_points=sum(p['missing'] for p in points))


def score(seed=42):
    rows=prepare()
    directory=RAW/f'S{seed}'
    lock_path=directory/'PREDICTIONS_LOCK.json'
    lock=C.read(lock_path)
    for b in lock['files']+lock['sources']:
        C.verify(b)
    result_path=C.DOC/f'TRAIN_TARGET_FOLLOWING_S{seed}.json'
    if result_path.exists():
        for b in C.read(result_path)['inputs']:
            C.verify(b)
        return
    predictions={arm:C.read(directory/f'{arm}.json')['predictions'] for arm in ('R0',*E.ARMS)}
    points=[]
    for row in rows:
        fid=row['id']
        for arm,values in predictions.items():
            pred=selected(values[fid])
            for target in ('raw_target','ref_target'):
                for corner,valid in enumerate(row['common_support']):
                    if not valid:
                        continue
                    point=np.asarray(pred['keypoints_xy'][corner]) if pred is not None else np.array([np.nan,np.nan])
                    missing=not np.isfinite(point).all()
                    error=float(math.hypot(*row['hw'])) if missing else float(np.linalg.norm(point-row[target][corner]))
                    points.append(dict(id=fid,recording=row['recording'],arm=arm,target=target,corner=corner,
                        occurrences_per_epoch=row['occurrences_per_epoch'],error_px=error,missing=missing))
    groups={'ALL78':[r['id'] for r in rows]}
    groups.update({rec:[r['id'] for r in rows if r['recording']==rec] for rec in sorted({r['recording'] for r in rows})})
    output={}
    for group,ids in groups.items():
        output[group]={}
        for arm in predictions:
            output[group][arm]={}
            for target in ('raw_target','ref_target'):
                samples=[p for p in points if p['id'] in ids and p['arm']==arm and p['target']==target]
                pooled=[p for p in samples if p['corner']<8]
                per_image=[float(np.mean([p['error_px'] for p in pooled if p['id']==fid])) for fid in ids if any(p['id']==fid for p in pooled)]
                output[group][arm][target]=dict(unique_corner_pooled=summarize_errors(pooled),
                    occurrence_weighted_corner_pooled=summarize_errors(pooled,True),
                    unique_image_uniform_mean_px=float(np.mean(per_image)) if per_image else None,
                    center_separate=summarize_errors([p for p in samples if p['corner']==8]),
                    by_corner={str(c):summarize_errors([p for p in samples if p['corner']==c]) for c in range(8)})
    private=directory/'ERRORS_PRIVATE.json'
    E.save(private,points)
    result=dict(status='NATIVE_TRAIN_TARGET_FOLLOWING_ONLY',images=len(rows),real_slots_per_epoch=sum(r['occurrences_per_epoch'] for r in rows),
        groups=output,inputs=[C.bind(lock_path),C.bind(RAW/'TARGETS_LOCK.json'),C.bind(private),C.bind(Path(__file__))],
        meaning='저장된의사타깃을따르는정도이며실제물리정답정확도가아님',
        coordinate_contract='native unaugmented RGB + original reflection pad100 inference; storedpaddedlabel→native기존변환검산',
        selection='highest confidence; no target/GT-nearest candidate replacement',
        visibility='actual storedcommonv2 support,corner0..7; center8별도',
        limitations=['실제학습affine/HSV/가림입력전수fit이아님','post-affine공통support/exposure는trainingtrace에서별도감사',
            'native타깃잔차만으로학습량부족원인을확정할수없음','이미지균등/코너균등/occurrence가중집계가서로다름'],
        new_fits=0,optimizer_updates=0)
    E.save(result_path,result)
    lines=['# Clean78 학습 타깃 전달: native 입력 진단','',
        '실제78장과epoch512 real slot에서복원했다. 새학습은없다. 이수치는의사타깃추종이지정답정확도나가림증강입력의실제TRAIN loss가아니다.','',
        '| 모델 | raw타깃 코너평균px | corrected타깃 코너평균px | corrected타깃 occurrence가중평균px | corrected타깃 이미지균등평균px |',
        '|---|---:|---:|---:|---:|']
    for arm,values in output['ALL78'].items():
        lines.append(f'| {arm} | {values["raw_target"]["unique_corner_pooled"]["mean_px"]:.4f} | {values["ref_target"]["unique_corner_pooled"]["mean_px"]:.4f} | {values["ref_target"]["occurrence_weighted_corner_pooled"]["mean_px"]:.4f} | {values["ref_target"]["unique_image_uniform_mean_px"]:.4f} |')
    lines+=['','최고confidence 검출을 그대로사용했다. teacher타깃에가장가까운후보로교체하지않았다. center는별도집계이며v1 ignored점은감독점으로세지않는다.',
        '원래nativeRGB에reflection100 padding을정확히한번적용한다. 이미padded된export RGB에다시padding을더하지않는다.',
        '가림/affine실제TRAIN입력에대한검증은아니므로이진단만으로최적화량·표현한계·전이실패중하나를확정하지않는다.','',
        f'[recording·코너·중앙값/P90·가중집계](TRAIN_TARGET_FOLLOWING_S{seed}.json)','']
    C.save(result_path.with_suffix('.md'),'\n'.join(lines),True)
    print('CLEAN_TRAIN_TARGET_FOLLOWING_COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','infer','score'])
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.phase=='prepare':prepare()
    else:(infer if args.phase=='infer' else score)(args.seed)

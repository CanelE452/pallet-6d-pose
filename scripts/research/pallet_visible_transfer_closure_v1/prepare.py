from collections import Counter
from pathlib import Path
import subprocess
import numpy as np
import cv2
from . import common as C

def main():
    if (C.DOC/'INPUT_BINDINGS.json').exists():
        for b in C.read(C.DOC/'INPUT_BINDINGS.json')['files']:C.verify(b)
        print('PREPARED_ALREADY');return
    P=C.P;files=set()
    required=('REPORT_KO.md','CORE_COMPARABILITY_AUDIT.json','CORE_RESULTS.json','PSEUDO_LABEL_QUALITY.json','MEASUREMENT_ROWS.json','PAIRED_ANALYSIS.json','INPUT_BINDINGS.json','PREDICTIONS_LOCK.json','MANUAL_SUPERVISION_BUDGET.json','INDEPENDENT_CONFIRMATION_AUDIT.json')
    for n in required:
        path=P.DOC/n;files.add(path)
        if n.endswith('.json'):C.read(path)
        else:path.read_text()
    for b in C.read(P.DOC/'INPUT_BINDINGS.json')['files']:C.verify(b)
    protocol=C.read(P.REC/'pose_only/PROTOCOL.json');accepted={r['id']:r for r in C.read(P.CACHE/'PSEUDO_ACCEPTED.json')}
    lists={a:[Path(x) for x in (C.ROOT/protocol['datasets'][a]['train_list']['path']).read_text().splitlines()] for a in ('RAW','REF')}
    assert [p.name for p in lists['RAW']]==[p.name for p in lists['REF']]
    counts=Counter(p.stem for p in lists['RAW'] if p.stem.startswith('PLASTIC__'))
    assert len(counts)==217 and sum(counts.values())==512
    rows=[];parity=[]
    for fid,n in sorted(counts.items()):
        r=accepted[fid];h,w=r['raw_hw'];xy={};masks={};labels={}
        for a in ('RAW','REF'):
            image=next(p for p in lists[a] if p.stem==fid);label=image.parent.parent/'labels'/f'{fid}.txt'
            vec=np.array(label.read_text().split(),float);q=vec[5:].reshape(9,3);masks[a]=q[:,2]==2
            xy[a]=q[:,:2]*[w+200,h+200]-100;labels[a]=C.bind(label);files.add(label)
            stored=cv2.imread(str(image));native=cv2.imread(str(C.ROOT/r['image']['path']))
            assert np.array_equal(stored,cv2.copyMakeBorder(native,100,100,100,100,cv2.BORDER_REFLECT_101))
            nativeq=np.array(P.selected(r['raw' if a=='RAW' else 'refined'])['keypoints_xy'])
            assert np.allclose(xy[a][masks[a]],nativeq[masks[a]],atol=1e-5,rtol=0)
            files.update([image.resolve(),C.ROOT/r['image']['path']])
        assert np.array_equal(masks['RAW'],masks['REF'])
        raw=P.selected(r['raw']);ref=P.selected(r['refined'])
        assert raw['keypoints_conf']==ref['keypoints_conf'] and raw['box_xyxy']==ref['box_xyxy'] and raw['keypoints_xy'][8]==ref['keypoints_xy'][8]
        rows.append(dict(id=fid,image=r['image'],hw=[h,w],recording=r['recording_id'],occurrences_per_epoch=n,
            exposures_5epochs=n*5,common_support=masks['RAW'],raw_target=xy['RAW'],ref_target=xy['REF'],
            padded_label_bindings=labels,predicted_box=raw['box_xyxy'],keypoint_confidence=raw['keypoints_conf']))
    for a in C.ARMS:
        files.add(C.ROOT/C.checkpoint(a)['path'])
        if a!='R0':
            fp=P.REC/'pose_only'/f'FIT_{a}.json';f=C.read(fp);files.update([fp,C.ROOT/f['results_csv']['path']])
    for f in (P.REC/'pose_only/PROTOCOL.json',P.CACHE/'PSEUDO_ACCEPTED.json',P.FINAL,P.TRUTH,P.RAW/'ANCHOR_POINTS.json',P.RAW/'PREDICTIONS.json',P.RAW/'POSE_PREDICTIONS.json'):files.add(f)
    for n in ('recovery_pose.py','recovery_pose_trainer.py','evaluate.py','train.py','pseudo.py'):files.add(C.ROOT/'scripts/research/pallet_type_selftrain_v1'/n)
    for n in ('true_ignore_pose_loss.py','true_ignore_trainer.py'):files.add(C.ROOT/'scripts/self_training_yolo/v3'/n)
    C.save(C.RAW/'TRAIN_TARGETS_PRIVATE.json',rows,True)
    C.save(C.DOC/'PREFLIGHT.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch='main',remote_start='9238735e2702bf817c6325811eb0d48ed2c0ade3',
        status=subprocess.check_output(['git','status','--short','--branch'],text=True),log=subprocess.check_output(['git','log','-5','--oneline'],text=True),
        unique_real=217,real_slots_per_epoch=512,images_added=0,manual_points_added=0,
        target_native_roundtrip_tolerance_px=1e-5,padding_once_bit_exact=True,raw_ref_boxes_support_and_preserved_teacher_confidence=True,
        augmented_tensor_cache_available=False,augmentation_scope='Native unaugmented inference only. Historical first-batch fingerprints are NOT a cache of all augmented training inputs.',max_new_fits=2,max_updates_per_arm=640,total_update_cap=1280),True)
    C.save(C.DOC/'GOAL_PATH.md','# 목적과 가지치기\n\n'+C.table(['행동','왜 필요한가','기존으로 대체'],[
        ['66점 paired/난도/연속오차','동률이 같은 점 유지인지 진입·이탈 상쇄인지 구분','기존 frozen예측/참조 재계산'],
        ['같은66점 legacy↔verified','표본 구성과 참조 변경 효과 분리','같은prediction/fixedID/no-match-gate로 통일'],
        ['TRAIN217 native 타깃 추종','원래 감독도 못 따라간 것과 학습밖 전이 구분','일치하는 TRAIN예측 cache 없음:3frozen모델 추론'],
        ['학습곡선·loss reduction','학습량 개입 근거/경쟁설명','기존5epoch csv·loss코드'],
        ['선택기/새모듈/전면대칭감사','native2D 질문과 관계없음','제외'],
        ['추가hard labels/모든지표비악화','이번 종료기준 아님','제외']])+ '\n가장 강한 반론: 작은 반복DEV의10px문턱에 맞춘 변경 아닌가? TRAIN 증거로 한 개입만 잠그고 모든 결과/손상을 보고한다. 그래도 독립 검증은 아니다. confidence 보존은 corrected좌표의 새 calibration을 뜻하지 않는다. 관찰적잔차는 원인 증명이 아니다.\n',True)
    files.update(Path(__file__).parent.glob('*.py'))
    C.save(C.DOC/'INPUT_BINDINGS.json',dict(files=[C.bind(f) for f in sorted(files)],targets=C.bind(C.RAW/'TRAIN_TARGETS_PRIVATE.json')),True)
    print('READY_NATIVE_TRAIN217',flush=True)

if __name__=='__main__':main()

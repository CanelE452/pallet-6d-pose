"""Distill fixed per-material corrected Clean19 labels into two R0 students."""
import argparse
import copy
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from scripts.research import pallet_replay_clean19_v1 as P
from scripts.research import pallet_visible_refine_hidden_pnp_v1 as V
from scripts.research.pallet_type_selftrain_v1 import common as C, train as T, evaluate as E

ROOT=P.ROOT
NAME='pallet_clean19_student_v1'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
OUT=ROOT/'outputs'/NAME
OLD_DOC=C.DOC


def scope():
    C.DOC=DOC;C.RAW=RAW;C.OUT=OUT


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


@torch.no_grad()
def prepare():
    assert not DOC.exists() and not RAW.exists()
    scope();P.N.setup();assert torch.cuda.is_available()
    split=P.read(P.DOC/'SPLIT.json');train=split['train'];evaluation=split['evaluation']
    assert len(train)==19 and len(evaluation)==300
    assert not {r['image']['sha256'] for r in train}&{r['image']['sha256'] for r in evaluation}
    baseline=P.read(P.RAW/'BASELINE_PREDICTIONS.json')['R0']
    protected=[P.bind(Path(__file__)),P.bind(P.DOC/'SPLIT.json'),P.bind(P.RAW/'BASELINE_PREDICTIONS.json'),P.bind(Path(V.__file__))]
    pseudo=[]
    for material in ('plastic','wood'):
        fit=P.read(ROOT/f'_docs/experiments/pallet_replay_by_type_v1/{material}/FIT.json')
        C.verify(fit['checkpoint']);protected.append(fit['checkpoint'])
        model=P.C.load_model()
        model.load_state_dict(torch.load(ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)['model_state_dict'])
        model.eval()
        for r in train:
            if r['object_type']!=material:continue
            C.verify(r['image']);image=cv2.imread(str(ROOT/r['image']['path']));assert image is not None
            raw=baseline[r['id']];refined=P.C.predict(model,image,raw,None)
            candidate=P.C.selected(raw);assert candidate is not None
            cam=P.read(ROOT/r['annotation']['path'])['camera_data'];intr=cam['intrinsics']
            K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
            dims,_=V.registry_input(C.TYPES[material.upper()])
            points=np.array(candidate['keypoints_xy'],float);points[(points==-1).all(1)]=np.nan
            initial=V.Pose.infer(points,K,dims[[0,2,1]],False)
            final,visible,decision=V.pipeline(raw,refined,initial,K,image.shape[:2])
            pseudo.append(dict(**r,raw_hw=list(image.shape[:2]),raw=raw,refined=final,visible_only=visible,decision=decision))
            protected.extend([r['image'],r['annotation']])
        del model;torch.cuda.empty_cache()
    save(RAW/'PSEUDO_LABELS.json',pseudo)
    shared=RAW/'dataset';bindings=[];syn=[];real={'PLASTIC':[],'WOOD':[]}
    # Reuse byte-identical synthetic512 from the established self-training protocol.
    parent=P.read(OLD_DOC/'TRAIN_PROTOCOL.json')
    oldlist=ROOT/parent['datasets']['PLASTIC']['train_list']['path']
    C.verify(parent['datasets']['PLASTIC']['train_list'])
    oldsyn=oldlist.read_text().splitlines()[:512];assert len(set(oldsyn))==512
    for old in oldsyn:
        image=Path(old);label=image.parent.parent/'labels'/image.with_suffix('.txt').name
        assert image.name.startswith('syn__')
        dest=shared/'images'/image.name;dl=shared/'labels'/label.name
        T.link(image,dest);T.link(label,dl);syn.append(str(dest));bindings.extend([C.bound(dest),C.bound(dl)])
    for i,row in enumerate(pseudo):
        candidate=P.C.selected(row['refined']);text=T.label(candidate,row['raw_hw']);assert text is not None
        dest=shared/'images'/f'real_{i:02d}.png';dest.parent.mkdir(parents=True,exist_ok=True)
        image=cv2.imread(str(ROOT/row['image']['path']))
        assert cv2.imwrite(str(dest),cv2.copyMakeBorder(image,100,100,100,100,cv2.BORDER_REFLECT_101))
        label=shared/'labels'/f'real_{i:02d}.txt';C.write_text(label,text)
        real[row['object_type'].upper()].append(str(dest));bindings.extend([C.bound(dest),C.bound(label)])
    assert [len(real[k]) for k in ('PLASTIC','WOOD')]==[10,9]
    oldval=Path(oldsyn[0]).parent.parent/'val.txt';assert oldval.exists()
    C.write_text(shared/'val.txt',oldval.read_text())
    for p in oldval.read_text().splitlines():
        im=Path(p);bindings.extend([C.bound(im),C.bound(im.parent.parent.parent/'labels/val'/im.with_suffix('.txt').name)])
    datasets={}
    for arm in real:
        replacement=np.random.default_rng(9021).choice(sorted(real[arm]),512,replace=True).tolist()
        assert len(set(replacement))==len(real[arm])
        trainlist=shared/f'{arm}_train.txt';C.write_text(trainlist,'\n'.join(syn+replacement)+'\n')
        yaml=shared/f'{arm}.yaml'
        C.write_text(yaml,f'path: {shared}\ntrain: {trainlist}\nval: {shared/"val.txt"}\nnc: 1\nnames: [pallet]\nkpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n')
        datasets[arm]=dict(data=C.bound(yaml),train_list=C.bound(trainlist),pseudo_unique=len(real[arm]),slots=1024)
        bindings.extend([C.bound(trainlist),C.bound(yaml)])
    protocol=dict(args=T.ARGS,arms=list(real),datasets=datasets,initialization=C.bound(C.N.E.R0),
        inputs=bindings,sources=protected+[C.bound(RAW/'PSEUDO_LABELS.json')],code=C.bound(T.__file__),common=C.bound(C.__file__),test_code=C.bound(C.HERE/'test_contract.py'),
        selection='fixed last.pt 5 epochs / 320 updates per student; no test selection',
        labels='R0 -> material Replay visible correction -> self-hidden PnP fill; original predicted boxes/confidence; no manual coordinate substitution',
        caveat='Refiner was supervised on these19, so this is supervised-refiner pseudo-label distillation, not fully unlabeled learning. Same-session adaptation, not independent generalization.',
        evaluation='fixed plastic184 + wood116; students alone, no inference refiner or filter',synthetic='same512 mixed-material replay, 1:1 epoch slots',
        tests=V.tests(),auto_promote=False)
    C.freeze(DOC/'TRAIN_PROTOCOL.json',protocol)
    records=[dict(**r,kind=r['object_type'].upper()) for r in evaluation]
    C.freeze(DOC/'EVAL_PROTOCOL.json',dict(records=records,arms=list(real),sources=[C.bound(DOC/'TRAIN_PROTOCOL.json'),P.bind(P.DOC/'SPLIT.json')]))
    print('PREPARED',len(pseudo),'hidden filled',sum(r['decision']['applied'] for r in pseudo),flush=True)


def evaluate(arm):
    scope()
    # Only evaluate each student on its own material; unchanged inference recipe.
    original=C.read
    def read(path):
        result=original(path)
        if Path(path)==DOC/'EVAL_PROTOCOL.json':
            result=copy.deepcopy(result);result['records']=[r for r in result['records'] if r['kind']==arm]
        return result
    C.read=read
    E.infer(arm)


def score():
    scope();split=P.read(P.DOC/'SPLIT.json');records=split['evaluation']
    files=[C.bound(RAW/f'EVAL_PREDICTIONS_{a}.json') for a in ('PLASTIC','WOOD')]
    C.freeze(DOC/'EVAL_OUTPUTS_LOCK.json',dict(files=files,frozen_before_scoring=True))
    truth=P.read(P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json')
    prior=P.read(P.RAW/'FRAME_METRICS.json')['R0']
    teacher=P.read(V.RAW/'FRAME_METRICS.json')['TYPE_REPLAY_PIPELINE']
    student={};predictions={}
    for a in ('PLASTIC','WOOD'):
        for r in P.read(RAW/f'EVAL_PREDICTIONS_{a}.json')['records']:
            fid=r['id'];t=truth[fid];p=P.C.selected(r['prediction'])
            matched=p is not None and E.O.iou(p['box_xyxy'],t['box'])>=.5
            q=np.full((9,2),np.nan) if p is None else p['keypoints_xy']
            student[fid]=dict(id=fid,**P.M.measure(q,t['gt'],t['valid'],t['permutations'],t['hw'],matched,p is not None))
            predictions[fid]=r['prediction']
    assert set(student)=={r['id'] for r in records}
    groups={'ALL300':[r['id'] for r in records]}
    groups.update({m:[r['id'] for r in records if r['object_type']==m] for m in ('plastic','wood')})
    groups.update({s:[r['id'] for r in records if r['severity']==s] for s in P.SEVERITIES})
    results={};lines=['# Clean19 보정 수도레이블 종류별 student 학습','',
        '플라스틱10장/목재9장의 보정 수도레이블 → 각각 R0부터5epoch/320step. 합성512+실사복원추출512 슬롯. 최종last만 평가. 평가300장 학생 단독출력; 보정기/PnP/필터 없음.',
        '', '보정기는19장의 수동 정답으로 이미 학습되었다. 따라서 완전 무라벨 self-training이 아닌 보정 지식 증류이다. 같은세션 적응이며 독립 일반화 검증이 아니다. AP/6D pose는 이번에 측정하지 않았다.', '']
    for g,ids in groups.items():
        results[g]={};lines += ['## '+g,'','| 모델 | PCK10% | PCK20% | median px | P90 px | matched |','|---|---:|---:|---:|---:|---:|']
        for name,mm in [('R0',prior),('CORRECTED_TEACHER',teacher),('STUDENT',student)]:
            s=P.M.summary([mm[i] for i in ids]);results[g][name]=s
            lines.append(f'| {name} | {100*s["PCK"]["10"]:.2f} | {100*s["PCK"]["20"]:.2f} | {s["matched_pooled_corner8_median_px"]:.2f} | {s["matched_pooled_corner8_P90_px"]:.2f} | {s["matched"]}/{len(ids)} |')
        results[g]['damage']=P.M.damage([prior[i] for i in ids],[student[i] for i in ids]);lines.append('')
    C.freeze(RAW/'FRAME_METRICS.json',student);C.freeze(DOC/'RESULTS.json',results)
    # Representative best/worst changes, chosen for display only after all metrics frozen.
    baseline=P.read(P.RAW/'BASELINE_PREDICTIONS.json')['R0'];edges=[(0,1),(1,5),(5,4),(4,0),(3,2),(2,6),(6,7),(7,3),(0,3),(1,2),(5,6),(4,7)]
    for material in ('plastic','wood'):
        rr=[r for r in records if r['object_type']==material]
        rr.sort(key=lambda r:student[r['id']]['frame_mean_px']-prior[r['id']]['frame_mean_px'])
        show=rr[:2]+rr[-2:];panels=[]
        for r in show:
            fid=r['id'];image=cv2.imread(str(ROOT/r['image']['path']));h,w=image.shape[:2];scale=500/w;pair=[]
            for name,pred in [('R0',baseline[fid]),('STUDENT',predictions[fid])]:
                im=cv2.resize(image,(500,round(h*scale)));p=P.C.selected(pred)
                if p is not None:
                    points=np.array(p['keypoints_xy'])*scale
                    for a,b in edges:
                        if np.isfinite(points[[a,b]]).all():cv2.line(im,tuple(points[a].astype(int)),tuple(points[b].astype(int)),(0,220,255),1)
                canvas=cv2.copyMakeBorder(im,44,0,0,0,cv2.BORDER_CONSTANT)
                err=(prior if name=='R0' else student)[fid]['frame_mean_px']
                cv2.putText(canvas,f'{name} mean {err:.2f}px',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                cv2.putText(canvas,fid,(8,37),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1);pair.append(canvas)
            panels.append(np.hstack(pair))
        # Mixed aspect ratios are padded, never stretch the overlays.
        width=max(x.shape[1] for x in panels)
        montage=np.vstack([cv2.copyMakeBorder(x,0,0,0,width-x.shape[1],cv2.BORDER_CONSTANT) for x in panels])
        dest=DOC/'images'/f'{material}.jpg';dest.parent.mkdir(parents=True,exist_ok=True);assert cv2.imwrite(str(dest),montage)
        lines += ['## '+material+' 시각 비교','', '위2개: 평균 코너오차 개선 최대 / 아래2개: 악화 최대. 노랑=각 모델 예측; GT overlay는 없음.', '',f'![{material}](images/{material}.jpg)','']
    save(DOC/'RESULTS_KO.md','\n'.join(lines)+'\n')
    C.freeze(DOC/'AUDIT.json',dict(complete=True,train19_eval300_disjoint=True,students=2,steps_each=320,no_evaluation_refiner=True))
    print('RESULTS',json.dumps(results,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','train','eval','score']);ap.add_argument('--arm',choices=['PLASTIC','WOOD']);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='train':scope();T.train(a.arm)
    elif a.stage=='eval':evaluate(a.arm)
    else:score()

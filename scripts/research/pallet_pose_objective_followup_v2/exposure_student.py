"""Cycle C changes only the scheduled mask-attempt probability .5 -> 1."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from . import common as C
from . import occlusion as O
from . import student as S

def frequent_plan(points, mask, box, shape, seed):
    rng=np.random.default_rng(seed); p=np.asarray(points); mask=np.asarray(mask,bool).copy(); mask[8]=False
    area=float(np.prod(np.maximum(0,np.asarray(box)[2:]-np.asarray(box)[:2])))
    ratio=float(rng.choice([.5,1.,2.])); fraction=float(rng.choice([.1,.2,.3]))
    w=max(1,round(np.sqrt(area*fraction*ratio))); h=max(1,round(np.sqrt(area*fraction/ratio)))
    previous_scheduled=bool(rng.random()<.5)  # Preserve identical downstream draws, even when gate is removed.
    result=dict(seed=int(seed),scheduled=True,previous_scheduled=previous_scheduled,area_fraction=fraction,aspect=ratio,
        size=[w,h],fill_seed=int(rng.integers(0,2**31-1)),applied=False,reason='not_scheduled',
        rectangle=None,covered=[],remaining=int(mask.sum()),supervised=int(mask.sum()),candidates_checked=0)
    if mask.sum()<3: result['reason']='cannot_cover1_leave2'; return result
    if w>shape[1] or h>shape[0]: result['reason']='shape_exceeds_input'; return result
    for _ in range(32):
        l=int(rng.integers(0,shape[1]-w+1)); t=int(rng.integers(0,shape[0]-h+1)); rect=[l,t,w,h]
        covered=O.cover(p,rect)&mask; result['candidates_checked']+=1
        if covered.sum()>=1 and mask.sum()-covered.sum()>=2 and O.overlap(rect,box)>0:
            result.update(applied=True,reason='random_valid',rectangle=rect,
                covered=np.flatnonzero(covered).tolist(),remaining=int(mask.sum()-covered.sum()),
                bbox_fraction=O.overlap(rect,box)/max(area,1e-12))
            return result
    result['reason']='no_valid_random_position_32'; return result

def preflight(material='PLASTIC'):
    dest=C.DOC/f'EXPOSURE_PREFLIGHT_{material}.json'
    if dest.exists(): assert C.read(dest)['passed']; return
    pp,p=S.protocol(material); ds=S.make_dataset(material,'REF'); base=copy.deepcopy(ds.transforms)
    canonical=O.reference_labels(p,C.ROOT); transform=O.SharedOcclusion(base,canonical)
    original_plan=O.random_plan; rows=[]
    for role in ('SOURCE','REAL'):
        indices=[i for i,f in enumerate(ds.im_files) if Path(f).name.startswith('syn__')==(role=='SOURCE')]
        for j in range(64):
            index=indices[int((j+.5)*len(indices)/64)]; examples=[]
            for plan in (original_plan,frequent_plan):
                O.random_plan=plan; S.seed(280901+index)
                sample=transform(copy.deepcopy(ds.get_image_and_label(index)))
                examples.append((S.sample_fingerprint(sample),sample['occlusion_info'],S.rng_digest()))
            a,b=examples
            assert all(a[0][k]==b[0][k] for k in ('boxes','support','xy')) and a[2]==b[2]
            if role=='SOURCE' or a[1]['applied']: assert a[0]==b[0]
            if a[1]['applied']:
                for k in ('rectangle','fill_seed','covered','remaining'): assert a[1][k]==b[1][k]
            rows.append(dict(role=role,index=index,A=a,C=b))
    O.random_plan=original_plan
    real=[r for r in rows if r['role']=='REAL']
    result=dict(passed=True,protocol=C.bind(pp),changed_variable='schedule probability0.5to1',
        old_applied=sum(r['A'][1]['applied'] for r in real),new_applied=sum(r['C'][1]['applied'] for r in real),real_probe=64,
        old_covered=sum(r['A'][1].get('actual_covered',0) for r in real),new_covered=sum(r['C'][1].get('actual_covered',0) for r in real),
        source_exact=64,previously_applied_pixels_identical=True,targets_boxes_RNG_exact=128,
        new_images=0,new_manual=0,fit=0,updates=0)
    assert result['new_applied']>result['old_applied']
    C.save(C.RAW/f'EXPOSURE_PREFLIGHT_{material}_PRIVATE.json',rows,True)
    result['private_detail']=C.bind(C.RAW/f'EXPOSURE_PREFLIGHT_{material}_PRIVATE.json')
    C.save(dest,result,True); print('EXPOSURE_PREFLIGHT',result,flush=True)

def lock():
    folder=C.DOC/'cycles/C_EXPOSURE'
    if (folder/'PROTOCOL.json').exists(): return
    preflight()
    a=C.read(C.DOC/'cycles/A_INPUT_OCCLUSION/PROTOCOL.json')
    C.save(folder/'SPEC.md','''# C — 가림 시도 빈도 한 변수

A에서 real2,560회 중542회(21.17%)만 실제 가림, REF감독21,823점 중824점(3.78%)만 덮였다. 작은 TRAIN probe에서 가린 점의 location gradient는 살아 있다. 이것만으로 노출 부족이 원인임을 확정할 수는 없지만, 더 많은 유효 가림 입력의 추가 가치를 단일 대조로 확인할 근거다.

변경은 A의 schedule probability0.5→1이다. 같은 seed에서 shape/aspect/fill/최대32위치proposal 및 covered≥1/remaining≥2/기존RGB기하증강을 그대로 둔다. gate용 random draw도 소비해 이전 적용 입력은 bit-exact, 새로 시도하는 occurrence만 달라진다. 무효 placement는 원본 입력 유지. 원본 supervision/teacher좌표/source비율/update/lr는 안 바꾼다. 오류정답/오답1:1, 실제 자연가림 등급 균형이라고 부르지 않는다. 1은 시도 빈도의 상한 한 값이며 DEV threshold 최적값이 아니다.

주 control은 이미 완료된 A RAW/REF이며 두 recipe의 baseline OLD_RAW/OLD_REF도 보존한다. C RAW/REF는 동일RGB/계획이며 타깃값만 다르다. 원래 affine의 RAW/REF 경계support차이는 그대로 공개한다. A+B 결합은 B가 기존REF 대비 두 주축 모두 개선하지 않아 실행하지 않는다. F는 기존자료 feasibility를 확인하되 teacher+학생4fit보다 현 학생 경로의 작은 exposure 대조를 먼저 선택했다. 이는 F 성능 실패가 아니다.

Plastic2fit×320update/seed42, 실제세번째이자마지막 주cycle. 주99 T/R, full128/66, Clean/Moderate/Severe·recording·P90·coverage·source를 같은 기준으로 본다. C−A를 따로 계산해 가림 자체 효과와 시도빈도 효과를 구분한다. 결과가 나빠도 빈도·크기·손실 재조정은 없다. 이후에는 사전 선택 규칙에 따른 한 recipe의 재현/control·Wood적용성과 마무리만 한다.
''',True)
    spec=dict(a,cycle='C_EXPOSURE',created_utc=C.now(),source_cycle='A_INPUT_OCCLUSION',
        masking=dict(a['masking'],schedule=1.),exposure=True,
        comparator=C.bind(C.DOC/'cycles/A_INPUT_OCCLUSION/RESULTS_PLASTIC_S42.json'),
        competing_B=C.bind(C.DOC/'cycles/B_COORDINATE_SUPPLEMENT/RESULTS_PLASTIC_S42.json'),
        exposure_preflight=C.bind(C.DOC/'EXPOSURE_PREFLIGHT_PLASTIC.json'),spec=C.bind(folder/'SPEC.md'),
        implementation=a['implementation']+[C.bind(Path(__file__))])
    C.save(folder/'PROTOCOL.json',spec,True); print('C_PROTOCOL_LOCKED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','lock','train']);p.add_argument('--cycle',default='C_EXPOSURE')
    p.add_argument('--material',default='PLASTIC');p.add_argument('--target',choices=['RAW','REF'],default='RAW');p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.phase=='preflight':preflight(args.material)
    elif args.phase=='lock':lock()
    else:
        assert C.read(C.DOC/f'EXPOSURE_PREFLIGHT_{args.material}.json')['passed']
        O.random_plan=frequent_plan
        S.train(args.cycle,args.material,args.target,args.seed)

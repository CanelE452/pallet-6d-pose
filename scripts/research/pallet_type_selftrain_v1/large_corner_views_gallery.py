"""Display automatic consensus recoveries and damage, never oracle output."""
import json
import cv2
import numpy as np
from . import large_corner_views as L
from .recovery_damage import draw

C=L.C
OUT=C.OUT/'large_corner_recovery_v1/views'


def main():
    results=C.read(L.RAW/'views/SUMMARY.json')
    records=C.read(L.DOC/'views/EVAL_PROTOCOL.json')['records'];metadata={r['id']:r for r in records}
    base={r['id']:r for r in C.read(L.RAW/'views/SCREEN_R0.json')['metrics']}
    after={r['id']:r for r in C.read(L.RAW/'views/SCREEN_CONSENSUS.json')['metrics']}
    bpred={r['id']:r for r in C.read(L.RAW/'views/EVAL_PREDICTIONS_R0.json')['records']}
    npred={r['id']:r for r in C.read(L.RAW/'views/EVAL_PREDICTIONS_CONSENSUS.json')['records']}
    decisions={r['id']:r['selected'] for r in C.read(L.RAW/'views/INFERENCE.json')}
    pe,pop=L.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in metadata}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    (OUT/'images').mkdir(parents=True,exist_ok=True);cards=[]
    for i,key in enumerate(sorted(metadata)):
        b,n=base[key],after[key];t=targets[key];valid=np.array(t.keypoint_supervision_mask,bool);gt=np.array(t.keypoints_xy)
        recovered=[j for j,(x,y) in enumerate(zip(b['canonical_errors'],n['canonical_errors'])) if x is not None and x>20 and y<=10] if b['matched'] else []
        damaged=[j for j,(x,y) in enumerate(zip(b['canonical_errors'],n['canonical_errors'])) if x is not None and x<5 and y>10] if b['matched'] else []
        C.verify(metadata[key]['image']);C.verify(metadata[key]['annotation']);im=cv2.imread(str(C.ROOT/metadata[key]['image']['path']))
        p=L.top(bpred[key]['prediction']);q=L.top(npred[key]['prediction'])
        left=draw(im,p['keypoints_xy'],gt,valid,perms[b['branch']],'R0 / original',b['frame_mean_px'])
        right=draw(im,q['keypoints_xy'],gt,valid,perms[n['branch']],'Automatic consensus: '+decisions[key],n['frame_mean_px'])
        filename=f'images/{i:03d}.jpg';dest=OUT/filename
        if not dest.exists():assert cv2.imwrite(str(dest),np.concatenate([left,right],axis=1),[cv2.IMWRITE_JPEG_QUALITY,90])
        cards.append(dict(id=key,image=filename,selected=decisions[key],matched=b['matched'],recovered=recovered,damaged=damaged,
            before=b['frame_mean_px'],after=n['frame_mean_px'],errors_before=b['canonical_errors'],errors_after=n['canonical_errors']))
    d=results['arms']['CONSENSUS']['hard_recovery_matched']
    assert sum(len(r['recovered']) for r in cards)==d['recovered'] and sum(len(r['damaged']) for r in cards)==d['damaged']
    payload=json.dumps(cards,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>큰 코너 오류 복구 — 자동 선택 결과</title>
<style>body{background:#101820;color:#edf2f7;font:17px sans-serif;max-width:1320px;margin:24px auto;padding:16px}select,input{font:inherit;padding:8px;margin:6px}article{border:1px solid #52616c;margin:24px 0;padding:12px}img{width:100%;height:auto}pre{white-space:pre-wrap}.warning{background:#502d23;padding:16px}</style>
<h1>큰 코너 오류: R0 → 자동 후보 선택</h1>
<p>왼쪽 R0, 오른쪽 크기·크롭·반전 후보의 자동 합의 선택입니다. 오른쪽은 정답을 보고 고른 oracle가 아닙니다. 선택은 모든 평가 정답을 읽기 전에 고정했습니다.</p>
<p class="warning">현재 채택 불가: 큰 오차281개 중23개를10px 이내로 복구했지만, 원래5px 이내511개 중24개를10px 밖으로 망가뜨렸습니다. 성공 예시와 실패 예시를 모두 표시합니다. 기존 최종 모델과 학습 수도레이블은 바꾸지 않았습니다.</p>
<p>노랑=예측, 자홍 X=기존 어노테이션 참조점. P=모델 번호, G=공식 C2 전체 대칭으로 대응한 참조점 번호. GT는 채점·그림에만 쓰이며 추론 입력이 아닙니다. 평균 오차는 대칭8점 기준이고 검출 매칭 실패에는 벌점이 포함됩니다. 반복 사용한 DEV194이며 독립 시험은 아닙니다.</p>
<select id="filter"><option value="all">전체194장</option><option value="recovered">큰 오류 복구가 있는 이미지</option><option value="damaged">좋던 점을 훼손한 이미지</option></select>
<select id="sort"><option value="recover">복구 코너 많은 순</option><option value="harm">훼손 코너 많은 순</option><option value="residual">남은 오차 큰 순</option></select><input id="query" placeholder="ID 검색"><p id="count"></p><main id="cards"></main>
<script>const data=__DATA__;const el=id=>document.getElementById(id);function render(){let f=el('filter').value,s=el('sort').value;let rows=data.filter(r=>r.id.includes(el('query').value)&&(f==='all'||(f==='recovered'?r.recovered.length>0:r.damaged.length>0)));rows.sort((a,b)=>s==='recover'?b.recovered.length-a.recovered.length||a.damaged.length-b.damaged.length:s==='harm'?b.damaged.length-a.damaged.length:b.after-a.after);el('count').textContent=rows.length+' / 194장';el('cards').replaceChildren();for(const r of rows){let a=document.createElement('article'),h=document.createElement('h3'),im=document.createElement('img'),dt=document.createElement('details'),su=document.createElement('summary'),pr=document.createElement('pre');h.textContent=r.id+' | 복구 '+r.recovered.length+'점 / 훼손 '+r.damaged.length+'점 | '+r.before.toFixed(2)+' → '+r.after.toFixed(2)+'px | '+r.selected;im.src=r.image;im.loading='lazy';im.alt=r.id;su.textContent='같은 GT 코너별 오차 확인';pr.textContent=JSON.stringify(r,null,2);dt.append(su,pr);a.append(h,im,dt);el('cards').append(a)}}for(const id of ['filter','sort','query'])el(id).addEventListener('input',render);render();</script></html>'''.replace('__DATA__',payload)
    C.write_text(OUT/'index.html',page);C.freeze(OUT/'CASES.json',cards)
    C.freeze(OUT/'VERIFICATION.json',dict(images=194,oracle_output_displayed=False,recovered=d['recovered'],damaged=d['damaged'],
        recovered_images=sum(bool(r['recovered']) for r in cards),damaged_images=sum(bool(r['damaged']) for r in cards),
        sources=[C.bound(__file__),C.bound(L.RAW/'views/SUMMARY.json'),C.bound(OUT/'CASES.json')],page=C.bound(OUT/'index.html')))
    print('GALLERY',OUT/'index.html');print('RECOVERY EXAMPLES',[(r['id'],r['image'],r['recovered'],r['damaged']) for r in sorted(cards,key=lambda r:(-len(r['recovered']),len(r['damaged'])))[:4]])


if __name__=='__main__':main()

"""All ordinary-plastic frames, with real successes AND damage visible."""
import json
import cv2
import numpy as np
from . import recovery_pseudo_denoise as D
from .recovery_damage import draw

C=D.C
OUT=C.OUT/'selftrain_recovery_v1/pseudo_denoise'


def main():
    result=C.read(D.DOC/'RESULTS.json')
    lock=C.read(D.DOC/'OUTPUTS_LOCK.json')
    for b in lock['artifacts']:C.verify(b)
    records=C.read(D.DOC/'EVAL_PROTOCOL.json')['records']
    metrics={a:{r['id']:r for r in C.read(D.RAW/f'SCREEN_{a}.json')['metrics']} for a in D.ARMS}
    metrics['R0']={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    preds={a:{r['id']:r['prediction'] for r in C.read(D.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']} for a in D.ARMS}
    preds['R0']={r['id']:r['prediction'] for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    pe,pop=D.R.E.O.population_metadata()
    targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in metrics['R0']}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    (OUT/'images').mkdir(parents=True,exist_ok=True);cards=[]
    for i,record in enumerate(records):
        key=record['id'];b=metrics['R0'][key];n=metrics['DENOISE'][key];t=targets[key]
        C.verify(record['image']);C.verify(record['annotation'])
        im=cv2.imread(str(C.ROOT/record['image']['path']));assert im is not None
        panels=[]
        for arm in ['R0',*D.ARMS]:
            row=metrics[arm][key];p=D.top(preds[arm][key]);assert p is not None
            panels.append(draw(im,p['keypoints_xy'],np.asarray(t.keypoints_xy),np.asarray(t.keypoint_supervision_mask,bool),perms[row['branch']],arm,row['frame_mean_px']))
        tile=np.concatenate([np.concatenate(panels[:2],axis=1),np.concatenate(panels[2:],axis=1)],axis=0)
        name=f'images/{i:03d}.jpg';dest=OUT/name
        if not dest.exists():assert cv2.imwrite(str(dest),tile,[cv2.IMWRITE_JPEG_QUALITY,90])
        pairs=list(zip(b['canonical_errors'],n['canonical_errors']))
        recovered=[j for j,(x,y) in enumerate(pairs) if x is not None and x>20 and y<=10] if b['matched'] else []
        damaged=[j for j,(x,y) in enumerate(pairs) if x is not None and x<5 and y>10] if b['matched'] else []
        cards.append(dict(id=key,image=name,matched=b['matched'],recovered=recovered,damaged=damaged,
            errors={a:metrics[a][key]['canonical_errors'] for a in ['R0',*D.ARMS]},
            mean={a:metrics[a][key]['frame_mean_px'] for a in ['R0',*D.ARMS]}))
    counts=result['results']['full194']['DENOISE']['recovery']
    assert sum(len(r['recovered']) for r in cards)==counts['recovered']
    assert sum(len(r['damaged']) for r in cards)==counts['damaged']
    payload=json.dumps(cards,ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>수도레이블 큰 입력 오차 학습</title>
<style>body{background:#101820;color:#eee;font:17px sans-serif;max-width:1300px;margin:24px auto;padding:12px}article{border:1px solid #789;padding:12px;margin:24px 0}img{width:100%}select,input{font:inherit;padding:8px}pre{white-space:pre-wrap}</style>
<h1>기존 수도레이블로 큰 입력 오차 복구 학습</h1>
<p>위: R0 / 기존 Replay(BASE). 아래: 입력 그대로 학습(CLEAN) / 입력 코너를 크게 교란하며 학습(DENOISE).</p>
<p>노랑=실제 모델 예측, 자홍 X=기존 평가 참조점. 정답은 채점·그림에만 쓰이며 모델 입력·학습·후보 선택에 쓰지 않았습니다. 추가 어노테이션·태그 없음. 성공과 실패를 모두 보여줍니다. 아래 복구·손상은 R0 대비 DENOISE의 동일 GT 코너 기준입니다.</p>
<p>반복 사용한 DEV194이며 기존 교사 학습과 겹친3장이 있습니다. 독립 확인이나 최종 모델 교체가 아닙니다.</p>
<select id="filter"><option value="all">전체194장</option><option value="recover">큰 오차 복구가 있는 사진</option><option value="damage">좋던 점을 훼손한 사진</option></select>
<select id="sort"><option value="recover">복구 많은 순</option><option value="damage">손상 많은 순</option><option value="tail">남은 오차 큰 순</option></select><input id="query" placeholder="ID 검색"><p id="count"></p><main></main>
<script>const data=__DATA__;const el=id=>document.getElementById(id);function render(){let rows=data.filter(r=>r.id.includes(el('query').value)&&(el('filter').value==='all'||(el('filter').value==='recover'?r.recovered.length:r.damaged.length)));let s=el('sort').value;rows.sort((a,b)=>s==='recover'?b.recovered.length-a.recovered.length||a.damaged.length-b.damaged.length:s==='damage'?b.damaged.length-a.damaged.length:b.mean.DENOISE-a.mean.DENOISE);el('count').textContent=rows.length+' /194장';const root=document.querySelector('main');root.replaceChildren();for(const r of rows){let a=document.createElement('article'),h=document.createElement('h3'),im=document.createElement('img'),d=document.createElement('details'),su=document.createElement('summary'),pre=document.createElement('pre');h.textContent=r.id+' | 복구 '+r.recovered.length+' / 손상 '+r.damaged.length+' | '+r.mean.R0.toFixed(2)+' → '+r.mean.DENOISE.toFixed(2)+'px';im.src=r.image;im.loading='lazy';im.alt=r.id;su.textContent='모델별 동일 코너 오차';pre.textContent=JSON.stringify(r,null,2);d.append(su,pre);a.append(h,im,d);root.append(a)}}for(const id of ['filter','sort','query'])el(id).addEventListener('input',render);render();</script></html>'''.replace('__DATA__',payload)
    C.write_text(OUT/'index.html',html);C.freeze(OUT/'CASES.json',cards)
    C.freeze(OUT/'VERIFICATION.json',dict(images=len(cards),counts=counts,all_frames_included=True,
        source=C.bound(D.DOC/'RESULTS.json'),page=C.bound(OUT/'index.html'),cards=C.bound(OUT/'CASES.json'),code=C.bound(__file__)))
    print('PSEUDO_DENOISE_GALLERY',OUT/'index.html',flush=True)


if __name__=='__main__':main()

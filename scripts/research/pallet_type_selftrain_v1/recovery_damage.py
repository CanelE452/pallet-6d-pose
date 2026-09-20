"""Posthoc visual/error audit. GT is used only for scoring and display, never training."""
import html
import json
from pathlib import Path
import cv2
import numpy as np
from . import recovery_common as R
from . import recovery_pose as P
from .pseudo import top
from .stability_review import EDGES
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

C=R.C
OUT=C.OUT/'selftrain_recovery_v1/damage'
RAW=R.RAW/'damage'


def stats(values):
    a=np.asarray(values,float)
    return dict(n=len(a),quantiles=dict(zip(['p50','p90','p99','max'],np.quantile(a,[.5,.9,.99,1]).tolist())),
                over20=int((a>20).sum()))


def draw(im,points,gt,valid,perm,title,error):
    canvas=im.copy();q=np.asarray(points,float)
    for a,b in EDGES:
        if np.isfinite(q[[a,b]]).all():cv2.line(canvas,tuple(np.rint(q[a]).astype(int)),tuple(np.rint(q[b]).astype(int)),(0,220,255),1,cv2.LINE_AA)
    for native,gid in enumerate(perm[:8]):
        if not valid[gid]:continue
        g=tuple(np.rint(gt[gid]).astype(int));p=tuple(np.rint(q[native]).astype(int))
        cv2.drawMarker(canvas,g,(230,50,230),cv2.MARKER_TILTED_CROSS,11,2)
        cv2.circle(canvas,p,4,(0,220,255),2,cv2.LINE_AA)
        cv2.line(canvas,p,g,(200,200,200),1,cv2.LINE_AA)
        label=f'P{native}/G{gid}'
        cv2.putText(canvas,label,(p[0]+5,p[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(canvas,label,(p[0]+5,p[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,220,255),1,cv2.LINE_AA)
    width=640;height=round(canvas.shape[0]*width/canvas.shape[1])
    canvas=cv2.resize(canvas,(width,height));banner=np.zeros((55,width,3),np.uint8)
    for line,y in [(title,21),(f'8-corner mean: {error:.2f}px | yellow=prediction, magenta=GT',43)]:
        cv2.putText(banner,line,(8,y),cv2.FONT_HERSHEY_SIMPLEX,.47,(245,245,245),1,cv2.LINE_AA)
    return np.concatenate([banner,canvas])


def main():
    metadata={r['id']:r for r in C.read(R.BASE_DOC/'EVAL_PROTOCOL.json')['records'] if r['kind']=='PLASTIC'}
    baseline={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    pred0={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    paths=[('pose_only','REF_LR5'),('pose_repeat','REF_ORDER43'),('pose_repeat','REF_ORDER44')]
    repeats={arm:C.read(R.RAW/phase/f'RESULTS_{arm}.json')['metrics'] for phase,arm in paths}
    pred1={r['id']:r for r in C.read(R.RAW/'pose_only/EVAL_PREDICTIONS_REF_LR5.json')['records']}
    summary={}
    for arm,rows in repeats.items():
        before=[baseline[r['id']] for r in rows]
        a=np.array([e for r in before for e in r['canonical_errors'] if e is not None]);b=np.array([e for r in rows for e in r['canonical_errors'] if e is not None])
        matched=[(x,y) for x,y in zip(before,rows) if x['matched'] and y['matched']]
        summary[arm]=dict(damage=EM.damage(before,rows),
            tail_counts_including_miss_penalties={str(t):dict(R0=int((a>t).sum()),student=int((b>t).sum())) for t in [20,40,80,100]},
            matched_only_tail_counts={str(t):dict(R0=sum(e>t for x,y in matched for e in x['errors']),student=sum(e>t for x,y in matched for e in y['errors'])) for t in [20,40,80,100]},
            unmatched_frames=sum(not r['matched'] for r in rows))
    pool=[];sampled=[]
    train_list=C.ROOT/C.read(R.DOC/'pose_only/PROTOCOL.json')['datasets']['REF']['train_list']['path']
    selected={Path(p).stem for p in train_list.read_text().splitlines()[512:]}
    for row in C.read(R.BASE_RAW/'PSEUDO_ACCEPTED.json'):
        if row['kind']!='PLASTIC':continue
        labels,_=P.paired_labels(row);valid=np.array(labels['REF'].split(),float)[5:].reshape(9,3)[:8,2]==2
        a=np.array(top(row['raw'])['keypoints_xy']);b=np.array(top(row['refined'])['keypoints_xy'])
        shift=np.linalg.norm(a[:8]-b[:8],axis=1)[valid];pool.extend(shift)
        if row['id'] in selected:sampled.extend(shift)
    assert sampled and len(selected)==217
    pe,pop=R.E.O.population_metadata();targets={item.frame_id:pe.E._legacy_forbidden_target(item) for item,meta in pop if item.frame_id in metadata}
    perms={r['object_type']:r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}[C.TYPES['PLASTIC']]
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'images').mkdir(exist_ok=True)
    cases=[];movements=[];reference={r['id']:r for r in repeats['REF_LR5']}
    for i,key in enumerate(sorted(metadata)):
        old,new=baseline[key],reference[key];p,q=top(pred0[key]['prediction']),top(pred1[key]['prediction'])
        shift=np.linalg.norm(np.array(p['keypoints_xy'])[:8]-np.array(q['keypoints_xy'])[:8],axis=1)
        movements.extend(shift);t=targets[key];gt=np.array(t.keypoints_xy);valid=np.array(t.keypoint_supervision_mask,bool)
        C.verify(metadata[key]['image']);C.verify(metadata[key]['annotation'])
        im=cv2.imread(str(C.ROOT/metadata[key]['image']['path']));assert im is not None
        panels=[draw(im,p['keypoints_xy'],gt,valid,perms[old['branch']],'R0 / before self-training',old['frame_mean_px']),
                draw(im,q['keypoints_xy'],gt,valid,perms[new['branch']],'REF_LR5 / student only, NO refiner at inference',new['frame_mean_px'])]
        filename=f'images/{i:03d}.jpg';dest=OUT/filename
        if not dest.exists():assert cv2.imwrite(str(dest),np.concatenate(panels,axis=1),[cv2.IMWRITE_JPEG_QUALITY,90])
        deltas=[n-b if n is not None else None for b,n in zip(old['canonical_errors'],new['canonical_errors'])]
        cases.append(dict(id=key,image=filename,session=metadata[key]['session'],before=old['frame_mean_px'],after=new['frame_mean_px'],
            delta=new['frame_mean_px']-old['frame_mean_px'],matched=old['matched'] and new['matched'],
            native_displacements_px=shift.tolist(),canonical_error_delta_px=deltas,branches=[old['branch'],new['branch']],
            errors_before=old['canonical_errors'],errors_after=new['canonical_errors']))
    report=dict(repeats=summary,pool_trusted_correction=stats(pool),sampled_unique_trusted_correction=stats(sampled),
        student_native_displacement=stats(movements),real_unique=217,
        original_configuration_REF_LR5_for_display=True,no_best_repeat_selection=True,
        GT_used_only_for_posthoc_scoring_display=True,no_automatic_training_change=True,
        interpretation='Mostly small corrections. This audit does not establish a single causal explanation; small targets, limited updates, accepted-pool bias and model restrictions may all contribute.',
        sources=[C.bound(__file__),C.bound(R.DOC/'RECOVERY_AUDIT.json')]+[C.bound(R.RAW/phase/f'RESULTS_{arm}.json') for phase,arm in paths])
    C.freeze(RAW/'AUDIT.json',report);C.freeze(RAW/'CASES.json',cases)
    payload=json.dumps(cases,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>일반 플라스틱 self-training 전후</title>
<style>body{background:#101820;color:#edf2f7;font:17px sans-serif;max-width:1320px;margin:24px auto;padding:16px}button,select,input{font:inherit;padding:8px;margin:6px}article{border:1px solid #4c5963;margin:24px 0;padding:12px}img{width:100%;height:auto}small{color:#b4c0ce}summary{cursor:pointer}pre{white-space:pre-wrap}</style>
<h1>일반 플라스틱: R0 → 보정 수도레이블 self-training</h1>
<p>왼쪽 R0, 오른쪽 학생 단독 추론. 추론할 때 보정기는 사용하지 않습니다. 노랑=모델 예측, 자홍 X=감독 GT. P는 모델의 native 번호, G는 대칭 정렬한 GT 번호입니다. GT는 비교·정렬·점수에만 사용했습니다.</p>
<p>전체 194장 모두 표시합니다. 오차는 첫 8코너 대칭 기준이며 논문 fixed-index 9점 median과 다릅니다. 검출 매칭 실패는 점수에 이미지 대각선 벌점이 들어갑니다. 기존 보정기 학습 이미지 3장이 포함된 반복 사용 DEV이며 독립 시험이 아닙니다.</p>
<p>같은 모델 REF_LR5의 고정 마지막 checkpoint를 표시하며 세 반복 중 가장 좋은 모델을 고른 것이 아닙니다.</p>
<select id="sort"><option value="harm">악화 큰 순</option><option value="gain">개선 큰 순</option><option value="residual">학생 오차 큰 순</option><option value="id">전체 ID 순</option></select>
<input id="query" placeholder="세션 또는 ID 검색"><label><input id="matched" type="checkbox">매칭 성공만</label><p id="count"></p><main id="cards"></main>
<script>const data=__DATA__;const el=id=>document.getElementById(id);function render(){let rows=data.filter(r=>r.id.includes(el('query').value)&&(!el('matched').checked||r.matched));let s=el('sort').value;rows.sort((a,b)=>s==='harm'?b.delta-a.delta:s==='gain'?a.delta-b.delta:s==='residual'?b.after-a.after:a.id.localeCompare(b.id));el('count').textContent=rows.length+' / 194장';el('cards').replaceChildren();for(const r of rows){let a=document.createElement('article');let h=document.createElement('h3');h.textContent=r.id+' | '+r.before.toFixed(2)+' → '+r.after.toFixed(2)+' px | Δ '+r.delta.toFixed(2)+' px'+(r.matched?'':' [검출 매칭 실패: 벌점]');let im=document.createElement('img');im.src=r.image;im.loading='lazy';im.alt=r.id;let d=document.createElement('details'),sm=document.createElement('summary'),pr=document.createElement('pre');sm.textContent='코너별 원시 수치';pr.textContent=JSON.stringify(r,null,2);d.append(sm,pr);a.append(h,im,d);el('cards').append(a)}}for(const id of ['sort','query','matched'])el(id).addEventListener('input',render);render();</script></html>'''.replace('__DATA__',payload)
    C.write_text(OUT/'index.html',page)
    lines=['# 일반 플라스틱 self-training: 큰 오차 감사','',
        '학습 완료 후 GT를 이용한 진단이다. 평가 이미지/코너를 학습 대상으로 바꾸지 않았고 새 필터나 모델 교체도 하지 않았다.','',
        '## 학습 신호와 학생 이동량','', '```json',json.dumps({k:report[k] for k in ['pool_trusted_correction','sampled_unique_trusted_correction','student_native_displacement']},indent=2),'```','',
        '## 세 반복의 개선·악화 및 큰 오차','', '```json',json.dumps(summary,indent=2),'```','',
        '8개의 매칭 실패 이미지 벌점과 실제 매칭된 코너의 큰 오차를 분리해 기록했다. 대칭 분기 변화는 canonical GT identity 기준으로 비교했다.','',
        '작은 보정 신호, 제한된 학습량, 필터를 통과한 쉬운 이미지의 편향, 동결한 모델 부분 등이 원인 후보다. 이 감사만으로 한 가지 원인으로 확정할 수 없다.','',
        f'전체 194장 비교 HTML: {C.bound(OUT/"index.html")["path"]}','']
    C.write_text(RAW/'REPORT_KO.md','\n'.join(lines))
    print(json.dumps(report,ensure_ascii=False,indent=2));print('GALLERY',OUT/'index.html')


if __name__=='__main__':main()

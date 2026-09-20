"""Audit raw argmax proposals ONLY; failed calibration keeps deployment at R0.

This adds no real-threshold sweep, no model selection, and no pseudo targets.
All raw decisions were already fixed in DECISIONS.json before GT scoring.
"""
import copy
import json
import cv2
import numpy as np
from . import identity_rank as I
from . import identity_rank_hard as H
from .recovery_damage import draw
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

C=I.C
OUT=C.OUT/'selftrain_recovery_v1/identity_rank_hard'


def main():
    for b in C.read(H.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    decisions=C.read(H.RAW/'DECISIONS.json')
    before={r['id']:r for r in C.read(I.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    predictions={a:[] for a in I.ARMS}
    for r in decisions:
        for a in I.ARMS:
            p=r['choices'][a]['probabilities'];choice=int(p[1]>p[0])
            old=before[r['id']];out=copy.deepcopy(old['prediction'])
            if choice:I.top(out)['keypoints_xy'][:8]=np.asarray(I.top(out)['keypoints_xy'])[I.F.QUARTER][:8].tolist()
            I.assert_preserved(old['prediction'],out)
            predictions[a].append(dict(id=r['id'],prediction=out,choice=choice,raw_hw=old['raw_hw']))
    C.freeze(H.RAW/'UNDEPLOYED_ARGMAX_PREDICTIONS.json',dict(status='DIAGNOSTIC_ONLY_FAILED_SOURCE_CALIBRATION',
        method='Single raw model argmax, never a real-GT-selected threshold; NOT accepted output or pseudo label',
        records=predictions,decisions=C.bound(H.RAW/'DECISIONS.json')))
    # Actual reference contents are accessed only after proposals are frozen.
    metadata={r['id']:r for r in C.read(H.DOC/'EVAL_PROTOCOL.json')['records']}
    pe,pop=I.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in metadata}
    groups=C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']
    perms=next(r['permutations'] for r in groups if r['object_type']==C.TYPES['PLASTIC'])
    base_lookup={r['id']:r for r in C.read(I.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    rows={a:[] for a in I.ARMS}
    for a,rr in predictions.items():
        for r in rr:
            t=targets[r['id']];b=base_lookup[r['id']]
            s=EM.measure(I.top(r['prediction'])['keypoints_xy'],t.keypoints_xy,t.keypoint_supervision_mask,
                perms,r['raw_hw'],b['matched'],b['detected'])
            rows[a].append(dict(id=r['id'],**s))
    base=[base_lookup[r['id']] for r in rows['SYN']];results={}
    for a,rr in rows.items():
        recovery=I.recovery_damage(base,rr,True)
        over100=sum(v is not None and v>100 and w<=10 for b,n in zip(base,rr) if b['matched'] for v,w in zip(b['canonical_errors'],n['canonical_errors']))
        results[a]=dict(summary=EM.summary(rr),recovery=recovery,recovered_over100=over100,
            changed_frames=sum(r['choice'] for r in predictions[a]))
    example='eval_pallet07:1778652166837872128'
    examples={a:next(r for r in rr if r['id']==example) for a,rr in rows.items()}
    fit={a:C.read(H.DOC/f'FIT_{a}.json') for a in I.ARMS}
    result=dict(status='UNDEPLOYED_ARGMAX_DIAGNOSTIC_ONLY',results=results,example=examples,
        actual_accepted_pipeline=C.bound(H.DOC/'RESULTS.json'),actual_accepted_changes=0,
        warning='Both classifiers failed source safety calibration. Raw argmax is not promoted, not used as pseudo labels, not counted as solved goal.',
        sources=[C.bound(__file__),C.bound(H.RAW/'UNDEPLOYED_ARGMAX_PREDICTIONS.json')])
    C.freeze(H.DOC/'UNDEPLOYED_ARGMAX_DIAGNOSTIC.json',result)
    # Fixed user example plus all unfiltered cases, not only successful figures.
    (OUT/'images').mkdir(parents=True,exist_ok=True);cards=[]
    lookup={a:{r['id']:r for r in rr} for a,rr in predictions.items()}
    metric={a:{r['id']:r for r in rr} for a,rr in rows.items()}
    for i,(key,record) in enumerate(metadata.items()):
        C.verify(record['image']);C.verify(record['annotation']);im=cv2.imread(str(C.ROOT/record['image']['path']));assert im is not None
        t=targets[key];b=base_lookup[key];panels=[draw(im,I.top(before[key]['prediction'])['keypoints_xy'],
            t.keypoints_xy,t.keypoint_supervision_mask,perms[b['branch']],'R0 / actual accepted output',b['frame_mean_px'])]
        for a in I.ARMS:
            r=metric[a][key];panels.append(draw(im,I.top(lookup[a][key]['prediction'])['keypoints_xy'],
                t.keypoints_xy,t.keypoint_supervision_mask,perms[r['branch']],'NOT DEPLOYED: '+a+' raw argmax',r['frame_mean_px']))
        name=f'images/{i:03d}.jpg';dest=OUT/name
        if not dest.exists():assert cv2.imwrite(str(dest),np.concatenate(panels,axis=1),[cv2.IMWRITE_JPEG_QUALITY,90])
        rr=metric['MIX'][key]
        recovered=[j for j,(x,y) in enumerate(zip(b['canonical_errors'],rr['canonical_errors'])) if x is not None and x>20 and y<=10] if b['matched'] else []
        damaged=[j for j,(x,y) in enumerate(zip(b['canonical_errors'],rr['canonical_errors'])) if x is not None and x<5 and y>10] if b['matched'] else []
        cards.append(dict(id=key,image=name,recovered=recovered,damaged=damaged,
            means=dict(R0=b['frame_mean_px'],**{a:metric[a][key]['frame_mean_px'] for a in I.ARMS}),
            changed={a:lookup[a][key]['choice'] for a in I.ARMS}))
    assert sum(len(r['recovered']) for r in cards)==results['MIX']['recovery']['recovered']
    assert sum(len(r['damaged']) for r in cards)==results['MIX']['recovery']['damaged']
    payload=json.dumps(cards,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>미적용 코너 대응 후보 진단</title>
<style>body{background:#101820;color:#eee;font:17px sans-serif;max-width:1600px;margin:24px auto;padding:12px}.warning{padding:20px;background:#692d27}article{border:1px solid #789;padding:12px;margin:24px 0}img{width:100%}select,input{font:inherit;padding:8px}</style>
<h1>미적용 후보 진단 — 새 모델의 최종 결과가 아닙니다</h1><p class="warning">두 학습기는 합성 보류 데이터의 정상 코너 보존 기준에 실패했습니다. 실제 출력은 R0 그대로입니다. 가운데·오른쪽은 선택 기준을 통과하지 않은 학습기 원시 argmax 후보이며, 모델 교체·수도레이블·학습에 사용하지 않았습니다.</p>
<p>왼쪽: 실제 출력 R0. 가운데: 합성만 학습한 미적용 후보. 오른쪽: 합성+기존 수도레이블 학습한 미적용 후보. 모두 GT를 읽기 전에 결정된 후보입니다. 정답은 채점·표시에만 쓰며, 공식 C2 평가를 유지합니다. 노랑=후보 예측, 자홍 X=기존 평가 참조점.</p>
<select id="filter"><option value="example">사용자가 보여준 의자·콘 사진</option><option value="all">전체194장</option><option value="recover">복구 사례</option><option value="damage">손상 사례</option></select><p id="count"></p><main></main>
<script>const data=__DATA__;function render(){const f=document.getElementById('filter').value;let rows=data.filter(r=>f==='all'||(f==='example'?r.id==='eval_pallet07:1778652166837872128':f==='recover'?r.recovered.length:r.damaged.length));document.getElementById('count').textContent=rows.length+' /194장';const root=document.querySelector('main');root.replaceChildren();for(const r of rows){const a=document.createElement('article'),h=document.createElement('h3'),im=document.createElement('img');h.textContent=r.id+' | R0 '+r.means.R0.toFixed(2)+'px / 미적용 SYN '+r.means.SYN.toFixed(2)+'px / 미적용 MIX '+r.means.MIX.toFixed(2)+'px | MIX 복구 '+r.recovered.length+' 손상 '+r.damaged.length;im.src=r.image;im.loading='lazy';im.alt=r.id;a.append(h,im);root.append(a)}}document.getElementById('filter').addEventListener('input',render);render();</script></html>'''.replace('__DATA__',payload)
    C.write_text(OUT/'index.html',page);C.freeze(OUT/'CASES.json',cards)
    lines=['# 미적용 원시 후보 진단','',result['warning'],'',
        '| 학습기 | 변경 후보 이미지 | >20→≤10 복구 | >100→≤10 복구 | 정상 코너 손상 | PCK20 |','|---|---:|---:|---:|---:|---:|']
    for a,r in results.items():
        d=r['recovery'];lines.append(f'| {a} | {r["changed_frames"]} | {d["recovered"]}/{d["hard"]} | {r["recovered_over100"]} | {d["damaged"]}/{d["good"]} | {100*r["summary"]["PCK"]["20"]:.2f}% |')
    lines+=['','의자·콘 사진: '+str({a:r['frame_mean_px'] for a,r in examples.items()}),
        '현재 실제 채택 출력은 R0 그대로이며, 이 후보 진단으로 안전 기준을 완화하지 않았다. 새 정답/태그0. 반복 DEV의 진단이지 독립 검증이 아니다.']
    C.write_text(H.RAW/'UNDEPLOYED_DIAGNOSTIC_KO.md','\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':main()

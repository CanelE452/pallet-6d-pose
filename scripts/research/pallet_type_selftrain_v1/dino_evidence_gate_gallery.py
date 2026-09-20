"""All194: R0, unconditional image candidate, learned accepted correction."""
import html
from collections import defaultdict
import cv2
import numpy as np
from . import dino_evidence_gate as E
from .dino_localization_gallery import panel

C=E.C;V=E.V;OUT=C.OUT/'selftrain_recovery_v1'/E.PHASE


def main():
    E.verify();audit=C.read(E.DOC/'COMPLETION_AUDIT.json')
    for b in audit['evidence']:C.verify(b)
    preds={'R0':{r['id']:r['prediction'] for r in C.read(E.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}}
    scores={'R0':{r['id']:r for r in C.read(E.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}}
    for arm in E.ARMS:
        for tag,root in [('IMAGE',V.RAW),('GATE',E.RAW)]:
            key=arm+'_'+tag;preds[key]={r['id']:r['prediction'] for r in C.read(root/f'EVAL_PREDICTIONS_{arm}.json')['records']}
            scores[key]={r['id']:r for r in C.read(root/f'SCREEN_{arm}.json')['metrics']}
    pe,pop=E.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in preds['R0']}
    receipts={r['id']:r for r in C.read(E.RAW/'INFERENCE_RECEIPTS.json')};byid=defaultdict(list)
    for c in C.read(E.RAW/'CORNER_DIAGNOSTICS.json')['rows']:byid[c['id']].append(c)
    recovered={k for k,rows in byid.items() if any(c['before']>40 and c['after']<=10 for c in rows)}
    damaged={k for k,rows in byid.items() if any(c['before']<5 and c['after']>10 for c in rows)}
    changed={k for k,r in receipts.items() if any(any(d['accepted']) for d in r['decisions'].values())}
    example='eval_pallet07:1778652166837872128'
    ordered=sorted(C.read(E.DOC/'EVAL_PROTOCOL.json')['records'],key=lambda r:(r['id'] not in recovered,r['id']!=example,r['id'] not in damaged,r['id'] not in changed,r['id']))
    OUT.mkdir(parents=True,exist_ok=True);sections=[];bindings=[]
    for i,r in enumerate(ordered):
        key=r['id'];C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
        panels=[];details=[]
        for arm in E.ARMS:
            tags=['R0',arm+'_IMAGE',arm+'_GATE']
            panels.append(np.concatenate([panel(im,preds[t][key],targets[key],scores[t][key],t) for t in tags],axis=1))
            accepted=np.flatnonzero(receipts[key]['decisions'][arm]['accepted']).tolist()
            details.append(arm+' 적용 native 코너: '+str(accepted))
        for c in byid[key]:
            if (c['before']>20 and c['after']<=10) or (c['before']<5 and c['after']>10):
                details.append('%s G%d %.2f→%.2fpx'%(c['arm'],c['GT_corner'],c['before'],c['after']))
        path=OUT/f'{i:03d}.jpg';assert not path.exists()
        assert cv2.imwrite(str(path),np.concatenate(panels,axis=0),[cv2.IMWRITE_JPEG_QUALITY,90])
        sections.append('<section><h2>'+html.escape(key)+'</h2><p>'+html.escape(' / '.join(details))+'</p><a href="'+path.name+'"><img loading="lazy" src="'+path.name+'"></a></section>')
        bindings.append(dict(id=key,image=C.bound(path),changed=key in changed,normal_damage=key in damaged))
    page='''<!doctype html><meta charset="utf-8"><title>영상 근거 기반 선택 보정: 전체194장</title>
<style>body{background:#101a21;color:#e1e8ef;font:16px sans-serif;margin:24px}section{border-top:1px solid #65727e;padding:20px 0}img{width:100%;height:auto}h2{font-size:18px}p{line-height:1.6}.warning{background:#53351f;padding:16px}</style>
<h1>영상 근거 기반 선택 보정 · 일반 플라스틱194장</h1>
<p class="warning">목표 미완료·최종 모델 미교체. SYN 정상 손상92→4개, >20→≤10px 복구10개.
MIX 정상 손상101→0개, 복구7개. 각 조건의40px초과 복구2개,80/100px초과 복구0개.
모든 기존 점이 정답에서40px 넘게 떨어진64코너의 실제 큰 위치 복구는 각각1개뿐입니다.</p>
<p>각 사진의 위 행 SYN(합성 학습), 아래 행 MIX(합성+기존 수도레이블).
각 행 왼쪽 R0 / 가운데 IMAGE(무조건 영상 보정) / 오른쪽 GATE(합성 정답으로만 학습·기준 설정한 선택 보정).
초록 G=GT, 청록 P=실제 출력. 번호와 평균 오차는 C2 대칭 대응에 따라 다를 수 있습니다.
GT는 점수·그림에만 사용했습니다. 거절한 코너는 원래 좌표 그대로이며, 모든194프레임을 포함합니다.
큰 복구→원래 요청 사진→손상→그 외 변경→유지 순서입니다. 그림을 누르면 확대됩니다.</p>'''+''.join(sections)
    C.write_text(OUT/'index.html',page)
    C.freeze(OUT/'GALLERY_RECEIPT.json',dict(rows=bindings,frames=194,all194_included=True,GT_for_display_only=True,
        source=C.bound(__file__),audit=C.bound(E.DOC/'COMPLETION_AUDIT.json')))
    print('EVIDENCE_GALLERY',OUT/'index.html',flush=True)


if __name__=='__main__':main()

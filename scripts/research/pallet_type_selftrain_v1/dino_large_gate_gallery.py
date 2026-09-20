"""All194 before/parent/large-error-trained gates, with both arms visible."""
import html
from collections import defaultdict
import cv2
import numpy as np
from . import dino_large_gate as L
from .dino_localization_gallery import panel

C=L.C;E=L.E;OUT=C.OUT/'selftrain_recovery_v1'/L.PHASE


def main():
    L.verify();audit=C.read(L.DOC/'COMPLETION_AUDIT.json');intervention=C.read(L.DOC/'INTERVENTION_AUDIT.json')
    for b in audit['evidence']+intervention['evidence']:C.verify(b)
    preds={'R0':{r['id']:r['prediction'] for r in C.read(E.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}}
    scores={'R0':{r['id']:r for r in C.read(E.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}}
    for arm in E.ARMS:
        for tag,root in [('PREVIOUS',E.RAW),('LARGE',L.RAW)]:
            name=arm+'_'+tag;preds[name]={r['id']:r['prediction'] for r in C.read(root/f'EVAL_PREDICTIONS_{arm}.json')['records']}
            scores[name]={r['id']:r for r in C.read(root/f'SCREEN_{arm}.json')['metrics']}
    pe,pop=E.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in preds['R0']}
    receipts={r['id']:r for r in C.read(L.RAW/'INFERENCE_RECEIPTS.json')};byid=defaultdict(list)
    for c in C.read(L.RAW/'CORNER_DIAGNOSTICS.json')['rows']:byid[c['id']].append(c)
    recovered={k for k,rr in byid.items() if any(c['before']>100 and c['after']<=10 for c in rr)}
    damaged={k for k,rr in byid.items() if any(c['before']<5 and c['after']>10 for c in rr)}
    changed={k for k,r in receipts.items() if any(any(d['accepted']) for d in r['decisions'].values())}
    example='eval_pallet07:1778652166837872128'
    ordered=sorted(C.read(L.DOC/'EVAL_PROTOCOL.json')['records'],key=lambda r:(r['id'] not in recovered,r['id']!=example,r['id'] not in damaged,r['id'] not in changed,r['id']))
    OUT.mkdir(parents=True,exist_ok=True);sections=[];bindings=[]
    for i,r in enumerate(ordered):
        key=r['id'];C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
        panels=[];details=[]
        for arm in E.ARMS:
            tags=['R0',arm+'_PREVIOUS',arm+'_LARGE']
            panels.append(np.concatenate([panel(im,preds[t][key],targets[key],scores[t][key],t) for t in tags],axis=1))
            details.append(arm+' 적용 native 코너: '+str(np.flatnonzero(receipts[key]['decisions'][arm]['accepted']).tolist()))
        for c in byid[key]:
            if (c['before']>20 and c['after']<=10) or (c['before']<5 and c['after']>10):
                details.append('%s G%d %.2f→%.2fpx; 최근접 기존 점%.2fpx'%(c['arm'],c['GT_corner'],c['before'],c['after'],c['nearest_R0_point_distance']))
        path=OUT/f'{i:03d}.jpg';assert not path.exists()
        assert cv2.imwrite(str(path),np.concatenate(panels,axis=0),[cv2.IMWRITE_JPEG_QUALITY,90])
        sections.append('<section><h2>'+html.escape(key)+'</h2><p>'+html.escape(' / '.join(details))+'</p><a href="'+path.name+'"><img loading="lazy" src="'+path.name+'"></a></section>')
        bindings.append(dict(id=key,image=C.bound(path),changed=key in changed,normal_damage=key in damaged))
    result=C.read(L.DOC/'RESULTS.json');summary=[]
    for a in E.ARMS:
        r=result['results']['full194'][a];s=audit['spatial_decomposition'][a]['no_old_point_within40']
        summary.append('%s: >20→≤10px %d/281, >100→≤10px %d/58, 정상 손상%d/511, 실제 큰 위치복구%d/64'%(a,r['recovery']['recovered'],r['tails']['100']['recovered'],r['recovery']['damaged'],s['recovered']))
    page='''<!doctype html><meta charset="utf-8"><title>큰 인공 오류 학습: 전체194장</title>
<style>body{background:#101a21;color:#e1e8ef;font:16px sans-serif;margin:24px}section{border-top:1px solid #65727e;padding:20px 0}img{width:100%;height:auto}h2{font-size:18px}p{line-height:1.6}.warning{background:#53351f;padding:16px}</style>
<h1>큰 인공 입력 오류로 학습한 선택기 · 일반 플라스틱194장</h1>
<p class="warning">목표 미완료·최종 모델 미교체. '''+html.escape(' / '.join(summary))+'''</p>
<p>위 행 SYN, 아래 행 MIX. 각 행 왼쪽 R0 / 가운데 PREVIOUS(이전 선택기) / 오른쪽 LARGE(큰 오류를 포함해 합성 정답으로 학습한 선택기).
GT는 초록 G, 실제 모델 출력은 청록 P입니다. 평가 GT는 점수·그림에만 사용했습니다.
큰 번호 오차 복구와 모든 기존 점이 정답에서 먼 실제 새 위치 복구를 구분합니다. 일부 코너 복구는 전체 포즈 성공이 아닙니다.
기준은 합성32장에서만 정했습니다. 탈락한 프레임은 없으며 전체194장을 포함합니다. 그림을 누르면 확대됩니다.</p>'''+''.join(sections)
    C.write_text(OUT/'index.html',page)
    C.freeze(OUT/'GALLERY_RECEIPT.json',dict(frames=194,rows=bindings,all194_included=True,GT_for_display_only=True,
        source=C.bound(__file__),audit=C.bound(L.DOC/'COMPLETION_AUDIT.json'),intervention=C.bound(L.DOC/'INTERVENTION_AUDIT.json')))
    print('LARGE_GATE_GALLERY',OUT/'index.html',flush=True)


if __name__=='__main__':main()

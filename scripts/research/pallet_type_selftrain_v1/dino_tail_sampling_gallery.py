"""All194 comparisons for uniform versus natural-tail sampling, saved outputs only."""
import argparse
import html
from collections import defaultdict
import cv2
import numpy as np
from . import dino_tail_sampling as L
from .dino_localization_gallery import panel

C=L.C;D=L.D


def main(phase):
    X=L;assert phase==L.PHASE;X.verify()
    audit=C.read(X.DOC/'COMPLETION_AUDIT.json');result=C.read(X.DOC/'RESULTS.json')
    for b in audit['evidence']:C.verify(b)
    for b in C.read(X.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    out=C.OUT/'selftrain_recovery_v1'/phase
    scores={'R0':{r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}}
    preds={'R0':{r['id']:r['prediction'] for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}}
    for a in L.ARMS:
        scores[a]={r['id']:r for r in C.read(X.RAW/f'SCREEN_{a}.json')['metrics']}
        preds[a]={r['id']:r['prediction'] for r in C.read(X.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']}
    pe,pop=D.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in preds['R0']}
    rows=C.read(X.RAW/'CORNER_DIAGNOSTICS.json')['rows'];by_id=defaultdict(list)
    for r in rows:by_id[r['id']].append(r)
    far={r['id'] for r in rows if r['nearest_R0_point_distance']>40 and r['after']<=10}
    damaged={r['id'] for r in rows if r['before']<5 and r['after']>10}
    example='eval_pallet07:1778652166837872128'
    metadata=C.read(X.DOC/'EVAL_PROTOCOL.json')['records']
    ordered=sorted(metadata,key=lambda r:(r['id']!=example,r['id'] not in far,r['id'] not in damaged,r['id']))
    entries=[];bindings=[];out.mkdir(parents=True,exist_ok=True)
    for i,r in enumerate(ordered):
        key=r['id'];C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
        views=[panel(im,preds[a][key],targets[key],scores[a][key],a) for a in ['R0',*L.ARMS]]
        path=out/f'{i:03d}.jpg';assert not path.exists()
        assert cv2.imwrite(str(path),np.concatenate(views,axis=1),[cv2.IMWRITE_JPEG_QUALITY,92])
        details=[]
        for c in by_id[key]:
            if (c['before']>40 and c['after']<=10) or (c['before']<5 and c['after']>10):
                details.append('%s G%d %.2f→%.2fpx (最近接 R0 %.2fpx)'%(c['arm'],c['GT_corner'],c['before'],c['after'],c['nearest_R0_point_distance']))
        label=('원래 요청 사진 · ' if key==example else '')+('실제 큰 위치 이동 복구 포함 · ' if key in far else '')+('정상 코너 손상 포함' if key in damaged else '')
        entries.append('<section><h2>'+html.escape(key)+'</h2><p>'+html.escape(label)+'</p><a href="'+path.name+'"><img loading="lazy" src="'+path.name+'"></a><p>'+html.escape(' / '.join(details))+'</p></section>')
        bindings.append(dict(id=key,image=C.bound(path),far_spatial_recovery=key in far,normal_damage=key in damaged))
    summary=[]
    for a in L.ARMS:
        t=result['results']['full194'][a];s=audit['spatial_decomposition'][a]['no_old_point_within40']
        summary.append('%s: 실제 큰 위치 이동 %d/64 복구; >20→≤10px %d/281; 정상 코너 손상%d/511; PCK20 %.2f%%; 채택 게이트 %s.'%(a,s['recovered'],t['recovery']['recovered'],t['recovery']['damaged'],100*t['summary']['PCK']['20'],result['passed'][a]))
    page='''<!doctype html><meta charset="utf-8"><title>보정기 일반화 검증 전체194장</title>
<style>body{background:#101a21;color:#e1e8ef;font:16px sans-serif;margin:24px}section{border-top:1px solid #65727e;padding:20px 0}img{width:100%;height:auto}h2{font-size:18px}p{line-height:1.6}.warning{background:#53351f;padding:16px}</style>
<h1>'''+html.escape(phase)+''' · 전체194장</h1><p class="warning">'''+html.escape(' / '.join(summary))+'''</p>
<p>왼쪽 R0 / 가운데 UNIFORM 균등 샘플링 / 오른쪽 TAIL 큰 오류50% 샘플링. 두 모델 모두 합성만 학습했습니다. 초록 G=GT, 청록 P=실제 출력 채널.
평가 GT는 점수·시각화에만 사용했습니다. GT로 좋은 점을 선택해 모델 출력에 적용하지 않았습니다.
번호와 평균 오차는 C2 대칭 대응에 따라 다를 수 있습니다. 한 코너 복구는 프레임 전체 성공이 아닙니다.
원래 요청 사진→실제 큰 위치 이동 복구 포함→손상 포함→나머지 순서이며 전체194장을 빠짐없이 포함합니다.
반복 DEV이며 독립 확증이 아닙니다. 최종 모델 교체 없음.</p>'''+''.join(entries)
    C.write_text(out/'index.html',page)
    C.freeze(out/'GALLERY_RECEIPT.json',dict(frames=len(bindings),rows=bindings,all194_included=True,GT_for_display_only=True,
        source=C.bound(__file__),audit=C.bound(X.DOC/'COMPLETION_AUDIT.json')))
    print('GENERALIZATION_GALLERY',out/'index.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['dino_tail_sampling']);main(p.parse_args().phase)


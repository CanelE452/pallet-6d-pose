"""Whole-population review of wide-field experiments, including damage cases."""
import argparse
import html
from collections import defaultdict
import cv2
import numpy as np
from . import dino_wide as W
from . import dino_wide_visual as V
from .dino_localization_gallery import panel

C=W.C;D=W.D


def main(phase):
    X={'dino_wide':W,'dino_wide_visual':V}[phase];X.verify()
    audit=C.read(X.DOC/'COMPLETION_AUDIT.json');result=C.read(X.DOC/'RESULTS.json')
    for b in audit['evidence']:C.verify(b)
    for b in C.read(X.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    out=C.OUT/'selftrain_recovery_v1'/phase
    scores={'R0':{r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}}
    preds={'R0':{r['id']:r['prediction'] for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}}
    for a in D.ARMS:
        scores[a]={r['id']:r for r in C.read(X.RAW/f'SCREEN_{a}.json')['metrics']}
        preds[a]={r['id']:r['prediction'] for r in C.read(X.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']}
    pe,pop=D.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in preds['R0']}
    rows=C.read(X.RAW/'CORNER_DIAGNOSTICS.json')['rows'];by_id=defaultdict(list)
    for r in rows:by_id[r['id']].append(r)
    far={r['id'] for r in rows if r['nearest_R0_point_distance']>40 and r['after']<=10}
    damaged={r['id'] for r in rows if r['before']<5 and r['after']>10}
    example='eval_pallet07:1778652166837872128'
    metadata=C.read(X.DOC/'EVAL_PROTOCOL.json')['records']
    ordered=sorted(metadata,key=lambda r:(r['id'] not in far,r['id']!=example,r['id'] not in damaged,r['id']))
    entries=[];bindings=[];out.mkdir(parents=True,exist_ok=True)
    for i,r in enumerate(ordered):
        key=r['id'];C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
        views=[panel(im,preds[a][key],targets[key],scores[a][key],a) for a in ['R0',*D.ARMS]]
        path=out/f'{i:03d}.jpg';assert not path.exists()
        assert cv2.imwrite(str(path),np.concatenate(views,axis=1),[cv2.IMWRITE_JPEG_QUALITY,92])
        details=[]
        for c in by_id[key]:
            if (c['before']>40 and c['after']<=10) or (c['before']<5 and c['after']>10):
                details.append('%s G%d %.2f→%.2fpx (최근접 기존 점 %.2fpx%s)'%(c['arm'],c['GT_corner'],c['before'],c['after'],c['nearest_R0_point_distance'],', 기존 출력 영역으로 10px 이내 도달 불가' if c['parent_unreachable_within10'] else ''))
        label=('실제 큰 위치 이동 복구 포함 · ' if key in far else '')+('정상 코너 손상 포함 · ' if key in damaged else '')+('원래 요청 사진' if key==example else '')
        entries.append('<section><h2>'+html.escape(key)+'</h2><p>'+html.escape(label)+'</p><a href="'+path.name+'"><img loading="lazy" src="'+path.name+'"></a><p>'+html.escape(' / '.join(details))+'</p></section>')
        bindings.append(dict(id=key,image=C.bound(path),far_spatial_recovery=key in far,normal_damage=key in damaged))
    summary=[]
    for a in D.ARMS:
        t=result['results']['full194'][a];s=audit['spatial_decomposition'][a]['no_old_point_within40']
        summary.append('%s: 큰 위치 이동64개 중%d개 복구; >20→≤10px %d/281, 정상 코너 손상%d/511; PCK20 %.2f%%; 채택 게이트 %s.'%(a,s['recovered'],t['recovery']['recovered'],t['recovery']['damaged'],100*t['summary']['PCK']['20'],result['passed'][a]))
    page='''<!doctype html><meta charset="utf-8"><title>넓은 시야 보정 전체194장</title>
<style>body{background:#101a21;color:#e1e8ef;font:16px sans-serif;margin:24px}section{border-top:1px solid #65727e;padding:20px 0}img{width:100%;height:auto}h2{font-size:18px}p{line-height:1.6}.warning{background:#53351f;padding:16px}</style>
<h1>'''+html.escape(phase)+''' · 전체194장</h1><p class="warning">'''+html.escape(' / '.join(summary))+'''</p>
<p>왼쪽 R0 / 가운데 합성만 SYN / 오른쪽 합성+기존 수도레이블 MIX. 초록 G=GT, 청록 P=모델의 실제 출력 채널.
평가 GT는 점수와 시각화에만 사용했습니다. GT로 코너나 프레임을 골라 모델 출력에 적용하지 않았습니다.
번호와 평균 오차는 C2 대칭 대응에 따라 다를 수 있습니다. 한 코너 복구는 프레임 전체 성공이 아닙니다.
모든 기존 예측점이 정답에서40px 넘게 떨어져 있던 코너의 복구→원래 요청 사진→손상→나머지 순서입니다. 전체194장을 빠짐없이 포함합니다.</p>'''+''.join(entries)
    C.write_text(out/'index.html',page)
    C.freeze(out/'GALLERY_RECEIPT.json',dict(frames=len(bindings),rows=bindings,all194_included=True,GT_for_display_only=True,
        source=C.bound(__file__),audit=C.bound(X.DOC/'COMPLETION_AUDIT.json')))
    print('WIDE_GALLERY',out/'index.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['dino_wide','dino_wide_visual']);main(p.parse_args().phase)

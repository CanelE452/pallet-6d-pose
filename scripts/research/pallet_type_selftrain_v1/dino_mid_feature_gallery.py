"""Saved output comparisons: R0 / SYN5k / mid-layer+last-layer, all194 frames."""
import argparse
import html
from collections import defaultdict
import cv2
import numpy as np
from . import dino_diverse_continue as L
from . import dino_mid_feature as T
from .dino_localization_gallery import panel

C=L.C;D=L.D;X=L.X


def main(phase):
    module={'dino_diverse_continue':L,'dino_mid_feature':T}[phase];module.verify()
    audit=C.read(module.DOC/'COMPLETION_AUDIT.json');result=C.read(module.DOC/'RESULTS.json')
    for b in audit['evidence']:C.verify(b)
    for owner in [X,module]:
        for b in C.read(owner.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    label={'dino_diverse_continue':'SYN 20k','dino_mid_feature':'SYN MID 5k'}[phase]
    scores={'R0':{r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}}
    preds={'R0':{r['id']:r['prediction'] for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}}
    for arm,owner in [('SYN 5k',X),(label,module)]:
        scores[arm]={r['id']:r for r in C.read(owner.RAW/'SCREEN_SYN.json')['metrics']}
        preds[arm]={r['id']:r['prediction'] for r in C.read(owner.RAW/'EVAL_PREDICTIONS_SYN.json')['records']}
    pe,pop=D.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in preds['R0']}
    rows=C.read(module.RAW/'CORNER_DIAGNOSTICS.json')['rows'];byid=defaultdict(list)
    for r in rows:byid[r['id']].append(r)
    far={r['id'] for r in rows if r['nearest_R0_point_distance']>40 and r['after']<=10}
    damaged={r['id'] for r in rows if r['before']<5 and r['after']>10}
    example='eval_pallet07:1778652166837872128'
    records=sorted(C.read(module.DOC/'EVAL_PROTOCOL.json')['records'],key=lambda r:(r['id']!=example,r['id'] not in far,r['id'] not in damaged,r['id']))
    out=C.OUT/'selftrain_recovery_v1'/phase;out.mkdir(parents=True,exist_ok=True);entries=[];bindings=[]
    for i,r in enumerate(records):
        key=r['id'];C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
        views=[panel(im,preds[a][key],targets[key],scores[a][key],a) for a in ['R0','SYN 5k',label]]
        path=out/f'{i:03d}.jpg';assert not path.exists();assert cv2.imwrite(str(path),np.concatenate(views,1),[cv2.IMWRITE_JPEG_QUALITY,92])
        changes=['G%d %.2f→%.2fpx; nearest any R0 %.2fpx'%(c['GT_corner'],c['before'],c['after'],c['nearest_R0_point_distance'])
            for c in byid[key] if (c['before']>40 and c['after']<=10) or (c['before']<5 and c['after']>10)]
        flag=('원래 요청 사진 / ' if key==example else '')+('실제 큰 위치 복구 포함 / ' if key in far else '')+('정상 코너 손상 포함' if key in damaged else '')
        entries.append('<section><h2>'+html.escape(key)+'</h2><p>'+html.escape(flag)+'</p><a href="'+path.name+'"><img loading="lazy" src="'+path.name+'"></a><p>'+html.escape(' | '.join(changes))+'</p></section>')
        bindings.append(dict(id=key,image=C.bound(path),far_spatial_recovery=key in far,normal_damage=key in damaged))
    t=result['results']['full194']['SYN'];s=audit['spatial_decomposition']['no_old_point_within40']
    summary='%s: >20→≤10px %d/281; normal damage %d/511; genuine far recovery %d/64; PCK20 %.2f%%; gate %s.'%(label,
        t['recovery']['recovered'],t['recovery']['damaged'],s['recovered'],100*t['summary']['PCK']['20'],result['passed']['SYN'])
    page='''<!doctype html><meta charset="utf-8"><title>큰 코너 복구 전체 비교</title><style>
body{background:#101a21;color:#e1e8ef;font:16px sans-serif;margin:24px}section{border-top:1px solid #65727e;padding:20px 0}img{width:100%;height:auto}h2{font-size:18px}p{line-height:1.6}.warning{background:#53351f;padding:16px}</style>
<h1>'''+html.escape(phase)+''' · 일반 플라스틱 전체194장</h1><p class="warning">'''+html.escape(summary)+'''</p><p>왼쪽 R0 / 가운데 기존 SYN5k / 오른쪽 새 실험.
초록 G=평가 GT, 청록 P=모델 출력. GT는 점수·시각화에만 사용했으며 출력 선택/학습에 쓰지 않았습니다.
기존 어떤 점에서도 40px 넘게 떨어진 정답의 복구를 별도로 셉니다. C2 번호 대응 변화만을 큰 위치 복구로 세지 않습니다.
한 코너 복구는 전체 자세 성공이 아닙니다. 원래 요청 사진→큰 위치 복구 포함→손상 포함→나머지 순서로194장 모두 표시합니다.
반복 DEV이며 최종 모델 교체는 없습니다. 새 레이블/태그는 없습니다.</p>'''+''.join(entries)
    C.write_text(out/'index.html',page)
    C.freeze(out/'GALLERY_RECEIPT.json',dict(frames=len(bindings),all194_included=True,rows=bindings,GT_for_display_only=True,
        source=C.bound(__file__),audit=C.bound(module.DOC/'COMPLETION_AUDIT.json')))
    print('FOLLOWUP_GALLERY',out/'index.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['dino_diverse_continue','dino_mid_feature']);main(p.parse_args().phase)


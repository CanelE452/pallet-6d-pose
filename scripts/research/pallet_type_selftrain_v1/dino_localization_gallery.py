"""All-frame diagnostic gallery: immutable outputs, GT for display only."""
import html
from collections import defaultdict
import cv2
import numpy as np
from . import dino_localization as D

C=D.C;P=D.P
OUT=C.OUT/'selftrain_recovery_v1/dino_localization'
EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def text(im,label,xy,color):
    cv2.putText(im,label,xy,cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(im,label,xy,cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)


def panel(image,pred,target,score,arm):
    factor=540/image.shape[1];canvas=cv2.resize(image,None,fx=factor,fy=factor)
    q=np.asarray(P.top(pred)['keypoints_xy'])[:8]*factor
    g=np.asarray(target.keypoints_xy)[:8]*factor;gv=np.asarray(target.keypoint_supervision_mask)[:8]
    for pts,valid,color,prefix in [(g,gv,(80,245,80),'G'),(q,np.isfinite(q).all(-1),(255,220,30),'P')]:
        for a,b in EDGES:
            if valid[a] and valid[b]:cv2.line(canvas,tuple(np.rint(pts[a]).astype(int)),tuple(np.rint(pts[b]).astype(int)),color,1,cv2.LINE_AA)
        for j,point in enumerate(pts):
            if valid[j]:
                xy=tuple(np.rint(point).astype(int));cv2.circle(canvas,xy,4,color,1,cv2.LINE_AA)
                text(canvas,prefix+str(j),(xy[0]+4,xy[1]-5),color)
    header=np.full((66,540,3),22,np.uint8)
    text(header,arm+' | C2 frame mean %.2f px'%score['frame_mean_px'],(12,22),(245,245,245))
    text(header,'GT green G / predicted cyan P (native channels)',(12,44),(210,210,210))
    return np.vstack([header,canvas])


def main():
    D.verify();audit=C.read(D.DOC/'COMPLETION_AUDIT.json')
    for b in C.read(D.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    scores={'R0':{r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}}
    preds={'R0':{r['id']:r['prediction'] for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}}
    for a in D.ARMS:
        scores[a]={r['id']:r for r in C.read(D.RAW/f'SCREEN_{a}.json')['metrics']}
        preds[a]={r['id']:r['prediction'] for r in C.read(D.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']}
    pe,pop=D.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in preds['R0']}
    corners=C.read(D.RAW/'CORNER_DIAGNOSTICS.json')['rows'];by_id=defaultdict(list)
    for r in corners:by_id[r['id']].append(r)
    metadata=C.read(D.DOC/'EVAL_PROTOCOL.json')['records'];example='eval_pallet07:1778652166837872128'
    recovered={r['id'] for r in corners if r['before']>100 and r['after']<=10}
    damaged={r['id'] for r in corners if r['before']<5 and r['after']>10}
    ordered=sorted(metadata,key=lambda r:(r['id']!=example,r['id'] not in recovered,r['id'] not in damaged,r['id']))
    entries=[];bindings=[];OUT.mkdir(parents=True,exist_ok=True)
    for i,r in enumerate(ordered):
        key=r['id'];C.verify(r['image']);image=cv2.imread(str(C.ROOT/r['image']['path']));assert image is not None
        views=[panel(image,preds[a][key],targets[key],scores[a][key],a) for a in ['R0',*D.ARMS]]
        path=OUT/f'{i:03d}.jpg';assert not path.exists();assert cv2.imwrite(str(path),np.concatenate(views,axis=1),[cv2.IMWRITE_JPEG_QUALITY,92])
        details=[]
        for c in by_id[key]:
            if c['before']>100 and c['after']<=10 or c['before']<5 and c['after']>10:
                details.append('%s G%d: %.2f→%.2fpx; 최근접 기존 점 %.2fpx'%(c['arm'],c['GT_corner'],c['before'],c['after'],c['nearest_R0_point_distance']))
        labels=('원래 요청 사진 · ' if key==example else '')+('100px 오류 일부 복구 · ' if key in recovered else '')+('정상 코너 손상 있음' if key in damaged else '')
        entries.append('<section><h2>'+html.escape(key)+'</h2><p>'+html.escape(labels)+'</p><a href="'+path.name+'"><img loading="lazy" src="'+path.name+'"></a><p>'+html.escape(' / '.join(details))+'</p></section>')
        bindings.append(dict(id=key,image=C.bound(path)))
    page='''<!doctype html><meta charset="utf-8"><title>DINO 보정 전체194장</title>
<style>body{background:#101a21;color:#e1e8ef;font:16px sans-serif;margin:24px}section{border-top:1px solid #65727e;padding:20px 0}img{width:100%;height:auto}h2{font-size:18px}p{line-height:1.6}.warning{background:#53351f;padding:16px}</style>
<h1>DINO 영상 특징 보정: 전체194장</h1>
<p class="warning">미채택 실험입니다. SYN은 큰 오류20개 복구/정상35개 손상, MIX는30개 복구/50개 손상입니다.
100px 초과→10px 이내는 각각4개/9개지만, 대부분 다른 R0 점이 이미 정답 근처에 있습니다. 한 코너 복구는 프레임 전체 성공이 아닙니다.
평가 GT는 점수·그림에만 사용했고 모델 추론에는 사용하지 않았습니다. GT로 좋은 점만 골라 출력하지 않았습니다.</p>
<p>왼쪽 R0 / 가운데 합성만 SYN / 오른쪽 합성+기존 수도레이블 MIX. 초록 G는 감독 가능한 GT, 청록 P는 실제 출력의 원래 채널입니다.
상단 점수는 기존 C2 대칭 평가값이므로 P/G 번호가 다른 대칭 분기에서 대응할 수 있습니다. 원래 사진→일부 큰 오류 복구→손상→나머지 순서이며 전체194장을 포함합니다. 이미지를 누르면 확대됩니다.</p>'''+''.join(entries)
    C.write_text(OUT/'index.html',page)
    C.freeze(OUT/'GALLERY_RECEIPT.json',dict(frames=len(bindings),rows=bindings,
        GT_for_display_only=True,all194_included=True,source=C.bound(__file__),audit=C.bound(D.DOC/'COMPLETION_AUDIT.json')))
    print('DINO_GALLERY',OUT/'index.html',flush=True)


if __name__=='__main__':main()

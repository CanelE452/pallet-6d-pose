"""Post-hoc reporting only, never trains or changes experiment decisions."""
from collections import Counter
import html
import json
from pathlib import Path
import cv2
import numpy as np
from . import common as C
from . import augmentation as A


def fmt(x,n=2):return '—' if x is None else f'{x:.{n}f}'


def main():
    assert C.read(C.DOC/'DRIVER_COMPLETE.json')['complete']
    C.immutable();C.DOC.joinpath('images').mkdir(exist_ok=True)
    result=C.read(C.DOC/'RESULTS.json');groups=result['groups']
    old=C.read(C.DOC/'OLD_TRAIN_FIT.json')['summary'];aug=C.read(C.DOC/'AUGMENTATION_AUDIT.json');audit=C.read(C.DOC/'TRAIN_TARGET_AUDIT.json')
    names=['R0','TEACHER','OLD_STUDENT','S0','S1','S2'];sevs=C.H.P.SEVERITIES
    lines=['# CLEAN19 구조적 EASY→HARD 증류 결정 실험','',
        '[확인] 같은 촬영 세션의 Clean19(플라스틱10·목재9) 감독, 자연평가300장 고정. 실사87개 직접클릭 근거 채널의 frozen teacher 좌표 사용. S0/S1/S2 각각두종류,총6fits×320=1,920 update. seed42, last만 평가. 이하 단일seed·반복DEV이며 독립 일반화/통계적 유의성 주장이 아니다.', '',
        '| method | Clean PCK10 개수/분모(%) | Moderate PCK10 | Severe PCK10 | Mod/Sev ADDsym AUC | Mod/Sev 축정확도% | R0정답 손실/획득 | 검출/매칭 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        vals=[]
        for s in sevs:
            t=groups[s][name]['twoD'];vals.append(f'{t["correct"]["10"]}/{t["corners"]} ({100*t["PCK"]["10"]:.2f})')
        mod=groups[sevs[1]][name]['sixD'];sev=groups[sevs[2]][name]['sixD'];whole=groups['ALL300'][name];d=whole['vs_R0'];t=whole['twoD']
        lines.append(f'| {name} | '+ ' | '.join(vals)+f' | {mod["ADDsym_AUC"]:.4f}/{sev["ADDsym_AUC"]:.4f} | {100*mod["axis_accuracy"]:.2f}/{100*sev["axis_accuracy"]:.2f} | {d["lost_correct10"]}/{d["gained_correct10"]} | {t["detected"]}/{t["matched"]} |')
    lines += ['', 'TEACHER=기존 종류별Replay+자기가림PnP, OLD_STUDENT=이전171점 수도레이블 학생(역사적 참고). 새S0와 old는 마스크/기본증강샘플동결이 다르므로 old 대비 변화를 가림효과라 하지 않는다. R0정답 손실=R0≤10px→해당모델>10px, 획득=반대; canonical GT identity로 비교. 검출/매칭 분모300.', '',
        '## 기존 학생 TRAIN-fit: 학습 실패인가 전이 실패인가','',
        '| 종류 | 모델 | 원영상 pseudo8 PCK10 | reflect100 pseudo8 PCK10 | reflect100 manual PCK10 |', '|---|---|---:|---:|---:|']
    for mat in C.MATERIALS:
        for name in ('R0','OLD_STUDENT','TEACHER'):
            r=old[mat][name]
            lines.append(f'| {mat} | {name} | {100*r["native"]["pseudo8"]["PCK"]["10"]["fraction"]:.2f}% | {100*r["reflect100"]["pseudo8"]["PCK"]["10"]["fraction"]:.2f}% | {100*r["reflect100"]["manual"]["PCK"]["10"]["fraction"]:.2f}% |')
    lines += ['', '[확인] 교사는 저장결과로서 원영상/native와reflect100열에서 동일하다. 비교 학생의 두 입력 추론은 별도로 실행했다. 8코너·중심·PnP대체·새마스크별PCK5/10/20,median/P90,전체물체대칭 보조값은 OLD_TRAIN_FIT.json에 있다. native channel score와 symmetry score를 혼동하지 않는다.',
        '', '## 감독·가림 계약','',
        f'- [확인] 원래 pseudo {audit["counts"]["old_pseudo"]}좌표 중 코너152+중심19. 직접클릭87개; 새좌표감독87개. PnP대체19개는 전부 비수동 채널로 제외. 비수동코너65개(그 중PnP19)+중심19가 제외되어84개. identity_unknown 명시필드는0개이나 비수동 출처를 수동근거로 간주하지 않았다.',
        '- [확인] 좌표값은 교사 yT 그대로이며 수동GT로 대체하지 않음. source=manual_click은 provenance이고 실제 가시성 보장 아님. 인공가림은 masked-target 복원이라고 해석한다.',
        '- [확인] R0 새시작,5epoch,batch=nbs16,AdamW lr1e-4/lrf0.1/cosine,seed42. 학습별 합성512unique+실사512복원추출/epoch,각2,560 exposure. 원래학생처럼 stock이 고정하는 층 외 새freeze없음; 실제 inventory와 초기값 해시6fit일치검사.',
        '- [확인] 기본HSV/affine은 설치YOLO로 occurrence별생성해640 RGB/타깃 텐서를 동결. S0/S1/S2가 같은텐서를 읽는다. workers2여도순서·RGB·target이변하지않도록epoch/slot명시. 기존 과거학생과 기본텐서가 같다는 주장은 하지 않는다.',
        '- [확인] 초기preflight는 stock의경계clipping을 unclipped affine와비교해실패. clipping을 반영한검사로수정했고학습전기존partial cache를bit-exact재검증했다. fit재시작/seed변경은없음. 선택적Albumentations는설치API quality_range불일치로비활성화(기존경로와같음);패키지수정없음.',
        f'- [확인] 실사{aug["counts"]["real"]} occurrence, scheduled {aug["counts"]["scheduled"]}, 실제paired apply {aug["counts"]["applied"]} ({100*aug["counts"]["applied"]/aug["counts"]["real"]:.2f}%). 미적용은이미지삭제가아닌RGB원본유지. 합성추가가림0.',
        f'- [확인] S1 masked target {aug["counts"]["S1_masked"]}, S2 {aug["counts"]["S2_masked"]}. 크기·fill·apply빈도·bbox overlap tolerance를맞췄지만 masked-point 수와실제전경가림면적동일은주장하지않는다.',
        '- [확인] S1은adjacency없는무작위위치, S2는native 3D graph인접쌍을포함. shape선정후최대64위치후실패시paired skip. 실제bbox-overlap차이와종류/채널빈도는보조JSON.',
        '- [확인] supervised2 / true-ignore1 유지. 인공가림은좌표·v를바꾸지않는다. 제외좌표변화loss불변,가린감독좌표변화loss변화,allignore좌표/RLE/kobj0 및keypoint gradient0,합성stock parity검사통과.', '',
        '## 주 대조: 자연 가림 전이와 정상점 손실','',
        '| 그룹 | 비교 | PCK10 Δ%p | ADDsym Δ | 축정확도 Δ%p | 기존≤10손실 | >10→≤10획득 | >20→<10복구 | <5→>10손상 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for group in sevs:
        for contrast,d in result['contrasts'][group].items():
            lines.append(f'| {group} | {contrast} | {d["PCK10_delta_pp"]:+.2f} | {d["ADDsym_delta"]:+.4f} | {d["axis_delta_pp"]:+.2f} | {d["lost_correct10"]} | {d["gained_correct10"]} | {d["bad20_to_good10"]} | {d["good5_to_bad10"]} |')
    probe_summary={};source_summary={}
    before=C.read(C.RAW/'PROBE_BEFORE_PREDICTIONS.json')
    lines += ['', '## 새학생 TRAIN-fit 및 고정 인공가림 probe','',
        '아래는현재새공통마스크(87채널)의입력640좌표기준PCK10이다. 원영상native/reflect100의기존pseudo8·manual·center별값은 DIAGNOSTICS 파일에별도로있다. 원영상픽셀PCK와이표를직접동일척도로비교하지않는다.', '',
        '| 종류 | 모델 | 기본 입력 | 무작위 가림 | 구조적 가림 | 구조적 가린점 | 구조적 남은점 |', '|---|---|---:|---:|---:|---:|---:|']
    for mat in C.MATERIALS:
        probe_summary[mat]={};source_summary[mat]={}
        for name in ['R0','OLD_STUDENT',*C.ARMS]:
            d=before[mat][name] if name in before[mat] else C.read(C.RAW/f'DIAGNOSTICS_{mat}_{name}.json')
            modes={}
            for mode in ('native','reflect100',*C.ARMS):
                rr=[r for r in d['train'] if r['mode']==mode]
                if not rr:continue
                modes[mode]={key:C.summary([r['metrics'][key] for r in rr]) for key in rr[0]['metrics']}
            probe_summary[mat][name]=modes
            vals=[modes[m]['all']['PCK']['10']['fraction'] for m in C.ARMS]+[modes['S2'][k]['PCK']['10']['fraction'] for k in ('masked','remaining')]
            lines.append(f'| {mat} | {name} | '+' | '.join(fmt(None if v is None else 100*v) for v in vals)+' |')
            source_summary[mat][name]=dict(**C.summary(d['source']),matched=sum(r['matched'] for r in d['source']),frames=len(d['source']))
    lines += ['', '## 기존 합성 heldout256','',
        '[확인] 기존Replay ORDERS의held256이미지 그대로 사용,합성train512와image교차0. 동일혼합종류source pool에각material학생을각각적용한보조검사이며실사종류routing성과와혼합하지않는다. 기존synthetic validation bookkeeping과겹칠수있으므로독립test라하지않는다. 기존보정기keypoint입력jitter stress는RGB-only학생입력에존재하지않아학생stress로전용하지않았고,새합성가림probe는추가하지않았다.', '',
        '| 종류 모델 | method | PCK5 | PCK10 | PCK20 | median/P90 px | match/256 |','|---|---|---:|---:|---:|---:|---:|']
    for mat,arms in source_summary.items():
        for name,s in arms.items():lines.append(f'| {mat} | {name} | {100*s["PCK"]["5"]["fraction"]:.2f} | {100*s["PCK"]["10"]["fraction"]:.2f} | {100*s["PCK"]["20"]["fraction"]:.2f} | {s["median"]:.2f}/{s["P90"]:.2f} | {s["matched"]}/256 |')
    for group in ['ALL300',*sevs,'PLASTIC','WOOD',*[m+'_'+s for m in C.MATERIALS for s in sevs]]:
        if group not in groups:continue
        lines += ['', '## '+group+' 전체 지표','', '| 모델 | PCK5/10/20% | matched med/P90 px | 검출/매칭 | R/Yaw med° | t med cm | IoU3D med | ADD AUC | pose성공/전체 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name,d in groups[group].items():
            a=d['twoD'];b=d['sixD'];pck='/'.join(f'{100*a["PCK"][str(k)]:.2f}' for k in (5,10,20))
            lines.append(f'| {name} | {pck} | {fmt(a["matched_pooled_corner8_median_px"])}/{fmt(a["matched_pooled_corner8_P90_px"])} | {a["detected"]}/{a["matched"]} | {fmt(b["rotation_deg"]["median"])}/{fmt(b["yaw_deg"]["median"])} | {fmt(b["translation_cm"]["median"])} | {fmt(b["IoU3D"]["median"],4)} | {b["ADDsym_AUC"]:.4f} | {b["available"]}/{b["frames"]} |')
    lines += ['', 'PCK는 이미지 대각선 failure penalty를 포함한 고정 전체 코너 분모; matched median/P90은 성공 조건부다. 6D는 기존 prediction-only W/D+SQPnP/LM, canonical C2/물리축 규약을 사용했다. 참조는 기하 복원이며 독립 실측이 아니다. 자세 median은 산출 가능한 사례 기준이고, AUC는 실패를0으로 전체 분모에 포함한다. 종류는 외부에서 제공해 routing하며 미지 종류 인식은 검증하지 않았다.', '',
        '## セッション別 paired差分','', '| session | 対比 | PCK10 Δ%p | ADD Δ | 失った≤10 | 得た≤10 |', '|---|---|---:|---:|---:|---:|']
    for group,dd in result['contrasts'].items():
        if not group.startswith('SESSION_'):continue
        for name,d in dd.items():lines.append(f'| {group} | {name} | {d["PCK10_delta_pp"]:+.2f} | {d["ADDsym_delta"]:+.4f} | {d["lost_correct10"]} | {d["gained_correct10"]} |')
    lines += ['', '[확인] 모든 모델의 공통 매칭 집합에 대한2D 보조표와 축 정답/오답별 회전median은 RESULTS.json에 저장했다. GT 조건부 사후분석을 primary로 바꾸지 않는다. 축 정확도가50% 부근이면 median이 크게 바뀔 수 있어 각도 변화만으로 원리를 입증하지 않는다.', '',
        '## 과정·재현 자료','', '[PROTOCOL](PROTOCOL.json) · [타깃출처](TRAIN_TARGET_AUDIT.json) · [가림계획](AUGMENTATION_PLAN.jsonl) · [증강검증](AUGMENTATION_AUDIT.json) · [loss검사](LOSS_TEST.json) · [실학습parity](TRAINING_PARITY.json) · [전체결과](RESULTS.json)', '',
        '## 해석 제한과 종료','',
        '[확인] 같은감독예산의직접87점학생/raw-pseudo대조군/새세션검증은이번미실행. 신규성·최종우월성 미확정. 가림된좌표를감독하지만실제물리적으로보이던코너인지독립검증하지않았다. 인공가림난도와자연가림성과를분리한다. 합성가림학습은이번주비교밖이다.',
        '[미검증] 다음 행동은 이번에고정한마지막모델을새촬영세션에서독립평가하는계획1개만남긴다. 새학습이나threshold/seed/가림량sweep 자동실행없음. GT·기존모델·논문표수정없음. commit/push없음. STOP.', '',
        '## 비교 gallery','', 'TRAIN가림: 고정계획의applied occurrence에서seed20260922로무작위10개. 평가: 각난도의S2−S0개선/악화최대와고정random1개. 모두채점후설명용선택이며blind검토나대표평균이아니다. [스크롤갤러리](GALLERY.html).']
    C.save(C.DOC/'DIAGNOSTIC_SUMMARY.json',dict(train=probe_summary,source=source_summary))
    gallery(lines)
    C.save(C.DOC/'REPORT_KO.md','\n'.join(lines)+'\n')
    C.immutable()
    C.save(C.DOC/'AUDIT.json',dict(complete=True,fits=6,total_updates=1920,originals_unchanged=True,protocol=C.bind(C.DOC/'PROTOCOL.json'),
        reports=[C.bind(C.DOC/'REPORT_KO.md'),C.bind(C.DOC/'RESULTS.json')],no_model_promotion=True,no_push=True))
    print('REPORT_COMPLETE',C.DOC/'REPORT_KO.md',flush=True)


def gallery(lines):
    plans=[json.loads(x) for x in (C.DOC/'AUGMENTATION_PLAN.jsonl').read_text().splitlines()]
    applied=[p for p in plans if p['plan']['applied']];rng=np.random.default_rng(20260922)
    selected=[applied[i] for i in rng.choice(len(applied),10,replace=False)];images=[];selection=[]
    for i,p in enumerate(selected):
        z=np.load(C.ROOT/p['cache']['path']);q=z['keypoints'][0,:,:2]*640;mask=z['keypoints'][0,:,2]==2;panels=[]
        for arm in C.ARMS:
            im=A.apply(z['img'],p['plan'],arm).transpose(1,2,0)[:,:,::-1].copy();covered=A.cover(q,p['plan'][arm]) if arm!='S0' else np.zeros(9,bool)
            for k in np.flatnonzero(mask):cv2.circle(im,tuple(q[k].astype(int)),5,(0,0,255) if covered[k] else (0,255,0),-1)
            im=cv2.resize(im,(420,420));cv2.putText(im,arm,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2);panels.append(im)
        dest=C.DOC/'images'/f'train_{i:02d}.jpg';assert cv2.imwrite(str(dest),np.hstack(panels));images.append(dest)
        selection.append(dict(kind='TRAIN_RANDOM',material=p['material'],epoch=p['epoch'],slot=p['slot'],image=dest.name))
    records=C.read(C.H.P.DOC/'SPLIT.json')['evaluation'];metrics=C.read(C.RAW/'FRAME_METRICS.json')
    pred={'R0':C.read(C.H.P.RAW/'BASELINE_PREDICTIONS.json')['R0'], 'TEACHER':C.read(C.H.V.RAW/'PREDICTIONS.json')['predictions']['TYPE_REPLAY_PIPELINE'],'OLD_STUDENT':{}}
    for mat in C.MATERIALS:
        pred['OLD_STUDENT'].update({r['id']:r['prediction'] for r in C.read(C.H.RAW/f'EVAL_PREDICTIONS_{mat}.json')['records']})
    for arm in C.ARMS:
        pred[arm]={}
        for mat in C.MATERIALS:pred[arm].update(C.read(C.RAW/f'EVAL_{mat}_{arm}.json')['predictions'])
    gt=C.read(C.H.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json')
    for sev in C.H.P.SEVERITIES:
        rr=[r for r in records if r['severity']==sev]
        rr.sort(key=lambda r:metrics['S2'][r['id']]['frame_mean_px']-metrics['S0'][r['id']]['frame_mean_px'])
        for tag,r in [('improved',rr[0]),('harmed',rr[-1]),('random',rr[int(rng.integers(len(rr)))])]:
            fid=r['id'];im=cv2.imread(str(C.ROOT/r['image']['path']));h,w=im.shape[:2];scale=420/w;panels=[]
            for arm in pred:
                view=cv2.resize(im,(420,round(h*scale)));c=C.H.P.C.selected(pred[arm][fid])
                if c is not None:
                    q=np.array(c['keypoints_xy'])*scale
                    for a,b in A.EDGES:
                        if np.isfinite(q[[a,b]]).all():cv2.line(view,tuple(q[a].astype(int)),tuple(q[b].astype(int)),(0,220,255),1)
                for k in range(8):
                    if gt[fid]['valid'][k]:cv2.circle(view,tuple((np.array(gt[fid]['gt'][k])*scale).astype(int)),2,(0,255,0),-1)
                view=cv2.copyMakeBorder(view,35,0,0,0,cv2.BORDER_CONSTANT)
                cv2.putText(view,f'{arm} mean {metrics[arm][fid]["frame_mean_px"]:.2f}px',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);panels.append(view)
            # Two rows of3 to preserve readable pixels on normal desktop widths.
            dest=C.DOC/'images'/f'{sev}_{tag}.jpg';assert cv2.imwrite(str(dest),np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:])]))
            images.append(dest);selection.append(dict(kind='EVAL_POSTHOC',severity=sev,selection=tag,id=fid,image=dest.name))
    # Put outcome examples before augmentation examples, with explicit comparators.
    lines.extend(['', '### 결과 이미지 읽는 법', '',
        '각 그림의 윗줄은 **R0 원본 / 보정 Teacher / 이전 학생**, 아랫줄은 **S0 기본 학습 / S1 일반 가림 / S2 구조적 가림**이다. 초록 점은 GT 참조, 노란 선은 각 모델의 2D 키포인트 연결이며 PnP 재투영선이 아니다. GT는 채점·표시에만 사용했다.', '',
        '아래 개선·악화는 **S0 대비 S2의 유효 코너 평균 오차(px)** 기준이다. R0 대비 개선이나 6D 자세 개선과 동일한 뜻이 아니다. 검출·매칭 실패에는 이미지 대각선 벌점이 포함된다. 극단 사례를 사후 선택했으므로 전체 성능은 위 집계표로 판단해야 한다.'])
    captions={}
    severity_names={'CLEAN':'Clean', 'MODERATE_OCCLUSION':'중간 가림', 'SEVERE_OCCLUSION':'심한 가림'}
    for tag,title in [('improved','개선이 큰 사례'),('harmed','악화된 사례'),('random','고정 무작위 사례')]:
        lines.extend(['',f'### {title}',''])
        for item in selection:
            if item.get('selection')!=tag:continue
            fid=item['id']; values={arm:metrics[arm][fid]['frame_mean_px'] for arm in pred}
            delta=values['S2']-values['S0'];vs_r0=values['S2']-values['R0'];vs_s1=values['S2']-values['S1']
            caption=f"{severity_names[item['severity']]} · {fid} — S0 {values['S0']:.2f} → S2 {values['S2']:.2f}px (변화 {delta:+.2f}px)"
            captions[item['image']]=caption
            lines.extend(['',f'#### {caption}','',
                '| R0 | 보정 Teacher | 이전 학생 | S0 기본 | S1 일반 가림 | S2 구조적 가림 |',
                '|---:|---:|---:|---:|---:|---:|',
                '| '+' | '.join(f'{values[arm]:.2f}px' for arm in pred)+' |','',
                f"S2의 평균 오차는 R0 대비 {vs_r0:+.2f}px, S1 대비 {vs_s1:+.2f}px다(음수=개선). 이 수치는 해당 이미지의 2D 평균 오차이며 PCK10이나 6D 정확도가 아니다.",'',
                f"![{caption}](images/{item['image']})"])
    lines.extend(['','### 학습에 실제 사용한 가림 예시','',
        '왼쪽 S0 / 가운데 S1 / 오른쪽 S2. 빨간 점은 가림 영역에 포함된 감독점, 초록 점은 남아 있는 감독점이다. 결과 예측이 아니라 학습 입력·타깃 표시다.'])
    for item in selection:
        if item['kind']=='TRAIN_RANDOM':
            caption=f"{item['material']} · epoch {item['epoch']+1} · slot {item['slot']}"
            captions[item['image']]=caption
            lines.extend(['',caption,'',f"![{caption}](images/{item['image']})"])
    page='<!doctype html><meta charset="utf-8"><title>EASY HARD 결과</title><style>body{background:#14212b;color:white;font:16px system-ui;max-width:1400px;margin:25px auto}img{width:100%}figure{margin:30px 0}</style><h1>TRAIN S0/S1/S2와 자연평가 비교</h1><p>TRAIN: 빨강=가린감독점,초록=남은점. 평가: 노랑=예측,초록=정답참조. 800px는검출매칭실패벌점. 평가6모델은윗줄R0/TEACHER/OLD_STUDENT,아랫줄S0/S1/S2.</p>'
    for tag in ['improved','harmed','random',None]:
        for item in selection:
            if item.get('selection')==tag:
                page+=f'<figure><figcaption>{html.escape(captions[item["image"]])}</figcaption><img loading="lazy" src="images/{item["image"]}"></figure>'
    C.save(C.DOC/'GALLERY.html',page);C.save(C.DOC/'GALLERY_SELECTION.json',selection)


if __name__=='__main__':main()

"""Reviewable full-image result report; all crops are display-only."""
from pathlib import Path
import csv,hashlib,html,io,json,sys,types
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
import torch
from torch.nn import functional as F

HERE=Path(__file__).absolute().parent
NAMES={'FULL_CLIP':'전체 CLIP','FULL_CLIP_SURGERY_ARCH':'전체 CLIP 개선형','V264_HEATMAP_LOCAL':'이전 V264','v61':'이전 v61','CAL_LEGACY':'과거 실사학습 모델'}


def safe(p):
    p=Path(p);assert p.is_absolute()and not any(x in str(p).lower()for x in ('v112','cache','.pyc','..'))
    for a in reversed((p,*p.parents)):assert not a.is_symlink()
    return p
def load(n):return json.loads(safe(HERE/n).read_text())
def read(b):
    raw=safe(b['path']).read_bytes();assert len(raw)==b['size_bytes']and hashlib.sha256(raw).hexdigest()==b['sha256'];return raw
def font(n):return ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',n)
def write(n,s):
    with safe(HERE/n).open('x')as f:f.write(s)
def pct(x):return '—'if x is None or x=='' else f'{float(x)*100:.2f}'


def main():
    torch.set_num_threads(1)
    protocol=load('PROTOCOL_BEFORE_FIT.json');seal=load('CURRENT_PREDICTIONS_FROZEN.json')
    metrics=list(csv.DictReader(safe(HERE/'evaluation_v2/CLASSIFICATION.csv').open()))
    comparison=list(csv.DictReader(safe(HERE/'evaluation_v2/ALL1784_COMPARISON.csv').open()))
    frames=[json.loads(x)for x in read(seal['current_frame_inputs']).splitlines()]
    queries=[json.loads(x)for x in read(seal['current_query_inputs']).splitlines()]
    masks=np.load(io.BytesIO(read(seal['frame_masks'])),allow_pickle=False)
    b=protocol['current_input_helper'];m=types.ModuleType('v267_readout_fullrgb');m.__file__=b['path'];sys.modules[m.__name__]=m;exec(compile(read(b),m.__file__,'exec'),m.__dict__)
    # Two prespecified diagnostic points: previous cone miss and prior manual-D near-q case.
    selected=[370,200]
    write('ROOT_VISUAL_SELECTION_BEFORE_RGB.json',json.dumps(dict(indices=selected,
        reason='G370 previous cone-whiteband error; G200 previously discussed manualDinitialerror1.77px. Selected before these fullRGBreads, not representative frequency estimates.'),indent=2)+'\n')
    im=Image.new('RGB',(1560,1280),'#f7f9fc');d=ImageDraw.Draw(im)
    d.text((25,18),'V267 · 원래 합성 마스크 + 전체 이미지 CLIP 분할',font=font(34),fill='#14243c')
    d.text((25,72),'KP crop 없이 전체512 입력 · 합성768학습/235검증 · 실사223 평가만',font=font(23),fill='#43546e')
    primary=[r for r in metrics if r['group']=='ALL'and r['cohort']=='PRIMARY']
    for x,s in [(25,'방법'),(460,'보임 precision'),(750,'보임 recall'),(1010,'가림 주석 recall'),(1300,'67점 회복')]:d.text((x,115),s,font=font(22),fill='#14243c')
    for j,arm in enumerate(('V264_HEATMAP_LOCAL','FULL_CLIP','FULL_CLIP_SURGERY_ARCH')):
        r=next(r for r in primary if r['model']==arm);y=155+42*j
        d.text((25,y),NAMES[arm],font=font(23),fill='#14243c')
        for x,key in [(460,'D_precision'),(750,'D_recall'),(1010,'E_recall')]:d.text((x,y),pct(r[key])+'%',font=font(24),fill='#14243c')
        d.text((1300,y),{'V264_HEATMAP_LOCAL':'—','FULL_CLIP':'59 / 67','FULL_CLIP_SURGERY_ARCH':'61 / 67'}[arm],font=font(24),fill='#14243c')
    d.text((25,292),'비대상 마스크를 E로 대응한 비교입니다. 배경·좌표 오류와 외부 가림 원인은 별개입니다.',font=font(23),fill='#a24c16')
    for j,index in enumerate(selected):
        q=queries[index];frame=frames[q['frame_index']];rgb=m.read_rgb(frame);h,w=rgb.shape[:2];xy=np.asarray(q['original_q0'])
        top=350+j*413
        title='콘 가림 회복 · 흰 띠를 팔레트에서 제외'if index==370 else'보이는 모서리 놓침 · 초기 오차1.77px'
        value=float(comparison[index]['FULL_CLIP_SURGERY_ARCH_p_target'])
        d.text((25,top),f'G{index} · {title} · P(target)={value:.4f}',font=font(25),fill='#14243c')
        source=Image.fromarray(rgb)
        raw=masks['FULL_CLIP_SURGERY_ARCH_logit'][q['frame_index']]
        prob=torch.as_tensor(raw).sigmoid().numpy()
        resized=F.interpolate(torch.as_tensor(prob).reshape(1,1,32,32),size=(h,w),mode='bilinear',align_corners=False)[0,0].numpy()
        pm=np.zeros((h,w,3),np.uint8);pm[:]=[15,30,21];pm[:,:,1]=np.clip(30+225*resized,0,255).astype(np.uint8)
        mask=Image.fromarray(pm)
        side=q['original_crop_side'];box=(xy[0]-side/2,xy[1]-side/2,xy[0]+side/2,xy[1]+side/2)
        zoom=source.crop(box)
        for x,panel,label in [(25,source,'전체 RGB · 모델 입력 범위'),(540,mask,'전체 팔레트 확률 · 고정0–1'),(1055,zoom,'초기 q 확대 · 표시 전용')]:
            d.text((x,top+42),label,font=font(21),fill='#43546e')
            canvas=Image.new('RGB',(480,320),'#1b2534')
            shown=ImageOps.contain(panel,(480,320),Image.Resampling.BILINEAR);offset=((480-shown.width)//2,(320-shown.height)//2);canvas.paste(shown,offset)
            draw=ImageDraw.Draw(canvas)
            if x!=1055:
                for native,point in enumerate(frame['original_query_xy8']):
                    if point is None or not frame['original_query_available8'][native]:continue
                    px=offset[0]+(point[0]+.5)*shown.width/w-.5;py=offset[1]+(point[1]+.5)*shown.height/h-.5
                    col='#fff124'if native==q['native_slot']else'#1cddff';r=4 if native==q['native_slot']else 2
                    draw.ellipse((px-r,py-r,px+r,py+r),outline=col,width=2)
            else:
                px=offset[0]+shown.width/2;py=offset[1]+shown.height/2
                draw.line((px-7,py,px+7,py),fill='#fff124',width=2);draw.line((px,py-7,px,py+7),fill='#fff124',width=2)
            im.paste(canvas,(x,top+76))
    d.text((25,1203),'전체 분할은 진전 · 보이는 점552/1147을 비대상으로 읽는 문제는 남음 · 새 모델 채택 없음',font=font(24),fill='#a24c16')
    im.save(safe(HERE/'RESULT_OVERVIEW_KO.png'))
    tables=[]
    for group,title in [('ALL','전체'),('WOOD_PRIMARY105','목재105장'),('SQUARE118','정사각형118장')]:
        body=[]
        for r in metrics:
            if r['group']!=group or r['cohort']!='PRIMARY':continue
            vals=[NAMES[r['model']],'/'.join(r[k]for k in ('D_D','D_E','D_U')),'/'.join(r[k]for k in ('E_D','E_E','E_U')),*[pct(r[k])for k in ('D_precision','D_recall','E_precision','E_recall','E_F1','U_rate')]]
            body.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>'for x in vals)+'</tr>')
        tables.append('<h2>'+title+'</h2><table><tr>'+''.join('<th>'+x+'</th>'for x in ['방식','D→D/E/U','E→D/E/U','D P%','D R%','E P%','E R%','E F1%','U%'])+'</tr>'+''.join(body)+'</table>')
    syn=load('syn_training_v1/SYN_DEV_RESULTS.json')['results']
    summary=dict(completed=True,problem_solved=False,new_model_adopted=False,full_image_scope=True,
        original_user_SYN=True,SYN_FIT_DEV=[768,235],new_fits=2,new_optimizer_updates=2000,
        current_fit_or_PnP_or_postcurrent_threshold_change=0,PRIMARY_N=1237,D=1147,E=90,primary_metrics=primary,
        CS_prior67_recovered=61,CS_prior67_remain=6,CS_new_external_misses=0,CS_visible_false_non_target=552,
        CS_predictedD=601,CS_D_true=595,CS_E_selected_D=6,CS_manualD_initial_LE3_total=376,CS_manualD_initial_LE3_false_non_target=182,
        prediction_semantics='MASK_PRESENCE_PROXY_E_NOT_CAUSE',current223_repeated_development_evaluation=True,
        official_architecture_surgery=True,text_feature_surgery=False,extra_automatic_fits=False)
    write('FINAL_RESULT_SUMMARY.json',json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V267 전체 이미지 CLIP 팔레트 분할</title><style>body{font-family:sans-serif;max-width:1560px;margin:30px auto;padding:0 24px;line-height:1.7;color:#17253c;background:#f5f7fb}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d4deeb;padding:9px}.note{padding:18px;background:#fff0ce}a{color:#165db8}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}</style>
<h1>사용자 원래 합성 마스크로 전체 이미지 CLIP 분할 비교 완료</h1>
<p class="note"><b>전체 영상에서 콘을 제외하는 분할은 진전했습니다.</b> CLIP 개선형이 V264의 기존 가림 누락67점 중61점을 E로 바꾸고6점은 남겼습니다. 맞히던 E23점의 새 누락은0입니다. 다만 보이는 점1147개 중552개도 비대상으로 읽어 보임 recall은51.87%입니다. 완성된 두 상태 판별기로 채택하지 않았습니다.</p>
<p><b>모델 입력은 전체 RGB입니다.</b> 원래 합성1003장(학습768/검증235)의 M4만 분할 정답으로 사용했습니다. 내부 진단용440장과 M0는 새 학습에 쓰지 않았습니다. 전체 영상을512×512로 변환하고 분할 지도를 만든 뒤에만 기존 초기 키포인트를 읽었습니다. 아래 확대 crop은 표시 전용입니다.</p>
<p><a href="evaluation_v2/INDEX_KO.html">전체 영상·확률 지도·마스크·확대104사례</a> · <a href="RUN_GUIDE_KO.md">재현 안내</a> · <a href="PROTOCOL_BEFORE_FIT.json">학습 전 동결 조건</a> · <a href="syn_mask_diagnostic_v1/INDEX_KO.html">원래 합성 RGB/M4/예측 대조</a> · <a href="mask_boundary_diagnostic_v1/INDEX_KO.html">보임 누락의 경계 진단</a></p>
<img src="RESULT_OVERVIEW_KO.png">
<p class="note"><b>출력의 의미:</b> 새 모델이 배운 것은 target/non-target입니다. 현재 점의 비대상을 E에 대응해 기존 human D/E와 비교한 <b>MASK_PRESENCE_PROXY</b>입니다. 비대상은 외부 가림뿐 아니라 배경·초기 좌표 오차·다른 native 대응도 포함할 수 있습니다. 외부 가림 원인을 학습한 별도 분류기가 생긴 것은 아닙니다. 실사 pixel GT가 없으므로 실사 segmentation IoU를 주장하지 않습니다.</p>
<p>표의 행별 열은 D/E/U, 비율은%입니다. D=보임, E=외부가림 주석에 대한 대응, U=입력 불가. 자기 가림은 E로 합치지 않았습니다. PRIMARY1237=D1147/E90에서 새 모델U0이며, 전체1784점의 기존 입력불가45U는 보존했습니다. 과거 실사학습 모델은 historical CAL 학습 이력이 있어 합성 전용 신모델로 부르지 않습니다.</p>'''+''.join(tables)+'''
<p><a href="evaluation_v2/CLASSIFICATION.csv">평가군60행 CSV</a> · <a href="evaluation_v2/SELF_RESPONSES.csv">자기 가림 응답</a> · <a href="evaluation_v2/ALL1784_COMPARISON.csv">1784점 결과</a> · <a href="evaluation_v2/EXTERNAL90_TRANSITIONS.csv">외부 가림90점 전이</a></p>
<h2>무엇이 됐고 무엇이 남았는가</h2><p>G370의 콘 흰 띠와 G908의 큰 콘 몸체를 마스크에서 제외합니다. G1515도 이전 crop-분할 모델의 새 누락에서 E로 돌아왔습니다. G370에는 콘 받침 일부를 팔레트로 표시하는 잔여 오류가 있습니다. G996은 사용자 시각 판단과 기존 E 주석이 다른 사례라 원 주석을 유지하고 명백한 성공 근거로 단정하지 않습니다.</p>
<p>개선형이 D로 선택한601점 중595점은 human D,6점은 E여서 D precision99.00%입니다. 그러나 실제 D1147점 중595점만 남습니다. 이는 같은 동결0.5 기준의 결과이며 새 confidence 게이트를 적합하거나 채택한 것이 아닙니다. 가림 정밀도13.21%와 보임 recall51.87%를 함께 봐야 합니다.</p>
<p>수동 보임점 중 초기 좌표 오차≤3px인376점에서도182점이 비대상입니다. 위치 오차만으로 전체 실패를 설명할 수 없습니다.32×32 분할의 경계 정밀도, 팔레트 영역과 의미상 보이는 코너의 차이, 일부 분할 전이 실패를 실제 영상과 경계 거리로 나눠 진단합니다. 실사 pixel GT가 없으므로 경계 가까움 자체를 원인 확정으로 쓰지 않습니다.</p>
<div class="grid"><a href="evaluation_v2/CASE_G0370.png"><img src="evaluation_v2/CASE_G0370.png"></a><a href="evaluation_v2/CASE_G0908.png"><img src="evaluation_v2/CASE_G0908.png"></a></div>
<h2>학습 방법과 비교 범위</h2><p>일반 CLIP ViT-B/16과 공식 CLIP Surgery의 영상 구조 변경(V–V attention/dual path)을 동결하고 같은26721개 파라미터 mask head만 학습했습니다. 두 방식은 같은 일반 사전학습 가중치·초기값·배치·1000업데이트를 사용합니다. 개선형은 <b>architecture surgery + M4 지도 학습</b>이며 텍스트 feature surgery나 전체 zero-shot CLIP Surgery를 실행한 것은 아닙니다. <a href="https://github.com/xmed-lab/CLIP_Surgery">공식 방법·코드</a></p>
<p>M4>127을 전체512로 변환한 뒤 patch16 평균으로32×32 soft target을 만듭니다. 손실은 학습 마스크에서만 정한 class-balanced BCE+soft Dice입니다. q/native/pose/기존 가림 점수는 head 입력에 없습니다. 체크포인트는 final1000 둘을 먼저 동결했고 DEV/실사 성능으로 바꾸지 않았습니다.</p>
<p>전체 일반 CLIP↔개선형의 차이는 같은 조건에서 영상 구조의 차이입니다. V264/v61 등과의 차이에는 입력 범위·기반 특징·감독·학습 자료도 달라지므로 모든 개선을 전체 영상 하나의 인과 효과로 단정하지 않습니다. 실사223장은 반복 개발 평가이며 새 학습·통계·문턱·체크포인트 적합에 쓰지 않았습니다. 독립 블라인드 검증은 아닙니다.</p>
<h2>합성 검증과 실제 정답</h2><p>다음은32×32 full-frame fractional target 기준의 합성 DEV235장 결과입니다. 원해상도 M4 기준 및 원본 RGB/GT/예측은 별도 진단에 제공합니다.</p><table><tr><th>방식</th><th>micro IoU%</th><th>평균 영상 IoU%</th><th>micro Dice%</th></tr>'''
    for arm in ('PLAIN_CLIP','CLIP_SURGERY_ARCH'):
        r=syn[arm]['fractional_target'];doc+='<tr>'+''.join('<td>'+str(v)+'</td>'for v in [arm,pct(r['micro_IoU']),pct(r['mean_frame_IoU']),pct(r['micro_Dice'])])+'</tr>'
    doc+='''</table><p><a href="syn_mask_diagnostic_v1/INDEX_KO.html">원래 합성 전체 RGB와 M4 정답 비교</a> · <a href="data_audit/AUDIT_SUMMARY.json">원본1003 데이터 감사</a> · <a href="fullmask_targets_v1/COMPLETE.json">마스크 변환·학습 분할</a></p>
<h2>검산·출처·한계</h2><p><a href="FULLFRAME_SCOPE_INDEPENDENT_REVIEW.json">전체 영상 범위 독립 검토</a> · <a href="MASK_TRAINING_INDEPENDENT_REVIEW.json">마스크 학습 독립 검토</a> · <a href="ROOT_INDEPENDENT_METRICS_CHECK.json">60행 독립 검산</a> · <a href="fullframe_replay_v1/COMPLETE.json">CPU mask head/1784점 재현</a> · <a href="METRICS_REPLAY_BYTECHECK.json">지표6파일 바이트 재현</a> · <a href="POSTSEAL_EVALUATOR_IMPLEMENTATION_FIX.json">HTML 구현 오류 수정 기록</a></p>
<p>첫 입력 감사에서 JPEG decoder 차이를 원래 MLPY 환경으로 해결했고, 평가 HTML append 오류는 표시 코드만 수정했습니다. 원본 실패 기록을 보존했으며 모델·수치·문턱·주석은 바꾸지 않았습니다. 새 head fit2회/총2000업데이트, backbone 학습·실사 적합·새 PnP·사후 threshold 조정0입니다. 이 고정 비교 뒤 추가 fit은 진행하지 않았습니다.</p></html>'''
    write('ROOT_READOUT_KO.html',doc)
    guide='''# V267 재현 안내

원래 사용자 합성 RGB1003/M4로 학습한 전체 이미지 CLIP mask head 두 모델입니다. 전체 영상512를 입력하고, 예측된32×32 지도에서 원래 query를 나중에 읽습니다. 모델 입력에 초기 KP crop은 없습니다. 확대 그림은 표시 전용입니다.

## 공개 패키지만으로 검산

Python3.10, NumPy1.26.4, PyTorch2.1.1, Pillow12.2 환경에서 확인했습니다. 지표 replay는 Pillow도 사용합니다. 아래 새 출력 폴더 이름은 이미 존재하면 안 됩니다.

```bash
python -B replay_fullframe_v267.py --out replay_cpu_check
python -B evaluate_fullframe_v267.py --replay-metrics --inputs evaluation_v2/compact_inputs --out replay_metrics_check
```

첫 명령은 고정5개 전체 영상의 mask-head forward/arm과 전체1784개의 raw 지도 lookup을 재현합니다. frozen CLIP backbone 자체를 다시 실행하는 것은 아닙니다. 둘째 명령은5방법×3집단×4평가군=60행의 CM/P/R/F1/U와 외부 가림 전이를 재계산합니다. 기존 지표6출력은 바이트 일치했습니다.

`ROOT_READOUT_KO.html`은 폴더 구조를 유지해 브라우저에서 여세요. GitHub에서는 README와 PNG를 직접 볼 수 있습니다.

## 전체 학습 재실행

원래 RGB/M4 데이터와 각각약1.05GB의 SYN frozen 특징 두 개,350.8MB 일반 CLIP 가중치는 Git에 중복 복사하지 않았습니다. 원본 경로·SHA·학습 분할과 공식 소스 commit은 JSON에 있습니다. 저장소만으로 원래 데이터의 전체 학습을 즉시 재실행할 수 있다고 주장하지 않습니다. 위 compact 검산은 원본 데이터와 backbone 가중치 없이 가능합니다.

공식 소스: https://github.com/xmed-lab/CLIP_Surgery/tree/d4696d47f49cfe70f49140afe5eb94f94c5f59bc

`fullframe_clip_v267.py`의 OFFICIAL_CODE와 WEIGHTS_URL/SHA를 확인하세요. official_source/clip/clip_model.py 및 clip_surgery_model.py, generic pretrained/ViT-B-16.pt를 해당 버전으로 준비해야 backbone 추론을 다시 실행할 수 있습니다. 이 실험은 공식 image architecture surgery를 사용했으며 text feature surgery/zero-shot pipeline은 실행하지 않았습니다.

실행 순서: 원본 데이터 감사→M4전체 soft-target→두 frozen encoder 특징→프로토콜 동결→두1000업데이트 fit→두 final 동결→SYNDEV scoring→실사223 전체 mask/1784 lookup 저장·동결→late GT join→평가/시각화. 원 경로에 묶인 protocol은 수정하지 말고 새 환경에서 명시적 재바인딩과 새 output directory로 기록하세요.

## 해석

M4는 target/non-target입니다. non-target를 E로 대응한 점 단위 수치는 human D/E와의 의미 전이 평가이며, 배경과 실제 occluder를 구분하는 별도 E 정답으로 학습하지 않았습니다. 낮은 mask 확률만으로 외부 가림 원인을 확정하면 안 됩니다. 실사 pixel GT는 없으므로 real IoU를 주장하지 않습니다.

같은0.5 기준의 개선형 D precision99.0%는595D/601선택점이며, 보임 recall51.87%와 E precision13.21%도 함께 봐야 합니다. 실사223은 반복 개발 평가이고 블라인드 검증이 아닙니다. 원래45 입력불가U와 자기 가림 별도 집계를 유지했습니다.
'''
    write('RUN_GUIDE_KO.md',guide)
    print(json.dumps(dict(completed=True,root_report=str(HERE/'ROOT_READOUT_KO.html'))))


if __name__=='__main__':main()

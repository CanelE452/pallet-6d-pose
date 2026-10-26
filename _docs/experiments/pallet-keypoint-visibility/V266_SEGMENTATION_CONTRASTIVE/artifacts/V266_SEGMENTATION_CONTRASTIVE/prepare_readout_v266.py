"""Build the reviewable V266 root report from frozen results and actual crops."""
from pathlib import Path
import csv
import html
import json
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).absolute().parent
NAMES={'CURRENT_V264':'현재 V264','SEGMENTATION':'분할 추가','SEGMENTATION_CONTRAST':'분할+대조 학습'}


def safe(value):
    p=Path(value)
    assert p.is_absolute() and not any(x in str(p).lower() for x in ('v112','cache','.pyc','..'))
    for a in reversed((p,*p.parents)):assert not a.is_symlink(),str(a)
    return p


def text(path):return safe(HERE/path).read_text()
def font(size):return ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size)
def write(path,value):
    with safe(HERE/path).open('x') as f:f.write(value)
def pct(x):return f'{float(x)*100:.2f}'


def main():
    metric=list(csv.DictReader(text('evaluation_v1/CLASSIFICATION.csv').splitlines()))
    syn=json.loads(text('syn_training_v1/SYN_DEV_RESULTS.json'))['results']
    primary=[r for r in metric if r['group']=='ALL'and r['cohort']=='PRIMARY']
    im=Image.new('RGB',(1560,1500),'#f7f9fc');d=ImageDraw.Draw(im)
    d.text((28,20),'V266 · 영역 분할 / 분할+대조 학습 비교',font=font(36),fill='#14243c')
    d.text((28,77),'합성만 학습 · 실사223장은 평가만 · 동일 주평가 D1,147 / E90점',font=font(24),fill='#43546e')
    d.text((28,119),'방식',font=font(23),fill='#14243c')
    for x,label in [(460,'보임 recall'),(705,'가림 recall'),(950,'가림 precision'),(1240,'기존누락 회복')]:d.text((x,119),label,font=font(23),fill='#14243c')
    for j,r in enumerate(primary):
        y=161+j*44;d.text((28,y),NAMES[r['model']],font=font(24),fill='#14243c')
        for x,k in [(460,'D_recall'),(705,'E_recall'),(950,'E_precision')]:d.text((x,y),pct(r[k])+'%',font=font(25),fill='#14243c')
        d.text((1240,y),'—'if j==0 else'0 / 67점',font=font(25),fill='#a12b24')
    d.text((28,303),'초록=모델이 예측한 대상 팔레트 · 주황=외부 물체 · 파랑=배경 (실사 분할 정답 아님)',font=font(23),fill='#43546e')
    # Preserve actual original/overlay panels, display crop only; no pixels regenerated.
    rows=[(0,'보임 복구 · 기존 E → 두 새 방식 D','주석 D. 보임 오판 일부를 줄임.'),
          (370,'기존 가림 누락 유지 · D → D / D','콘의 흰 띠를 대상 팔레트로 분할함.'),
          (1515,'새 가림 누락 · E → D / D','콘 받침을 팔레트로 분할함.')]
    for j,(index,title,caption) in enumerate(rows):
        top=360+j*354;d.text((28,top),f'G{index} · '+title,font=font(25),fill='#14243c')
        with Image.open(safe(HERE/f'evaluation_v1/CASE_G{index:04d}.png')) as src:
            for x,box,label in [(28,(24,226,376,578),'실제 crop'),(334,(820,226,1172,578),'분할 추가'),(640,(422,652,774,1004),'분할+대조')]:
                patch=src.crop(box).resize((258,258),Image.Resampling.LANCZOS)
                im.paste(patch,(x,top+68));d.text((x,top+40),label,font=font(18),fill='#43546e')
        d.text((935,top+75),caption,font=font(22),fill='#14243c')
        if index==370:d.text((935,top+113),'P(가림): 45.7 → 5.4 / 4.9%',font=font(22),fill='#a12b24')
        if index==1515:d.text((935,top+113),'P(가림): 67.2 → 34.6 / 34.8%',font=font(22),fill='#a12b24')
    d.text((28,1450),'이번 구성은 외부 가림 개선 실패 · 새 모델 채택 없음 · PnP/실사 적합/사후 문턱 조정 없음',font=font(23),fill='#a12b24')
    im.save(safe(HERE/'RESULT_OVERVIEW_KO.png'))
    tableheaders=['방식','D→D/E/U','E→D/E/U','D precision','D recall','E precision','E recall','E F1','보류율']
    tables=[]
    for group,title in [('ALL','전체'),('WOOD_PRIMARY105','목재105장'),('SQUARE118','정사각형118장')]:
        rows2=[]
        for r in metric:
            if r['group']!=group or r['cohort']!='PRIMARY':continue
            v=[NAMES[r['model']],'/'.join(r[k]for k in ['D_D','D_E','D_U']),'/'.join(r[k]for k in ['E_D','E_E','E_U']),*[pct(r[k])for k in ['D_precision','D_recall','E_precision','E_recall','E_F1','U_rate']]]
            rows2.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>'for x in v)+'</tr>')
        tables.append('<h2>'+title+'</h2><table><tr>'+''.join('<th>'+h+'</th>'for h in tableheaders)+'</tr>'+''.join(rows2)+'</table>')
    synrows=[]
    for arm,label in [('CURRENT_HEATMAP_LOCAL','현재 V264'),('SEGMENTATION','분할 추가'),('SEGMENTATION_CONTRAST','분할+대조')]:
        s=syn[arm];owner=s.get('segmentation',{}).get('per_class',{}).get('FOREIGN_OBJECT',{}).get('IoU')
        synrows.append('<tr>'+''.join('<td>'+str(v)+'</td>'for v in [label,s['CM_rows_D_E_cols_D_E_U'],f"{s['D']['recall']*100:.2f}",f"{s['E']['recall']*100:.2f}",'—'if owner is None else f'{owner*100:.2f}'])+'</tr>')
    body='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V266 분할·대조 학습 비교</title>
<style>body{font-family:sans-serif;line-height:1.7;max-width:1500px;margin:30px auto;padding:0 25px;background:#f5f7fb;color:#16243a}table{border-collapse:collapse;background:white;width:100%;font-size:15px}th,td{border:1px solid #d7dfed;padding:8px}img{max-width:100%;height:auto}a{color:#135cb4}.note{background:#fff0ce;padding:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}</style>
<h1>현재 방식 / 영역 분할 / 분할+대조 학습 비교 완료</h1>
<p class="note"><b>이번 구성은 외부 가림 개선에 실패했습니다.</b> 보임 recall은 올랐지만 기존 누락67점은 한 점도 회복하지 못했고, 맞히던 가림3점도 추가로 놓쳤습니다. 새 모델을 채택하지 않았으며 이 결과 뒤 추가 학습·threshold 변경은 없습니다.</p>
<p>초기 query·crop·동결 DINO 특징·heatmap 위치는 유지했습니다. 분할 두 방식은 동일23,204개 추가 파라미터, 같은 초기값·FIT 표본·1,000업데이트입니다. 차이는 픽셀 대조 손실0.1 추가입니다. 현재 V264는 저장된 기존 모델이며, 분할 추가와의 차이에는 추가 용량·감독·업데이트도 포함됩니다. CLIP Surgery 자체를 실행한 실험은 아닙니다.</p>
<p><a href="PROTOCOL_BEFORE_FIT.json">학습 전 조건</a> · <a href="RUN_GUIDE_KO.md">재현 안내</a> · <a href="evaluation_v1/INDEX_KO.html">실제97사례 / 외부 가림90점 전체</a> · <a href="syn_visual_v1/INDEX_KO.html">합성 RGB·분할 정답·예측</a></p>
<img src="RESULT_OVERVIEW_KO.png">
<p>혼동행렬 각 행의 열은 D/E/U입니다. D=보임, E=외부 가림, U=보류. 비율은 %이고 recall은 U를 분모에 포함합니다. 전체1784 query중 입력 불가45점은 모든 방식에서 그대로 U입니다. 아래 PRIMARY1237점에서는 U=0입니다. 자기 가림은 E로 합치지 않았습니다.</p>'''
    body+='\n'.join(tables)
    body+='''<p><a href="evaluation_v1/CLASSIFICATION.csv">36개 평가군별 전체 표</a> · <a href="evaluation_v1/SELF_RESPONSES.csv">자기 가림 별도 응답</a> · <a href="evaluation_v1/ALL1784_COMPARISON.csv">1784점 행별 결과</a> · <a href="evaluation_v1/EXTERNAL90_TRANSITIONS.csv">가림90점 전이</a></p>
<h2>기존 누락67점과 새 누락</h2><p>분할 / 분할+대조 모두 회복0·잔존67·새 누락3점입니다. 새 누락은 정사각형 G1515·G1536·G1677입니다. 전체1784개에서도 새 E 발견은 없고 변경은 기존 E→D51/50점뿐입니다. 전체 정확도 상승만으로 외부 가림 개선이라고 해석할 수 없습니다.</p>
<h2>무엇이 안 됐는가</h2><p><b>① 분할 전이:</b> G370에서 콘의 몸체 일부는 외부 물체로 잡지만 흰 띠는 팔레트로 나눕니다. G908·G1515에서도 콘의 받침·경계가 팔레트와 이어집니다. 한 콘의 표면을 같은 외부 물체로 일관되게 묶지 못했습니다. 실사에는 dense pixel 정답이 없어 분할 IoU를 주장하지 않습니다. G996은 사용자 판단과 기존 E 주석이 다른 사례로 유지하며 명백한 실패 근거로 단정하지 않습니다.</p>
<p><b>② 분할→native 가림 판정:</b> 합성 DEV의 첫 가림 누락 source1928은 q주변 외부 물체 확률이 약75/73%인데도 두 방식이 모두 D입니다. 정답 분할에 가깝더라도 그 정보를 가림 판단으로 연결하는 처리도 실패할 수 있습니다. G370에서도 owner 정보를 강제로 전부 외부 물체로 바꾸는 사후 진단은 E를 회복하지 못했습니다. 이 가상 descriptor 조작은 배포 규칙·성능 개선으로 채택하지 않았습니다.</p>
<p><b>해석:</b> 합성 학습에서는 기존 분류기만으로 BCE가 매우 작아, 추가 분류기가 기존 점수를 강하게 따르는 경로가 남아 있습니다. 위 관측은 합성→실사 분할 전이와 최종 판정 양쪽의 병목을 지지합니다. 다른 분할 방법 전체의 불가능성을 입증하지는 않습니다. 후속 검증을 한다면 합성의 콘 전체·흰 띠·받침 및 팔레트/가리는 물체의 역할 교대와 같은 점의 가림 전후 쌍을 충분히 구성하고, 분할 정보가 실제 결정을 바꾸는지부터 확인해야 합니다. 이번 비교에서 추가 fit은 두 번으로 끝냈습니다.</p>
<div class="grid"><a href="evaluation_v1/CASE_G0370.png"><img src="evaluation_v1/CASE_G0370.png"></a><a href="evaluation_v1/CASE_G1515.png"><img src="evaluation_v1/CASE_G1515.png"></a></div>
<h2>합성 학습·검증 구성과 결과</h2><p>110개 base / 4개 scene variant / 두 초기 query 구성으로7040행입니다. FIT86개 base5504행, DEV24개1536행이며 base·원영상 경로·원영상 SHA 중복0입니다. 실제 D/E 학습 가능점은 FIT D4572/E188, DEV D1288/E56입니다. D/E를16개씩 뽑는 동일 학습 순서를 사용했습니다. 자기 가림·불명확 라벨은 E로 바꾸지 않았습니다.</p>
<p>렌더러의 part_id0…18=대상,999=외부 물체,-1=배경을 사용했습니다. 영상 밖·모호함은 무시합니다. 외부 물체가 crop에 있다는 이유만으로 키포인트 E를 부여하지 않고, 기존 native D/E 정답을 유지했습니다. 정답 지도는 원래 crop과 같은96샘플→224resize→14평균으로16×16에 맞췄습니다. 실제 학습 FIT에서 외부 물체의 알려진 토큰 질량은1.31%이며, 합성 FIT만으로 역제곱근 클래스 가중치를 고정했습니다.</p>
<table><tr><th>방식</th><th>SYNDEV CM(D/E/U)</th><th>D recall%</th><th>E recall%</th><th>외부 물체 분할 IoU%</th></tr>'''+''.join(synrows)+'''</table>
<p>분할 IoU는 합성16×16 토큰의 fractional area 기준입니다. 실사 성능이 아닙니다. 대조 학습으로 합성 외부 물체 IoU는 소폭 올랐지만 실사 가림 회복으로 연결되지 않았습니다.</p>
<a href="syn_visual_v1/INDEX_KO.html"><img src="syn_visual_v1/SYN_RGB_GT_PRED_FIXED6_KO.png"></a>
<h2>검산과 한계</h2><p>실사223장의 이미지·주석·파생 통계로 새 모델·문턱·체크포인트를 적합하지 않았습니다. 두 final을 먼저 동결하고 모든1784 예측을 저장한 뒤 주석을 결합했습니다. 다만 이223장은 여러 차례 본 개발 평가 자료로, 독립 블라인드 검증은 아닙니다. 원래 주석과 native 대응·위치 출처를 유지했습니다. 숨은 E의 주석 좌표는 직접 보이는 위치 정답과 다릅니다.</p>
<p><a href="ROOT_INDEPENDENT_METRICS_CHECK.json">독립36행 수치 검산</a> · <a href="DATA_INDEPENDENT_REVIEW.json">데이터 독립 검토</a> · <a href="TRAINING_INDEPENDENT_REVIEW.json">학습 독립 검토</a> · <a href="prediction_replay_v1/COMPLETE.json">CPU 예측 재현</a> · <a href="METRICS_REPLAY_BYTECHECK.json">지표 바이트 재현</a> · <a href="readout_diagnostic_v1/RESULTS.json">분할 정보 사용 진단</a> · <a href="data_v1/PREFLIGHT_NUMERIC_FAILURE.json">학습 전 수치 검사 수정 기록</a></p>
<p>추가 학습2회/총2000업데이트. 새 backbone·위치 모델 학습·PnP·실사 적합·사후 문턱 조정0입니다. CPU에서 전체1784개 분류 출력을 재현했고, 고정32개 입력의 전체 dense forward도 검산했습니다. 원래1.38GB 합성 특징과350MB 실사 특징은 원본 SHA·경로를 기록하고 Git에서 제외하며, compact 출력/검산용 입력을 제공합니다.</p></html>'''
    write('ROOT_READOUT_KO.html',body)
    guide='''# V266 실행·재현 안내

현재 V264 / 영역 분할 추가 / 분할+대조 학습을 비교했습니다. 학습은 합성뿐입니다. 결과는 외부 가림 개선 실패이며 새 모델을 채택하지 않았습니다.

## 다른 컴퓨터에서 결과 검산

이 폴더를 복사한 뒤 Python3, NumPy, PyTorch CPU를 사용합니다. 실제 확인 환경은 Python3.10.20, NumPy1.26.4, PyTorch2.1.1+cu118입니다. 지표 검산에는 Pillow도 필요합니다. 다음 명령은 학습·PnP·대형 데이터·원본 실사 접근 없이 동작합니다. 결과 디렉토리 이름은 새 이름이어야 합니다.

```bash
python -B replay_predictions_v266.py --out replay_cpu_check
python -B evaluate_v266.py --replay-metrics --inputs evaluation_v1/compact_inputs --out replay_metrics_check
```

첫 명령은 전체1784 분류를22D 저장 descriptor와 체크포인트로 재현하고, 고정32개 입력의 dense branch 전체 forward도 재현합니다. 전체1784 dense map의 SHA를 검산하지만, 전체 dense forward를 다시 실행하는 것은 아닙니다. 둘째 명령은36개 지표행, 자기 가림 응답, 가림90점 전이 등을 재계산합니다. `METRICS_REPLAY_BYTECHECK.json`에 주요6출력의 바이트 일치를 기록했습니다.

`ROOT_READOUT_KO.html`과 하위 HTML은 폴더 구조를 유지해서 브라우저로 열면 이미지·표를 함께 볼 수 있습니다. GitHub에서는 패키지 README와 PNG를 바로 볼 수 있습니다.

## 학습 전체 재실행에 필요한 외부 입력

`PROTOCOL_BEFORE_FIT.json`, `data_v1/SOURCE_BUFFER_BINDINGS.json`, `DATA_INDEPENDENT_REVIEW.json`의 경로와 SHA가 원본 근거입니다. 대형 합성 DINO 특징1,384,120,448바이트와 실사 특징 약350MB는 Git에 중복 복사하지 않았습니다. 따라서 저장소만으로 원래 특징 추출과 전체 학습을 즉시 재실행할 수 있다고 주장하지 않습니다. 원본 렌더 buffer·동결 모델·특징이 있어야 가능합니다. 공개 패키지만으로 위의 분류/지표 검산은 가능합니다.

원 실행 순서는 `build_dense_labels_v266.py` → `prepare_training_packet_v266.py` → 조건 동결 → `train_syn_v266.py` → 두 final 동결 → `infer_current_v266.py` →1784예측 동결 → `evaluate_v266.py`입니다. 모든 출력은 새 폴더에만 기록됩니다. 학습·추론 스크립트는 원 경로/SHA로 입력을 검증하므로 새 환경에서는 경로를 명시적으로 재바인딩하고 새 실행으로 기록해야 합니다. 기존 봉인 protocol을 수정하지 마세요.

두 새 모델은 각1000업데이트, seed17, 동일초기값/학습행, AdamW lr.001/wd.0001, D16/E16 균형 표본입니다. 손실은 BCE+가중분할CE, 대조 방식만 +.1 pixel contrast(tau.1)입니다. 기본V264는 고정하고 새23,204개 파라미터만 학습합니다. 보류는 기존 입력불가만 유지하며 성능에 맞춘 confidence 문턱이 아닙니다.

## 평가 해석

실사223장은 반복 개발 평가입니다. 새 학습·통계 적합·문턱·체크포인트 선택에는 사용하지 않았지만, 독립 블라인드 검증으로 부르면 안 됩니다. PRIMARY1237은D1147/E90이며, SELF는별도입니다. 전체1784의45U와 PRIMARY의0U를 혼동하지 마세요. 실사 dense분할 정답은 없으므로 실사분할IoU도 없습니다.

초기 q와 위치 후보는 유지했습니다. 가려진 E의 참조좌표는 수동/투영/외삽 출처가 섞이며 직접 보이는 위치 정답이 아닙니다. 사용자 이견G996도 원 주석을 유지한 채 별도 표시합니다.

학습 전 pooled known_mass가1보다최대2.15e-6 큰 float32 누적오차를 확인해 수치검사 허용오차와[0,1]clip을 기록했습니다. 추가 적합이나 데이터 선택 변경이 아닙니다. 데이터 출처 COMPLETE의 basename충돌로 메타데이터 한 항목이 누락된 것은 독립 검토 JSON에 원 SHA로 보완했습니다. 학습 라벨/입력 바이트는 바꾸지 않았습니다.
'''
    write('RUN_GUIDE_KO.md',guide)
    result=dict(completed=True,problem_solved=False,new_model_adopted=False,new_fits=2,new_optimizer_updates=2000,
        current_fit_or_threshold_or_PnP=0,PRIMARY_N=1237,D=1147,E=90,primary_metrics=primary,
        prior67_recovered_each=0,new_external_misses_each=3,new_external_miss_indices=[1515,1536,1677],
        diagnosis=['Cone whiteband/base is often predicted target: real segmentation transfer failure.',
                   'SYNDEV source1928 has high foreign ownership yet predictsD: owner-to-native-visibility readout failure.',
                   'All1784 label changes only oldE->D, no newE discovery.'],
        all1784_label_transitions=dict(SEGMENTATION=dict(E_D=51,D_E=0,D_D=1508,E_E=180,U_U=45),SEGMENTATION_CONTRAST=dict(E_D=50,D_E=0,D_D=1508,E_E=181,U_U=45)),
        current223_repeated_development_evaluation=True,further_automatic_training=False)
    write('FINAL_RESULT_SUMMARY.json',json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'completed':True,'report':str(HERE/'ROOT_READOUT_KO.html')}))


if __name__=='__main__':main()

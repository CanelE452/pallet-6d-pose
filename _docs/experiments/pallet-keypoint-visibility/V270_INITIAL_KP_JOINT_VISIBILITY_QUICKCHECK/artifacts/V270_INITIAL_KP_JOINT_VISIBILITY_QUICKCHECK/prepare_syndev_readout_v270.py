"""Display the sealed SYNDEV result; no additional data/model evaluation."""
from pathlib import Path
import hashlib
import html
import json

HERE=Path(__file__).absolute().parent


def safe(value):
    p=Path(value)
    if not p.is_absolute() or any(x in str(p).lower() for x in ('v112','cache','.pyc','..')):raise ValueError(str(p))
    for a in reversed((p,*p.parents)):
        if a.is_symlink():raise ValueError(str(a))
    return p


def binding(path):
    p=safe(path);b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}


def main():
    result=json.loads(safe(HERE/'syndev_analysis_v1/RESULTS.json').read_text())
    audit=json.loads(safe(HERE/'syndev_analysis_v1/TRAINING_AUDIT.json').read_text())
    assert result['PASS'] and audit['PASS']
    primary=[r for r in result['primary_classification'] if r['availability_policy']=='ALL_SAVED_DETECTION_ANCHORS']
    cr=[]
    for r in primary:
        cr.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['method'],
            f"[[{r['D_to_D']},{r['D_to_E']}],[{r['E_to_D']},{r['E_to_E']}]]",
            f"{100*r['D_recall']:.2f}%",f"{100*r['E_recall']:.2f}%",
            '-' if r['E_precision'] is None else f"{100*r['E_precision']:.2f}%",
            f"{100*r['balanced_accuracy']:.2f}%",r['U']])+'</tr>')
    lr=[]
    for r in result['all_location_rows']:
        if r['cohort']!='ALL_NATIVE_INFRAME':continue
        lr.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['method'],r['N'],
            f"{r['median_px']:.2f}",f"{r['p90_px']:.2f}",r['LE3']])+'</tr>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V270 합성 DEV 공동학습 진단</title>
<style>body{font:17px sans-serif;line-height:1.65;max-width:1280px;margin:32px auto;padding:12px;background:#111820;color:#e6edf5}table{border-collapse:collapse}th,td{border:1px solid #526071;padding:8px}a{color:#8ed0ff}strong{color:#ffcf87}code{white-space:pre-wrap}</style>
<h1>V270 합성 DEV235: 초기 전체 이미지 좌표·가림 공동학습</h1>
<p><strong>JOINT는 좌표 기능을 XY_ONLY 수준에 가깝게 유지하며 가림 proxy도 학습했다. 다만 이번 짧은 실험에서 VIS_ONLY보다 외부가림 재현율이 좋아지지는 않았다.</strong> 실사 전이를 입증하는 결과는 아니며, 별도 실사 평가를 봐야 한다.</p>
<p>동일 초기 R0·동일 SYN768 학습 분할·동일 baseline TAL anchor·동일 seed17/300×8 draws로 세 번만 학습했다. 전체 이미지 frozen backbone/neck의 원래 pose stem을 조정하고 별도 native8 E 출력층을 추가했다. M4·crop·키포인트 query를 입력/가림 교사로 사용하지 않았다. 원 전체 네트워크를 전부 finetune한 실험은 아니다.</p>
<h2>보임·외부가림 proxy 분류</h2>
<p>기존 DEV235의 strict D733/E192=925점. source fraction=0인 D와 ≥0.5인 E만 집계했다. 행=정답[D,E], 열=예측[D,E]. P(E)≥0.5는 E로 고정했다. 동일 저장 detection-anchor 전체와 사전 고정 detection≥0.001 정책의 결과는 같으며 보류0이다.</p>
<table><tr><th>방법</th><th>혼동행렬</th><th>D recall</th><th>E recall</th><th>E precision</th><th>균형정확도</th><th>보류</th></tr>'''+''.join(cr)+'''
</table><p>VIS_ONLY→JOINT에서 D 오판14점이 맞아지고 D 정답1점이 틀려졌다. E 오판이 맞아진 점은0이며 E 정답6점이 틀려졌다. precision 상승과 recall 하락의 tradeoff로, 가림 판정의 일관된 개선은 아니다. 학습 FIT 비율의 상수 예측보다 두 모델이 낫지만 E의 약 절반을 여전히 놓친다. 완전차단 fraction1의144점도 VIS_ONLY73점/JOINT68점만 E로 맞췄다.</p>
<h2>좌표 drift: 분류 성공 지표와 분리</h2>
<p>DEV235 원본 화면 안의 native8 가상 GT1,750점에서 원본 raw pixel 오차를 계산했다. Centroid는 아래 표에 포함하지 않았다. 이 표는 저장 모델 후보 anchor의 좌표이며 새로운 PnP나 좌표 후처리는 없다.</p>
<table><tr><th>방법</th><th>N</th><th>중앙값 px</th><th>P90 px</th><th>≤3px</th></tr>'''+''.join(lr)+'''
</table><p>공유 pose stem에 가림 loss만 학습한 VIS_ONLY는 명시적 위치 제약이 없어 좌표 오차가 크게 늘었다. JOINT의 위치 loss는 이 drift를 막아 XY_ONLY와 비슷한 수준을 유지했다. <strong>이는 위치를 함께 보호할 이유에 대한 근거이며, JOINT의 가림 성능 향상에 대한 근거와는 별개다.</strong> 원 R0의 raw DEV 좌표 전체는 추출 시 따로 저장하지 않았으므로 R0 대비 좌표 개선을 주장하지 않는다.</p>
<h2>제외한 상태와 교사 한계</h2>
<p>자기 가림294점, 0&lt;fraction&lt;0.5의 부분 차단531점, 화면 밖 native130점은 D/E 정답 집계에서 제외했다. 자기 가림294점에서 VIS_ONLY는294E, JOINT는293E를 냈다. 정답이 없는 상태의 분포이므로 이를 외부가림 오류율로 세지 않았다. E 출력을 외부 원인 판정으로 확대하지 말고 예정된 기하 self 필터와 구분해야 한다.</p>
<p>가림 교사는 저장된 virtual cuboid 주변9ray 외부 차단 proxy다. 실제 표면의 native 가시성이나 사람의 D/E 정의와 같다고 인증하지 않았다. 기존 반복 개발 SYN DEV이며 새 독립 검증셋도 아니다. 재질별·fraction별 원표를 함께 제공하며 이 결과를 보고 새 학습·문턱·정책 선택을 하지 않았다.</p>
<h2>실행·재현 감사</h2><p>최종 checkpoint3개 모두300회, 총900 optimizer update이며 동일 초기 pose/E state와 draws를 사용했다. frozen body 및 BN buffer가 동일하고 변하지 않았고, XY_ONLY의 학습하지 않은 E층은 초기값 그대로다. 모든 checkpoint가 저장된 뒤에만 DEV 예측을 수행했다. 작은 head checkpoint·로그·숫자 교사/예측을 포함한 compact input으로 CPU metrics와 checkpoint scope를 다시 계산했고7개 결과 파일이 byte 단위로 같다. 재계산에 RGB/M4/current223/model forward/optimizer가 필요하지 않다.</p>
<p><a href="syndev_analysis_v1/CLASSIFICATION.csv">전체 재질·분수별 분류표</a> · <a href="syndev_analysis_v1/LOCATION_SUMMARY.csv">위치 오차표</a> · <a href="syndev_analysis_v1/NONDE_DISTRIBUTIONS.csv">self·unknown 분포</a> · <a href="syndev_analysis_v1/TRAINING_AUDIT.json">300×3 checkpoint/log 감사</a> · <a href="SYNDEV_METRICS_REPLAY_BYTECHECK.json">CPU replay7파일 일치</a> · <a href="LABEL_AUDIT_READOUT_KO.html">실제 합성 crop과 교사 감사</a></p>
<p>재현: <code>python -B analyze_syndev_joint_v270.py --replay syndev_compact_inputs_v1 --out fresh_replay_output</code></p></html>'''
    out=safe(HERE/'SYNDEV_READOUT_KO.html');assert not out.exists();out.write_text(page+'\n')
    print(json.dumps({'readout':binding(out)}),flush=True)


if __name__=='__main__':main()

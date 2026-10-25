"""Render saved V248 SOURCE results. No model, fitting, or private evaluation I/O."""
from pathlib import Path
import csv
import hashlib
import json
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).absolute().parent
H = R / 'V248_ROOT_PUBLIC_PIXEL_AUX'
FIT = H / 'matched_source_public_pixel_aux_fit_v1'
ARMS = ['CONTROL_NO_PUBLIC_AUX', 'TREAT_PUBLIC_PIXEL_AUX']


def safe(p):
    p = Path(p)
    if any(x in str(p).lower() for x in ['v112', '__pycache__', '.pyc']):
        raise ValueError('Forbidden explicit artifact path')
    if any(a.is_symlink() for a in [p, *p.parents]):
        raise ValueError('Symlink artifact')
    return p


def read_json(p):
    return json.loads(safe(p).read_bytes())


def write(p, text):
    with safe(p).open('x') as f:
        f.write(text)


def pct(x):
    return '정의 불가' if x is None else f'{100*x:.2f}%'


def run():
    complete = read_json(FIT / 'COMPLETE.json')
    freeze = read_json(FIT / 'BOTH_FINAL1000_AND_LOGS_FROZEN_BEFORE_ANY_DEV_OUTPUT.json')
    assert complete['PASS'] and freeze['PASS']
    s = read_json(FIT / 'SOURCE_RESULTS.json')
    feats = read_json(H / 'aux_canonical_pixel_features_v1/COMPLETE.json')
    rows = []
    for arm in ARMS:
        for name, rr, cmkey, visible in [
            ('SYN intended-native DEV', s['arms'][arm]['family0_DEV'], 'CM_D_E_by_D_E_U', 'D'),
            ('YCB associated-r DEV', s['arms'][arm]['family1_DEV'], 'CM_D_E_by_D_E_U', 'D'),
            ('Pallox conditional-pixel AUX_DEV', s['public_AUX'][arm]['AUX_DEV'], 'CM_V_E_by_V_E_U', 'V'),
        ]:
            cm = rr[cmkey]
            rows.append(dict(arm=arm, target=name, rows=rr['rows'],
                             visible_recall=rr[visible]['recall'], E_recall=rr['E']['recall'],
                             visible_precision=rr[visible]['precision'], E_precision=rr['E']['precision'],
                             U_rate=sum(r[2] for r in cm)/rr['rows'], CM=json.dumps(cm)))
    with safe(H / 'DEV_COMPARISON.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fig, axes = plt.subplots(3, 2, figsize=(10, 11), constrained_layout=True)
    for ax, row in zip(axes.flat, [rows[0], rows[3], rows[1], rows[4], rows[2], rows[5]]):
        cm = np.array(json.loads(row['CM']))[:, :2]
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        for y in range(2):
            for x in range(2):
                ax.text(x, y, str(cm[y, x]), ha='center', va='center', fontsize=20,
                        color='white' if norm[y, x] > .55 else 'black')
        ax.set_xticks([0, 1], ['Visible', 'Occluded'])
        ax.set_yticks([0, 1], ['Visible', 'Occluded'])
        ax.set_xlabel('Prediction'); ax.set_ylabel('Conditional / SOURCE target')
        ax.set_title(('CONTROL' if row['arm']==ARMS[0] else 'PUBLIC AUX TREATMENT')+'\n'+row['target'], fontsize=10)
    fig.suptitle('V248 saved SOURCE outputs: counts, row-normalized color\nNot current pallet evaluation or native public-pallet gold', fontsize=13)
    fig.savefig(safe(H / 'SOURCE_CONFUSION_OVERVIEW.png'), dpi=145)
    plt.close(fig)
    gate = s['SOURCE_gate']
    lines = [
        '# V248 — 공개 실사 픽셀 보조학습은 성공, 기존 SOURCE 판정으로 전이는 확인하지 못함', '',
        '**판정: SOURCE 채택 기준 실패. 새 모델 채택 없음, 새 current/v61 평가 없음. 목표는 아직 미해결이다.**', '',
        'V241 검토 문서가 구분한 “질문 픽셀”, “연결된 표면점”, “지정 native 키포인트”를 별도 목표로 유지했다. '
        '이번에 추가한 것은 공개 Pallox의 조건부 픽셀 감독을 공유 시각 처리층에 주는 손실이다. '
        '보조 head 자체의 높은 점수를 팔레트 native 가시성 정확도로 해석하지 않는다.', '',
        '## 실제 수행', '',
        '- 공개 TRAIN 22쌍·44 view의 기존 조건부 certificate만 사용했다. 원래 352 query의 좌표·crop 크기·native·U·분할은 바꾸지 않았다.',
        '- 새로운 AUX query를 별도 생성했다. 각 pair/view의 V 최대32개, 완전히 영상 안에 있는 5×5 E core 최대32개, L=64px, seed17, 부족분 보충 없음이다.',
        '- pair 단위로 미리 정한 17 FIT/5 DEV를 모든 view·파생 픽셀에 상속했다. 실제 생성은 FIT 1,120개(V832/E288), DEV 448개(V320/E128), 총1,568개이며 입력 지원은 모두 유효했다.',
        '- 고정 DINOv2 vits14_reg4로 RGB/valid crop의 pixel1154 특징만 추출했다. encoder 상태 전후 동일·같은 batch 반복 바이트 일치를 확인했다. fitted normalizer는 없다.',
        '- 양군은 같은 V239 final1000 core와 같은 새 보조 head에서 시작했다. 기존 main 입력·손실·draw는 같고, 처리군만 보조손실 가중치0.25를 추가했다. 각1,000회, 총2,000회 새 optimizer update를 실제 수행했다. core 누적 update는 각2,000회이다.',
        '- 보조손실이 공유층의 시각1154열·bias에 도달하고, 조건32열·기존 출력층·관계 분기에는 직접 도달하지 않는지 optimizer 전 확인했다. 대조군 보조 readout은 업데이트·weight decay가 없었다.',
        '- 양 최종 checkpoint와 학습 log를 고정한 뒤에만 새 DEV 출력과 이전 baseline CSV를 해석했다. 문턱은 미리 정한 E logit≥0이며 sweep·early stop·재학습을 하지 않았다.', '',
        '## 같은 저장 출력의 비교', '',
        '| 평가 목표 | 학습군 | 보임 recall | 가림 recall | 가림 precision | 보류율 |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(f"| {row['target']} | {'대조' if row['arm']==ARMS[0] else '보조학습'} | {pct(row['visible_recall'])} | {pct(row['E_recall'])} | {pct(row['E_precision'])} | {pct(row['U_rate'])} |")
    lines += ['', '![SOURCE 혼동행렬](SOURCE_CONFUSION_OVERVIEW.png)', '',
              '행은 정답 보임/가림, 열은 예측 보임/가림이다. 원 SOURCE의 S/U 7,423행은 binary 감독에서 제외된 행이며 모델의 성공적인 보류로 세지 않았다. 위 보류율은 표에 포함된 입력 지원이 유효한 행에서의 모델 U 비율이다.', '',
              '대조군의 새 보조 readout은 학습하지 않았으므로 AUX 점수 차이는 보조학습 연결 확인용이다. 방법의 유효한 전이 비교는 동일하게 학습한 기존 SOURCE 경로끼리의 비교이다.\n\n보조학습 처리군은 AUX_FIT 1,120개를 모두 맞혔고 내부 AUX_DEV에서는 보임319/320·가림128/128개를 맞혔다. '
              '그러나 YCB associated-r DEV의 가림은 양군 모두0/57개였다. SYN DEV에서는 대조군50/56 → 처리군51/56으로 가림 한 점이 늘었다.', '',
              f"기존 두 family의 DEV balanced recall 평균에서 처리군의 대조군 대비 이득은 **{gate['matched_treatment_vs_control_gain_pp']:.4f}pp**로, 사전 +1pp 기준을 통과하지 못했다. "
              f"V239 대비 이득은 **{gate['legacy_SOURCE_vs_V239']['mean_family_DEV_balanced_recall_gain_pp']:.4f}pp**이고 YCB E recall은0%라 기존 SOURCE 기준도 실패했다. AUX_DEV V/E≥80% 조건만 통과했다.", '',
              '## 무엇을 알게 됐는가', '',
              '**관측:** 이번 작은 head와 고정 특징으로 공개 Pallox의 조건부 픽셀 V/E를 학습할 수 있었다. gradient 단절이나 학습을 시작하지 못한 기술 문제로 이번 결과를 설명할 근거는 없다. '
              '**아직 미확인:** 그 감독으로 지정 native 점의 가시성을 판별할 수 있는지, 새로운 물체·장면·카메라에서 유지되는지는 검증하지 못했다. '
              '공개 보조 문제에서 성공한 사실과 기존 SOURCE 판정에서의 실패가 함께 남았다.', '',
              '**범위:** Pallox는 같은 물리 bin의 조건부 SAM+depth+mesh 교사 감독이다. RGB-depth 정합·정적 카메라/물체·교정 가정이 남으며 native 키포인트 대응과 SELF는 UNKNOWN이다. '
              'pair를 나눴지만 물리 장면이 독립인 blind 평가가 아니다. 기존 SOURCE DEV도 반복 노출된 개발 자료다. '
              '따라서 447/448을 일반적인 실사 팔레트 정확도나 외부 가림 gold 정확도로 부르지 않는다.', '',
              '이번 모델은 고정 SOURCE 기준에 실패했으므로 current223/v61 비교 단계에 진입하지 않았다. '
              '목재/정사각형·외부/자기 가림·초기 좌표 오차별 현재 평가 수치는 이번에 갱신하지 않았다. '
              '현재 평가 자료와 파생 점수로 학습·문턱·모델·데이터·구조를 고르지 않았고, PnP·위치 회귀·원 query의 GT 중심 재crop도 하지 않았다.', '',
              '## 실제 crop과 검증 자료', '',
              '- [미리 정한 SOURCE crop 12개·빈 stratum 4개](visuals/PORTABLE_GALLERY_KO.html) — 새 출력 전에 고른 사례이며 조건부 픽셀 목표만 보여준다.',
              '- [비교표 CSV](DEV_COMPARISON.csv)',
              '- [원 CM·precision/recall·고정 gate JSON](artifacts/V248_ROOT_PUBLIC_PIXEL_AUX/matched_source_public_pixel_aux_fit_v1/SOURCE_RESULTS.json)',
              '- [실제 실행과 terminal 기록](artifacts/V248_ROOT_PUBLIC_PIXEL_AUX/ACTUAL_EXECUTION_TERMINALS.json)',
              '- [학습 전 명세](artifacts/V248_ROOT_PUBLIC_PIXEL_AUX/PROTOCOL_BEFORE_NEW_AUX_ARRAYS_OR_FEATURES_OR_FIT.json)',
              '- [crop 명세 정정 — 실제 데이터 읽기 전](artifacts/V248_ROOT_PUBLIC_PIXEL_AUX/CROP_CONTRACT_CLARIFICATION_BEFORE_PAYLOADS.json)',
              '- [독립 CODE QA](artifacts/V248_PUBLIC_AUX_TRAINER_CODE_QA_C/REVIEW.json)',
              '- [V241 첨부 문서 시점과 최신 완료 작업의 대조](artifacts/V248_REFERENCE_DOCUMENT_QA_C/KO.md)', '',
              '## 공개 자료와 재현 범위', '',
              'Pallox 원자료: Yanming Wu·Eric Demeester, KU Leuven, CC-BY 4.0, [DOI 10.48804/DJLDFS](https://doi.org/10.48804/DJLDFS). '
              '사용자가 직접 촬영한 평가 자료와 원본을 공유하지 않는다고 확인했다. 새 기록·특징·checkpoint·draw·log·출력·코드와 실제 crop을 함께 보존한다. '
              '기존 SOURCE 의존 파일은 저장소의 앞선 실험 패키지와 SHA로 연결하며, 공식 encoder 가중치를 이번 패키지에 재배포하지 않는다. '
              '저장 결과 검산과 모델 재학습은 별개이다. GPU 모델 forward를 다시 하지 않고도 공개된 행별 CSV·log·manifest를 검산할 수 있다.', '',
              'V241 첨부 문서는 진단 참고자료로 읽었다. 그 안의 조건부 실행 제안을 별도 사용자 지시로 취급하지 않았다. '
              '이번 실험 권한은 기존 공개 실사+합성 SOURCE 학습 및 지속 수행 요청에 따른다.', '',
              '과거 평가 aggregate 접근에 대한 V247 공개 범위 설명은 유지한다. 이번 선택에는 사용하지 않았다. '
              '명시적으로 지정한 artifact 경로의 접근을 통제했으며, 설치 라이브러리 내부의 모든 filesystem/cache 활동을 감사했다는 주장은 하지 않는다.', '']
    md='\n'.join(lines)
    write(H/'PUBLIC_README_KO.md',md)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['target'], '대조군' if r['arm']==ARMS[0] else '보조학습',pct(r['visible_recall']),pct(r['E_recall']),pct(r['E_precision']),pct(r['U_rate'])])+'</tr>' for r in rows)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V248 공개 픽셀 보조학습 실제 결과</title><style>body{font:17px/1.7 sans-serif;max-width:1100px;margin:40px auto;padding:0 22px;background:#f7f8fb;color:#182232}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #d8dfe9;padding:8px;font-size:14px}img{max-width:100%}.card{padding:22px;background:white;margin:20px 0;border-radius:14px}.fail{border-left:6px solid #b83f3f}</style><h1>V248 실제 결과</h1><div class="card fail"><b>공개 픽셀 보조 문제는 학습했지만, 기존 SOURCE 판정 개선은 기준 미달입니다.</b><p>AUX_DEV: 보임319/320, 가림128/128. 기존 YCB associated-r 가림: 두 학습군 모두0/57. 모델 채택·새 current 평가 없음.</p></div><div class="card"><h2>같은 저장 출력의 비교</h2><table><tr><th>목표</th><th>학습군</th><th>보임 recall</th><th>가림 recall</th><th>가림 precision</th><th>보류</th></tr>'''+table+'''</table><p>이 값은 SOURCE와 조건부 AUX 개발 자료의 결과입니다. 현재 팔레트 평가 정확도가 아닙니다. 제외된 S/U 7,423행은 모델 보류로 세지 않았습니다.</p></div><div class="card"><img src="SOURCE_CONFUSION_OVERVIEW.png" alt="SOURCE 혼동행렬"></div><div class="card"><h2>실제 crop과 근거</h2><p><a href="visuals/PORTABLE_GALLERY_KO.html">미리 정한 실제 SOURCE crop 보기</a> · <a href="DEV_COMPARISON.csv">비교 CSV</a> · <a href="README_KO.md">전체 한국어 보고서</a></p><p>별도 생성한 AUX 픽셀 query이며, 원 native 키포인트를 옮기거나 그 정답으로 바꾸지 않았습니다. 같은 물리 bin의 조건부 교사 라벨이라 native·SELF 대응 및 독립 물리 장면 일반화는 미확인입니다.</p></div></html>'''
    write(H/'ROOT_READOUT_KO.html',page)
    print(json.dumps({'completed':True,'saved_result_readout_only':True,'new_model_forward':0,'SOURCE_gate_PASS':gate['PASS'],'feature_records':feats['counts']['records']}))


if __name__=='__main__':
    run()

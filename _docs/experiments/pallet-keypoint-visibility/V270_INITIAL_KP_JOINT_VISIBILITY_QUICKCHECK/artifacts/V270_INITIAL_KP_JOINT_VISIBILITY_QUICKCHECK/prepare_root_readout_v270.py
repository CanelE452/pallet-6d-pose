"""Build reviewable V270 summary from frozen numerical outputs; no model execution."""
from pathlib import Path
import csv, collections, hashlib, html, json
import numpy as np

W=Path(__file__).absolute().parent
def safe(p):
    p=Path(p)
    assert p.is_absolute() and not any(x in str(p).lower() for x in ('v112','cache','.pyc','..'))
    for a in reversed((p,*p.parents)): assert not a.is_symlink(),str(a)
    return p
def readj(rel):return json.loads(safe(W/rel).read_text())
def write(rel,text):
    with safe(W/rel).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
def dump(rel,obj):write(rel,json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
def bind(rel):
    p=safe(W/rel);return dict(file=rel,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)

r=readj('evaluation_v4/RESULTS.json');metrics=r['metrics']['PRIMARY_PAIRED1237']
rows=list(csv.DictReader(safe(W/'evaluation_v4/ALL1784_CLASSIFICATION_COORDINATE_JOIN.csv').open()))
assert len(rows)==1784
with np.load(safe(W/'CURRENT_RAW223_PREDICTIONS.npz'),allow_pickle=False) as p:
    verified={}
    for arm in ('VIS_ONLY','JOINT'):
        c=collections.Counter()
        for row in rows:
            if row['primary']!='True' or row['truth'] not in ('D','E'):continue
            x='E' if p[arm+'_E_probability'][int(row['frame_index']),int(row['native_slot'])]>=.5 else 'D'
            assert x==row[arm+'_D_E_U'];c[row['truth']+'_'+x]+=1
        target=metrics['ALL'][arm]['confusion_matrix']
        assert all(c[k]==target[k] for k in ('D_D','D_E','E_D','E_E'))
        verified[arm]=dict(c)
    slots={str(k):dict(D=int((p['JOINT_E_probability'][:,k]<.5).sum()),E=int((p['JOINT_E_probability'][:,k]>=.5).sum())) for k in range(8)}
assert all(slots[str(k)]['E']==223 for k in (2,3,6,7))
assert all(slots[str(k)]['D']==223 for k in (0,1))
qa=dict(PASS=True,independent_NPZ_threshold_vs_CSV=verified,threshold=.5,raw_to_native_permutation_changes=0,
        rows=1784,primary=1237,slots_all223=slots,new_fit=0,new_model_forward=0,
        verified_inputs=[bind(x) for x in ('CURRENT_RAW223_PREDICTIONS.npz','evaluation_v4/RESULTS.json','evaluation_v4/ALL1784_CLASSIFICATION_COORDINATE_JOIN.csv')],
        actual_pixels_reviewed=[2,370,800],visual_interpretation_KO='G2와 G370은 같은 native2의 노출 목재와 콘 뒤 점인데 둘 다 E. G800은 native0 콘 가림인데 D. 표시 확대는 모델 입력이 아니다.')
dump('ROOT_INDEPENDENT_RESULT_CHECK.json',qa)

names={'VIS_ONLY':'가림만 학습','JOINT':'위치+가림 공동 학습','EVA_DECLIP':'이전 V269 DeCLIP (참고)','FULL_CLIP_SURGERY_ARCH':'이전 V267 Surgery (참고)','CAL_LEGACY':'기존 실사 학습 가림 모델 (참고)'}
def pct(v):return f'{100*v:.2f}%'
def table(group,arms):
    head='<tr><th>방법</th><th>보임 정답 / 오판 / 보류</th><th>가림 정답 / 누락 / 보류</th><th>보임 precision / recall</th><th>가림 precision / recall</th><th>균형 recall</th><th>보류율</th></tr>'
    body=[]
    for arm in arms:
        d=metrics[group][arm];c=d['confusion_matrix']
        body.append(f'<tr><td>{names[arm]}</td><td>{c["D_D"]} / {c["D_E"]} / {c["D_U"]}</td><td>{c["E_E"]} / {c["E_D"]} / {c["E_U"]}</td><td>{pct(d["D_precision"])} / {pct(d["D_recall"])}</td><td>{pct(d["E_precision"])} / {pct(d["E_recall"])}</td><td>{pct(d["balanced_recall"])}</td><td>{pct(d["U_rate"])}</td></tr>')
    return '<div class="scroll"><table>'+head+''.join(body)+'</table></div>'
summary=dict(schema='V270_INITIAL_KP_JOINT_VISIBILITY_QUICKCHECK_COMPLETED',completed=True,
             adopted=False,conclusion_KO='구현과 공동 gradient는 작동하지만 이 300-step frozen-backbone pilot은 외부 가림 판정을 개선하지 못했다. 합성 슬롯별 D/E 지원 부족과 슬롯 사전값 의존이 확인됐다.',
             scope='Full-image initial YOLO26n one2one KP shared cv4 stem + new E head; backbone/neck and final xy readout frozen; not full-model/from-scratch joint training.',
             new_fits=3,optimizer_steps_per_fit=300,new_optimizer_steps=900,SYN_fit_frames=768,SYN_dev_frames=235,current_eval_frames=223,
             current223_used_for_fit_or_threshold_or_checkpoint_selection=0,current223_repeated_development_evaluation=True,
             primary=dict(N=1237,D=1147,E=90,VIS_ONLY=metrics['ALL']['VIS_ONLY'],JOINT=metrics['ALL']['JOINT']),
             material_primary={g:{a:metrics[g][a] for a in ('VIS_ONLY','JOINT')} for g in ('WOOD_PRIMARY105','SQUARE118')},
             no_new_PnP=True,new_GT_matching=0,raw_native_identity9=True,new_gate_or_abstention_search=False,
             semantic_target='Existing SYN external 9-ray blockage proxy, D=fraction0; E>=0.5; partial/self/outside unknown. Not an M4 pixel label or certified real native visibility.',
             slot_issue=dict(SYN_fit_D_counts_bottom_slots=[0,0,0,0],native_slots=[2,3,6,7],real_all223_predicted_E=True,floor_cause_unconfirmed=True,strict_D0_selection_excludes_partial_negative_candidates=[582,595,218,252]),
             root_independent_check=bind('ROOT_INDEPENDENT_RESULT_CHECK.json'),
             canonical=dict(evaluation='evaluation_v4',visual='visual_v3',cpu_head_replay='cpu_replay_v2'),
             further_automatic_training=False)
dump('FINAL_RESULT_SUMMARY.json',summary)

doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V270 초기 KP 위치·가림 공동 학습 빠른 확인</title><style>body{font:17px/1.7 system-ui,sans-serif;color:#182e46;background:#f2f5fa;max-width:1220px;margin:24px auto;padding:0 20px 60px}section{background:white;border-radius:12px;padding:20px;margin:18px 0}h1{font-size:29px;line-height:1.35}h2{font-size:22px}table{border-collapse:collapse;font-size:15px;min-width:930px;width:100%}td,th{border:1px solid #c6d2e0;padding:9px;text-align:left}th{background:#eaf0f8}.scroll{overflow:auto}img{max-width:100%;height:auto}a{color:#125cb0}.lead{border-left:6px solid #d67a28}.note{color:#465e76}code{overflow-wrap:anywhere}</style>
<h1>V270 · 초기 KP 모델이 위치와 가림을 함께 출력하게 한 빠른 확인</h1>
<section class="lead"><p><strong>공동 출력과 학습은 구현됐지만, 이번 짧은 실험은 외부 가림 판정을 개선하지 못했습니다.</strong> 가림만 학습한 대조군→공동 학습에서 보임 오판은511→471개로 줄고, 가림 누락은51→59개로 늘었습니다. 새 모델은 채택하지 않습니다.</p><p>더 중요한 원인이 확인됐습니다. 합성 학습에서 아래쪽 native2/3/6/7의 strict 보임 D가 모두0개였고, 새 모델은 실사223장 모두에서 이 네 점을 가림 E로 냈습니다. 이는 이미지의 가림 단서보다 슬롯 번호의 사전값에 의존하는 문제입니다. <strong>초기 모델의 위치+가림 공동 학습 일반이 불가능하다는 결론은 아닙니다.</strong></p></section>
<section><h2>실제 영상: 같은 번호인데 콘 유무를 구분하지 못함</h2><img src="SAME_SLOT_DIAGNOSIS.png" alt="같은 native2의 보임과 콘 가림을 모두 가림으로 판정하며 native0 콘 가림은 보임으로 놓치는 사례"><p>G2는 가림 없는 목재 native2인데 E(.951), G370은 콘 뒤 native2인데 E(.948)입니다. G800의 native0은 콘 위 위치를 유지해도 D(.323)입니다. 성공 사례 G370 하나만으로 콘 구분을 배웠다고 해석할 수 없습니다. 확대는 표시용이며 실제 모델은 전체 이미지를 입력받습니다.</p><p><a href="visual_v3/INDEX_KO.html">실제19사례 전체 이미지·고정 확대</a> · <a href="evaluation_v4/SLOT_PRIOR_DIAGNOSTIC.json">실사 슬롯별 출력223장 전체</a> · <a href="EVALUATION_AND_VISUAL_QA.json">픽셀 검토와 해석 한계</a></p><p class="note">G300·G996은 기존 라벨과 표시 위치 해석에 이견이 있는 경계 사례입니다. 원래 평가 라벨은 보존했으며, 이 둘을 명백한 가림 전이 실패의 대표 근거로 사용하지 않았습니다.</p></section>
<section><h2>같은 PRIMARY1237점: 보임1147·외부 가림90</h2>'''+table('ALL',('VIS_ONLY','JOINT'))+'''<p>행의 보임 정답/오판/보류는 GT D→예측 D/E/U, 가림 정답/누락/보류는 GT E→예측 E/D/U입니다. 임계값은 실사 평가 전 고정한 E확률0.5이며 새 보류 게이트는 없습니다. 전체1784 native 모서리 중 등록 D/E1260점에서는 공동 학습 보임681/1168(58.30%), 가림33/92(35.87%)입니다. 과거 crop/query 가용성 제한을 새 전체 이미지 모델에 강제로 적용하지 않았고, 이전 모델과 같은 비교는 PRIMARY1237로 따로 고정했습니다.</p><h2>목재105장</h2>'''+table('WOOD_PRIMARY105',('VIS_ONLY','JOINT'))+'''<h2>정사각형118장</h2>'''+table('SQUARE118',('VIS_ONLY','JOINT'))+'''<p>자기 가림은 사용자 계획대로 별도 기하 판정 대상이며 D/E 학습·주 성능에서 제외했습니다. 자기 가림에서 E를 내더라도 외부 물체 구분 성공으로 집계하지 않습니다. <a href="evaluation_v4/RESULTS.json">모든 구간·자기 가림·좌표 진단 원표</a> · <a href="evaluation_v4/METRICS.csv">전체 비교 CSV</a> · <a href="evaluation_v4/PRIMARY_EXTERNAL90.csv">외부 가림90점 전수</a></p><h2>기존 방식은 같은 점의 참고 비교</h2>'''+table('ALL',('EVA_DECLIP','FULL_CLIP_SURGERY_ARCH','CAL_LEGACY'))+'''<p class="note">이전 DeCLIP/Surgery는 원래 query의 대상 마스크 값을 읽는 proxy이고, 이번 모델은 전체 이미지의 초기 KP branch에서 직접 semantic E를 냅니다. 입력·목표·구조가 달라 이 표는 역사적 참고입니다. 공동 학습 효과의 주 대조는 동일 초기화·자료·300회인 VIS_ONLY↔JOINT입니다. CAL_LEGACY는 기존 실사 학습 이력이 있는 비교군입니다.</p></section>
<section><h2>무엇을 실제 학습했나</h2><p>원래 YOLO26n R0 가중치를 시작점으로 위치만(XY_ONLY), 가림만(VIS_ONLY), 위치+가림(JOINT)을 각각300회, 총900 optimizer update 학습했습니다. 전체 RGB에 reflect101 테두리100px 후 square640 letterbox를 적용하고, 같은 합성768장·seed17·batch8·고정 draw를 썼습니다. 추가 fit의 검증235장은 이번 학습에서 제외했습니다. R0의 원래 사전 학습과 중복되지 않는 독립 영상인지는 확인하지 않았습니다.</p><p><strong>빠른 확인 범위:</strong> backbone/neck·검출·최종27채널(x/y/기존 canvas confidence)은 동결하고, 기존 KP shared cv4 stem과 새9채널 E head를 학습했습니다. centroid E는 제외합니다. 전체 네트워크를 처음부터 공동 학습한 실험은 아닙니다. 원래 R0 confidence는 좌표가 라벨되고 이미지 안에 있는지에 대한 값이므로 보임/가림 정답으로 재사용하지 않았습니다.</p><p>새 E 손실은 existing external9ray fraction=0을 D,≥0.5를 E로 삼고 partial·self·out-of-frame은 unknown 처리합니다. 가려져도 알려진 좌표는 위치 손실에 포함하며 E 유효성에 따라 좌표를 지우지 않습니다. M4 마스크 픽셀을 가림 GT로 사용한 실험도 아닙니다. <a href="PROTOCOL_BEFORE_FIT.json">학습 전 프로토콜</a> · <a href="PROSPECTIVE_OPTIMIZER_AND_SCOPE_CLARIFICATION_V1.json">학습 범위·optimizer 명세</a> · <a href="INDEPENDENT_SCOPE_AND_GRADIENT_AUDIT.json">공유 stem의 위치/E gradient 독립 확인</a> · <a href="INPUT_CHANNEL_NATIVE_IDENTITY_BEFORE_GT.json">채널→현재 native 대응223장</a></p><p>현재 raw채널은 native 순서와 identity이며 perm_v4를 다시 적용하지 않았습니다. 실제223장 어노테이션은 fit·정규화·임계값·체크포인트 선택에 사용하지 않았습니다. 다만223장은 여러 차례 확인한 개발 평가이며 독립 최종 검증셋은 아닙니다.</p></section>
<section><h2>왜 실패했나: 합성 슬롯의 양쪽 상태 지원이 없음</h2><p>TRAIN6144모서리 중 D2417/E556만 strict 가림 학습 대상입니다. partial1758·self962·outside451은 E 손실에서 제외했습니다. 아래쪽 native2/3/6/7은 TRAIN D=0/0/0/0, E=95/98/28/26입니다. 위쪽0/1/4/5는 D=598/601/608/610, E=74/74/77/84입니다. 이는 원본 합성 전체에 아래쪽 보임이 없다는 뜻이 아닙니다. 이번에 strict D=0만 남기고0&lt;fraction&lt;0.5를 unknown으로 제외하면서, 원래0.5 기준이라면 음성 후보인582/595/218/252개도 빠졌습니다. 이 감독 선택이 E-only 슬롯을 만든 점이 이번 설계의 한계입니다. 그렇다고 이를 검증 없이 실제 보임 정답으로 바꿀 수도 없습니다. 전체 D/E class balancing만으로 각 슬롯의 지원 부족을 없앨 수 없습니다.</p><p>추가 학습 DEV925점에서 공동 학습 D705/733(96.18%)·E96/192(50.00%), 균형 recall73.09%입니다. 이미지를 전혀 안 보고 위쪽=D·아래쪽=E로만 찍는 사전값도 D733/733·E86/192, 균형 recall72.40%를 냅니다. <strong>이번 합성 점수만으로 시각적인 가림 구분을 학습했다고 주장할 수 없습니다.</strong> 실사 ALL등록 E정답33개 중26개도 항상E인 네 슬롯에서 나오고, 같은 슬롯이 보임442개를 오판합니다.</p><p>첫 hit의 물체 owner가 저장돼 있지 않아 바닥 접촉이 external proxy에 섞였는지는 확정하지 못했습니다. 확인된 것은 슬롯별 D지원 부족과 실제 출력 편향입니다. 라벨 proxy·가상 cuboid 모서리와 눈으로 보이는 모서리의 의미 차이, 짧은 학습, 실사 전이의 기여를 이번 한 실험으로 각각 확정할 수는 없습니다. 합성 DEV부터 E recall이50%이므로 모든 실패를 실사 전이 탓으로 돌려서도 안 됩니다.</p><p><a href="slot_prior_diagnosis_v1/READOUT_KO.html">추가 슬롯 교사 감사·정적 사전값 비교</a> · <a href="SYNDEV_READOUT_KO.html">원래 SYNDEV 결과</a> · <a href="LABEL_AUDIT_READOUT_KO.html">합성 원본 RGB·cuboid·M0/M4·9ray 정의 대조</a></p></section>
<section><h2>위치 손실은 좌표 유지에 도움, 가림 분류와 별도</h2><p>수동 보임857점의 위치 오차 중앙값은 동일 square640 R0 5.02px, 위치만4.33px, 공동4.56px, 가림만62.80px입니다. 위치 손실 없이 공유 KP stem을 바꾼 VIS_ONLY는 좌표가 손상됐고 JOINT는 이를 억제했습니다. 이는 공동 가림 분류의 성공을 뜻하지 않습니다. 기존 입력좌표 오차≤3px인 수동 보임376점에서도 공동 모델은93점을 가림으로 오판했습니다. 새 PnP나 자세 최적화는 하지 않았습니다.</p></section>
<section><h2>다음 시도의 선행 조건</h2><ol><li>합성 라벨을 픽셀/first-hit depth·owner와 대조해 ‘보이는 native 모서리’와 external9ray proxy의 의미를 먼저 맞춥니다. 바닥·자기 가림·가상 모서리를 외부 물체와 분리합니다.</li><li>각 native 슬롯마다 보임과 외부 가림을 모두 확보합니다. 같은 팔레트·시점·조명에서 외부 물체만 추가/제거한 쌍을 만들면 번호만 보고 답하는 shortcut을 검사할 수 있습니다.</li><li>학습 전에 슬롯별 상태 표와 영상 없는 슬롯 사전값 기준선을 고정합니다. 그 기준선을 이기지 못하면 실사 평가 전에 중단합니다.</li><li>라벨과 자료가 통과한 뒤에만 초기 KP 모델의 더 넓은 층까지 공동 학습할지 결정합니다. 이번에는 추가 학습·임계값 탐색을 하지 않았습니다.</li></ol><p>현재는 새 가림 판별기를 적용할 단계가 아닙니다. 공동 출력 구조는 사용할 수 있고, 다음 검증의 병목은 합성 감독 데이터입니다.</p></section>
<section><h2>재현과 실패 기록</h2><p><a href="PORTABLE_REPLAY_GUIDE_KO.md">설치·저장 데이터 CPU 재현 명령</a> · <a href="FINAL_RESULT_SUMMARY.json">기계 판독 요약</a> · <a href="ROOT_INDEPENDENT_RESULT_CHECK.json">루트 독립 NPZ/CSV 수치 검산</a> · <a href="syn_training_v1/COMPLETE.json">3종 실제 학습 완료·SHA</a> · <a href="CURRENT_PREDICTIONS_FROZEN.json">실사 출력 동결</a></p><p>보관한 실패: 평가 parser가 비D/E 상태를 거부한 v1~v3, 시각화 Pillow 환경 차이 v1, CPU sigmoid 재계산의 최초8e-8 허용오차 실패입니다. 평가 v4와 시각화 v3가 최종입니다. CPU replay v2는 sigmoid 산술만 float32 epsilon1.2e-7로 명시하며 원래 실패와 변경 계약을 보존했습니다. 모델·예측·0.5 결정·학습은 바뀌지 않았고 모든1784점의 결정은 정확히 같습니다.5개 실제 pyramid fixture의3종 head CPU 오차는 원래1e-3 기준을 통과했습니다.</p><p><a href="cpu_replay_v1/FAILURE_REASON.json">최초 CPU 실패</a> · <a href="REPLAY_FLOAT32_SIGMOID_CONTRACT_V2.json">명시한 산술 계약</a> · <a href="cpu_replay_v2/REPLAY_RESULT.json">CPU head replay PASS</a> · <a href="evaluation_cpu_replay_v1/COMPLETE.json">실사 전체 metric CPU replay PASS</a> · <a href="SYNDEV_METRICS_REPLAY_BYTECHECK.json">SYN7산출물 byte 동일</a></p></section></html>'''
write('ROOT_READOUT_KO.html',doc)

md='''# V270 초기 KP 위치·가림 공동 출력 빠른 확인

초기 전체 이미지 KP branch가 위치와 가림을 함께 출력하도록 구현해, 위치만 / 가림만 / 공동 학습을 합성768장으로 각각300회 학습했습니다. **공동 출력은 작동하지만 이번 짧은 pilot은 가림 판정을 개선하지 못해 채택하지 않습니다.**

| PRIMARY1237 | 보임 정답 /1147 | 보임→가림 오판 | 가림 정답 /90 | 가림→보임 누락 | 균형 recall |
|---|---:|---:|---:|---:|---:|
| 가림만 |636|511|39|51|49.39%|
| 위치+가림 |676|471|31|59|46.69%|

이번 strict D=fraction0 선택으로 partial 음성 후보를 제외하면서 합성 아래쪽 native2/3/6/7의 보임 D 학습 표본이 **각0개**가 됐고, 모델도 실사223장 모두 이 네 점을 E로 출력했습니다. 같은 native2에서 노출 목재 G2와 콘 뒤 G370이 모두 E입니다. 정적 슬롯 사전값의 SYNDEV 균형 recall72.40%와 공동73.09%가 비슷해, 합성 점수만으로 영상 단서 학습을 주장할 수 없습니다. 바닥이 external label에 섞였는지는 first-hit owner 미저장으로 확정하지 못했습니다.

![같은 번호의 보임·콘 가림과 다른 번호의 콘 누락](SAME_SLOT_DIAGNOSIS.png)

- [전체 보고서·목재/정사각형·precision/recall·혼동행렬](ROOT_READOUT_KO.html)
- [실제19사례](visual_v3/INDEX_KO.html)
- [합성 슬롯 교사 감사](slot_prior_diagnosis_v1/READOUT_KO.html)
- [합성 라벨 원본 대조](LABEL_AUDIT_READOUT_KO.html)
- [전체 평가 CSV](evaluation_v4/METRICS.csv)
- [저장 데이터 CPU 재현](PORTABLE_REPLAY_GUIDE_KO.md)
- [학습 전 프로토콜](PROTOCOL_BEFORE_FIT.json)

이번 실험은 backbone/neck을 동결하고 기존 KP shared stem와 새 E head를 학습한 300-step pilot입니다. 전체 네트워크의 처음부터 공동 학습 일반을 부정하는 실험은 아닙니다. 새 E 교사는 M4 픽셀 대신 기존 external9ray proxy입니다. partial/self/outside는 E 손실에서 제외했고 위치 유효성과 분리했습니다. 실제223장은 동결 후 평가에만 사용했으며 반복 개발 평가입니다. 새 PnP·실사 fit·threshold/체크포인트 선택0, 추가 fit0입니다.

다음에는 먼저 각 슬롯의 D/E 양쪽 표본과 같은 시점의 가림 추가/제거 쌍을 확보하고 의미 정의·first-hit owner를 감사해야 합니다. 이번 요청에서는 추가 학습을 진행하지 않습니다.
'''
write('PUBLICATION_README_KO.md',md)
print(json.dumps({'PASS':True,'independent_counts':verified,'report':'ROOT_READOUT_KO.html','new_fits':0},ensure_ascii=False))

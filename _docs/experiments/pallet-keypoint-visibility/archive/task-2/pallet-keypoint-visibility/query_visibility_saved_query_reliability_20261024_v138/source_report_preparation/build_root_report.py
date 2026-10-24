"""Saved source diagnosis report; no model/GT/image/private body reads."""
from pathlib import Path
import datetime, hashlib, html, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V=Path(__file__).resolve().parents[1]; P=V.parent
D=V/'saved_observability_diagnosis_v1'
def read(p): return json.loads(Path(p).read_text())
def bind(p): return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def write(p,x):
    with Path(p).open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def esc(x): return html.escape(str(x))
def auc(x): return x['AUC_fixed_higher_score_positive']
def fmt(x): return '—' if x is None else f'{x:.3f}'

seal=read(D/'DIAGNOSIS_SEAL.json'); r=read(D/'SOURCE_RESULTS.json')
assert seal['completed'] and seal['PASS'] and r['all26624_preserved'] and r['records']==15168
assert seal['results']==bind(D/'SOURCE_RESULTS.json')
next_assessment=V/'saved_risk_interpretation_v1/NEXT_STEP_ASSESSMENT.json'
review=V/'independent_design_review/SAVED_DIAGNOSIS_REVIEW.json'
assert next_assessment.exists() and review.exists()
assert read(review)['PASS']
risks=r['fixed_primary_signal_order']
corpora=('OLD_NATIVE','CONTINUOUS_NATIVE')
labels=['Low detector confidence','Low native keypoint confidence','Invalid local crop fraction',
        'Invalid central crop fraction','Invalid context fraction','Input unavailable',
        'Native bounds invalid','Outside predicted bbox','Global edge inconsistency','Native held-out edge residual']
columns=['Position >3px\nwithin D','Position >3px\nwithin E','C29 mistake\nwithin D',
         'C29 mistake\nwithin E','External E vs D\nall q errors','External E vs D\nerror <=3px']
def six(s,n):
    return [auc(s[g][t][n]) for t in ('GT_ERROR_GT3','C29_DE_MISTAKE_WHEN_INPUT_AVAILABLE') for g in ('D','E')]+[
        auc(s['EXTERNAL_E1_D0_BY_GTERROR'][e][n]) for e in ('ALL_GT_ERROR','GT_ERROR_LE3')]
matrices={c:[[v for v in six(r['summary'][c]['VALIDATION'],n)] for n in risks] for c in corpora}

plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans'})
fig,axes=plt.subplots(1,2,figsize=(18,7),layout='constrained',sharey=True)
for ax,c,title in zip(axes,corpora,['Original synthetic validation','Broader synthetic validation']):
    z=np.array(matrices[c],float); im=ax.imshow(z,vmin=0,vmax=1,cmap='RdBu_r',aspect='auto')
    ax.set_xticks(range(6),columns,rotation=35,ha='right');ax.set_yticks(range(10),labels)
    ax.set_title(title,pad=15,fontweight='bold')
    for i in range(10):
        for j in range(6): ax.text(j,i,fmt(matrices[c][i][j]),ha='center',va='center',color='white' if abs(z[i,j]-.5)>.3 else '#142433')
    ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,10,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',bottom=False,left=False)
fig.colorbar(im,ax=axes,shrink=.75,label='Fixed-direction AUC (0.5 = no ranking; not recall)')
fig.suptitle('Saved input signals: position error, visibility mistakes and external occlusion',fontweight='bold',fontsize=16)
plot=V/'FIXED_SIGNAL_VAL_AUC.png'; assert not plot.exists();fig.savefig(plot,dpi=160);plt.close(fig)

entries=[]
with (D/'ALL_FIXED_SIGN_AUC_AND_TRAIN_BIN_COUNTS.jsonl').open() as f:
    for line in f:
        a=json.loads(line)
        if a['corpus']=='CONTINUOUS_NATIVE' and a['fold']=='VALIDATION' and a['teacher'] in ('D','E') and a['availability_scope']=='ALL_ORIGINAL' and a['target']=='C29_DE_MISTAKE_WHEN_INPUT_AVAILABLE' and a['kind']=='PRIMARY_FIXED_HIGHER_RISK': entries.append(a)
assert len(entries)==20
previous=P/'query_visibility_continuous_source_coverage_control_20261024_v137'
history=read(previous/'ROOT_RESULTS_KO.json')
result=dict(completed=True,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='V138_SAVED_SOURCE_INPUT_OBSERVABILITY_DIAGNOSIS',
    goal_status='active',fully_solved=False,classification='PROGRESS',new_training_updates=0,new_model_forwards=0,new_private_access=0,PnP=0,query_or_L_correction=0,
    outcome_KO='자료 범위 확장의 합성 개선은 확인했다. 이번 고정 입력 신호는 위치 오류·보임 오판과 일부 연관되지만, 외부 가림을 단독으로 충분히 구별한다는 근거는 없다.',
    rows=26624,original_native=6144,continuous_native=20480,input_defined_confidence_geometry=26512,NO_QUERY_rows=112,
    fixed10risks=risks,validation_six_column_order=columns,validation_fixed_direction_AUC=matrices,
    accounting=r['accounting'],new_VAL_teacher_conditioned_mistake_bins=entries,
    source_progress=history['source_new_native_VAL'],source_near_diagnostic=history['source_new_native_VAL_error_LE3'],
    source_eligibility=history['eligibility'],latest_real_evaluation_unchanged=history['latest_real_evaluation_unchanged'],
    decisions=dict(confidence_only_occlusion_rule_adopted=False,geometry_only_coordinate_certificate=False,new_threshold_or_abstention_rule=None,new_selected_model=None,
        next_step='Source-only controlled conditioning of frozen visual evidence with query input, separating position reliability from external-visibility target. Predeclare actual experiment before fitting; no private-guided model choice.'),
    limitations_KO=['AUC는 위험 신호의 순위 연관성이다. 모델 recall/precision이나 실사 성능이 아니다.',
        '10개 위험 신호의 방향을 결과에 맞춰 뒤집지 않았다. G42 등 58개 원시 변수는 방향 없는 별도 설명 자료다.',
        'AUC는 정의된 입력·알려진 타깃에서 계산하며 누락·정답 S/U·예측 보류를 전체 집계에 남겼다.',
        '입력 없음/경계 오류가 예측 보류를 정의하는 부분은 새로운 실패 발견으로 해석하지 않는다.',
        'TRAIN 입력 분포로만 다섯 구간을 먼저 고정했다. GT 좌표 오차는 진단 타깃이며 입력·배포 필터가 아니다.',
        '같은 기본 장면과 네 변형은 서로 상관된다. 단변량 AUC가 낮아도 영상·다변량 정보의 부재를 증명하지 않는다.',
        '이번 결과는 원래 두 합성 형상에 대한 진단이다. 실사 전이 개선이나 배포 성공은 아직 입증되지 않았다.',
        '2D 모서리 일관성은 공통 이동·투영 변형·일부 대응 오류를 놓친다. 카메라/PnP/좌표 최적화를 하지 않았다.'],
    bindings=dict(input_seal=bind(V/'input_only_reliability_v1/INPUT_SEAL.json'),diagnosis_seal=bind(D/'DIAGNOSIS_SEAL.json'),
        TRAIN_input_bins_before_GT=bind(D/'BINNING_SEAL_BEFORE_GT.json'),independent_review=bind(review),next_assessment=bind(next_assessment),
        earlier_source_comparison=bind(previous/'ROOT_RESULTS_KO.json'),plot=bind(plot),full_fixed_sign_table=bind(D/'ALL_FIXED_SIGN_AUC_AND_TRAIN_BIN_COUNTS.jsonl')))
# Korean report prose is maintained directly; no standalone prose document.
write(V/'ROOT_RESULTS_KO.json',result)
parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
 '<title>V138 — 진전과 남은 신뢰도 문제</title><style>body{font:17px/1.65 system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 20px;background:#f6f8fa;color:#182432}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}th,td{border:1px solid #d4dce4;padding:9px;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%;height:auto}.note{padding:18px;background:#e8f0fa;border-left:5px solid #3974af}code{font-size:13px;overflow-wrap:anywhere}a{color:#195ba0}</style>',
 '<h1>진전은 있지만 실사 성공은 아직 확인하지 못했습니다</h1>',
 '<p class="note">같은 학습 예산으로 기존 합성을 반복한 대조군보다 자료 범위를 넓힌 모델이 합성 검증에서 보임·외부 가림을 함께 개선했습니다. 전체 기준은 미통과이고 새 실사 평가는 하지 않았습니다. 마지막 실사 평가는 보임 1137/1147, 외부 가림 35/90입니다.</p>',
 '<h2>이미 확인한 성과: 합성 자료 범위 확장</h2><table><tr><th>같은 새 합성 검증</th><th>보임 recall</th><th>외부 가림 recall</th><th>외부 가림 precision</th></tr>']
for m in ('C16','C28','C29'):
    z=history['source_new_native_VAL'][m];parts.append('<tr><td>'+m+'</td>'+''.join(f'<td>{100*z[k]:.2f}%</td>' for k in ('D_recall','E_recall','E_precision'))+'</tr>')
parts.extend(['</table><p>C16은 이전 모델, C28은 기존 자료 반복, C29는 같은 추가 배치에 넓은 치수·시점 자료를 넣은 모델입니다. C29는 64개 source 조건 중 44개를 통과했습니다. 보임 보존·일부 같은 재질 가림·공개 source 조건은 아직 실패합니다. 새 모델을 채택하지 않았습니다.</p>',
 '<p>초기 좌표의 GT 오차가 3px 이하인 합성 구간에서는 C29 보임 947/950, 외부 가림 36/37을 맞혔습니다. GT로 나눈 사후 진단이며 전체 성능을 대신하지 않습니다. 전체 새 검증의 이진 오판 74개 중 70개는 3px 초과 구간입니다. 이 연관성만으로 좌표 오류가 유일한 원인이라고 단정할 수는 없습니다.</p>',
 '<h2>이번 확인: 초기 입력만으로 남은 오류를 알 수 있는가?</h2><p>기존 6144개와 새 20480개, 총 26624개를 유지했습니다. 초기 검출 confidence·키포인트 confidence, 원본 q8·L·영상 경계, 저장 crop의 유효 마스크, 고정된 2D 모서리 관계에서 10개 위험 신호와 58개 설명 변수를 만들었습니다. 입력 표와 TRAIN 입력 기반 구간을 정답·C29 출력 결합 전에 저장했습니다. 새 학습·모델 추론·실사 접근·PnP·좌표 보정은 0회입니다.</p>',
 '<p><b>핵심:</b> 낮은 confidence나 기하 불일치는 보임 오판과 일부 연관됩니다. 그러나 외부 가림 자체나 놓친 가림을 같은 단일 규칙으로 충분히 구분하지는 못했습니다. 입력 경계·crop 유효도는 대부분 정상이라 남은 오류의 주된 설명이 되지 못했습니다.</p>',
 '<p>AUC 0.5는 해당 방향에서 순위 구별이 없는 수준입니다. 1에 가까울수록 선언한 위험 방향과 연관됩니다. 0.5 미만도 그대로 표시하며 결과를 보고 부호를 바꾸지 않았습니다. 아래는 recall/정확도가 아닙니다.</p><img src="FIXED_SIGNAL_VAL_AUC.png" alt="고정 위험 신호의 기존·확장 합성 검증 AUC">'])
ko_cols=['위치 오류 >3px / 정답 D','위치 오류 >3px / 정답 E','C29 오판 / 정답 D','C29 오판 / 정답 E','외부 E 대 보임 D / 전체','외부 E 대 보임 D / ≤3px']
ko_names=['낮은 검출 confidence','낮은 키포인트 confidence','local crop 유효 영역 부족','중심 crop 유효 영역 부족','context 유효 영역 부족','입력 없음','native 경계 오류','예측 bbox 밖 거리','전체 모서리 불일치','해당 점의 모서리 잔차']
for c,title in zip(corpora,['기존 합성 검증','확장 합성 검증']):
    a=r['accounting'][c]['VALIDATION'];st=a['teacher_states'];parts.append('<h2>'+title+': 고정 방향 AUC</h2><p>'+f"전체 {a['original_rows']}점: 보임 D {st['D']}, 외부 가림 E {st['E']}, 자기 가림 S {st['S']}, 정답 불확실 U {st['U']}. 알려진 D/E의 입력 보류 {a['known_DE_input_U']}점, 전체 좌표 없음 {a['NO_QUERY_qerror_unknown']}점."+'</p><table><tr><th>위험 신호</th>'+''.join('<th>'+esc(k)+'</th>' for k in ko_cols)+'</tr>')
    for n,z in zip(ko_names,matrices[c]):parts.append('<tr><td>'+n+'</td>'+''.join('<td>'+fmt(x)+'</td>' for x in z)+'</tr>')
    parts.append('</table>')
parts.append('<h2>확장 검증: 미리 정한 입력 구간별 C29 오판</h2><p>모든 10개 신호와 D/E를 표시합니다. Q1→Q5는 TRAIN 입력에서만 정한 위험 순위 구간이며 배포 기준이 아닙니다. 분모는 이 표의 타깃인 입력 사용 가능 D/E입니다. 전체 정답 D1772 중 입력 보류 2개는 본문 전체 집계와 보류 포함 실패 타깃에 남아 있습니다.</p><table><tr><th>정답</th><th>신호</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5</th></tr>')
for e in entries:
    parts.append('<tr><td>'+e['teacher']+'</td><td>'+ko_names[risks.index(e['signal'])]+'</td>')
    for q in range(1,6):
        z=e['TRAIN_input_quantile_bins']['Q'+str(q)];parts.append('<td>'+f"{z['positive']}/{z['target_known']}"+'</td>')
    parts.append('</tr>')
parts.append('</table><h2>해석과 다음 단계</h2><p>목표가 과도해서 실패한다고 단정할 결과는 아닙니다. 고정 위치 근처의 합성 가림 판단은 개선 가능했으며, 넓은 학습 자료의 효과도 같은 예산 대조로 확인했습니다. 반면 실제 사진에서 같은 개선이 나는지, 초기 위치가 빗나간 경우 영상과 위치 신뢰도를 함께 써서 보임 오판과 가림 누락을 줄일지는 아직 검증해야 합니다.</p><p>다음 source 실험은 위치 신뢰도와 외부 가림을 다른 타깃으로 유지하고, 고정된 영상 증거와 초기 입력 정보를 함께 쓰는 통제 실험으로 준비합니다. confidence 하나로 가림을 결정하거나 GT 오차로 보류시키지 않습니다. 이전 G42 추가 실험은 외부 가림 개선 없이 보임 오판이 늘었으므로, 단순 기하 추가의 성공을 전제하지 않습니다. 현재 실사 자료로 학습·문턱·모델을 선택하지 않습니다.</p>')
parts.append('<p><a href="../'+previous.name+'/ROOT_READOUT_KO.html">이전 자료 범위 대조 결과와 실제 저장 crop 비교 보기</a></p><details><summary>제약과 근거</summary><ul>')
for s in result['limitations_KO']:parts.append('<li>'+esc(s)+'</li>')
for k,b in result['bindings'].items():parts.append('<li>'+esc(k)+': <code>'+esc(b['path'])+'</code> <code>'+esc(b['sha256'])+'</code></li>')
parts.append('</ul></details></html>')
report=V/'ROOT_READOUT_KO.html'
with report.open('x') as f:f.write('\n'.join(parts)+'\n')
print(json.dumps(dict(results=bind(V/'ROOT_RESULTS_KO.json'),report=bind(report),plot=bind(plot)),ensure_ascii=False))

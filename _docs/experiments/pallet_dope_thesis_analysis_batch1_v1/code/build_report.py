"""Integrate frozen Batch1 outputs, with no inference or raw data reads."""
from common import *
from collections import Counter
import statistics

def loadcsv(name):
    with (OUT/'tables'/name).open() as f:return list(csv.DictReader(f))

def pct(x):return f'{100*float(x):.2f}'
def val(x):return 'NA' if x in ('',None) else f'{float(x):.2f}'

def main():
    inputs=[NOFLIP/'SCORES.json',NOFLIP/'PLASTIC184_SCORES.json',
            OUT/'preflight/T3_FILTER_EQUIVALENCE.json',OUT/'tables/T3_FILTER_STAGES.json',
            OUT/'tables/T4_PNP_AVAILABILITY.json',OUT/'cache/T2_BOOTSTRAP.json',
            OUT/'hashes/COMPARISON.json',OUT/'cache/METADATA_RECORDINGS.json',
            OUT/'preflight/GATE5_RESOLUTION.json']
    start('integrated_report',inputs)
    authorization=read(OUT/'preflight/GATE5_RESOLUTION.json')
    assert authorization['status']=='RESOLVED_BY_EXPLICIT_USER_AUTHORIZATION'
    assert authorization['user_answer']=='기존 파생 SCORES 사용만 허용하고 push'
    assert not authorization['raw_protected_FINAL_RGB_opening_allowed']
    assert not authorization['raw_protected_FINAL_annotation_body_opening_allowed']
    for rel,expected in authorization['frozen_quantitative_outputs'].items():
        assert sha(OUT/rel)==expected,('Frozen quantitative result changed',rel)
    proof=read(OUT/'preflight/T3_FILTER_EQUIVALENCE.json')
    assert proof['complete'] and proof['pass_all']
    assert read(OUT/'hashes/COMPARISON.json')['all_unchanged']
    scores=read(NOFLIP/'SCORES.json');extra=read(NOFLIP/'PLASTIC184_SCORES.json')
    metrics=[]
    for population,kind,panel,payload in [(f'DEV292_{k}',k,'all_annotated_reference',scores) for k in KINDS]+[
                ('GREEN_manual600','GREEN','verified_manual_only',scores),
                ('PLASTIC184','PLASTIC','all_annotated_reference',extra)]:
        for role,model in roles(kind).items():
            frames=[r for r in payload['models'][model]['frames'] if kind_of(r)==kind]
            points=[p for f in frames for p in f[panel]]
            errors=[p['error_px'] for p in points if p['error_px'] is not None]
            n=len(points);obs=len(errors)
            r=dict(population=population,kind=kind,role=role,model=model,n_frames=len(frames),
                   n_reference=n,n_observed=obs,coverage=obs/n,
                   conditional_median_px=statistics.median(errors) if errors else None,
                   conditional_precision10=sum(e<=10 for e in errors)/obs if obs else None)
            r.update({f'PCK{tau}':sum(e<=tau for e in errors)/n for tau in (5,10,20)})
            metrics.append(r)
    csvwrite(OUT/'tables/T1_EXISTING_METRICS.csv',metrics)
    h=['모집단','모델','영상/기준점','PCK5 %','PCK10 %','PCK20 %','coverage %','조건부 정밀도10 %','조건부 중앙값px']
    (OUT/'tables/T1_EXISTING_METRICS.md').write_text('[확인] 저장된 SCORES만 재집계했다. 현재 평가 전처리·디코더의 값이며, 미실행 T1(b)의 원래 경로 결과가 아니다. 누락은 PCK 실패이고 조건부 통계에서 제외한다.\n\n'+
        mdtable(h,[[r['population'],r['role'],f"{r['n_frames']}/{r['n_reference']}",
                   pct(r['PCK5']),pct(r['PCK10']),pct(r['PCK20']),pct(r['coverage']),
                   pct(r['conditional_precision10']),val(r['conditional_median_px'])] for r in metrics]))
    stages=read(OUT/'tables/T3_FILTER_STAGES.json')['rows']
    assert all(r['annotated_eligible_selected_ref_count']==r['selected_ref_observed_count'] for r in stages)
    for tau in (5,10,20):
        headers=['종류','교사','필터','단계','기준점','통과영상(1+/6+)','통과코너(전체/채점가능)',
                 f'통과 정밀도≤{tau}px %',f'탈락 PCK≤{tau}px(누락실패) %',f'탈락 관측점 정밀도≤{tau}px %','탈락점(전체/관측)']
        rows=[]
        for r in stages:
            def pp(v):return 'NA' if v is None else pct(v)
            rows.append([r['kind'],r['teacher_role'],r['filter'],r['stage'],r['panel'],
                         f"{r['frames_nonempty']}/{r['frames_ge6']}",
                         f"{r['total_mask_corners']}/{r['annotated_eligible_selected_ref_count']}",
                         pp(r[f'selected_ref_precision_le{tau}px']),
                         pp(r[f'rejected_ref_PCK_le{tau}px_full_denominator']),
                         pp(r[f'rejected_ref_precision_le{tau}px_observed_only']),
                         f"{r['rejected_ref_count']}/{r['rejected_ref_observed_count']}"])
        (OUT/f'tables/T3_DETAIL_{tau}PX.md').write_text('[확인] 원 함수와 최종 판정 일치 확인 후 계산. 최종 선택 가설과 원 RANSAC 합의·LOO/flip 중앙값을 고정한 누적 진단이다. 단계별 가설 재선택 실험이 아니다. 전체 통과 코너 중 기준점이 있는 점만 정밀도를 채점한다. 탈락 점은 누락을 실패로 센 값과 관측점에 한한 정밀도를 구분했다.\n\n'+mdtable(headers,rows))
    ci=loadcsv('T2_PRIMARY_CI.csv')
    cirows=[r for r in ci if r['unit']=='frame' and r['metric']=='PCK10']
    concise=mdtable(['종류','비교(후보−비교군)','ΔPCK10 %p','영상 단위 95% CI','판정'],[
        [r['kind'],r['comparison'],val(r['delta_pp']),f"[{val(r['ci_low_pp'])}, {val(r['ci_high_pp'])}]",r['decision']] for r in cirows])
    (OUT/'tables/T2_PCK10_OVERVIEW.md').write_text('[확인] 영상 단위 CI. 같은 녹화의 상관을 반영하지 않아 좁을 수 있다. 녹화 단위 CI는 모든 모집단에서 미계산이다. 다중 비교 보정 없음.\n\n'+concise)
    t6=loadcsv('T6_CROSS_TYPE.csv')
    cross=[r for r in t6 if r['training_kind'] in KINDS and r['training_kind']!=r['evaluation_kind']]
    assert len(t6)==72 and len(cross)==36
    csvwrite(OUT/'tables/T6_CROSS_ONLY.csv',cross)
    (OUT/'tables/T6_CROSS_ONLY.md').write_text('[확인] 두 기존 SCORES의 전체72개 모델×종류 행 중 실제 교차 종류36행만 표시한다. 공통18행과 동일종류18행은 T6_CROSS_TYPE.csv에 문맥으로 보존했다. 새 추론0회.\n\n'+mdtable(
        ['소스 실험','학습 종류','평가 종류','모델','PCK5 %','PCK10 %','PCK20 %','coverage %'],
        [[r['source_experiment'],r['training_kind'],r['evaluation_kind'],r['model'],
          val(r['PCK5_percent']),val(r['PCK10_percent']),val(r['PCK20_percent']),val(r['coverage_percent'])] for r in cross]))
    absrows=[r for r in metrics if r['population'].startswith('DEV292_')]
    primary=mdtable(['종류','모델','PCK10 %','coverage %','조건부 정밀도10 %'],[
        [r['kind'],r['role'],pct(r['PCK10']),pct(r['coverage']),pct(r['conditional_precision10'])] for r in absrows])
    summary='''# DOPE 자기학습 졸업논문 분석 Batch 1

[확인] 기존 `79ba5b3` 결과를 분석했다. 학습·재학습·임계값·하이퍼파라미터 변경은 0회다. 새 신경망 추론은 T3의 고정 교사 7개에 대한 원본·반전 1,752회뿐이다. T1(b)는 원래 추론 경로를 인증하지 못해 미실행했고 T4는 기존 예측으로 계산했다.

[확인] 주 평가 DEV292는 PLASTIC128장/985점, WOOD45장/346점, GREEN119장/943점이다. PCK는 원본 픽셀의 같은 의미 채널0–7에서 오차≤τ인 점/전체 기준점이며 누락은 실패다. 중심점과 대칭 재정렬은 제외했다. 합성 추가학습 대조군은 종류 공통 R1/R2 한 모델씩이다. T3의 자기학습 모델 교사는 종류별 R1만 사용했고, 공통 R0를 별도로 검사했다.

## 먼저 확인된 계획의 오류와 한계

[확인] **7×7 soft-argmax(T=.1)가 R0의 원래 추론 디코더였다는 전제는 확인되지 않았다.** 기존 INVENTORY의 해당 항목은 학습 보조 손실을 설명한다. 원래 추론과 같다고 간주한 재평가는 하지 않았다. `preflight/T1B_SKIPPED.md`에 실제 코드 위치와 근거를 남겼다.

[확인] **녹화 단위 CI는 전부 미계산이다.** PLASTIC은 확인된 ID7개와 출처가 상충하는5장이 있고 GREEN119장은 ID 근거가 없다. WOOD는 녹화2개로 최소3개 조건을 충족하지 않는다. PLASTIC184도 ID9개와 상충5장이 있다. 확인된 일부만으로 모집단을 바꾸거나 세션명을 녹화 ID로 대체하지 않았다. 아래 영상 단위 CI는 연속 프레임 상관 때문에 좁을 수 있으며, 독립 녹화에 대한 일반화 근거로 사용할 수 없다.

[확인] 보호 물리 FINAL은 양성229장(PLASTIC149/WOOD80)이며 독립6D 정답 존재는 허용 메타데이터만으로 확인하지 못했다. WOOD 학습1000장은 메타데이터 registry상 FINAL과 REC_001/002가 겹친다. 선언 이미지 SHA 교집합은0이다. 이 registry는 이미지 중첩의 연결 성분이므로 카메라 녹화 로그를 새로 인증한 결과가 아니다. FINAL의 RGB·정답 본문은 열지 않았다. PLASTIC184 보조 분석은 이미 생성된 SCORES만 사용했다.

[확인] **PLASTIC184에는 보호 FINAL과 이미지 SHA가 같은56장이 포함된다.** T2의 모델 간 기준점 대응 확인에서 기존 파생 SCORES의 `reference.xy`를 읽었고, CI는 저장된 `error_px`만으로 계산했다. 이 사실을 확인한 후 게이트5에 따라 게시를 중단했다. 이후 2026-10-27 사용자가 **“기존 파생 SCORES 사용만 허용하고 push”**라고 명시해 해당 보조 분석을 포함했다. `preflight/GATE5_STOP.md`에 중단 경위를, `preflight/GATE5_RESOLUTION.json/md`에 허용 범위를 보존했다. 독립 감사도 같은 파생 점수를 읽었다. 원본 FINAL RGB·annotation 본문 열람은0회이며 이 허용은 원본 정답 열람으로 확대하지 않는다. CI·seed·사례를 바꾸지 않았고 승인 시점18개 CSV의 SHA를 고정했다.

[확인] 실내·실외의 명시 촬영 메타데이터는 없다. 주간·야간과 출처 도메인은 `tables/T0_CONDITIONS.md`에 별도로 보존했다. 모르는 조건을 영상 외관으로 추정하지 않았다.

[확인] T5의 정답 기반 최대·중앙값·최소 사례 선정과 원 게이트3는 충돌했으며, 사용자가 **T5의 사전 고정 규칙에 한해서만 예외를 허용**했다. 그 외 정답은 채점에만 사용하고 임계값·체크포인트를 고르지 않았다.

## 현재 평가 경로의 주 결과

'''+primary+'''
관측점이 늘어 coverage가 높아지는 것과 검출된 점의 조건부 정밀도가 높아지는 것을 구분해야 한다. 부호 오차와 조건부 중앙값은 `tables/T1_NATIVE_BIAS.md`, 모든 절대 지표는 `tables/T1_EXISTING_METRICS.md`에 있다.

## 고정한 paired bootstrap

'''+concise+'''
[확인] 영상 복원추출10,000회, seed20261026, 같은 표본을 두 모델에 적용한 percentile95 CI다. 하한>0이면 “높았다”, 상한<0이면 “낮았다”, 0을 포함하면 “차이 불명확”이다. **다중 비교 보정은 하지 않았다.** PCK5/20·coverage·조건부 정밀도 및 모든 미계산 녹화 행은 `tables/T2_PRIMARY_CI.md`에 있다. GREEN manual-click600점과 PLASTIC184는 `tables/T2_GREEN_MANUAL_CI.md`, `tables/T2_PLASTIC184_CI.md`의 별도 결과다.

## 필터 검증과 해석

'''
    c=proof['checked']
    summary+=f"[확인] 원 필터와 래퍼의 최종 판정이 main/no-flip 각각 {c['main']['frames']}/{c['noflip']['frames']} 교사·프레임, 각각 {c['main']['nine_channel_comparisons']}/{c['noflip']['nine_channel_comparisons']} 채널 비교에서 100% 일치했다. 불일치0이며 정밀도 계산 이전 검증이다. 모든 등록 가설도 별도 검사했다. `preflight/T3_FILTER_EQUIVALENCE.json`.\n\n"
    summary+='''[확인] main의 s0는 원 함수대로 원본과 flip-back의 유효·신뢰도≥.3·영상 내부 조건의 교집합이다. s1은 원 RANSAC 합의≥6, s2는 LOO 점·중앙값, s3는 flip 점·중앙값, final은≥6점이다. 원 함수의 **최종 선택 가설에 조건부인 누적 단계 진단**이며, 각 단계에서 가설·합의·중앙값을 다시 맞추는 독립 ablation은 아니다. 비교용 신뢰도만 선별은 original-only와 both-view를 구분했다. 정밀도≤5/10/20px와 탈락 점 통계는 `tables/T3_DETAIL_5PX.md`, `T3_DETAIL_10PX.md`, `T3_DETAIL_20PX.md` 및 전체 CSV에 있다. GREEN manual-only600점도 따로 표기했다.

## PnP와 정성 사례, 교차 종류

[확인] T4는 모서리≥6개 검출과 등록 치수·기존 K로 EPnP+LM을 수행하고 모든8개 모서리 깊이가 양수인지를 계산했다. 정답을 사용하지 않았다. **PnP 성립률은 독립6D 자세 정확도가 아니다.** 주21개·교차 종류 보조24개 모델/종류 행을 `tables/T4_PNP_AVAILABILITY.md`에서 확인할 수 있다. 카메라 K는 DEV annotation의 기존 camera_data만 읽어 크기를 대조했으며 새로 추정하지 않았다. WOOD의 K는 기존 SENSOR_PROFILE_SCALED 메타데이터라는 제한이 있다. 근거는 저장소 `data/evaluation/pallet_eval_v1/dev_existing/sessions/wood_183705/session.json:8–9` 및 `wood_184309/session.json:8–9`다.

[확인] T5는 종류별 영상당 ΔPCK10(flipR2−controlR2)의 최대·중앙값에 가장 가까운 영상·최소를 ID 오름차순 동률 규칙으로 고정해9장을 선정했다. 예쁜 사례로 교체하지 않았다. 모든292장 차이 CSV와 선정표, `figures/`의9개 PNG·PDF 및 렌더 QA를 보존했다. T6는 두 기존 SCORES의 전체 모델×종류72행(공통18·동일종류18·교차종류36)을 보존했고, 실제 교차 종류36행은 `tables/T6_CROSS_ONLY.csv/md`에 따로 정리했다. 새 추론은 없다.

## 논문에 사용할 수 있는 문장

[확인] “종류별 자기학습을2회 수행한 DOPE는 합성 데이터만 추가학습한 대조군과 비교할 때, 영상 단위 paired bootstrap에서 목재 PCK@10이6.65%p 높았고(95% CI1.73–11.58), 일반 플라스틱은3.55%p 낮았으며(−6.36–−0.81), 녹색 플라스틱의 차이는 불명확했다(−0.53%p, −3.94–2.86). 이 신뢰구간에는 녹화 내 프레임 상관과 다중 비교 보정이 반영되지 않았다.”

[추정] 필터가 거른 점의 오차 분포와 coverage 변화는 성능 차이를 설명하는 단서가 될 수 있다. 본 분석만으로 전처리·디코더·자기학습의 개별 인과 효과나 독립6D 자세 개선을 입증했다고 해석하지 않는다. GREEN에는 PnP 생성 기준점이 포함되므로 manual600점 보조 결과를 함께 제시해야 한다.

## 재현·무결성

[확인] `REQUEST.md`가 원 지시문, `FIXED_PLAN.json`이 계산 전 고정 규칙이다. `logs/RUN_LOG.jsonl`에 실제 명령·seed·입력 및 코드 SHA256, `code/`에 계산·검증 코드를 남겼다. `hashes/SNAPSHOT_BEFORE.json`, `SNAPSHOT_AFTER.json`, `COMPARISON.json`에서 두 실험 폴더598개 파일 및 R0가 모두 변경되지 않았음을 확인했다.

코드는 원 실행 위치 `/home/minjae/Documents/pallet-research/dope-thesis-analysis-batch1-v1/code/`와 기존 소스 실험 두 폴더를 전제로 한다. 저장소 사본을 재현하려면 이 새 분석 폴더로 복원해 실행한다. 기존 산출물을 덮어쓰거나 완료된 추론을 다시 수행하는 용도로 실행하지 않는다. 정확한 실행 명령은 RUN_LOG에 있다.
'''
    (OUT/'README_KO.md').write_text(summary)
    (OUT/'README.md').write_text('# DOPE thesis analysis Batch1\n\n[국문 분석 보고서](README_KO.md) · [원 요청](REQUEST.md) · [표](tables/) · [그림](figures/) · [실행 로그](logs/RUN_LOG.jsonl) · [무결성 비교](hashes/COMPARISON.json)\n')
    auxiliary=OUT/'tables/T2_PLASTIC184_CI.md'
    marker='## 보호 FINAL 파생 점수 사용 범위'
    if marker not in auxiliary.read_text():
        with auxiliary.open('a') as f:
            f.write('\n'+marker+'\n\n[확인] PLASTIC184 중56장은 보호 FINAL과 선언 이미지 SHA가 같다. 기존 파생 SCORES의 reference.xy는 모델 간 대응 검사에, error_px는 CI 계산에 사용했다. 원본 FINAL RGB·annotation 본문 열람은0회다. 이 사실을 공개하고 2026-10-27 사용자의 “기존 파생 SCORES 사용만 허용하고 push” 답변에 따라 포함했다. preflight/GATE5_RESOLUTION.json의 좁은 허용 범위이며 주 DEV292 결과와 분리한다.\n')
    for rel,expected in authorization['frozen_quantitative_outputs'].items():
        assert sha(OUT/rel)==expected,('Report rebuild changed frozen quantitative result',rel)
    log('step_complete',task='integrated_report',absolute_metric_rows=len(metrics),T3_detail_tables=3)

if __name__=='__main__':main()

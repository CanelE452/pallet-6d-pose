"""Publish post-E2 diagnosis solely from computed machine-readable evidence."""
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np

from analyze import C, OUT, ROOT, ARMS, SHOW, save, stats


def pct(x):return 'NA' if x is None else f'{100*x:.2f}'
def num(x):return 'NA' if x is None else f'{x:.3f}'
def table(headers, rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows]+['']


def main():
    load=lambda n:C.read(OUT/n)
    d1=load('D1_TRANSITIONS.json');rows=d1['rows'];bands=load('D2_ERROR_BANDS.json');vis=load('D3_VISIBILITY_STATS.json')
    oracle=load('D4_ORACLE_HEADROOM.json');cluster=load('D5_FAILURE_CLUSTERS.json');decision=load('DECISION_REINTERPRETATION.json');gallery=load('D3_REVIEW_SELECTION.json')
    # M.summary has a fixed-branch field inherited from its scratch R0 rows. This
    # field has no defined meaning for a mixed-candidate corner oracle. Keep the
    # original draft evidence; publish an explicit, schema-clean validated copy.
    clean=copy.deepcopy(oracle);clean['ORACLE_CORNER'].pop('E_fixed',None)
    clean['schema_note']='The draft D4_ORACLE_HEADROOM.json ORACLE_CORNER.E_fixed is an inherited placeholder, not a valid mixed-candidate metric. Removed here; all requested PCK/median/P90/matching values unchanged.'
    save('D4_ORACLE_HEADROOM_VALIDATED.json',clean)
    matrix=d1['transition']['10'];matrix20=d1['transition']['20'];net=matrix['GOOD->BAD']-matrix['BAD->GOOD']
    lines=['# D1 — A10 → A11 코너 전이','',f'[확인] 동일 93장/713코너. A10 10px 정답 {d1["correct10"]["A10"]}, A11 {d1["correct10"]["A11"]}. 잃음 {matrix["GOOD->BAD"]} − 얻음 {matrix["BAD->GOOD"]} = 순손실 {net}.','']
    for name,m in [('10px',matrix),('20px',matrix20)]:
        lines+=['## '+name,'']+table(['[확인] A10 \\ A11','GOOD','BAD'],[['GOOD',m['GOOD->GOOD'],m['GOOD->BAD']],['BAD',m['BAD->GOOD'],m['BAD->BAD']]])
    lines+=['[확인] 25개 감소가 아니라 25개 손실과 19개 획득의 차이가 6개다. 각 코너의 좌표·오차·분기·매칭·가시성 provenance는 [전체 CSV](D1_TRANSITIONS.csv)에 있다.','',
        '[확인] canonical GT identity로 join했고, 각 후보의 승인된 whole-object branch를 역매핑했다. 매칭 실패의 저장 좌표 기하거리와 실제 채점 벌점 800px은 구분했다. 해당 실패도 동일 분모에 유지했다.','']
    save('D1_SUMMARY.md','\n'.join(lines))
    lines=['# D2 — 초기 R0 오류 구간','', '[확인] B0 ≤5, B1 >5–10, B2 >10–20, B3 >20–40, B4 >40px. B5는 매칭 실패를 우선 분리했다. good-damage 조건은 R0 <5px로 B0 경계와 다르다.','']
    lines+=table(['[확인] 구간','코너','A10 PCK10 %','A11 PCK10 %','획득','손실','hard 복구','good 손상','medium 손상'],
        [[k,v['count'],pct(v['models']['A10']['PCK10']),pct(v['models']['A11']['PCK10']),v['gained'],v['lost'],v['hard_recovery'],v['good_damage'],v['medium_damage']] for k,v in bands.items()])
    lines+=table(['[확인] 구간','A10 mean/median','A11 mean/median','A10 PCK20 %','A11 PCK20 %','개선/악화','≥5px 개선/악화','≥10px 개선/악화'],
        [[k,f'{num(v["models"]["A10"]["error"]["mean"])}/{num(v["models"]["A10"]["error"]["median"])}',f'{num(v["models"]["A11"]["error"]["mean"])}/{num(v["models"]["A11"]["error"]["median"])}',pct(v['models']['A10']['PCK20']),pct(v['models']['A11']['PCK20']),f'{v["improved"]}/{v["worsened"]}',f'{v["improved5"]}/{v["worsened5"]}',f'{v["improved10"]}/{v["worsened10"]}'] for k,v in bands.items()])
    hard=Counter(r['band'] for r in rows if r['hard_recovery']);hardnew=Counter(r['band'] for r in rows if r['hard_recovery'] and not r['A10']['PCK10'])
    lines += [f'[확인] R0 >20 → A11 ≤10 복구 {sum(hard.values())}개는 `{dict(hard)}`. 그중 A10도 이미 맞힌 코너를 제외하면 `{dict(hardnew)}`다. >40px 구간에서는 10px 이내 복구를 확인하지 못했다.','',
        f'[확인] A10-only 손실은 B1 {bands["B1"]["lost"]}, B2 {bands["B2"]["lost"]}개이며 다른 구간 손실은 {sum(v["lost"] for k,v in bands.items() if k not in ("B1","B2"))}개. R0 <5 → A11 >10 손상은 {sum(r["good_damage"] for r in rows)}개이며, A10도 이미 실패하던 점일 수 있어 전이 손실과 구분해야 한다.','',
        '[추정] 관측된 trade-off는 20–40px 일부 복구와 5–20px 보존 손실이다. 모든 large error를 복구하는 일반적 능력이나 실제 외부 가림 복원을 증명하지는 않는다.','']
    save('D2_ERROR_BANDS.md','\n'.join(lines))
    lines=['# D3 — 가시성/가림 출처 감사','',f'[확인] **{vis["status"]}**. 외부 가림 복구가 없다는 뜻이 아니라 외부/자기 subtype을 확인할 독립 판정이 부족하다는 뜻이다.','',
        f'[확인] 기존 prediction-blinded 사람 검토 {vis["manual_reviewed_corners"]}/713코너가 있다. 좌표 source=unknown과 가시성 검토 여부는 다른 정보다. 사람의 `o`는 occluded 일반 판정이며 subtype GT는 아니다.','']
    for key,title in [('verified_visibility_class','검증 가능한 subtype'),('manual_visibility','사람이 판정한 일반 가시성'),('auto_proxy_state','기존 자동 기하/depth 후보 — 정답 아님'),('stored_reason','annotation 표기 — 혼합 provenance')]:
        lines+=['## '+title,'']+table(['[확인] 그룹','코너','R0 %','N2 %','Replay %','A10 %','A11 %','획득/손실','hard 복구','good 손상'],
            [[k,v['count'],*[pct(v['models'][a]['PCK10']) for a in SHOW],f'{v["gained"]}/{v["lost"]}',v['hard_recovery'],v['good_damage']] for k,v in vis[key].items()])
    lines += [f'[확인] A11 hard recovery {vis["external_hard_recovery_unknown"]}개 중 manual external subtype으로 확인된 개수는 {vis["verified_external_hard_recovery"]}개다. 자동 SELF_VISIBLE/SELF_OCCLUDED/EXTERNAL_CANDIDATE는 보조 자료로만 집계했다. image-level occlusion을 코너별 GT로 복사하지 않았다.','',
        f'[확인] [검토 갤러리](D3_REVIEW_GALLERY.html): {gallery["corners"]}개 고유 코너, {gallery["frames"]}개 프레임. 모든 A11-only/A10-only/hard recovery/good damage와 고정 랜덤 20코너를 포함하고 중복만 합쳤다. full RGB, bbox crop, GT와 다섯 후보의 좌표/오차를 표시한다.','',
        '[확인] 현재 사람의 새 검토를 수행하지 않았으며 annotation도 변경하지 않았다. 모델 출력을 보여주는 사후 검토 자료이므로 향후 독립 visibility 정답으로 쓰려면 prediction-blinded review가 별도로 필요하다.','']
    save('D3_VISIBILITY_STATS.md','\n'.join(lines))
    lines=['# D4 — A10/A11 포함 후보 oracle','',
        '[확인] frame oracle은 **평균 whole-object 코너 오차 최소** 후보 하나를 선택한다. 과거 E1의 PCK 우선 선택과 목적이 다르므로 두 headroom의 차이를 오직 새 후보 효과라고 해석할 수 없다. 모든 oracle은 GT를 사용한 사후 진단이며 실제 배포 성능이 아니다.','']
    combined={**oracle['baselines'],'ORACLE_FRAME':oracle['ORACLE_FRAME'],'ORACLE_CORNER_SECONDARY':oracle['ORACLE_CORNER'],'A10_A11_FRAME_ORACLE':oracle['A10_A11_FRAME_ORACLE']}
    lines+=table(['[확인] 후보','PCK10 %','PCK20 %','matched median','matched P90','detected/matched/frames'],
        [[k,pct(v['PCK']['10']),pct(v['PCK']['20']),num(v['matched_pooled_corner8_median_px']),num(v['matched_pooled_corner8_P90_px']),f'{v["detected"]}/{v["matched"]}/{v["total_frames"]}'] for k,v in combined.items()])
    lines+=table(['[확인] 기준','frame headroom pp','corner headroom pp'],[[a,f'{oracle["headroom"][a]:+.6f}',f'{oracle["secondary_headroom"][a]:+.6f}'] for a in ('N2','A10','A11')])
    lines+=table(['[확인] 후보','deterministic best /93 (%)','unique best /85 matched (%)'],
        [[a,f'{oracle["deterministic_best_frequency"].get(a,0)} ({100*oracle["deterministic_best_frequency"].get(a,0)/93:.2f})',f'{oracle["unique_best_frequency"].get(a,0)} ({100*oracle["unique_best_frequency"].get(a,0)/85:.2f})'] for a in ARMS])
    lines += [f'[확인] A11 평균오차가 A10보다 낮은 프레임 {oracle["A11_lower_frame_error"]}, A10이 낮은 프레임 {oracle["A10_lower_frame_error"]}, 동률 {oracle["both_equal_frame_error"]}. 동률인 매칭 실패 프레임은 고정 후보 순서상 R0로 집계되므로 unique-best와 구분한다.','',
        f'[확인] 모든 후보가 10px 이내 좌표를 제공하지 못한 코너 {oracle["all_candidates_fail_PCK10"]}/713. secondary corner oracle도 동일 canonical GT identity의 승인된 whole-object 분기를 사용했지만 후보를 점별 혼합하므로 배포 가능한 방법이 아니다.','',
        f'[확인] N2 기준 frame headroom {oracle["headroom"]["N2"]:.6f}pp는 반올림 전 3pp 미만이다. A11은 {oracle["unique_best_frequency"].get("A11",0)}개 프레임에서 단독 best로 일부 보완성은 있지만, 이것만으로 selector 학습의 우선순위가 높다고 할 수 없다. 평균오차 최소 선택이 PCK10 최대 선택을 보장하지도 않는다.','',
        '[확인] [검증된 oracle JSON](D4_ORACLE_HEADROOM_VALIDATED.json)을 사용한다. 최초 계산 JSON의 secondary E_fixed는 scratch R0에서 상속된 비정의 필드라 검증본에서 제거했다. 요청된 PCK/median/P90/검출/매칭 수치는 동일하며 기존 E2 결과는 전혀 수정하지 않았다.','']
    save('D4_ORACLE_HEADROOM.md','\n'.join(lines))
    lines=['# D5 — 실패 그룹의 기술 통계','', '[확인] selector/classifier 학습이나 threshold 탐색은 하지 않았다. 코너들은 프레임에 군집되어 있으므로 독립 표본의 유의성 검정으로 해석하지 않는다.','']
    lines+=table(['[확인] 그룹','n','bbox area med','aspect med','R0 conf med','A11 move med','A10/A11 disagreement med'],
        [[g,v['count'],num(v['features']['bbox_area']['median']),num(v['features']['bbox_aspect']['median']),num(v['features']['R0_kp_confidence']['median']),num(v['A11_native_correction']['median']),num(v['features']['native_disagreement_px']['median'])] for g,v in cluster['groups'].items()])
    for g,v in cluster['groups'].items():
        lines += [f'[확인] `{g}` canonical 코너 빈도 `{v["corners"]}`; 세션 빈도 `{v["sessions"]}`.','']
    for field,bins in cluster['bins'].items():
        lines += ['## '+field+' (고정 기술통계 구간)','']+table(['[확인] 구간','n','획득','손실','good 손상/분모'],
            [[b,s['count'],s['gained'],s['lost'],f'{s["good_damage"]}/{s["good_denominator"]}'] for b,s in bins.items()])
    corr=cluster['correlation'];lines += [f'[확인] 매칭된 {corr["n"]}프레임에서 disagreement와 A11−A10 평균오차의 Pearson 상관은 {corr["pearson_signed_delta"]:.3f}, 차이 절댓값과는 {corr["pearson_absolute_delta"]:.3f}다. 큰 불일치는 차이의 크기와 연관되지만 어느 쪽이 맞는지의 방향 신호로 검증된 것은 아니다.','',
        '[확인] R0 confidence는 모든 그룹에서 높고 획득/손실 그룹의 bbox 크기·aspect 분포도 겹친다. bbox area는 거리의 직접 측정값이 아니다. verified visible point count는 일부만 검토된 점의 수이지 실제 visible 수가 아니다. primary93에 PnP 가능 flag는 없으므로 UNKNOWN으로 남겼다.','',
        '[추정] 큰 보정량 하나로 손상을 거르기 어렵다. 가장 큰 이동 구간에도 획득이 있고, good-damage 분모는 작은 구간에서 극히 작다. 기존 RGB utility selector의 GREEN 실패를 재현하는 동일 feature 재학습을 제안할 근거는 부족하다.','']
    save('D5_FAILURE_CLUSTERS.md','\n'.join(lines))
    lines=['# 역사적 gate와 인과 해석의 분리','',f'[확인] HISTORICAL_GATE_RESULT = **{decision["HISTORICAL_GATE_RESULT"]}**. 기존 E2_DECISION.json을 그대로 보존했다.','',
        '[확인] 원래 report.py의 recovery 조건은 A11≥R0+3pp 및 A11≥기존 N2/N3/Replay 최고였다. A11>A10 조건은 없다. 그래서 RECOVERY_OK는 인공 가림 학습 효과 입증과 동의어가 아니다.','',
        f'[확인] CAUSAL_INTERPRETATION = **{decision["CAUSAL_INTERPRETATION"]}**. A11−A10 PCK10 {decision["A11_minus_A10_PCK10_pp"]:+.6f}pp, PCK20 {decision["A11_minus_A10_PCK20_pp"]:+.6f}pp, matched median {decision["A11_minus_A10_matched_median_px"]:+.3f}px, P90 {decision["A11_minus_A10_matched_P90_px"]:+.3f}px.','',
        f'[확인] 전체 프레임 평균오차 차이는 {decision["A11_minus_A10_full_frame_mean_px"]:+.3f}px로 다른 방향일 수 있다. 하나의 지표 개선을 전체 성공으로 바꾸지 않는다. 단일 seed의 파일럿이며 통계적 효과 부재를 증명한 실험은 아니다.','']
    save('DECISION_REINTERPRETATION.md','\n'.join(lines))
    route=dict(status='[추정]',primary='C_DATA_OCCLUSION_GAP',secondary=None,
        A_conditions=dict(headroom_at_least_3=oracle['headroom']['N2']>=3,complementary=True,validated_inference_signal=False),
        B_conditions=dict(verified_external_recovery_attribution=False,primary_geometry_oracle_available=False),
        C_evidence=dict(single_recording=True,adjacent_clean_images=253,real_occ_A11_minus_A10_pp=decision['A11_minus_A10_PCK10_pp'],
                        source_stress=decision['source_stress_PCK10']),
        D_conditions=dict(low_frame_headroom=True,geometry_insufficiency_established=False),
        new_training_started=False,causal_data_diversity_failure_proven=False,
        proposal=dict(sizes=[8,32,128],inputs=['CLEAN','OCC actual frozen R0 rerun'],
            source_replay='same fixed source protocol',total_updates=300,real_exposures=2400,seed=1,
            split='unlabeled clean multi-session training disjoint from evaluation by recording/image hash',
            selection='fixed nested subsets; count recordings/viewpoints separately; no GT-based pseudo-label selection',
            inference='same frozen detector; refiner only; last checkpoint; no E6 student self-training',
            analysis='within-size A11-like vs A10-like direct contrast plus normal/source preservation, no best-of-6 deployment',
            if_insufficient_sessions='stop preparation and request dataset choice; do not duplicate adjacent frames to claim diversity'))
    save('NEXT_STAGE_PLAN.json',route)
    save('NEXT_STAGE_PLAN.md','\n'.join(['# 다음 분기 — 계획만 작성','',
        '[추정] **Primary: ROUTE C — 데이터 다양성/인공–실제 가림 차이 분리. 보조 route는 지정하지 않는다.**','',
        '[확인] 한 recording의 인접 253개 이미지를 사용했다. 실제 가림 A11−A10은 음수이고, 합성 stress 복구 향상과 실제 가림 성능은 분리되어 있다. 데이터 다양성이 원인이라고 확정한 것은 아니다.','',
        '[추정] 다음 한 실험은 **다중 세션 clean 8/32/128 × CLEAN/OCC refiner-only 통제 비교**다. 총 update/실사 노출, 초기 checkpoint, source replay, optimizer, seed, 가림 recipe를 고정한다. 각 크기의 subset은 사전 고정하고 recording/viewpoint 다양성을 함께 기록한다. 실제 가림 direct contrast와 정상/source 보존을 함께 평가한다.','',
        '[확인] 기존 한 세션 결과와 새 결과 차이만으로 다양성의 인과 효과를 증명하지 못한다. 규모별 nested subset에서도 이미지 수와 다양성이 함께 변할 수 있다. 순수 다양성 분리가 필요하면 별도의 matched-size 세션 통제가 필요하며, 이 계획 자체를 검증 완료로 주장하지 않는다.','',
        '[추정] 새로운 clean 세션이 충분하지 않으면 준비 단계에서 멈춘다. 같은 영상의 인접 프레임을 복제/추가해 독립 데이터 규모가 늘었다고 해석하지 않는다. 평가 recording 제외 및 고정 pseudo gate는 유지한다.','',
        f'[확인] ROUTE A는 frame headroom {oracle["headroom"]["N2"]:.6f}pp<3이며 검증된 inference-time 선택 신호가 없다. ROUTE B는 외부/자기 가림 독립 subtype와 primary93 geometry oracle이 부족하다. ROUTE D 역시 geometry로 해결 불가능함이 검증되지 않아 곧바로 구조를 바꿀 근거가 부족하다.','',
        '[확인] 이 문서는 계획이다. 새 모델 학습, 추가 fine-tuning, E3~E6, 새 pseudo-label 생성, commit/push는 실행하지 않았다.','']))
    gooddamage=[r for r in rows if r['good_damage']]
    post=['# Post-E2 원인분해 최종 보고','',
        f'[확인] 기존 결과를 재채점/분해한 진단이다. **{decision["CAUSAL_INTERPRETATION"]}**, 기존 판정 {decision["HISTORICAL_GATE_RESULT"]} 보존. 새 학습 없음.','',
        '## 1. 왜 A11이 A10보다 6코너 낮았나?','']
    post+=table(['[확인] A10 \\ A11','10px 이내','10px 초과'],[['10px 이내',matrix['GOOD->GOOD'],matrix['GOOD->BAD']],['10px 초과',matrix['BAD->GOOD'],matrix['BAD->BAD']]])
    post += [f'[확인] {matrix["GOOD->BAD"]}개 손실 − {matrix["BAD->GOOD"]}개 획득 = {net}개 순손실. A10 {d1["correct10"]["A10"]}/713 → A11 {d1["correct10"]["A11"]}/713. 20px 전이는 획득/손실 각각 {matrix20["BAD->GOOD"]}/{matrix20["GOOD->BAD"]}로 순변화가 없다. [D1](D1_SUMMARY.md).','',
        '## 2. 어느 초기 오류에서 복구했나?','',
        f'[확인] 10px 정답 획득 B0/B1/B2/B3/B4 = '+ '/'.join(str(bands[k]['gained']) for k in ('B0','B1','B2','B3','B4'))+'개. R0>20px hard recovery는 '+str(sum(hard.values()))+'개이며 '+str(dict(hard))+'에 속한다. >40px 복구는 확인하지 못했다. [D2](D2_ERROR_BANDS.md).','',
        '## 3. 실제 외부 가림 코너가 복구됐나?','',
        f'[확인] **확정 불가**. hard recovery {sum(hard.values())}개 중 manual generic occluded는 {vis["manual_visibility"].get("OCCLUDED_UNSPECIFIED",{}).get("hard_recovery",0)}개지만 external/self subtype은 검증되지 않았다. 사람 검토 가시성 기록은 {vis["manual_reviewed_corners"]}코너에 있고, 나머지는 auto/unknown과 분리했다. `EXTERNAL_OCCLUSION_CORNER_RECOVERY_UNVERIFIED`. [D3](D3_VISIBILITY_STATS.md).','',
        '## 4. 정상/중간 코너 손상은 어디에서 생겼나?','',
        f'[확인] A10-only 손실 {matrix["GOOD->BAD"]}개는 모두 R0 5–20px: B1 {bands["B1"]["lost"]}, B2 {bands["B2"]["lost"]}. R0<5 → A11>10 손상 {len(gooddamage)}개는 `'+', '.join(f'{r["frame_id"]}/G{r["corner_id"]}' for r in gooddamage)+'`이며 전이 상태 '+str([r['transition10'] for r in gooddamage])+'. 초기 정상점 손상과 A10 대비 손실은 다른 집계다.','',
        '## 5. 후보들은 보완적인가?','',
        f'[확인] A11이 A10보다 낮은 프레임 평균오차 {oracle["A11_lower_frame_error"]}장, 반대 {oracle["A10_lower_frame_error"]}장, 동률 {oracle["both_equal_frame_error"]}장. A11 단독 best {oracle["unique_best_frequency"].get("A11",0)}장으로 아주 소수의 hard 사례에서만 best인 것은 아니다. 모든 후보가 실패하는 코너는 {oracle["all_candidates_fail_PCK10"]}/713이다.','',
        '## 6. frame oracle headroom은?','',
        f'[확인] frame oracle PCK10 {pct(oracle["ORACLE_FRAME"]["PCK"]["10"])}%, N2/A10/A11 대비 '+ '/'.join(f'{oracle["headroom"][a]:+.6f}' for a in ('N2','A10','A11'))+'pp. 점별 비배포 upper bound는 '+pct(oracle['ORACLE_CORNER']['PCK']['10'])+'%. [D4](D4_ORACLE_HEADROOM.md).','',
        '## 7. selector를 만들 가치가 있나?','',
        f'[추정] 우선순위는 낮다. 지정된 frame oracle 기준 {oracle["headroom"]["N2"]:.6f}pp<3이다. 후보 차이의 크기는 관측되지만 방향을 고르는 신호는 검증하지 못했다. near-threshold인 점과 단일 DEV의 한계는 남는다. [D5](D5_FAILURE_CLUSTERS.md).','',
        '## 8. geometry/visibility 분리가 더 직접적인가?','',
        '[추정] 구조적으로 가능한 가설이지만 현재 subtype GT와 primary geometry headroom이 없어 다음 1순위로 확정할 근거가 부족하다. 현재 geometry proxy를 실제 외부 가림 복구 증거로 격상하지 않는다.','',
        '## 9. 데이터 다양성 문제는 남았나?','',
        '[확인] 253장은 한 촬영분의 인접 이미지다. 합성 stress에서 A10/A11이 초기보다 좋아졌지만 실제 가림 direct contrast는 음수다. [추정] 데이터/도메인 차이는 남은 후보 원인이지, 이 분석으로 확정된 단일 원인이 아니다.','',
        '## 10. 다음 한 가지 primary 실험','',
        '[추정] **ROUTE C: 다중 세션 clean 8/32/128 × CLEAN/OCC, 동일 총 노출의 보정기 통제 실험.** 보조 route 없음. [계획만 작성](NEXT_STAGE_PLAN.md).','',
        '## 검토 자료 및 무결성','',
        f'[확인] [코너 검토 갤러리](D3_REVIEW_GALLERY.html): {gallery["corners"]}코너/{gallery["frames"]}프레임. GT·R0·N2·Replay·A10·A11, full RGB와 bbox crop. 새 annotation 입력 없음.','',
        '[확인] [입력 감사](PRECHECK.md), [역사적 gate 재해석](DECISION_REINTERPRETATION.md), [검증](AUDIT.json). 원본 E2 및 checkpoint 불변, 새 학습/commit/push 없음. 추가 파일은 postmortem 하위에만 있다.','']
    save('POSTMORTEM_KO.md','\n'.join(post))
    print('REPORTS_READY')


if __name__=='__main__':main()

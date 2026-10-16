"""Report all saved LOO roots while preserving the original evaluated policy."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def save(p,r):
    with p.open('x') as f:json.dump(r,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def main():
    out=ROOT/'output_corrected';summary=json.loads((out/'SUMMARY.json').read_text());g=summary['groups']
    with gzip.open(out/'DECISION_ROWS.jsonl.gz','rt') as f:rows=[json.loads(s) for s in f]
    check=json.loads((ROOT/'PRIVATE_RAW_READER_RESULT.json').read_text());assert check['exit_code']==0 and check['checks']['passed']
    lines=['가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?',
        '', '**아직 전체 쉬움·중간245장에서 단순 대조보다 위치·회전 개선은 확인되지 않았다. LOO를 사용하되, 선택한 하나의 해와의 거리나 여러 해의 동의만으로 보정이 정확하다고 판단하면 안 된다는 것을 실제 저장 결과에서 확인했다.** 원래 채택133점 중69점에서는 최대 inlier 수가 같은 다른 LOO 해가 초기 N3를 더 선호했다. 반대로 그 해들이 모두 후보를 선호한64점 중22점도 참조 코너 오차가 악화됐다. 여러 해가 같은 관측 오차에 함께 영향을 받을 수 있으며, 동의 자체는 정확한 검증 기준이 아니다.',
        '', '자기 가림 H와 검사 대상 k는 LOO fit/score에서 제외됐고, 최종 NEW fit에는 H 초기좌표와center8이 들어가지 않았다. NEW R,t가 나오면 H를 재투영으로 대체하고 다시 관측처럼 넣어 refit하지 않는 계약은 유지한다. 마스크 한 점이 틀렸다는 이유로 frame 전체를 실패시키는 방법으로 바꾸지 않았다. 현재 문제는 이 제외 조건 이후의 **보정 후보 검증과 최종 가설 선택의 정확도**다.',
        '', '**실제 실행 범위.** 현재245 prediction 전부에서 실제 호출된 LOO423결정/189프레임, 저장된 raw/refit/LM root 진단12178개를 읽었다. 채택133점과 미채택290점 모두 남겼다. V27 채택133과 frame/corner/좌표/reference/gain을 대조했다. detector/head·새 PnP·새 projection·새 선 fit·새 학습·새 RGB·새 R,t 성능 채점·policy 변경은 모두0이다. 기존 full pipeline 시간에 이 감사 시간을 더해 새로운 처리시간처럼 보고하지 않는다.',
        '', '**두 root 집합은 GT 없이 실행 전에 고정했다.** A는 원 `dimension_constraint_allowed`와 기존 inlier≥4 조건을 만족하는 모든 저장 root다. B는 A에서 최대 inlier 수가 같은 root 전부다. 후보 gain, 참조 좌표, GOOD/BAD, 좌표 개선 여부로 집합을 고르지 않았다. 후보의 선호는 기존 `G=||native−p||²−||candidate−p||²`와1e−8 numerical tie만 사용했다. 두 집합은 대안의 local rank를 재검사하지 않은 **저장 합의 집합**이며 정확한 물리 자세나 calibrated predictive confidence set이라는 인증을 주지 않는다.',
        '', '코드 경로는 [V4 selection.py](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/selection.py#L135)의 H∪{k} LOO→full solved 저장, [V4 pose.py](../../../scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py#L203)의 root별 projected8/fit/inlier/score 저장이다. `observation_contract.cornerwise_records[].loo_solver.all_candidate_solutions[].projected[k]`를 읽었고 R,t로 투영을 새로 만들지 않았다. 원 선택기는 여러 root를 남기면서도 numerical score tie가 아니면 선택한 한 projection으로 후보를 검사한다. 같은 inlier 수에서 더 큰 SSE인 대안은 현재 선택되지 않지만 후보 선호를 뒤집을 수 있다.',
        '', '| 집합 | LOO 결정 수 | 저장 root entry 수 | A에서 모두 후보 선호 | A에서 N3 선호 해 존재 | B에서 모두 후보 선호 | B에서 N3 선호 해 존재 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,label in [('ALL_CALLED_LOO','실제 호출 전부'),('ORIGINAL_ADOPTED','기존 채택'),('ORIGINAL_NOT_ADOPTED','기존 미채택'),
        ('ADOPTED_CANDIDATE_BAD','기존 채택 BAD'),('ADOPTED_COORDINATE_BETTER','기존 채택·좌표 개선'),('ADOPTED_COORDINATE_WORSE','기존 채택·좌표 악화')]:
        d=g[key];a=d['pools']['DIM_ALLOWED_GE4'];b=d['pools']['DIM_ALLOWED_MAX_INLIER_COUNT']
        lines.append(f'| {label} | {d["decisions"]} | {d["raw_root_entries"]} | {a["all_candidate_preferred_decisions"]} | {a["any_native_preferred_decisions"]} | {b["all_candidate_preferred_decisions"]} | {b["any_native_preferred_decisions"]} |')
    lines+=['', 'A가 비어 있지 않은 결정은404개이며19개는 비었다. LOO state는 NEW401, 관측부족11, 수치 생성실패1, ambiguity3, 합의부족7이었다. **LOO NEW401은 최종 method의 NEW 산출 수가 아니다.** 기존 method의 전체245 최종 NEW240/fallback5/완전실패0은 그대로 유지된다. 미채택290은 LOO 불가19/ambiguity3/후보 residual cap137/N3 선호131로 나뉜다. 실패·미채택을 자료에서 제거하지 않았다.',
        '', '![전체133점 저장 LOO root 선호 범위](figures/ALL133_ROOT_PREFERENCE_ENVELOPES.png)',
        '', '모든133점의 원 선택 gain은 양수다. x는 해당 저장 root 집합의 최소 gain이다. x<0인 점에는 N3를 선호하는 다른 해가 있다. A에서9점 모두 후보 선호 중2점이 악화됐고, B에서는64점 중22점이 악화됐다. 색은 집합/선호 계산 뒤에 기존 DEV pixel reference로 분류했다. x축 symlog는 음수의 큰 범위를 읽기 쉽게 표시하기 위한 것이다. 점에 가려졌던 초기 legend/annotation은 데이터 영역 밖으로 옮겼으며 초기 그림·코드도 보존했다.',
        '', '**여러 해가 동의해도 틀리는 직접 사례.** A의 모든 저장 해가 후보를 선호하고 candidate residual≤8px인 경우에도 다음 두 점은 악화됐다.',
        '', '| frame/corner | N3 참조오차 px | 후보 참조오차 px | A root 수 | A 최소 gain px² | 최대 선택 projection 차이 px |',
        '|---|---:|---:|---:|---:|---:|---:|']
    stable_worse=[r for r in rows if r['original_adopted'] and r['coordinate_outcome']=='WORSE' and r['pools']['DIM_ALLOWED_GE4']['all_candidate_preferred']]
    assert len(stable_worse)==2
    for r in stable_worse:
        a=r['pools']['DIM_ALLOWED_GE4']
        lines.append(f'| {r["frame_id"]} / k{r["corner"]} | {r["native_error_px"]:.6f} | {r["candidate_error_px"]:.6f} | {a["entries"]} | {a["gain_min_px2"]:.6f} | {a["max_distance_from_selected_projection_px"]:.6f} |')
    lines+=['', '두 사례는 후보도8px 이내인 GOOD proxy지만 더 정확한 N3를 손상시켰다. A에서 모두 후보를 선호한 BAD 한 점은 eval_cad:1778653031275751936/k1이며 후보13.122491px/N3 16.189872px다. 좌표는 개선됐어도 BAD로 남았고, 기존 프레임 위치·회전은 둘 다 나빠졌다([V27 전체13표](../pallet_observation_uncertainty_20261011_v27/RESULT_KO.md)). 픽셀 분류, 픽셀 개선, 최종 자세 개선은 서로 다르다.',
        '', '**기존 채택 BAD13점 전부.** 원 선택 gain과 A/B 최소 gain을 함께 남긴다. 음수는 같은 집합에 N3를 선호하는 다른 root가 있다는 뜻이다. 이 표로 새 정책의 자세 성능을 계산하거나 주장하지 않는다.',
        '', '| frame/corner | 기존 선택 gain px² | A 최소 gain px² | B 최소 gain px² | B에서 N3 선호 witness의 SSE−선택 SSE px² | 후보 좌표 변화 |',
        '|---|---:|---:|---:|---:|---|']
    for r in summary['all_adopted_BAD_cases']:
        chosen=r['roots'][r['original_selected_root_index']];a=r['pools']['DIM_ALLOWED_GE4'];b=r['pools']['DIM_ALLOWED_MAX_INLIER_COUNT']
        w=r['native_preference_flip_witness']['DIM_ALLOWED_MAX_INLIER_COUNT'];gap='없음' if w is None else f'{w["witness_minus_selected_objective_px2"]:.6f}'
        lines.append(f'| {r["frame_id"]} / k{r["corner"]} | {chosen["gain_px2"]:.6f} | {a["gain_min_px2"]:.6f} | {b["gain_min_px2"]:.6f} | {gap} | {r["coordinate_outcome"]} |')
    lines+=['', '**전체 root 동의를 새 gate로 넣어서 해결됐다고 표시하지 않는다.** A에서 동의한9점 중7점은 개선됐지만 기존 개선80점 중73점은 A의 다른 해와 선호가 달랐다. B에서는 기존 개선80점 중38점이 선호가 달랐고, 동의한64점 중22점이 악화됐다. 분산된 root 안에는 낮은 합의·큰 SSE·수치상 동등한 entry·rank 미확인 대안도 있다. 이 계산은 동의의 진단이며 어떤 root까지 예측 불확실성 집합으로 볼지나 어느 policy가 전체 자세를 개선하는지의 답은 아니다. 최종 selected fit의 BAD 대응이 더 큰 합의를 얻는 [V23](../pallet_consensus_cause_audit_20261011_v23/RESULT_KO.md) 실패는 boundary 채택0에서도 존재해 별도로 남는다.',
        '', '**실제 고쳐야 할 부분과 남은 최소 실험.** 자기 가림 제외와 corner ID 정의는 유지한다. 검사한 source reference/ID 차이가8~19px 오차를 설명하지 않았고, 이번에는 LOO 단일해/다중해 선호도만으로 정확도를 인증할 수 없음을 확인했다. [V27](../pallet_observation_uncertainty_20261011_v27/RESULT_KO.md)의 후보 반경은 LOO 자세의 불확실성 반경이 아니며 실사 채택133점 중64점만 참조오차가 그 반경 안이었다. 따라서 후보 관측의 오차, 남은 N3 관측의 함께 치우친 오차, heldout prediction의 오차를 반영할 **독립적인 uncertainty 검증**이 먼저 필요하다. GT로 root나 반경을 사후 선택하는 gate는 배포 방법이 아니다.',
        '', '최소 다음 성능 실험은 고정 가중치·고정 자료로 검증 불확실성을 고려한 하나의 사전 고정 후보 채택 가설을 검증하고, 동일245장 전체 운용/공통 산출집합·T/R/ADDsym·NEW/fallback/fail·직접 가시 손상·H 재투영·실제 전체 경로 시간을 한 번 대조하는 것이다. 새 학습, seed 추가, 새 RGB가 필수는 아니다. 다만 root spread만을 오차 상한으로 쓰거나 이번 DEV GT로 B를 맞춰 성공을 보고하면 안 된다. 현재 읽기 전용 요청에서 새 정책 성능 재실험은 실행하지 않았으므로 **해결 완료가 아니다**.',
        '', '**확인된 과거 오류와 이번 감사 코드 오류를 혼동하지 않는다.** separate-head `targets()`의2164개 true query→NONE 오류는 실제 mesh/depth/wire POSITIVE/NONE/IGNORE로 이미 고쳤다. 이후 승인된3×3000 학습과 당시 준비/CPU gradient 상태를 구분하고, 이번 추가학습0이며 수정 전 IMAGE_ROLE0/319를 수정 후 결과로 주장하지 않는다. main N3 버그로 바꾸어 보고하지 않는다. 이번 root reader 첫 실행은 `best_candidate`와 같은 값의 entry가 하나뿐이라고 가정해 assertion 실패했다. 원 실행 코드·봉인 protocol·실패 receipt를 보존한 뒤 모든 matching indices와 entry를 유지하도록 감사기만 수정했다. 독립 raw reader에서306개 결정이 같은 best 진단값의 복수 entry를 갖는다고 확인했다. 기존 solver/선택 policy·root pool·학습 자료는 변경하지 않았다.',
        '', '**현재 전체245 성능은 그대로다.** [V21](../pallet_whole_u_closure_20261011_v21/RESULT_KO.md) mean T/R/ADDsym=9.745696cm/12.622492deg/18.580216cm, N3=9.754548cm/10.914842deg/17.028647cm, HYBRID ordinary=9.207154cm/10.613464deg/16.439686cm다. 실제 실사 수치는 기존 DEV GEOMETRIC_PROXY이며 독립 실측 physical truth는 아니다. 전체 분포·공통 산출집합·full path runtime은 기존 보고서에서 보존한다. 현재 범위는 Clean153/Moderate92이며 Severe74·역사적319는 변경하지 않았다.',
        '', '검산 자료: [245 frame/call/미호출 기록](FRAME_ROWS.jsonl.gz), [423결정·12178 entry 원행](DECISION_ROWS.jsonl.gz), [모든 분류·13 BAD 사례](SUMMARY.json), [실행 전 고정](PROTOCOL.json), [원본 불변·H∪k 검산](CHECKS.json). corrected primary와 독립 raw reader에서 전체245/423/12178를 다시 대조했다. compact reader는 각 root residual/gain, GT-free pool, envelope·witness·집계 및 V27 채택133을 독립 검산한다.',
        '', '```bash',
        'python3 -I -S -B scripts/research/pallet_loo_root_envelope_20261011_v28/check_public.py _docs/experiments/pallet_loo_root_envelope_20261011_v28 --checkout .',
        '```',
        '', '원 V21 prediction archive를 [기존 안내](../pallet_whole_u_closure_20261011_v21/PUBLIC_READER_KO.md)로 복원했다면 `--predictions RESTORED/evaluation/PREDICTIONS.jsonl.gz`를 추가해 모든423 decision/12178 root의 원 추출도 검사할 수 있다. 이 옵션 없는 compact PASS를 raw 원본 검산으로 표현하지 않는다. 실패 초기 code/protocol과 그림 수정 전 artifact도 공개해 재검토할 수 있게 했다.']
    text='\n'.join(lines)+'\n'
    with (out/'RESULT_KO.md').open('x') as f:f.write(text)
    save(out/'GOAL_STATUS.json',dict(status='active',accuracy_goal_achieved=False,
        progress='all423 saved LOO decision and12178 root audit; independent raw check; simple root agreement found insufficient',
        remaining='validated observation and heldout uncertainty; final consensus ranking; actual unchanged-policy/new-policy performance evidence',
        new_policy_or_pose_performance_experiment=False))
    save(ROOT/'REPORT_BUILD_CHECKS.json',dict(passed=True,report_lines=len(lines),all13_BAD_cases=True,
        report_sha256=hashlib.sha256(text.encode()).hexdigest(),accuracy_goal_achieved=False))
    print(json.dumps(dict(passed=True,report_lines=len(lines),all13_BAD_cases=True,accuracy_goal_achieved=False)))


if __name__=='__main__':main()

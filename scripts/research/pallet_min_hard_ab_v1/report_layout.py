"""Organize the preserved result material in directive section17 order."""
import re
from . import common as C


def assemble(text):
    chunks=re.split(r'^## \d+\. .*\n',text,flags=re.M)[1:]
    assert len(chunks)==9
    old={i+1:s.strip() for i,s in enumerate(chunks)}
    split=old[3].index('직접 클릭36개')
    sections=[('1. 한 줄 결론',old[1]),('2. 왜 이 실험을 했나','현재 best `S1 + GEO_LINEAR`에서 자연 가림 hard 오류가 남았다. 이전 clean-preservation adapter는 clean 회복과 severe 성능을 동시에 보존하지 못했다. 이번에는 모델 출력으로 고르지 않은 hard8장의 직접 클릭 감독이 고정 예산에서 도움이 되는지, hard 노출 효과와 좌표 source 효과를 분리했다.'),
              ('3. Model-blind difficulty tagging',old[3][:split]+'\n\n![촬영별 큐](figures/01_tagging_queue_recordings.png)\n\n![사람 난도 태그](figures/02_human_difficulty_distribution.png)'),
              ('4. Manual annotation',old[3][split:]),('5. Causal training contract',old[2]+'\n\n![동일 예산과 짝 조건](figures/05_pair_integrity.png)')]
    for i,g in enumerate(('CLEAN','MODERATE','SEVERE'),6):
        table='\n'.join(line for line in old[4].splitlines() if line.startswith('| 난도') or line.startswith('|---') or line.startswith('|'+g+'('))
        sections.append((f'{i}. {g}',table+f'\n\n![{g} PCK와 AUC](figures/{i:02d}_{g.lower()}_pck_auc.png)'))
    sections.append(('9. Candidate vs selector decomposition',old[5]+'\n\n![현재 선택과 사후 oracle](figures/09_oracle_vs_current.png)'))
    anchor='\n'.join(line for line in old[6].splitlines() if line.startswith('|그룹') or line.startswith('|---') or line.startswith('|ANCHOR'))
    sections.append(('10. Verified visible',anchor+'\n\n![fixed-ID 검증](figures/10_verified_visible.png)'))
    sections.append(('11. Pseudo vs manual','동일 RGB/박스/support에서 수동군은 수도레이블군보다 MODERATE·SEVERE PCK10 및 최종 AUC가 높다. 수도레이블만 추가한 군은 두 hard 난도의 PCK10이 BASE와 같고 최종 AUC는 악화됐다. 좌표 source의 추가 가치 신호는 있지만 기존 BASE의 hard 최종 AUC를 넘지는 못했다.\n\n![좌표 source별 비교](figures/12_pseudo_vs_manual.png)'))
    source='\n'.join(line for line in old[6].splitlines() if line.startswith('|그룹') or line.startswith('|---') or line.startswith('|SOURCE') or line.startswith('SOURCE256'))
    sections.append(('12. Synthetic source preservation',source+'\n\n![합성 보존](figures/13_source_preservation.png)'))
    sections.append(('13. Per-recording',old[7]+'\n\n![촬영별 차이](figures/11_recording_breakdown.png)'))
    sections.append(('14. Objective decision','**HARD_SUPERVISION_LOCALIZATION_SIGNAL_SELECTOR_LIMIT**. 사전 판정 규칙에 따라 hard의 PCK/oracle 개선이 CURRENT AUC로 전달되지 않은 경우다. **추가 hard labeling 중단, 기존 배포 S1 유지.**\n\n![판정 요약](figures/14_decision.png)'))
    sections.append(('15. Next step','새 학습·라벨 추가 없이 기존 예측에서 W/D 후보가 좋아졌지만 잘못 선택된 frame을 분석한다. 특히 MODERATE 선택 손실과 큰 오류 꼬리를 분리한다. 이 후속 원인분해는 이번 실험 범위 밖이며 자동 실행하지 않았다.'))
    sections.append(('16. Limitations',old[9]+'\n\n선택 사항인 10,000회 bootstrap은 미실행했다. 단일 seed의 유의성이나 독립 TEST 성능으로 해석하지 않는다.'))
    fit='\n'.join(line for line in old[6].splitlines() if line.startswith('|그룹') or line.startswith('|---') or line.startswith('|TRAIN'))
    sections.extend([('부록 A. 전체 2D 표',old[4]),('부록 B. 학습 적합도 — 일반화 아님',fit),('부록 C. 개선·악화 이미지',old[8]),
                     ('부록 D. 지시문 이행 및 재현','[항목별 검증표](DIRECTIVE_CHECKLIST.md) · [검사 상세](DIRECTIVE_AUDIT.json) · [14개 그림 manifest](FIGURE_MANIFEST.json) · [CLI 출력 스냅샷](FINAL_CLI_OUTPUT.txt) · [재개 회귀검사](RESUME_TESTS.json)\n\n마무리 단계에서 재학습·새 주석·재평가를 하지 않았다. 기존 체크포인트·예측·결과35개는 [보존 lock](FINALIZATION_INPUT_LOCK.json)과 대조했다. 사용자 승인 변경은 원래 조건의 PASS로 위장하지 않고 별도 표기한다.')])
    return '# Minimal hard supervision A/B — 최종 보고서\n\n'+'\n\n'.join('## '+h+'\n\n'+body for h,body in sections)+'\n'

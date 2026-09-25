"""Stage reports from completed locked results, never used for fitting."""
import argparse
from . import common as C

def stage1():
    j=C.read(C.sdoc(1)/'MODERATE_SELECTOR_DIAGNOSTIC.json');m=j['groups']['MODERATE']
    lines=['# Stage1 — Moderate W/D 후보 선택 진단','',
        '**Q1: YES. 현재 선택기가 더 좋은 후보를 실제로 버리는 사례가 있다.**','',
        f"고정 Moderate21, S1 PCK10={m['PCK10']:.6f}. CURRENT ADD AUC={m['CURRENT']['ADDsym_AUC']:.8f}, ORACLE={m['ORACLE']['ADDsym_AUC']:.8f}, gap={m['selection_loss']:.8f}. 기존 값을1e-7 이내 재현했다.",'',
        f"W/D parity {m['CURRENT']['axis_correct_count']}/21. 현재 axis wrong {m['axis_wrong']}장, alternate ADD-better {m['alternate_ADD_better']}장, 둘의 교집합 {m['current_wrong_alternate_better']}장. 현재 wrong인데 재투영오차는 오히려 더 낮은 사례 {m['wrong_lower_reprojection']}장. 두 후보의 invariant violation이 모두0인 사례 {m['both_invariant_zero']}/21장.",'',
        '재투영 잔차·좌우/상하 규칙만으로 올바른 W/D를 고르기 어려운 사례다. 이것이 synthetic 학습으로 해결된다는 뜻은 아니며 Stage2/3에서 따로 검증한다. real GT를 보고 특징을 고르지 않았고, Stage0 목록을 고정한 뒤 이 결과를 읽었다.','',
        '|frame|recording|current|best (사후)|best ADDnorm|ADD gap|category|','|---|---|---|---|---:|---:|---|']
    for r in j['moderate']:lines.append(f"|{r['id']}|{r['recording']}|{r['current']}|{r['best']}|{r['best_ADD_actual']:.5f}|{r['oracle_current_ADD_gap']:.5f}|{r['category']}|")
    for name in ('01_current_vs_oracle.png','02_selector_margin.png','03_component_distributions.png','04_wrong_candidate_cases.jpg','05_recording_breakdown.png'):
        lines+=['',f'![{name}](../figures/stage1/{name})']
    lines+=['','H10 role warning은 유지하지만 C4 remapping으로 점수를 만들거나 primary에서 사례를 제외하지 않았다. GT oracle은 비배포용이다. 이미 열람한 recording-disjoint DEV이며 독립 TEST가 아니다. 작은 ROI만 공개하고 원본/좌표캐시는 로컬에 보존한다.','']
    C.save(C.sdoc(1)/'STAGE1_REPORT_KO.md','\n'.join(lines))
    C.save(C.DOC/'REPORT_KO.md','# Selector recovery + frozen expert routing\n\nStage1 완료, Stage2–4 진행 중.\n\n[Stage1 보고서·이미지](stage1_diagnostic/STAGE1_REPORT_KO.md)\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=int);a=p.parse_args();globals()[f'stage{a.stage}']()

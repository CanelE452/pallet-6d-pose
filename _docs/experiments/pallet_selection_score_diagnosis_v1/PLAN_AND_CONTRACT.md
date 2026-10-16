# 고정 후보 선택 점수 진단 — 실행 계약

이번 분석은 원407de865의238장 고정 후보·관측·참조를 현재main2fe1969 소스와 해시로 연결한다. 네 집단114/10/22/46, 별도CAL46을 모두 보존한다. 기존 G10/G13 각family와 frozen-e 선택은 원순서·eligible/dedup·tie-break를 그대로 재현하며, 목적함수는 stage별로 분리한다.

학습·모델forward·PnP·수치최적화·후보생성·관측재생성·정답수정은0이다. 고정R/t의 원목적함수·잔차·제약과 기존 기하참조 오차만 평가한다. 원selector/비용parity는atol1e-9,rtol0이며 수치에 맞추어 완화하지 않는다. runtime profiler가 분석worker의 금지함수 진입을 감시하며 다른 프로세스나 원함수를 monkeypatch하지 않는다.

입력계약/원선택 재현/가산비용 복원 전량을 통과해야 항제외 진단을 수행한다. 실제활성 full_BI,partial_normal,outline을각각한번만제외하는G10단독·G13내부단독6경로를 성과표 작성 전에 INFLUENCE_PROTOCOL.json에 고정했다. prior실험·조합·점별삭제·가중치탐색은 없다. 원가중 총비용에서 해당 실제가중 항을뺀double값을사용하고 원정확tie-break를보존한다.

원CAL목록처음6장300retained후보에서처리량과정확scalar-only동등성을검사한뒤전량을진행한다. 전체고정bank11603후보를빠짐없이분해하고 실패32장만이아닌성공114/CAL/전체192의손익을보고한다. 새목적함수로후보를다시푼방법의성능으로주장하지않는다.

원 source·관측·후보·참조의 불일치 또는 해당비용/선택parity실패는관련분석만보류하고 독립분석은계속한다. 결측은NA와이유,semantic owner는기존미검증상태다. CPU분석단계elapsed7200초/GPU0/동시CPUworker최대3을유지한다. 비용계산은2worker로실행하고 root의순수기록분석에한worker여유를둔다.

새공개코드와작은장부·허용RGB그림·보고서는별도research/selection-score-diag-20261016-v1브랜치에게시한다. 기존main·원실험·논문·운영기본값·사용자수정은보존한다. 대형raw는해시연결만남기고복제공개하지않는다.
